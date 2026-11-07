"""Qwen3.5-2B wrapper: per-step vision encoding, embeddings, incremental LM calls, hybrid cache access.

Verified against transformers 5.20.2 (`models/qwen3_5/modeling_qwen3_5.py`):
- `model.model.visual(pixel_values, grid_thw)` returns `pooler_output` = merged tokens (N, 2048);
- `model.model.language_model(inputs_embeds, position_ids[3,B,L], past_key_values, use_cache)`;
- full-attention layers cache `DynamicLayer.keys/values` [B, kv_heads, T, head_dim] with RoPE applied,
  so evicting entries keeps the remaining positions valid; linear-attention (Gated DeltaNet) layers
  keep `recurrent_states` / `conv_states`, which are never evicted.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager

import numpy as np
import torch

from .config import BackboneConfig, StreamConfig
from .positions import PositionCounter

LORA_TARGET = (
    r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))"
)


def frame_size(height: int, width: int, pixels: int, unit: int) -> tuple[int, int]:
    """Aspect-preserving (h, w), multiples of `unit`, with about `pixels` pixels."""
    scale = math.sqrt(pixels / (height * width))
    h = max(unit, round(height * scale / unit) * unit)
    w = max(unit, round(width * scale / unit) * unit)
    return h, w


@contextmanager
def functional_linear_cache():
    """Make Gated DeltaNet cache updates functional (assign, not copy_) for gradient-carrying streaming.

    transformers updates the recurrent/conv states in place to keep static addresses for CUDA graphs. With
    autograd that overwrites the initial state the GDN kernel saved for backward ("modified by an inplace
    operation"). Stage 2 trains through cached streaming forwards, so inside this context the updates assign
    new tensors instead. Inference does not need it.
    """
    from transformers.cache_utils import LinearAttentionLayer

    original_recurrent = LinearAttentionLayer.update_recurrent_state
    original_conv = LinearAttentionLayer.update_conv_state

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
        if not self.has_previous_state[state_idx]:
            full = conv_states
            self.has_previous_state[state_idx] = True
            if full.shape[-1] < self.conv_kernel_size[state_idx]:
                full = torch.nn.functional.pad(full, (self.conv_kernel_size[state_idx] - full.shape[-1], 0), value=0)
        else:
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        self.conv_states[state_idx] = full[..., -self.conv_kernel_size[state_idx] :]
        return full

    LinearAttentionLayer.update_recurrent_state = update_recurrent_state
    LinearAttentionLayer.update_conv_state = update_conv_state
    try:
        yield
    finally:
        LinearAttentionLayer.update_recurrent_state = original_recurrent
        LinearAttentionLayer.update_conv_state = original_conv


def detach_cache(cache):
    """Detach every cached tensor in place on the cache object (truncated backprop between chunks)."""
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None and torch.is_tensor(layer.keys):
            layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
        for name in ("recurrent_states", "conv_states"):
            states = getattr(layer, name, None)
            if isinstance(states, dict):
                for k, v in states.items():
                    if torch.is_tensor(v):
                        states[k] = v.detach()


def branch_cache(cache):
    """Fork for a question during training: share the (large) full-attention KV tensors, clone the small GDN
    states. Safe because KV updates concatenate into new tensors and GDN states are cloned (autograd-friendly,
    unlike deepcopy of non-leaf tensors)."""
    fork = copy.copy(cache)
    fork.layers = []
    for layer in cache.layers:
        clone = copy.copy(layer)
        for name in ("recurrent_states", "conv_states", "is_conv_states_initialized", "is_recurrent_states_initialized",
                     "has_previous_state", "conv_kernel_size"):  # fmt: skip
            value = getattr(layer, name, None)
            if isinstance(value, dict):
                setattr(clone, name, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in value.items()})
        fork.layers.append(clone)
    return fork


class Backbone:
    def __init__(self, cfg: BackboneConfig, stream: StreamConfig, device="cuda"):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.cfg, self.stream, self.device = cfg, stream, torch.device(device)
        self.dtype = getattr(torch, cfg.dtype)
        self.processor = AutoProcessor.from_pretrained(cfg.name, local_files_only=cfg.local_files_only)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.name,
            dtype=self.dtype,
            attn_implementation=cfg.attn_implementation,
            local_files_only=cfg.local_files_only,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.peft = None
        config = self.model.config
        self.text_config = config.text_config
        self.hidden = self.text_config.hidden_size
        self.layer_types = list(self.text_config.layer_types)
        self.full_layers = [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]
        self.linear_layers = [i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"]
        tokenizer = self.processor.tokenizer
        self.tokenizer = tokenizer
        self.vision_start = config.vision_start_token_id
        self.vision_end = getattr(config, "vision_end_token_id", None) or tokenizer.convert_tokens_to_ids(
            "<|vision_end|>"
        )
        self.video_token = config.video_token_id
        self.head_text, self.tail_text = self._chat_parts()

    # ------------------------------------------------------------------ training
    def add_lora(self, rank: int, alpha: int, dropout: float = 0.05, adapter: str | None = None):
        """LoRA on the LM's attention and Gated DeltaNet projections (vision stays frozen).

        PEFT injects the adapters into the base modules in place, so `self.model` keeps working and
        `self.peft.disable_adapter()` gives the frozen-backbone teacher.
        """
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter:
            self.peft = PeftModel.from_pretrained(self.model, adapter, is_trainable=True)
        else:
            config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET)
            self.peft = get_peft_model(self.model, config)
        return self.peft

    def enable_checkpointing(self):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.language_model.train()  # checkpointing is only active in training mode (no dropout in Qwen3.5)

    # ------------------------------------------------------------------ modules
    @property
    def visual(self):
        return self.model.model.visual

    @property
    def language_model(self):
        return self.model.model.language_model

    @property
    def lm_head(self):
        return self.model.lm_head

    def embed_ids(self, ids: list[int] | torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device).view(1, -1)
        return self.model.get_input_embeddings()(ids)

    def ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _chat_parts(self) -> tuple[str, str]:
        marker = "\u0000TRACE_V3\u0000"
        messages = [{"role": "user", "content": marker}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.cfg.thinking
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text.count(marker) != 1:
            raise ValueError("chat template did not render the user content exactly once")
        head, tail = text.split(marker)
        return head, tail

    # ------------------------------------------------------------------ vision
    def frame_shape(self, height: int, width: int) -> tuple[int, int]:
        return frame_size(height, width, self.stream.pixels_per_frame, self.stream.unit)

    def resize_step(self, frames: np.ndarray) -> np.ndarray:
        """Resize a step's frames to the configured pixel budget (sides multiple of `unit`)."""
        import cv2

        h, w = self.frame_shape(frames.shape[1], frames.shape[2])
        if (frames.shape[1], frames.shape[2]) == (h, w):
            return frames
        return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])

    @torch.no_grad()
    def preprocess(self, frames: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """frames: uint8 [2, H, W, 3] already resized to multiples of `unit`."""
        if frames.ndim != 4 or frames.shape[0] != 2:
            raise ValueError("a step is exactly 2 frames [2, H, W, 3]")
        out = self.processor.video_processor(
            videos=[frames], do_sample_frames=False, do_resize=False, return_tensors="pt"
        )
        return out["pixel_values_videos"], out["video_grid_thw"]

    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Merged visual tokens for one step: [h*w, hidden] in raster order of the merged grid."""
        out = self.visual(pixel_values.to(self.device, self.dtype), grid_thw=grid_thw.to(self.device))
        return out.pooler_output

    def merged_grid(self, grid_thw: torch.Tensor) -> tuple[int, int]:
        merge = self.visual.spatial_merge_size
        return int(grid_thw[0, 1]) // merge, int(grid_thw[0, 2]) // merge

    # ------------------------------------------------------------------ language model
    def new_cache(self):
        from transformers import DynamicCache

        return DynamicCache(config=self.text_config)

    def lm(self, embeds: torch.Tensor, positions: torch.Tensor, cache=None, use_cache=True):
        """embeds [1, L, d]; positions [3, L]. Returns last hidden states [1, L, d]."""
        out = self.language_model(
            inputs_embeds=embeds,
            position_ids=positions.view(3, 1, -1).to(self.device),
            past_key_values=cache,
            use_cache=use_cache,
        )
        return out.last_hidden_state

    def lm_taps(self, embeds: torch.Tensor, positions: torch.Tensor, cache=None, use_cache=True, taps=()):
        """Like `lm`, also returning the outputs of decoder layers `taps`: ([1, L, d], [T, 1, L, d] or None).

        Uses forward hooks on the decoder layers, so it does not depend on how a transformers version
        indexes `output_hidden_states`.
        """
        if not taps:
            return self.lm(embeds, positions, cache, use_cache), None
        captured, handles = {}, []
        for index in taps:
            if not 0 <= index < len(self.layer_types):
                raise ValueError(f"tap layer {index} outside 0..{len(self.layer_types) - 1}")

            def hook(_module, _inputs, output, index=index):
                captured[index] = output[0] if isinstance(output, tuple) else output

            handles.append(self.language_model.layers[index].register_forward_hook(hook))
        try:
            last = self.lm(embeds, positions, cache, use_cache)
        finally:
            for handle in handles:
                handle.remove()
        return last, torch.stack([captured[i] for i in taps])

    def gdn_states(self, cache, layers) -> torch.Tensor | None:
        """Recurrent (associative-memory) states of the given Gated DeltaNet layers: [G, H, K, V] or None."""
        states = []
        for index in layers:
            if self.layer_types[index] != "linear_attention":
                raise ValueError(f"layer {index} is not a Gated DeltaNet layer")
            state = self.recurrent_state(cache, index)
            if state is None:
                return None
            states.append(state[0])
        return torch.stack(states) if states else None

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Mean last-layer hidden state of each text, each in its own LM pass (no cache): [len, d]."""
        pooled = []
        for text in texts:
            ids = self.ids(text) or [self.tokenizer.eos_token_id]
            hidden = self.lm(self.embed_ids(ids), PositionCounter().text(len(ids), self.device), cache=None,
                             use_cache=False)  # fmt: skip
            pooled.append(hidden[0].float().mean(0))
        return torch.stack(pooled)

    def letter_logprobs(self, hidden: torch.Tensor, letter_ids: list[int]) -> torch.Tensor:
        """Log-probs of option letters at one position, without bf16 output quantization.

        bf16 logits of magnitude ~16-32 are quantized in steps of 0.125, so close options tie exactly and
        argmax picks by index (PR #3 control: top-2 margins 0.0 / 0.125). The letter logits are computed in
        fp32 from the hidden state and the letter rows of lm_head; the full-vocabulary logsumexp (a shared
        constant across letters) comes from the regular logits.
        """
        hidden = hidden.reshape(-1)
        rows = self.lm_head.weight[letter_ids].float()
        letters = rows @ hidden.float()
        normalizer = self.lm_head(hidden[None].to(self.lm_head.weight.dtype)).float().logsumexp(-1)
        return letters - normalizer

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    # ------------------------------------------------------------------ hybrid cache
    def kv_length(self, cache) -> int:
        return cache.get_seq_length() if cache is not None else 0

    def evict(self, cache, keep: torch.Tensor):
        """Keep only `keep` (bool [T] or long indices) in every full-attention layer."""
        if keep.dtype == torch.bool:
            keep = keep.nonzero().flatten()
        keep = keep.to(self.device)
        for index in self.full_layers:
            layer = cache.layers[index]
            if getattr(layer, "keys", None) is None or layer.keys.numel() == 0:
                continue
            layer.keys = layer.keys.index_select(-2, keep)
            layer.values = layer.values.index_select(-2, keep)

    def recurrent_state(self, cache, layer: int | None = None):
        layer = self.linear_layers[0] if layer is None else layer
        states = getattr(cache.layers[layer], "recurrent_states", None)
        return None if states is None else states.get(0)

    def kv_bytes(self, cache) -> int:
        total = 0
        for index in self.full_layers:
            layer = cache.layers[index]
            if getattr(layer, "keys", None) is not None:
                total += layer.keys.numel() * layer.keys.element_size() * 2
        return total

    def state_bytes(self, cache) -> int:
        total = 0
        for index in self.linear_layers:
            layer = cache.layers[index]
            for states in (getattr(layer, "recurrent_states", {}), getattr(layer, "conv_states", {})):
                for value in (states or {}).values():
                    if value is not None:
                        total += value.numel() * value.element_size()
        return total

    @staticmethod
    def fork(cache):
        """Independent copy (questions must not leak into the stream)."""
        return copy.deepcopy(cache)
