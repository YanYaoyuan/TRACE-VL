"""Small offline backbones for verification and optional pretrained Hugging Face backbones."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ByteTokenizer:
    pad_token_id, bos_token_id, eos_token_id, vocab_size = 0, 1, 2, 259

    def encode(self, text: str, limit: int | None = None) -> list[int]:
        ids = [b + 3 for b in text.encode("utf-8")]
        return ids[:limit] if limit else ids

    def decode(self, ids) -> str:
        return bytes(int(i) - 3 for i in ids if 3 <= int(i) < 259).decode("utf-8", errors="replace")


class PromptFormat:
    """Token layout around the visual prefix: [head] [visual prefix] [question + suffix] [answer].

    `raw` reproduces the original layout (empty head, bare question). `chat` wraps the question in the
    tokenizer's chat template (or a plain Question/Answer template for the byte tokenizer) and marks the
    visual prefix, so a post-trained LLM sees the format it was tuned on.
    """

    def __init__(self, tokenizer, mode="raw", chat_template=None):
        self.tokenizer, self.mode = tokenizer, mode
        if mode == "raw":
            self.head, self.middle, self.suffix = [], [], []
            return
        if chat_template is None:
            head_text, middle_text, suffix_text = "Video:\n", "\nQuestion: ", "\nAnswer: "
        else:
            head_text, suffix_text = chat_template
            head_text, middle_text = head_text + "Video:\n", "\n"
        self.head = tokenizer.encode(head_text)
        self.middle = tokenizer.encode(middle_text)
        self.suffix = tokenizer.encode(suffix_text)

    def prompt(self, question, limit):
        # Truncate only the question so the template suffix that cues the answer always survives.
        return self.middle + self.tokenizer.encode(question, limit=limit) + self.suffix


def chat_template_parts(tokenizer):
    """Split a rendered single-turn chat prompt into the text before and after the user content."""
    marker = "\u0000TRACE_VL_QUESTION\u0000"
    messages = [{"role": "user", "content": marker}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if text.count(marker) != 1:
        raise ValueError("chat template did not render the user content exactly once")
    return tuple(text.split(marker))


def training_sequences(tokenizer, prompts, answers, limit, device, prompt_format=None):
    """Pad only at sequence ends; prompt and prefix positions never become loss targets."""
    sequences, labels = [], []
    for prompt, answer in zip(prompts, answers):
        q = (
            tokenizer.encode(prompt, limit=limit // 2)
            if prompt_format is None
            else prompt_format.prompt(prompt, limit // 2)
        )
        a = tokenizer.encode(answer, limit=max(1, limit - len(q) - 1)) + [tokenizer.eos_token_id]
        sequences.append(q + a)
        labels.append([-100] * len(q) + a)
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    targets = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        ids[i, : len(seq)] = torch.tensor(seq, device=device)
        targets[i, : len(lab)] = torch.tensor(lab, device=device)
        mask[i, : len(seq)] = 1
    return ids, targets, mask


class TinyVision(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, dim // 2, 4, stride=4), nn.GELU(), nn.Conv2d(dim // 2, dim, 2, stride=2), nn.GELU()
        )

    def forward(self, images):
        x = self.net(images)
        return x.flatten(2).transpose(1, 2), x.shape[-2:]


class HFVision(nn.Module):
    def __init__(self, name, local_files_only):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, SiglipVisionModel
        except ImportError as e:
            raise ImportError("Install trace-vl[hf] for pretrained backbones") from e
        self.processor = AutoImageProcessor.from_pretrained(
            name, local_files_only=local_files_only, use_fast=False
        )
        self.model = SiglipVisionModel.from_pretrained(name, local_files_only=local_files_only)
        self.patch_size = self.model.config.patch_size
        mean = self.processor.image_mean
        std = self.processor.image_std
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images):
        pixel_values = ((images - self.mean) / self.std).to(next(self.model.parameters()).dtype)
        out = self.model(pixel_values=pixel_values, return_dict=True).last_hidden_state
        grid = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        if out.shape[1] != grid[0] * grid[1]:
            raise ValueError("vision tokens do not match the configured patch grid")
        return out, grid


class TinyLanguage(nn.Module):
    def __init__(self, dim, text_limit, prompt_format="raw"):
        super().__init__()
        self.dim, self.text_limit = dim, text_limit
        self.tokenizer = ByteTokenizer()
        self.format = PromptFormat(self.tokenizer, prompt_format)
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, dim)
        self.position = nn.Embedding(text_limit + 1024, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, 3 * dim, dropout=0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.tokenizer.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def encode_text(self, texts, mode="embedding_mean"):
        rows = [self.tokenizer.encode(t, self.text_limit) or [1] for t in texts]
        device = self.embedding.weight.device
        return torch.stack([self.embedding(torch.tensor(r, device=device)).mean(0) for r in rows])

    def _logits(self, embeddings, mask=None):
        length = embeddings.shape[1]
        if length > self.position.num_embeddings:
            raise ValueError("tiny decoder context exceeded")
        p = torch.arange(length, device=embeddings.device)
        x = embeddings + self.position(p)[None]
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        x = self.decoder(x, mask=causal, src_key_padding_mask=None if mask is None else ~mask.bool())
        return self.head(self.norm(x))

    def _head(self, prefix):
        ids = torch.tensor(self.format.head, dtype=torch.long, device=prefix.device)
        return self.embedding(ids)[None].expand(prefix.shape[0], -1, -1)

    def loss(self, prefix, prompts, answers):
        ids, targets, mask = training_sequences(
            self.tokenizer, prompts, answers, self.text_limit, prefix.device, self.format
        )
        prefix = torch.cat([self._head(prefix), prefix], dim=1)
        x = torch.cat([prefix, self.embedding(ids)], dim=1)
        targets = torch.cat([targets.new_full(prefix.shape[:2], -100), targets], dim=1)
        mask = torch.cat([mask.new_ones(prefix.shape[:2]), mask], dim=1)
        logits = self._logits(x, mask)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, prefix, prompts, max_new_tokens=32):
        answers = []
        prefix = torch.cat([self._head(prefix), prefix], dim=1)
        for p, prompt in zip(prefix, prompts):
            ids = self.format.prompt(prompt, self.text_limit // 2) or [1]
            seq = torch.tensor(ids, device=p.device)[None]
            generated = []
            for _ in range(max_new_tokens):
                logits = self._logits(torch.cat([p[None], self.embedding(seq)], dim=1))
                nxt = int(logits[0, -1].argmax())
                if nxt == self.tokenizer.eos_token_id:
                    break
                generated.append(nxt)
                seq = torch.cat([seq, seq.new_tensor([[nxt]])], dim=1)
            answers.append(self.tokenizer.decode(generated))
        return answers


class HFTokenizer:
    def __init__(self, tokenizer):
        self.inner = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def encode(self, text, limit=None):
        return self.inner.encode(
            text, add_special_tokens=False, truncation=limit is not None, max_length=limit
        )


class HFLanguage(nn.Module):
    def __init__(self, name, local_files_only, text_limit, lora_rank=0, prompt_format="raw"):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install trace-vl[hf] for pretrained backbones") from e
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = HFTokenizer(tokenizer)
        template = (
            chat_template_parts(tokenizer) if prompt_format == "chat" and tokenizer.chat_template else None
        )
        self.format = PromptFormat(self.tokenizer, prompt_format, template)
        self.lm = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=local_files_only, torch_dtype="auto"
        )
        if lora_rank:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise ImportError("Install trace-vl[lora] for LoRA") from e
            self.lm = get_peft_model(
                self.lm,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=2 * lora_rank,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
        self.dim, self.text_limit = self.lm.config.hidden_size, text_limit

    def encode_text(self, texts, mode="last_hidden"):
        device = next(self.lm.parameters()).device
        nonempty = torch.tensor([bool(text.strip()) for text in texts], device=device)
        # Pure QA datasets have no monitoring rule. Avoid empty sequences in the decoder.
        fallback = self.tokenizer.inner.eos_token or "[empty]"
        texts = [text if text.strip() else fallback for text in texts]
        t = self.tokenizer.inner(
            texts, padding=True, truncation=True, max_length=self.text_limit, return_tensors="pt"
        ).to(device)
        if mode == "embedding_mean":
            x = self.lm.get_input_embeddings()(t.input_ids)
            pooled = (x * t.attention_mask[..., None]).sum(1) / t.attention_mask.sum(
                1, keepdim=True
            ).clamp_min(1)
            return pooled * nonempty[:, None]
        base = self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm
        x = base.model(**t, use_cache=False, return_dict=True).last_hidden_state
        indices = t.attention_mask.sum(1) - 1
        return x[torch.arange(x.shape[0], device=x.device), indices] * nonempty[:, None]

    def _with_head(self, prefix):
        ids = torch.tensor(self.format.head, dtype=torch.long, device=prefix.device)
        head = self.lm.get_input_embeddings()(ids)[None].expand(prefix.shape[0], -1, -1)
        return torch.cat([head, prefix.to(head.dtype)], dim=1)

    def loss(self, prefix, prompts, answers):
        ids, targets, mask = training_sequences(
            self.tokenizer, prompts, answers, self.text_limit, prefix.device, self.format
        )
        embeddings = self.lm.get_input_embeddings()(ids)
        prefix = self._with_head(prefix).to(embeddings.dtype)
        x = torch.cat([prefix, embeddings], dim=1)
        targets = torch.cat([targets.new_full(prefix.shape[:2], -100), targets], dim=1)
        mask = torch.cat([mask.new_ones(prefix.shape[:2]), mask], dim=1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return self.lm(
            inputs_embeds=x,
            attention_mask=mask,
            position_ids=positions,
            labels=targets,
            use_cache=False,
            return_dict=True,
        ).loss

    @torch.no_grad()
    def generate(self, prefix, prompts, max_new_tokens=64):
        answers = []
        prefix = self._with_head(prefix)
        for p, prompt in zip(prefix, prompts):
            ids = self.format.prompt(prompt, self.text_limit // 2) or [self.tokenizer.eos_token_id]
            ids = torch.tensor(ids, device=p.device)[None]
            embedding = self.lm.get_input_embeddings()(ids)
            x = torch.cat([p[None].to(embedding.dtype), embedding], dim=1)
            result = self.lm.generate(
                inputs_embeds=x,
                attention_mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.long),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            answers.append(self.tokenizer.inner.decode(result[0], skip_special_tokens=True))
        return answers
