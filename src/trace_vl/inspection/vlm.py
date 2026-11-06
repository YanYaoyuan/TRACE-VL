"""Pretrained video VLM wrapper (default Qwen3-VL) for option scoring, captioning and LoRA SFT."""

from __future__ import annotations

import torch

from .evidence import LETTERS

LORA_TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def messages_for(segments, answer=None):
    """Interleaved user content; every text item ends with a newline so pieces do not run together."""
    content = []
    for item in segments:
        if item["type"] == "text":
            content.append({"type": "text", "text": item["text"] + "\n"})
        else:
            content.append({"type": "image", "image": item["image"]})
    messages = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class VLM:
    def __init__(
        self,
        name="Qwen/Qwen3-VL-2B-Instruct",
        device="cuda",
        dtype="bfloat16",
        unit=32,
        adapter=None,
        attn_implementation="sdpa",
        local_files_only=False,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(name, local_files_only=local_files_only)
        self._allow_exact_sizes(unit)
        self.model = AutoModelForImageTextToText.from_pretrained(
            name,
            dtype=self.dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        ).to(self.device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        tokenizer = self.processor.tokenizer
        self.letter_ids = []
        for letter in LETTERS:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"option letter {letter!r} is not a single token")
            self.letter_ids.append(ids[0])
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def _allow_exact_sizes(self, unit):
        # Packed images already have sides that are multiples of `unit`; keep the processor from
        # resizing them again so the planned token budget is the real one.
        processor = self.processor.image_processor
        low, high = 4 * unit * unit, 16384 * unit * unit
        if isinstance(getattr(processor, "size", None), dict):
            processor.size = {"shortest_edge": low, "longest_edge": high}
        for key, value in (("min_pixels", low), ("max_pixels", high)):
            if hasattr(processor, key):
                setattr(processor, key, value)

    def inputs(self, messages, generation: bool):
        batch = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=generation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        batch = batch.to(self.device)
        if "pixel_values" in batch:
            # self.model may be wrapped (PEFT, DDP) during training; use the dtype chosen at load time.
            batch["pixel_values"] = batch["pixel_values"].to(self.dtype)
        return batch

    def visual_tokens(self, batch) -> int:
        return int((batch["input_ids"] == self.image_token_id).sum())

    @torch.no_grad()
    def choose(self, segments, n_options: int):
        """Pick the option letter with the highest next-token log-probability."""
        batch = self.inputs(messages_for(segments), generation=True)
        logits = self.model(**batch, logits_to_keep=1).logits[0, -1].float()
        scores = logits.log_softmax(-1)[self.letter_ids[:n_options]]
        return int(scores.argmax()), scores.tolist(), self.visual_tokens(batch)

    @torch.no_grad()
    def generate(self, segments, max_new_tokens=48) -> str:
        batch = self.inputs(messages_for(segments), generation=True)
        output = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        new = output[0, batch["input_ids"].shape[1] :]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()

    def loss(self, segments, answer: str):
        """Cross-entropy on the assistant answer only."""
        batch = self.inputs(messages_for(segments, answer), generation=False)
        ids = batch["input_ids"][0].tolist()
        header, start = self.assistant_header, None
        for index in range(len(ids) - len(header), -1, -1):
            if ids[index : index + len(header)] == header:
                start = index + len(header)
                break
        if start is None:
            raise ValueError("assistant header not found in the chat template output")
        labels = batch["input_ids"].clone()
        labels[:, :start] = -100
        return self.model(**batch, labels=labels).loss, self.visual_tokens(batch)

    def add_lora(self, rank=16, alpha=32, dropout=0.05, gradient_checkpointing=True):
        from peft import LoraConfig, get_peft_model

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET)
        self.model = get_peft_model(self.model, config)
        return self.model
