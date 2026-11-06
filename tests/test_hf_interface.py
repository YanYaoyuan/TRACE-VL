"""Real HF API roundtrip with miniature local weights; no network or 2B download."""

import pytest
import torch

transformers = pytest.importorskip("transformers", reason="optional hf extra is not installed")

from trace_vl.config import ModelConfig  # noqa: E402
from trace_vl.models.model import TraceVL  # noqa: E402


@pytest.mark.parametrize(
    "query_encoding,prompt_format",
    [("embedding_mean", "raw"), ("last_hidden", "raw"), ("embedding_mean", "chat")],
)
def test_local_siglip_qwen3_forward_and_generation(tmp_path, query_encoding, prompt_format):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3ForCausalLM,
        SiglipImageProcessor,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    vision_dir, language_dir = tmp_path / "vision", tmp_path / "language"
    vision = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=32,
            patch_size=8,
        )
    )
    vision.save_pretrained(vision_dir)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(vision_dir)
    vocab = {
        token: i
        for i, token in enumerate(["<pad>", "<eos>", "<unk>", "what", "happened", "yes", "no", "watch"])
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} </{{ m['role'] }}> {% endfor %}"
        "{% if add_generation_prompt %}<assistant> {% endif %}"
    )
    tokenizer.save_pretrained(language_dir)
    language = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            tie_word_embeddings=True,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    language.save_pretrained(language_dir)
    cfg = ModelConfig(
        backend="hf",
        dim=32,
        vision_dim=32,
        language_dim=32,
        heads=4,
        st_layers=1,
        spatial_tokens=4,
        motion_tokens=2,
        state_slots=2,
        event_capacity=2,
        read_events=1,
        read_tokens=4,
        text_limit=32,
        image_size=32,
        freeze_vision=True,
        freeze_language=True,
        local_files_only=True,
        vision_model=str(vision_dir),
        language_model=str(language_dir),
        query_encoding=query_encoding,
        prompt_format=prompt_format,
    )
    model = TraceVL(cfg).train()
    assert bool(model.language.format.head) == (prompt_format == "chat")
    torch.testing.assert_close(model.language.encode_text([""], query_encoding), torch.zeros(1, 32))
    batch = {
        "frames": torch.rand(1, 2, 3, 32, 32),
        "timestamps": torch.tensor([[0.0, 0.5]]),
        "questions": ["what happened"],
        "answers": ["yes"],
        "rules": ["watch"],
    }
    result = model(batch)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.projector[0].weight.grad.abs().sum() > 0
    model.eval()
    answers, info = model.answer(result["state"].detach(), batch["questions"], max_new_tokens=2)
    assert len(answers) == 1 and info["valid"].any()
