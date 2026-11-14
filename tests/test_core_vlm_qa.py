from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from trace_vl.core_vlm.qa import VideoQALoss, VisualStep, grid_pool


class TinyBackbone:
    """Real tiny Qwen3.5 hybrid LM, random weights; plumbing/gradient test only."""

    def __init__(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                  linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
                                  linear_num_key_heads=2, linear_num_value_heads=2,
                                  layer_types=["linear_attention", "full_attention"])
        torch.manual_seed(18)
        self.model = nn.ModuleDict({"lm": Qwen3_5TextModel(config), "head": nn.Linear(32, 64, bias=False)})
        self.model.eval()
        self.device, self.dtype, self.hidden = torch.device("cpu"), torch.float32, 32
        self.vision_start, self.vision_end = 1, 2
        self.head_text, self.tail_text = "USER:", "ASSISTANT:"
        self.last = SimpleNamespace()

    def ids(self, text):
        return [3 + ord(c) % 61 for c in text]

    def embed_ids(self, ids):
        return self.model["lm"].embed_tokens(torch.tensor([ids], dtype=torch.long))

    def lm(self, embeds, positions, cache, use_cache):
        assert cache is None and not use_cache
        hidden = self.model["lm"](inputs_embeds=embeds, position_ids=positions[:, None],
                                  use_cache=False).last_hidden_state
        self.last.hidden = hidden
        return hidden

    def logits(self, hidden):
        return self.model["head"](hidden)


def sample():
    torch.manual_seed(42)
    steps = [VisualStep(torch.randn(6, 32, requires_grad=True), (2, 3), end, (end - .5, end))
             for end in (1., 2.)]
    return steps, {"question": "What changed?", "answer_text": "moved", "query_time": 2.}


@pytest.mark.parametrize("mode,count", [("full", 12), ("grid", 4)])
def test_answer_loss_reaches_early_video_and_language_model(mode, count):
    bb = TinyBackbone()
    model = VideoQALoss(bb, mode=mode, pool_grid=(1, 2))
    steps, record = sample()
    loss, meta = model(steps, record)
    assert meta["visual_tokens_total"] == count
    assert meta["answer_tokens"] == 5
    # Independent answer-only CE: no supervision on prompt or visual positions.
    expected = F.cross_entropy(bb.logits(bb.last.hidden[:, -5:]).reshape(5, 64),
                               torch.tensor(bb.ids("moved")))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(torch.isfinite(s.tokens.grad).all() and s.tokens.grad.abs().sum() > 0 for s in steps)
    lm_grads = [p.grad for p in bb.model["lm"].parameters() if p.grad is not None]
    assert lm_grads and all(torch.isfinite(g).all() for g in lm_grads)
    assert sum(g.abs().sum() for g in lm_grads) > 0


def test_qa_loss_changes_when_observed_video_changes():
    model = VideoQALoss(TinyBackbone())
    steps, record = sample()
    a, _ = model(steps, record)
    changed = [VisualStep(torch.zeros_like(s.tokens), s.grid, s.end, s.frame_times) for s in steps]
    b, _ = model(changed, record)
    assert abs(float((a - b).detach())) > 1e-6  # plumbing sensitivity, NOT learned video understanding


def test_answer_termination_is_supervised_and_generation_never_consumes_gold():
    bb = TinyBackbone()
    model = VideoQALoss(bb, answer_end_token_id=2)
    steps, row = sample()
    loss, meta = model(steps, row)
    assert meta["answer_tokens"] == 6
    expected = F.cross_entropy(bb.logits(bb.last.hidden[:, -6:]).reshape(6, 64),
                               torch.tensor(bb.ids("moved") + [2]))
    torch.testing.assert_close(loss, expected)
    first = model.generate(steps, row, max_new_tokens=3)
    second = model.generate(steps, {**row, "answer_text": "a completely different gold answer"},
                            max_new_tokens=3)
    assert first == second


def test_answer_loss_trains_lora_with_base_weights_frozen():
    from peft import LoraConfig, get_peft_model

    bb = TinyBackbone()
    # PEFT injects into the original module in place, as Backbone.add_lora does.
    get_peft_model(bb.model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                      target_modules=["q_proj", "v_proj"]))
    model = VideoQALoss(bb, mode="grid", pool_grid=(1, 2))
    loss, _ = model(*sample())
    loss.backward()
    params = dict(model.named_parameters())
    trainable = {n: p for n, p in params.items() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    assert sum(float(p.grad.abs().sum()) for p in trainable.values()) > 0
    assert all(p.grad is None for p in params.values() if not p.requires_grad)


def test_grid_bins_cover_each_patch_once_without_overlap():
    basis = torch.eye(35)
    pooled, rows, cols = grid_pool(basis, (5, 7), (2, 3))
    assert pooled.shape == (6, 35)
    assert torch.equal((pooled > 0).sum(0), torch.ones(35, dtype=torch.long))
    torch.testing.assert_close(pooled.sum(1), torch.ones(6))
    assert rows.tolist() == [0, 0, 0, 3, 3, 3]
    assert cols.tolist() == [0, 2, 5, 0, 2, 5]


@pytest.mark.parametrize("defect", ["future_frame", "future_step", "reverse", "wrong_grid", "empty"])
def test_invalid_observations_refused_before_language_model(defect):
    bb = TinyBackbone()
    model = VideoQALoss(bb)
    steps, record = sample()
    if defect == "future_frame":
        steps[-1].frame_times = (1.5, 2.1)
    elif defect == "future_step":
        record["query_time"] = 1.9
    elif defect == "reverse":
        steps.reverse()
    elif defect == "wrong_grid":
        steps[0].grid = (3, 3)
    else:
        steps = []
    with pytest.raises(ValueError):
        model(steps, record)
    assert not hasattr(bb.last, "hidden")


def test_mc_target_does_not_leak_into_prompt():
    model = VideoQALoss(TinyBackbone())
    steps, record = sample()
    record.update(options=["move left", "move right"], answer=0)
    p0, pos0, _, answer0, _ = model.prefix(steps, record)
    record["answer"] = 1
    p1, pos1, _, answer1, _ = model.prefix(steps, record)
    assert answer0 == "A" and answer1 == "B"
    assert torch.equal(p0, p1) and torch.equal(pos0, pos1)
