"""Regression checks for causal evidence and the matched answer path."""
import json
import zipfile
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from trace_vl.core_vlm.qa import VisualStep
from trace_vl.history_reader.data import populations, tasks
from trace_vl.history_reader.model import IntervalReader, VisualHistory, evidence_indices


def captions():
    return {"duration": 30, "timestamps": [[0, 3], [5, 8], [12, 15]],
            "sentences": ["A person opens a door.", "The person washes a cup.", "They put the cup down."]}


def test_captions_ground_order_and_hold_out_by_source(tmp_path):
    raw = captions()
    rows = tasks(raw, "abcdefghijk", "dev", 17, 4, 30, 2)
    assert {row["task"] for row in rows} == {"before", "after"}
    for row in rows:
        gold_index = 0 if row["task"] == "before" else 2
        assert row["options"][row["answer"]] == raw["sentences"][gold_index]
        assert row["evidence"] == raw["timestamps"][gold_index]
        assert row["evidence"][1] <= row["query_time"] - 5
        assert row["observation"]["end"] == row["query_time"]
    archive = tmp_path / "captions.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("train.json", json.dumps({f"v_{i:011d}": raw for i in range(80)}))
        z.writestr("val_1.json", json.dumps({"v_00000000000": raw}))
        z.writestr("val_2.json", "{}")
    args = SimpleNamespace(seed=17, recent=4, lookback=30, block=2)
    result = populations(archive, {"00000000001"}, args)
    assert result == populations(archive, {"00000000001"}, args)
    assert len(result["train"]) + len(result["dev"]) == 78
    assert result["train"] and result["dev"]
    assert not result["train"].keys() & result["dev"].keys()
    assert {"00000000000", "00000000001"}.isdisjoint(result["train"].keys() | result["dev"].keys())


def test_history_is_bounded_and_selects_a_disjoint_matched_block():
    bank = VisualHistory(recent=2, block=2, capacity=2)
    for i in range(8):
        bank.append(VisualStep(torch.full((2, 5), float(i)), (1, 2), i + .5, (float(i), i + .5)))
    bank_tokens, spans = bank.tensors()
    assert [step.end for step in bank.recent] == [6.5, 7.5]
    assert spans.tolist() == [[2, 3.5], [4, 5.5]]
    assert bank_tokens.shape == (2, 8, 5)
    assert evidence_indices(bank.slots, (2.3, 3.4), "sample0") == (0, 1)
    assert evidence_indices(bank.slots, (2.3, 5.4), "sample0") is None
    with pytest.raises(ValueError, match="increasing"):
        bank.append(bank.recent[-1])


@pytest.mark.parametrize("mode", ["cosine", "learned"])
def test_reader_preserves_base_logits_and_has_answer_gradient(mode):
    torch.manual_seed(4)
    model = IntervalReader(16, 8, mode)
    q, mem, times, heads = torch.randn(16), torch.randn(3, 8, 16), torch.randn(3, 2), torch.randn(4, 16)
    logits, scores, index = model(q, mem, times, heads=heads)
    torch.testing.assert_close(logits, heads @ q, rtol=0, atol=0)
    empty, _, index_empty = model(q, mem[:0], times[:0], heads=heads)
    torch.testing.assert_close(empty, logits, rtol=0, atol=0)
    assert index_empty is None
    assert index == scores.argmax()
    F.cross_entropy(logits[None], torch.tensor([0])).backward()
    assert model.gate.grad.abs().item() > 0
    if mode == "learned":
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.gate.fill_(.1)
        logits, _, _ = model(q, mem, times, heads=heads)
        F.cross_entropy(logits[None], torch.tensor([0])).backward()
        assert model.key.weight.grad.norm().item() > 0
        assert model.query.weight.grad.norm().item() > 0
