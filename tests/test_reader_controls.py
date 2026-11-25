import itertools

import pytest
import torch

from trace_vl.reader_controls import analysis, behavior, policies, run

TIMES = [[2.0 * i, 2.0 * i + 0.5] for i in range(36)]
ROW = {"id": "q", "query_time": 80.0, "evidence": [20.0, 25.0]}


def test_policies_freeze_budgets_and_share_pair_fill():
    reads, audit = policies.build(ROW, TIMES, list(range(8)))
    again, _ = policies.build(ROW, TIMES, list(range(8)))
    assert reads == again
    hits = policies.evidence_hits(TIMES, ROW["evidence"])
    assert hits == [10, 11, 12]
    for name, read in reads.items():
        size = {"recent4": 4, "all_bank": 36}.get(name, 8)
        assert len(read) == size and read == sorted(set(read))
    for seed in policies.SEEDS:
        assert set(hits) <= set(reads[f"bank_evidence8:{seed}"])
        a, b = set(reads[f"pair_evidence:{seed}"]), set(reads[f"pair_distractor:{seed}"])
        assert a - b == set(hits) and len(b - a) == 3 and not b & set(hits)
    assert all(abs(g) <= policies.CALIPER_SECONDS for g in audit["pair"]["age_difference_seconds"])


def test_missing_evidence_is_kept_and_matching_failure_is_reported():
    reads, audit = policies.build({**ROW, "evidence": [0.6, 0.9]}, TIMES, list(range(8)))
    assert audit["pair"]["failure"] == "no_stored_evidence"
    assert len(reads["bank_evidence8:17"]) == 8 and "pair_evidence:17" not in reads
    _, far = policies.build({**ROW, "evidence": [0.0, 100.0]}, TIMES, list(range(8)))
    assert far["pair"]["failure"] == "no_match_within_caliper"


def test_spacing_keeps_distinct_steps():
    for n in range(8, 40):
        got = policies.spaced(list(range(n)))
        assert len(set(got)) == 8 and got == sorted(got) and got[0] == 0 and got[-1] == n - 1


def test_matching_is_maximal_then_nearest():
    times = [[float(t), float(t)] for t in (0, 3, 4, 9, 10, 21)]
    hits, others = [1, 3, 5], [0, 2, 4]
    expected = None
    for k in range(3, -1, -1):
        options = []
        for left in itertools.combinations(hits, k):
            for right in itertools.permutations(others, k):
                gaps = [abs(times[a][-1] - times[b][-1]) for a, b in zip(left, right)]
                if all(g <= 5 for g in gaps):
                    options.append(sum(gaps))
        if options:
            expected = (k, min(options))
            break
    pairs = policies.match_ages(hits, others, times, caliper=5)
    assert (len(pairs), sum(abs(times[a][-1] - times[b][-1]) for a, b in pairs)) == expected


def test_blank_read_replaces_values_only():
    features = torch.randn(36, 16, 8)
    blank = torch.zeros(16, 8)
    values, chosen, _ = run.fixed([1, 5], None)(features, TIMES, None, 80.0)
    empty, same, _ = run.fixed([1, 5], blank)(features, TIMES, None, 80.0)
    assert torch.equal(values, features[[1, 5]]) and torch.equal(chosen, same)
    assert empty.shape == values.shape and not empty.any()


def record(weights, source, good):
    reads, _ = policies.build({**ROW, "id": source}, TIMES, list(range(8)))
    if source == "s0":
        reads = {k: v for k, v in reads.items() if not k.startswith("pair_")}
    logits = {"text_only": [0.0, 0.0]}
    for name in reads:
        logits[name] = [good if name.startswith(("bank_evidence8", "all_bank")) else 0.0, 0.0]
        logits["blank:" + name] = [0.0, 0.0]
    return {"weights": weights, "source_group": source, "gold": 0, "logits": logits}


def test_analysis_averages_seeds_keeps_sources_and_reports_dependence_loss():
    sources = [f"s{i}" for i in range(12)]
    records = [
        record(w, s, (2.0 if w == "start" else 0.0) + (0.01 * i if w == "start" else 0.0))
        for w in analysis.WEIGHTS for i, s in enumerate(sources)
    ]
    result = analysis.analyze(records, sources)
    start = result["weights"]["start"]["contrasts"]
    assert start["evidence_vs_text"]["nll"]["sources"] == 12
    assert start["pair_content"]["nll"]["sources"] == 11
    out = analysis.outcome(result)
    assert out["status"].startswith("headroom_supported")
    changed = out["trained_minus_start"]["trained_learned"]["evidence_content"]
    assert changed == "reduced_measured_visual_dependence"
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze(records[1:], sources)


def test_seeds_are_averaged_as_metrics_within_source():
    r = {"gold": 0, "logits": {"random8:17": [1.0, 0.0], "random8:29": [0.0, 1.0]}}
    values = analysis.arm_values(r)["random8"]
    assert values["acc"] == 0.5
    expected = sum(analysis.question_metrics(x, 0)["nll"] for x in r["logits"].values()) / 2
    assert values["nll"] == pytest.approx(expected)


def test_text_accuracy_triggers_scrutiny_without_retiring_the_cohort():
    sources = [f"s{i}" for i in range(12)]
    records = [record(w, s, 2.0) for w in analysis.WEIGHTS for s in sources]
    for r in records:
        r["logits"]["text_only"] = [0.5, 0.0]
    out = analysis.outcome(analysis.analyze(records, sources))
    assert out["text_only_scrutiny"] and out["status"].startswith("headroom_supported")


def test_behavior_probe_separates_question_driven_and_age_driven_selection():
    def content(features, _times, query, _query_time):
        return (features.mean(1) @ query).topk(8).indices

    def oldest(_features, _times, _query, _query_time):
        return range(8)

    good, bad = behavior.probe(content), behavior.probe(oldest)
    assert behavior.accepted(good) and not behavior.accepted(bad)
    assert bad["question_change_rate"] == 0


def test_score_question_packs_every_control_through_the_native_path(monkeypatch):
    from test_core_vlm_native import tiny_example

    from trace_vl.core_vlm.native import NativeVideoQALoss

    bb, _, _ = tiny_example()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]

        def __call__(self, *, videos, text, **kwargs):
            n = len(videos[0]) // 2
            ids = torch.tensor([[5] + ([1] + [3] * 16 + [2, 8]) * n + [6, 7]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.where(ids == 3, 2, 0),
                "video_grid_thw": torch.tensor([[n, 8, 8]]),
            }

    bb.processor = Processor()
    monkeypatch.setattr(bb, "ids", lambda text: [5, 6] if len(text) > 1 else [9 + ord(text) - 65])
    entry = {
        "features": torch.randn(36, 16, bb.hidden), "times": TIMES, "hw": [128, 128], "grid": [4, 4],
        "fps": 2.0, "frame_count": 162, "indices": [[4 * i, 4 * i + 1] for i in range(36)],
    }
    row = {**ROW, "question": "Which came first?", "options": ["a", "b"], "answer": 0}
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    reads = {"latest8": list(range(28, 36)), "recent4": [32, 33, 34, 35], "all_bank": list(range(36))}
    logits, tokens = run.score_question(bb, net, entry, row, reads, torch.zeros(16, bb.hidden))
    assert tokens["latest8"] == tokens["blank:latest8"] == 128 and tokens["all_bank"] == 576
    assert set(logits) == {"text_only", *reads, *(f"blank:{k}" for k in reads)}
    assert all(len(v) == 2 for v in logits.values())
