"""Stage-2 memory pilot (experiments.md §14): does LEARNED retention beat mass / FIFO at the same KV budget?

Splits are frozen at source-video level BEFORE streams are stitched; every video the Stage 1 checkpoint saw
(its train and validation manifests) is excluded, so dev/test streams contain only unseen videos.

Comparison design: ONE Stage-2 model (B4 representation, trained with the TKM policy) is evaluated under several
memory policies at the same budget; only the policy changes at evaluation. A second, separate question (does
Stage-2 training help at all?) compares the Stage-1 B4 checkpoint under the heuristic policies.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .data import read_jsonl, write_jsonl

SPLITS = ("train", "dev", "test")


def split_of(video: str, seed: int, fractions: tuple[float, float, float]) -> str:
    u = int(hashlib.sha256(f"stage2:{seed}:{video}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < fractions[0] else "dev" if u < fractions[0] + fractions[1] else "test"


def make_splits(records: list[str], exclude: list[str], out_dir: str, fractions=(0.8, 0.1, 0.1), seed: int = 7,
                max_videos: dict | None = None) -> dict:  # fmt: skip
    """MC records (convert-llava-mc output) -> train/dev/test record files by source video; videos listed in
    `exclude` manifests (Stage 1 train/val) are removed first. Writes <split>.jsonl and manifest.json."""
    out = Path(out_dir)
    excluded = {r["video"] for path in exclude for r in read_jsonl(path)}
    rows = [r for path in records for r in read_jsonl(path) if r.get("video") and r["video"] not in excluded
            and r.get("question") and r.get("options")]  # fmt: skip
    by_split = defaultdict(list)
    for r in rows:
        split = split_of(r["video"], seed, fractions)
        by_split[split].append({**r, "split": split})
    rng = random.Random(seed)
    for split, cap in (max_videos or {}).items():  # seeded per-split video cap (keeps all questions of kept videos)
        pool = sorted({r["video"] for r in by_split[split]})
        if cap is not None and len(pool) > cap:
            kept = set(rng.sample(pool, cap))
            by_split[split] = [r for r in by_split[split] if r["video"] in kept]
    videos = {s: {r["video"] for r in by_split[s]} for s in SPLITS}
    if videos["train"] & videos["dev"] or videos["train"] & videos["test"] or videos["dev"] & videos["test"]:
        raise AssertionError("splits share a source video")
    manifest = {"records": [str(p) for p in records], "excluded_manifests": [str(p) for p in exclude],
                "excluded_videos": len(excluded), "fractions": list(fractions), "seed": seed,
                "max_videos": max_videos or {}, "splits": {}}  # fmt: skip
    for s in SPLITS:
        write_jsonl(out / f"{s}.jsonl", sorted(by_split[s], key=lambda r: r["id"]))
        manifest["splits"][s] = {"questions": len(by_split[s]), "videos": len(videos[s]),
                                 "sha256": hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest()}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_prep_request(target: str | Path, request: dict) -> None:
    """Record the data-preparation request; an existing directory prepared differently is refused (no stale reuse)."""
    target = Path(target)
    if target.exists():
        previous = json.loads(target.read_text())
        if previous != request:
            raise ValueError(f"{target.parent} was prepared with {previous}, not {request}; use a new output directory")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(request, indent=2) + "\n")


DEV_RULE = "dev-hash-prefix-v1"


def expand_dev_split(records: list[str], exclude: list[str], reuse_dir: str | Path, out_dir: str | Path,
                     dev_cap: int | None = None, seed: int = 7, fractions=(0.8, 0.1, 0.1)) -> dict:  # fmt: skip
    """Dev-only expansion that PRESERVES a previous run's train and frozen test manifests.

    `reuse_dir` is an earlier `make_splits` output (train/dev/test.jsonl + manifest.json). Its train.jsonl and
    test.jsonl are verified against their recorded hashes and copied byte for byte. The dev split is rebuilt from
    the videos the same frozen source-video split assigns to dev (same seed and fractions, Stage-1 exclusions),
    ordered by sha256(f"{seed}:dev:{video}") and capped at `dev_cap` (None = all available): a prefix rule, so a
    larger cap only adds videos, and train/test are never resampled (make_splits' sequential RNG would)."""
    import shutil

    reuse, out = Path(reuse_dir), Path(out_dir)
    old = json.loads((reuse / "manifest.json").read_text())
    if [float(f) for f in old["fractions"]] != [float(f) for f in fractions] or old["seed"] != seed:
        raise ValueError("reused split was made with a different seed or fractions")
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        if _sha(reuse / f"{split}.jsonl") != old["splits"][split]["sha256"]:
            raise ValueError(f"reused {split}.jsonl does not match its recorded hash")
        shutil.copyfile(reuse / f"{split}.jsonl", out / f"{split}.jsonl")
    kept = {s: {r["video"] for r in read_jsonl(out / f"{s}.jsonl")} for s in ("train", "test")}
    excluded = {r["video"] for path in exclude for r in read_jsonl(path)}
    rows = [r for path in records for r in read_jsonl(path) if r.get("video") and r["video"] not in excluded
            and r.get("question") and r.get("options") and split_of(r["video"], seed, fractions) == "dev"]  # fmt: skip
    pool = sorted({r["video"] for r in rows}, key=lambda v: hashlib.sha256(f"{seed}:dev:{v}".encode()).hexdigest())
    if set(pool) & (kept["train"] | kept["test"]):
        raise AssertionError("dev pool overlaps the reused train/test manifests")
    chosen = set(pool if dev_cap is None else pool[:dev_cap])
    dev = sorted(({**r, "split": "dev"} for r in rows if r["video"] in chosen), key=lambda r: r["id"])
    write_jsonl(out / "dev.jsonl", dev)
    manifest = {
        **{k: old[k] for k in ("records", "excluded_manifests", "excluded_videos", "fractions", "seed")},
        "dev_rule": DEV_RULE, "dev_cap": dev_cap, "dev_available_videos": len(pool),
        "previous": {"manifest_sha256": _sha(reuse / "manifest.json"), "max_videos": old.get("max_videos"),
                     "dev": old["splits"]["dev"]},
        "splits": {"train": old["splits"]["train"], "test": old["splits"]["test"],
                   "dev": {"questions": len(dev), "videos": len(chosen), "sha256": _sha(out / "dev.jsonl")}},
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------- report
def _median(values: list[float]):
    v = sorted(values)
    return v[len(v) // 2] if v else None


def _rows(eval_dir: Path) -> list[dict]:
    return read_jsonl(Path(eval_dir) / "predictions.jsonl")


def _stream_bootstrap(per_stream: dict, samples: int = 4000, seed: int = 7) -> list[float]:
    """per_stream: stream -> [sum, count]; CI of sum/count (x100) resampling whole streams."""
    keys = sorted(per_stream)
    if not keys:
        return [None, None]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        pick = rng.choices(keys, k=len(keys))
        num, den = sum(per_stream[k][0] for k in pick), sum(per_stream[k][1] for k in pick)
        draws.append(100 * num / den if den else 0.0)
    draws.sort()
    return [round(draws[int(0.025 * samples)], 3), round(draws[int(0.975 * samples) - 1], 3)]


def _accuracy(rows: list[dict], keep) -> dict:
    sel = [r for r in rows if keep(r)]
    per = defaultdict(lambda: [0, 0])
    for r in sel:
        per[r["stream"]][0] += r["correct"]
        per[r["stream"]][1] += 1
    acc = 100 * sum(r["correct"] for r in sel) / len(sel) if sel else None
    return {"n": len(sel), "streams": len(per), "accuracy_pct": None if acc is None else round(acc, 3),
            "ci95_pct": _stream_bootstrap(per)}  # fmt: skip


def _paired(a: list[dict], b: list[dict], keep) -> dict:
    ra, rb = {r["id"]: r for r in a if keep(r)}, {r["id"]: r for r in b}
    per = defaultdict(lambda: [0, 0])
    for key, r in ra.items():
        per[r["stream"]][0] += r["correct"] - rb[key]["correct"]
        per[r["stream"]][1] += 1
    num, den = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"n": den, "delta_pp": round(100 * num / den, 3) if den else None, "ci95_pp": _stream_bootstrap(per)}


STAGE2_CHECKPOINT_STEPS = ()  # Stage-2 artifact contract: final module/LoRA pair only (writer and verifier)
EXPECTED_RUNS = ("s2_tkm", "s2_mass", "s2_fifo", "s2_full", "s2_tkm_trainsched", "s1_mass", "s1_fifo", "s1_full")


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------- identities
def init_identity(checkpoint_dir: str | Path) -> dict:
    """Content identity of the Stage-1 module checkpoint and its LoRA adapter."""
    from .harness import _dir_identity

    d = Path(checkpoint_dir)
    return {"modules.pt": _sha(d / "modules.pt"), "lora": _dir_identity(str(d / "lora"))}


def source_identity(source_dir: str | Path = Path(__file__).parent) -> str:
    digest = hashlib.sha256()
    for p in sorted(Path(source_dir).glob("*.py")):
        digest.update(p.name.encode() + p.read_bytes())
    return digest.hexdigest()


def data_sources(mc_json: list[str], excluded: list[str], build: dict) -> dict:
    """Inputs that define the data (sizes/caps excluded, so smoke and pilot can be compared)."""
    return {"mc_json": {str(p): _sha(p) for p in mc_json}, "excluded": {str(p): _sha(p) for p in excluded}, "build": build}


def stream_media_identity(streams_path: str | Path, video_root: str | Path) -> str:
    from .harness import _video_identity

    clips = [{"video": c["video"]} for r in read_jsonl(streams_path) for c in r.get("clips", [])]
    return _video_identity(clips, str(video_root))


# ---------------------------------------------------------------------------- data sufficiency
def expected_questions(streams_path: str | Path, split: str, limit: int | None, cfg) -> dict:
    """Questions the stream evaluator will ask (same stream order, limit and due-time rules), id -> evidence age."""
    from .train_stage2 import chunk_plan, stream_times

    records = [r for r in read_jsonl(streams_path) if r.get("split") == split and r.get("questions")]
    records = records[:limit] if limit else records
    gap, out = cfg.stream.frame_gap, {}
    for r in records:
        times = stream_times(r, cfg.stream.step_seconds, gap, cfg.stage2.max_seconds)
        if not times:
            continue
        for _, due in chunk_plan(times, r["questions"], len(times), gap):
            for q in due:
                end = max((b for _, b in q.get("evidence") or []), default=None)
                out[q["id"]] = None if end is None else round(float(q["time"]) - end, 3)
    return out


PROTOCOL = "memory-pilot-v2"  # v2: explicit stream selection manifests (smoke eligibility rule; pilot first-n)
SMOKE_RULE = "smoke-eligible-hash-v1"
PILOT_RULE = "pilot-first-n-v1"


def stream_question_ages(record: dict, cfg) -> list[float | None]:
    """Actual evidence ages of the questions the evaluator asks in one stream (same due-time rules)."""
    from .train_stage2 import chunk_plan, stream_times

    times = stream_times(record, cfg.stream.step_seconds, cfg.stream.frame_gap, cfg.stage2.max_seconds)
    if not times:
        return []
    ages = []
    for _, due in chunk_plan(times, record.get("questions", []), len(times), cfg.stream.frame_gap):
        for q in due:
            end = max((b for _, b in q.get("evidence") or []), default=None)
            ages.append(None if end is None else round(float(q["time"]) - end, 3))
    return ages


def select_streams(streams_path: str | Path, split: str, cfg, rule: str, n: int | None, out_path: str | Path,
                   seed: int = 7, min_duration: float = 240.0, min_beyond: int = 1) -> dict:  # fmt: skip
    """Deterministic, metadata-only stream selection, fixed before any model is evaluated; never uses accuracy.

    SMOKE_RULE: eligible = duration >= min_duration and >= min_beyond actual beyond-window questions (evidence
    age > memory.window_seconds under the evaluator's due-time rules); eligible streams ordered by
    sha256(f"{seed}:{id}"); the first n are selected (n=None: all eligible).
    PILOT_RULE: the first n streams in file order (the formal pilot's unchanged evaluation population).
    Writes the selected records to out_path (reused by every run) and returns the selection report."""
    window = cfg.memory.window_seconds
    records = [r for r in read_jsonl(streams_path) if r.get("split") == split and r.get("questions")]
    report = {"rule": rule, "split": split, "seed": seed, "candidates": len(records), "window_seconds": window}
    if rule == PILOT_RULE:
        chosen = records[:n] if n else records
        report.update({"eligible": len(records), "excluded": {}})
    elif rule == SMOKE_RULE:
        excluded = {"too_short": 0, "no_beyond_window_question": 0}
        eligible = []
        for r in records:
            beyond = sum(a is not None and a > window for a in stream_question_ages(r, cfg))
            short = float(r["duration"]) < min_duration
            excluded["too_short"] += int(short)
            excluded["no_beyond_window_question"] += int(beyond < min_beyond)
            if not short and beyond >= min_beyond:
                eligible.append(r)
        eligible.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        chosen = eligible[:n] if n else eligible
        report.update({"eligible": len(eligible), "excluded": excluded, "min_duration": min_duration,
                       "min_beyond_window_questions": min_beyond})  # fmt: skip
    else:
        raise ValueError(f"unknown selection rule {rule}")
    write_jsonl(out_path, chosen)
    report["selected"] = [{"id": r["id"], "duration": r["duration"], "ages": stream_question_ages(r, cfg)} for r in chosen]
    report["selected_count"] = len(chosen)
    report["sha256"] = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    return report


def data_report(streams_dir: str | Path, splits_manifest: str | Path, dev_limit: int | None, window: float, cfg,
                dev_path: str | Path | None = None, train_path: str | Path | None = None) -> dict:  # fmt: skip
    """Counts per split plus the questions the evaluator will ask on the dev streams actually used (`dev_path`,
    e.g. the selected manifest, with dev_limit None) and the training streams actually sampled (`train_path`)."""
    streams_dir = Path(streams_dir)
    rep = {"splits": json.loads(Path(splits_manifest).read_text())["splits"], "streams": {}}
    for split in SPLITS:
        rows = read_jsonl(streams_dir / f"{split}.jsonl")
        durations = sorted(r["duration"] for r in rows)
        rep["streams"][split] = {
            "streams": len(rows), "questions": sum(len(r.get("questions", [])) for r in rows),
            "duration_median": durations[len(durations) // 2] if durations else None,
            "duration_min": durations[0] if durations else None,
            "distractor_clips": sum(not c.get("queried", True) for r in rows for c in r.get("clips", [])),
        }  # fmt: skip
    dev_file = Path(dev_path) if dev_path else streams_dir / "dev.jsonl"
    ages = expected_questions(dev_file, "dev", dev_limit, cfg)
    known = [a for a in ages.values() if a is not None]
    dev_count = len([r for r in read_jsonl(dev_file) if r.get("split") == "dev" and r.get("questions")])
    train_file = Path(train_path) if train_path else streams_dir / "train.jsonl"
    rep["train_used"] = {"file": train_file.name, "streams": len(read_jsonl(train_file))}
    rep["dev_selected"] = {"file": dev_file.name, "streams": min(dev_limit, dev_count) if dev_limit else dev_count,
                           "questions": len(ages),
                           "beyond_window": sum(a > window for a in known), "within_window": sum(a <= window for a in known),
                           "evidence_age_median": sorted(known)[len(known) // 2] if known else None}  # fmt: skip
    return rep


THRESHOLDS = {"smoke": {"train_streams": 2, "dev_streams": 2, "dev_beyond_window": 2},
              "pilot": {"train_streams": 50, "dev_streams": 30, "dev_beyond_window": 100}}


def data_problems(rep: dict, mode: str) -> list[str]:
    t, problems = THRESHOLDS[mode], []
    train_used = rep.get("train_used", {}).get("streams", rep["streams"]["train"]["streams"])
    if train_used < t["train_streams"]:
        problems.append(f"usable train streams {train_used} < {t['train_streams']}")
    if rep["dev_selected"]["streams"] < t["dev_streams"]:
        problems.append(f"selected dev streams {rep['dev_selected']['streams']} < {t['dev_streams']}")
    if rep["dev_selected"]["beyond_window"] < t["dev_beyond_window"]:
        problems.append(f"beyond-window dev questions {rep['dev_selected']['beyond_window']} < {t['dev_beyond_window']}")
    return problems


# ---------------------------------------------------------------------------- report and decision
def memory_report(eval_root: str, window_seconds: float, primary: str = "s2_tkm",
                  heuristics=("s2_mass", "s2_fifo"), upper: str = "s2_full", expected_runs=None,
                  expected_ids: set | None = None) -> dict:  # fmt: skip
    """Per-run statistics, paired deltas and a three-state decision:
    invalid (validity prerequisites fail: missing/incomplete runs, too few beyond-window questions, no memory
    pressure, unequal budgets), negative (valid, but the predeclared gain rule fails), supported (both pass)."""
    root = Path(eval_root)
    runs = {d.name: _rows(d) for d in sorted(root.iterdir()) if (d / "predictions.jsonl").is_file()}
    invalid = []
    for name in expected_runs or ():
        if name not in runs:
            invalid.append(f"missing run {name}")
    reference = expected_ids if expected_ids is not None else ({r["id"] for r in runs[primary]} if primary in runs else None)
    for name, rows in runs.items():
        if name in (expected_runs or runs) and reference is not None and {r["id"] for r in rows} != reference:
            invalid.append(f"{name}: question set differs from the expected set (incomplete coverage)")
    beyond = lambda r: r.get("evidence_age") is not None and r["evidence_age"] > window_seconds  # noqa: E731
    within = lambda r: r.get("evidence_age") is not None and r["evidence_age"] <= window_seconds  # noqa: E731
    summary = {}
    for name, rows in runs.items():
        n = max(1, len(rows))
        kept = [r["evidence_kept"] for r in rows if beyond(r) and r.get("evidence_kept") is not None]
        streams = {r["stream"]: r for r in rows}
        summary[name] = {
            "all": _accuracy(rows, lambda r: True), "beyond_window": _accuracy(rows, beyond),
            "within_window": _accuracy(rows, within),
            "evidence_kept_beyond_window": round(sum(kept) / len(kept), 4) if kept else None,
            "kv_tokens_mean": round(sum(r["kv_tokens"] for r in rows) / n, 1),
            "kv_mb_mean": round(sum(r.get("kv_mb", 0.0) for r in rows) / n, 3),
            "evicted_tokens_mean": round(sum(r.get("evicted_tokens", 0) for r in rows) / n, 1),
            # evicted and appended are both KV tokens (text + visual); visual_tokens_streamed is visual only
            "kv_appended_tokens_mean": round(sum(r.get("kv_appended_tokens", 0) for r in rows) / n, 1),
            "visual_tokens_streamed_mean": round(sum(r.get("visual_tokens_streamed", r.get("streamed_tokens", 0))
                                                     for r in rows) / n, 1),
            "evidence_age_median": _median([r["evidence_age"] for r in rows if r.get("evidence_age") is not None]),
            "seconds_per_stream": round(sum(r.get("stream_seconds_elapsed", 0) for r in streams.values()) / max(1, len(streams)), 2),
        }  # fmt: skip
    pairs = {}
    if invalid:  # incomplete or mismatched coverage: no paired statistics on partial question sets
        return {"window_seconds": window_seconds, "runs": summary, "paired": {},
                "decision": {"status": "invalid", "invalid_reasons": invalid, "gain_rule_met": None,
                             "meaning": "inconclusive: incomplete or mismatched prediction coverage; no paired comparison "
                                        "was computed and no conclusion about retention is drawn"}}  # fmt: skip
    if primary in runs:
        for other in [*heuristics, upper, f"{primary}_trainsched"]:
            if other in runs:
                pairs[f"{primary}_minus_{other}"] = {"beyond_window": _paired(runs[primary], runs[other], beyond),
                                                     "all": _paired(runs[primary], runs[other], lambda r: True)}  # fmt: skip
    validity, decision = {}, None
    if primary in runs and all(h in runs for h in heuristics):
        kv = [summary[x]["kv_tokens_mean"] for x in (primary, *heuristics)]
        fifo_kept = summary[heuristics[-1]]["evidence_kept_beyond_window"]
        validity = {
            "complete_coverage": not invalid,
            "enough_beyond_window": summary[primary]["beyond_window"]["n"] >= 100,
            "evictions": summary[primary]["evicted_tokens_mean"] > 0,
            "fifo_loses_evidence": fifo_kept is not None and fifo_kept <= 0.8,
            "equal_budget": max(kv) <= 1.02 * min(kv),
        }
        gains = [pairs[f"{primary}_minus_{h}"]["beyond_window"] for h in heuristics]
        gain = all(g["delta_pp"] is not None and g["delta_pp"] >= 3.0 and g["ci95_pp"][0] is not None
                   and g["ci95_pp"][0] > 0 for g in gains)  # fmt: skip
        status = "invalid" if not all(validity.values()) else ("supported" if gain else "negative")
        decision = {
            "status": status, "validity": validity, "gain_rule_met": gain,
            "invalid_reasons": invalid + [k for k, v in validity.items() if not v],
            "rule": "valid: complete coverage, >= 100 beyond-window questions, evictions, FIFO evidence kept <= 0.8, retained "
                    "KV within 2%; supported: valid and s2_tkm - each heuristic on beyond-window questions >= +3 pp with "
                    "stream-bootstrap CI lower > 0",
            "meaning": {"invalid": "inconclusive: the experiment did not test retention; fix the setup, no conclusion",
                        "negative": "valid test, gain rule failed: learned retention (over age/surprise statistics) is not "
                                    "a contribution; do not scale Stage 2 for retention",
                        "supported": "learned retention over age/surprise statistics beats both heuristics here"}[status],
        }  # fmt: skip
    elif invalid or primary not in runs:
        decision = {"status": "invalid", "invalid_reasons": invalid or [f"missing {primary}"],
                    "meaning": "inconclusive: the experiment did not test retention"}  # fmt: skip
    return {"window_seconds": window_seconds, "runs": summary, "paired": pairs, "decision": decision}


# ---------------------------------------------------------------------------- smoke authorization
def smoke_checks(report: dict, metrics: list[dict], data_rep: dict) -> dict:
    """Machine-checked smoke success (not just a readable digest)."""
    import math

    grads = [m.get("grad_norm_retention") for m in metrics]
    reasons = (report.get("decision") or {}).get("invalid_reasons", [])
    coverage = [r for r in reasons if r.startswith("missing run") or "question set" in r]
    return {
        "training_updates_logged": len(metrics) > 0,
        "retention_gradient_finite_nonzero": any(g is not None and math.isfinite(g) and g > 0 for g in grads)
        and all(g is None or math.isfinite(g) for g in grads),
        "retention_labels_both_classes": sum(m.get("ret_pos", 0) for m in metrics) > 0
        and sum(m.get("ret_neg", 0) for m in metrics) > 0,
        "evictions": (report["runs"].get("s2_tkm") or {}).get("evicted_tokens_mean", 0) > 0,
        "expected_runs_complete": not coverage and all(name in report["runs"] for name in EXPECTED_RUNS),
        "enough_smoke_questions": data_rep["dev_selected"]["beyond_window"] >= THRESHOLDS["smoke"]["dev_beyond_window"],
    }  # fmt: skip


def verify_smoke(smoke_ok: str | Path, expected: dict) -> list[str]:
    """Problems preventing a pilot (empty list = authorized). `expected` holds revision, config, init and data
    sources of the intended pilot; smoke and pilot may differ only in sizes (caps, steps, dev stream count)."""
    path = Path(smoke_ok)
    if not path.is_file():
        return [f"no smoke authorization at {path}"]
    try:
        record = json.loads(path.read_text())
    except ValueError:
        return [f"unreadable smoke authorization {path}"]
    problems = [] if record.get("passed") is True else ["smoke did not pass"]
    for key in ("protocol", "revision", "config", "init", "data_sources"):
        if record.get(key) != expected.get(key):
            problems.append(f"smoke {key} differs from the intended pilot")
    return problems


# ---------------------------------------------------------------------------- pilot follow-up (448adf1 review)
def memory_analysis(eval_root: str | Path, window_seconds: float) -> dict:
    """From existing predictions: paired S2 - S1 under each heuristic policy, per-question TKM vs FIFO/mass agreement,
    and the memory audit fields when present (KV split, unused capacity, FIFO overlap, score correlations)."""
    root = Path(eval_root)
    runs = {d.name: _rows(d) for d in sorted(root.iterdir()) if (d / "predictions.jsonl").is_file()}
    beyond = lambda r: r.get("evidence_age") is not None and r["evidence_age"] > window_seconds  # noqa: E731
    out = {"stage2_minus_stage1": {}, "agreement": {}, "audit": {}}
    for policy in ("fifo", "mass", "full"):
        a, b = f"s2_{policy}", f"s1_{policy}"
        if a in runs and b in runs:
            out["stage2_minus_stage1"][policy] = {"beyond_window": _paired(runs[a], runs[b], beyond),
                                                  "all": _paired(runs[a], runs[b], lambda r: True)}  # fmt: skip
    for other in ("s2_fifo", "s2_mass"):
        if "s2_tkm" in runs and other in runs:
            a, b = {r["id"]: r for r in runs["s2_tkm"]}, {r["id"]: r for r in runs[other]}
            counts = defaultdict(int)
            for key in a.keys() & b.keys():
                ca, cb = a[key]["correct"], b[key]["correct"]
                counts["both_correct" if ca and cb else "only_tkm" if ca else f"only_{other}" if cb else "both_wrong"] += 1
            out["agreement"][f"s2_tkm_vs_{other}"] = {**counts, "n": len(a.keys() & b.keys())}
    for name, rows in runs.items():
        fields = [k for k in ("kv_protected", "kv_window", "kv_older", "kv_older_unused", "kv_older_groups",
                              "audit_jaccard_vs_fifo_mean", "audit_score_recency_rho_mean", "audit_score_mass_rho_mean")
                  if any(r.get(k) is not None for r in rows)]  # fmt: skip
        if fields:
            out["audit"][name] = {k: round(sum(r[k] for r in rows if r.get(k) is not None)
                                           / max(1, sum(r.get(k) is not None for r in rows)), 4) for k in fields}  # fmt: skip
    return out


def budget_precheck(eval_root: str | Path, policies=("s2_tkm", "s2_mass", "s2_fifo"), tolerance: float = 0.02) -> dict:
    """Mean retained KV per policy on the pre-check streams; passes only if max/min - 1 <= tolerance (the original
    2% equal-budget criterion, applied before the complete evaluation)."""
    kv = {}
    for name in policies:
        rows = _rows(Path(eval_root) / name)
        kv[name] = sum(r["kv_tokens"] for r in rows) / max(1, len(rows))
    spread = max(kv.values()) / min(kv.values()) - 1 if kv and min(kv.values()) > 0 else float("inf")
    return {"kv_tokens_mean": kv, "spread": spread, "tolerance": tolerance, "passed": spread <= tolerance}


def precheck_streams(streams_path: str | Path, out_path: str | Path, per_end: int = 2) -> list[str]:
    """Budget pre-check manifest: the `per_end` shortest and longest streams of the evaluation manifest (by duration,
    ties by id), chosen from metadata only."""
    rows = sorted(read_jsonl(streams_path), key=lambda r: (r["duration"], r["id"]))
    chosen = rows[:per_end] + [r for r in rows[-per_end:] if r not in rows[:per_end]]
    write_jsonl(out_path, chosen)
    return [r["id"] for r in chosen]
