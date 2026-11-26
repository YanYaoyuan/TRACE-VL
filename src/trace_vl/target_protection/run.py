"""Phase 1c: privileged target protection with three seeds; references stay read-only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.complementary_memory import native_state as test_a
from trace_vl.complementary_memory.analysis import question_metrics
from trace_vl.complementary_memory.state_stream import Block, NativeStream, block_inputs, score
from trace_vl.core_vlm import longstream
from trace_vl.core_vlm import longstream_delayed as delayed
from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import run_jobs
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VisualStep, grid_pool
from trace_vl.core_vlm.train import checked_checkpoint, tensor_digest
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.positions import PositionCounter
from trace_vl.v3.run_io import ensure_run_manifest

from . import analysis
from .stream import TargetProtectedStream

REFERENCE = Path("/data/yyy/tracevl-runs/complementary-test-a-plain-fa8b205-001")
REFERENCE_ID = "a8a0a7c2aca226990e76fbd9840df92ef651a99fd130dec03b9a307fcdb5d23b"
REFERENCE_REPORT_SHA = "13b470fe2f5ac6bdb9b706c2b212f67244c27180b5a88feff384dc4454c595d9"
REPRODUCTION_TOLERANCE = .25
B0_REFERENCE = Path("/data/yyy/tracevl-runs/b0-window40-5e62524-001")
B0_ID = "e40fc6f3290603cf2086e0a24955d828346406a1f64cc86611d27a65c380c543"
B0_REPORT_SHA = "c3c0c324f058f8ae12d5fb95f12dff2944cb477a5e547496badc9ccbc3ed8654"


def read(path):
    return json.loads(Path(path).read_text())


def code_identity():
    root = Path(__file__).parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base, old_report = read(args.reference / "inputs.json"), read(args.reference / "report.json")
    if (sha256(args.reference / "report.json") != REFERENCE_REPORT_SHA
            or base["id"] != REFERENCE_ID or digest({k: v for k, v in base.items() if k != "id"}) != REFERENCE_ID
            or not old_report["completed"] or old_report["smoke"]
            or old_report["sources"] != 77 or old_report["questions"] != 385):
        raise ValueError("complete frozen Test A required")
    if base["weights"] != "plain" or base["memory_seeds"] != list(analysis.SEEDS):
        raise ValueError("wrong reference weights or seeds")
    if base["code"] != test_a.code_identity() or base["versions"] != test_a.versions():
        raise ValueError("original helper code or runtime changed")
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("base model changed")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("starting LoRA changed")
    b0, b0_report = read(args.b0_reference / "inputs.json"), read(args.b0_reference / "report.json")
    if (b0["id"] != B0_ID or digest({k: v for k, v in b0.items() if k != "id"}) != B0_ID
            or sha256(args.b0_reference / "report.json") != B0_REPORT_SHA
            or b0["reference"] != base or not b0_report["completed"] or b0_report["smoke"]
            or b0_report["sources"] != 77 or b0_report["questions"] != 385):
        raise ValueError("complete frozen B0 required")
    current = code_identity()
    if any(current.get(name) != value for name, value in b0["code"].items()):
        raise ValueError("B0 helper source changed")
    versions = {**test_a.versions(), "numpy": np.__version__}
    if b0["versions"] != versions:
        raise ValueError("B0 runtime changed")
    hashes, b0_hashes, targets = {}, {}, {}
    for source in base["sources"]:
        zero = [q for q in source["queries"] if q["delay_seconds"] == 0]
        if len(zero) != 1 or len({q["target_id"] for q in source["queries"]}) != 1:
            raise ValueError("one anchored fact and one zero-delay query required")
        for q in source["queries"]:
            name = f"{digest(q['id'])}.json"
            old, row = read(args.reference / "questions" / name), read(args.b0_reference / "questions" / name)
            if (not old["completed"] or not row["completed"] or old["input_id"] != base["id"]
                    or row["input_id"] != b0["id"] or old["id"] != q["id"] or row["id"] != q["id"]
                    or old["gold"] != q["answer"] or row["gold"] != q["answer"]
                    or old["source_group"] != source["source_group"] or row["source_group"] != source["source_group"]
                    or old["steps_seen"] != q["planned_encoder_calls"]
                    or row["steps_seen"] != q["planned_encoder_calls"]):
                raise ValueError("reference question identity differs")
            snapshot = old["snapshot_steps"]
            if len(snapshot) != 8 or len(set(snapshot)) != 8 or row["snapshot_steps"] != snapshot:
                raise ValueError("eight unique reference target steps required")
            previous = targets.setdefault(source["source_group"], snapshot)
            if previous != snapshot or (q["delay_seconds"] == 0 and snapshot != old["recent_steps"]):
                raise ValueError("causal target snapshot differs across queries")
            hashes[q["id"]] = sha256(args.reference / "questions" / name)
            b0_hashes[q["id"]] = sha256(args.b0_reference / "questions" / name)
    if len(base["sources"]) != 77 or len(hashes) != 385:
        raise ValueError("all 77 sources/385 questions required")
    inputs = {"protocol": "target-protected32-v1", "reference": base,
              "reference_report_sha256": REFERENCE_REPORT_SHA, "reference_question_sha256": hashes,
              "b0_input": B0_ID, "b0_report_sha256": B0_REPORT_SHA, "b0_question_sha256": b0_hashes,
              "target_steps": targets, "memory_seeds": list(analysis.SEEDS), "recent": 8, "history_capacity": 32,
              "protected_size": 8, "relevant_effect_nats": analysis.RELEVANT_EFFECT,
              "primary": "protected_minus_reservoir:300:nll", "bootstrap": {"draws": 10000, "seed": 7},
              "reproduction_tolerance": REPRODUCTION_TOLERANCE, "window8_reproduction": "smoke_sources_only",
              "code": current, "versions": versions,
              "protocol_sha256": sha256(Path(__file__).parents[3] / "docs/complementary_memory/target_protection_protocol.md")}
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)


def load_inputs(args):
    value = read(args.out / "inputs.json")
    if value["id"] != digest({k: v for k, v in value.items() if k != "id"}):
        raise ValueError("target-protection manifest corrupted")
    if value["code"] != code_identity() or value["versions"] != {**test_a.versions(), "numpy": np.__version__}:
        raise ValueError("target-protection code or versions changed")
    if sha256(Path(__file__).parents[3] / "docs/complementary_memory/target_protection_protocol.md") != value["protocol_sha256"]:
        raise ValueError("target-protection protocol changed")
    return value


def reference_row(args, inputs, identity):
    path = args.reference / "questions" / f"{digest(identity)}.json"
    if sha256(path) != inputs["reference_question_sha256"][identity]:
        raise ValueError("reference question bytes changed")
    return read(path)


def reproduction(current, saved, gold):
    a, b = current["option_logits_fp32"], saved["option_logits_fp32"]
    if len(a) != len(b) or not a or not all(math.isfinite(v) for v in [*a, *b]):
        raise ValueError("finite matching options required")
    gap = max(abs(x - y) for x, y in zip(a, b, strict=True))
    if gap > REPRODUCTION_TOLERANCE:
        raise ValueError(f"reference logit mismatch: {gap}")
    return {"max_logit_gap": gap, "choice_changed": current["choice"] != saved["choice"],
            "gold_nll_change": question_metrics(a, gold)["nll"] - question_metrics(b, gold)["nll"]}


def b0_row(args, inputs, identity):
    path = args.b0_reference / "questions" / f"{digest(identity)}.json"
    if sha256(path) != inputs["b0_question_sha256"][identity]:
        raise ValueError("B0 question bytes changed")
    return read(path)


@torch.no_grad()
def evaluate_source(bb, inputs, source, args, window_reproduction):
    """Same decode/schedule/packing; three protected streams and a smoke-only window control."""
    base, out = inputs["reference"], args.out
    if sha256(source["video"]) != source["video_sha256"] or sha256(source["metadata"]) != source["metadata_sha256"]:
        raise ValueError("frozen source media changed")
    metadata = read(source["metadata"])
    queries = {q["id"]: q for q in source["queries"]}
    indices, times, _ = longstream.schedule(metadata, max(q["query_time"] for q in queries.values()))
    if indices.tolist() != source["sampled_frame_indices"] or times.tolist() != source["sampled_frame_times"]:
        raise ValueError("original causal PTS schedule changed")
    meta = metadata["streams"][0]
    fps = float(Fraction(meta["avg_frame_rate"]))
    hw = frame_size(meta["height"], meta["width"], base["pixels"], 32)
    prompts = {k: test_a.native_prompt(bb, q, indices, times, metadata, hw) for k, q in queries.items()}
    head = next(iter(prompts.values()))[0]
    if any(h != head for h, _ in prompts.values()):
        raise ValueError("question leaked into the live prompt head")
    live = {f"protected{seed}": TargetProtectedStream(
        bb, head, seed=test_a.memory_seed(source["source_group"], seed)) for seed in analysis.SEEDS}
    if window_reproduction:
        live["window8"] = NativeStream(bb, head, "window", recent=8)
    recent, snapshots, completed = deque(maxlen=8), {}, []
    parity_parts, parity, encoded = ([], [], []), None, 0
    seconds = {"vision": 0., "lm_and_budget_audit_all_streams": 0., "questions": 0., "packing_audit": 0.}
    maxima = {name: {"steady_steps": 0, "before_eviction_steps": 0, "steady_kv_bytes": 0} for name in live}
    state_bytes = {}
    counter = PositionCounter()
    packing = hashlib.sha256(bytes.fromhex(tensor_digest({"embeds": bb.embed_ids(head),
                                                       "positions": counter.text(len(head), bb.device)})))
    torch.cuda.reset_peak_memory_stats()

    def ask(q):
        if encoded != q["planned_encoder_calls"]:
            raise ValueError("query saw the wrong prefix")
        old = reference_row(args, inputs, q["id"])
        if any(old[k] != q[k] for k in ("target_time", "query_time", "delay_seconds", "target_id")):
            raise ValueError("question timing differs from reference")
        if (list(live["protected17"].recent) != old["recent_steps"]
                or [b.index for b in snapshots[q["target_id"]]] != old["snapshot_steps"]
                or live["protected17"].counter.pos != old["streams"]["window_stream"]["next_position"]):
            raise ValueError("reference recent/snapshot indices or native coordinates differ")
        started = time.monotonic()
        tail = prompts[q["id"]][1]
        before = {name: s.snapshot() for name, s in live.items()}
        logical = {name: (s.audit(), s.rng.getstate()) for name, s in live.items()}
        if len({s.counter.pos for s in live.values()}) != 1:
            raise ValueError("live policy positions diverged")
        def run(stream):
            value = score(bb, stream.fork(), stream.counter.pos, tail, len(q["options"]))
            if len(value["option_logits_fp32"]) != len(q["options"]) or not all(math.isfinite(v) for v in value["option_logits_fp32"]):
                raise ValueError("nonfinite or wrong option logits")
            return value
        predictions = {f"protected{seed}": run(live[f"protected{seed}"]) for seed in analysis.SEEDS}
        checks = {}
        if q["delay_seconds"] == 0:
            for seed in analysis.SEEDS:
                name = f"protected{seed}"
                audit, saved = live[name].audit(), old["streams"][f"reservoir{seed}"]
                if audit["protected"] or any(audit[k] != saved[k] for k in (
                        "history", "recent", "kv_steps", "kv_tokens", "kv_bytes", "state_bytes", "next_position")):
                    raise ValueError("zero-delay Reservoir retained indices or budgets differ")
                checks[name] = reproduction(predictions[name], old["predictions"][f"reservoir{seed}:S1K1"], q["answer"])
        aux_prediction = None
        if window_reproduction:
            aux_prediction = run(live["window8"])
            checks["window8"] = reproduction(aux_prediction, old["predictions"]["window_stream"], q["answer"])
        for seed in analysis.SEEDS:
            name = f"protected{seed}"
            if run(live[name]) != predictions[name]:
                raise RuntimeError("protected-stream re-answer is not deterministic")
            audit = live[name].audit()
            if (set(audit["protected"]) | set(audit["recent"])) & set(old["snapshot_steps"]) != set(old["snapshot_steps"]):
                raise ValueError("a privileged target step was lost")
        for name, stream in live.items():
            stream.verify(before[name])
            if (stream.audit(), stream.rng.getstate()) != logical[name]:
                raise RuntimeError("question changed retention metadata or RNG")
        tail_positions = PositionCounter(live["protected17"].counter.pos).text(len(tail), bb.device)
        tail_hash = tensor_digest({"embeds": bb.embed_ids(tail), "positions": tail_positions})
        torch.cuda.synchronize()
        seconds["questions"] += time.monotonic() - started
        value = {"completed": True, "input_id": inputs["id"], "reference_input_id": base["id"],
                 "id": q["id"], "source_group": source["source_group"], "gold": q["answer"],
                 "task": q.get("task"), "target_id": q["target_id"], "delay_seconds": q["delay_seconds"],
                 "target_time": q["target_time"], "query_time": q["query_time"], "steps_seen": encoded,
                 "predictions": predictions, "reproduction": checks, "window8_prediction": aux_prediction,
                 "streams": {name: s.audit() for name, s in live.items()},
                 "snapshot_steps": [b.index for b in snapshots[q["target_id"]]],
                 "packing_prefix_chain_sha256": packing.hexdigest(), "head_ids_sha256": digest(head),
                 "tail_ids_sha256": digest(tail), "tail_embed_positions_sha256": tail_hash}
        write_json(out / "questions" / f"{digest(q['id'])}.json", value)
        completed.append(q["id"])

    def dispatch(event):
        when, kind, identity = event
        if kind == 1:
            return ask(queries[identity])
        blocks = list(recent)
        if not blocks or any(b.end > when for b in blocks):
            raise ValueError("snapshot-index audit is noncausal")
        snapshots[identity] = blocks
        indices_to_protect = [b.index for b in blocks]
        if indices_to_protect != inputs["target_steps"][source["source_group"]]:
            raise ValueError("protected target snapshot differs from frozen reference")
        for seed in analysis.SEEDS:
            live[f"protected{seed}"].protect(indices_to_protect)

    events = delayed.CausalEvents(delayed.events_for(source["queries"]), dispatch)
    pixels = longstream.stream_pixels(source["video"], indices, *hw)
    try:
        for i, frames in enumerate(pixels):
            end = float(times[i][-1])
            events.due(end, inclusive=False)
            started = time.monotonic()
            values, grid = bb.preprocess(frames)
            dense = bb.encode(values, grid)
            shape = bb.merged_grid(grid)
            tokens, rr, cc = grid_pool(dense, shape, (4, 4))
            torch.cuda.synchronize()
            seconds["vision"] += time.monotonic() - started
            encoded += 1
            block = Block(i, end, tuple(test_a.stamp_ids(bb, indices[2*i:2*i+2], fps)),
                          tokens.detach().clone(), rr, cc, shape)
            started = time.monotonic()
            audit_embeds, audit_pos = block_inputs(bb, block, copy.copy(live["protected17"].counter))
            packing.update(bytes.fromhex(tensor_digest({"embeds": audit_embeds, "positions": audit_pos})))
            del audit_embeds, audit_pos
            torch.cuda.synchronize()
            seconds["packing_audit"] += time.monotonic() - started
            started = time.monotonic()
            recent.append(block)
            for name, stream in live.items():
                maxima[name]["before_eviction_steps"] = max(maxima[name]["before_eviction_steps"], stream.audit()["kv_steps"] + 1)
                stream.append(block)
                audit = stream.audit()
                limit = 8 if name == "window8" else 40
                if audit["kv_steps"] > limit:
                    raise ValueError("steady KV budget exceeded")
                maxima[name]["steady_steps"] = max(maxima[name]["steady_steps"], audit["kv_steps"])
                maxima[name]["steady_kv_bytes"] = max(maxima[name]["steady_kv_bytes"], audit["kv_bytes"])
                previous = state_bytes.setdefault(name, audit["state_bytes"])
                if previous != audit["state_bytes"]:
                    raise ValueError("recurrent state bytes grew with time")
            torch.cuda.synchronize()
            seconds["lm_and_budget_audit_all_streams"] += time.monotonic() - started
            if parity is None:
                pair = tuple(float(t) for t in times[i])
                for store, item in zip(parity_parts, (block, VisualStep(dense, shape, end, pair), frames), strict=True):
                    store.append(item)
                if len(parity_parts[0]) == test_a.PARITY_STEPS:
                    q = min(queries.values(), key=lambda x: x["query_time"])
                    parity = test_a.check_parity(bb, head, prompts[q["id"]][1], *parity_parts, q, indices, times, metadata)
                    parity_parts = ([], [], [])
            events.due(end, inclusive=True)
            if i % 30 == 0:
                print(json.dumps({"source": source["source_group"], "step": encoded, "of": len(times)}), flush=True)
        events.due(float("inf"), inclusive=True)
    finally:
        pixels.close()
    if len(completed) != len(queries) or events.cursor != len(events.events) or parity is None:
        raise ValueError("incomplete source, queries or parity")
    write_json(out / "sources" / f"{digest(source['source_group'])}.json", {
        "completed": True, "input_id": inputs["id"], "source_group": source["source_group"], "steps": encoded,
        "parity": parity, "live_streams": len(live), "window8_reproduction": window_reproduction, "seconds": seconds,
        "budget_maxima": maxima, "state_bytes": state_bytes,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "packing_prefix_chain_sha256": packing.hexdigest(),
        "transient_scope": "logical before-eviction count from the append-one/evict-one path; CUDA peak separately measured"})


def worker(args):
    inputs = load_inputs(args)
    base = inputs["reference"]
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("model identity changed")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("LoRA identity changed")
    sources = test_a.selected_sources(base, args.smoke)[args.shard::args.shards]
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(BackboneConfig(name=base["model"], dtype="bfloat16", local_files_only=True),
                  StreamConfig(pixels_per_frame=base["pixels"]), device="cuda:0")
    bb.add_lora(8, 16, dropout=0., adapter=str(Path(base["plain"]) / "adapter"))
    bb.model.eval().requires_grad_(False)
    for source in sources:
        path = args.out / "sources" / f"{digest(source['source_group'])}.json"
        if path.exists():
            old = read(path)
            if not old["completed"] or old["input_id"] != inputs["id"]:
                raise ValueError("completed source belongs to another run")
            continue
        evaluate_source(bb, inputs, source, args, source["source_group"] in base["smoke_source_groups"])


def report(args):
    inputs = load_inputs(args)
    base = inputs["reference"]
    selected = test_a.selected_sources(base, args.smoke)
    rows, sources, refs, b0 = [], [], {}, {}
    for source in selected:
        audit = read(args.out / "sources" / f"{digest(source['source_group'])}.json")
        if not audit["completed"] or audit["input_id"] != inputs["id"]:
            raise ValueError("incomplete source")
        sources.append(audit)
        for q in source["queries"]:
            row = read(args.out / "questions" / f"{digest(q['id'])}.json")
            if (not row["completed"] or row["input_id"] != inputs["id"] or row["id"] != q["id"]
                    or row["gold"] != q["answer"] or row["source_group"] != source["source_group"]
                    or row["steps_seen"] != q["planned_encoder_calls"]
                    or set(row["predictions"]) != {f"protected{s}" for s in analysis.SEEDS}):
                raise ValueError("incomplete or wrong question")
            for seed in analysis.SEEDS:
                a = row["streams"][f"protected{seed}"]
                target = set(row["snapshot_steps"])
                if (a["kv_steps"] != min(q["planned_encoder_calls"], 40)
                        or not target.issubset(set(a["recent"]) | set(a["protected"]))):
                    raise ValueError("wrong protected budget or lost target")
            refs[q["id"]] = reference_row(args, inputs, q["id"])
            b0[q["id"]] = b0_row(args, inputs, q["id"])
            rows.append(row)
    by_arm = {}
    for name in sorted({name for row in rows for name in row["reproduction"]}):
        checks = [row["reproduction"][name] for row in rows if name in row["reproduction"]]
        gaps = [v["max_logit_gap"] for v in checks]
        by_arm[name] = {"queries": len(checks), "max_gap": max(gaps),
                        "q50_q95_q99": np.quantile(gaps, [.5, .95, .99]).tolist(),
                        "changed_choices": sum(v["choice_changed"] for v in checks)}
        if not args.smoke:
            drift = [v["gold_nll_change"] for v in checks]
            by_arm[name].update(mean_gold_nll_change=float(np.mean(drift)),
                                max_abs_gold_nll_change=max(abs(v) for v in drift))
    if any(by_arm.get(f"protected{s}", {}).get("queries") != len(selected) for s in analysis.SEEDS):
        raise ValueError("every source/seed zero-delay reference must be reproduced")
    if by_arm.get("window8", {}).get("queries") != 10:
        raise ValueError("exactly the two fixed smoke sources reproduce window8")
    result = {"completed": True, "smoke": args.smoke, "inputs": inputs["id"], "sources": len(sources),
              "questions": len(rows), "new_prediction_vectors": len(rows) * 3,
              "auxiliary_window8_vectors": sum(r["window8_prediction"] is not None for r in rows),
              "reproduction_by_arm": by_arm, "max_reproduction_gap": max(v["max_gap"] for v in by_arm.values()),
              "max_parity_gap": max(s["parity"]["max_letter_logit_gap"] for s in sources),
              "peak_allocated_gib": max(s["peak_allocated_gib"] for s in sources),
              "total_steps": sum(s["steps"] for s in sources),
              "total_seconds": {k: sum(s["seconds"][k] for s in sources) for k in sources[0]["seconds"]},
              "reference_scope": "All-source/three-seed 0 s reproduction only; smoke-only window8. Old packing digests absent; later unreproduced-reference drift remains unbounded. Privileged timing, separately evolved states, reused development data."}
    if not args.smoke:
        if len(sources) != 77 or len(rows) != 385:
            raise ValueError("all 77 sources/385 questions required")
        result["analysis"] = analysis.analyze(rows, refs, b0)
        result["outcome"] = analysis.outcome(result["analysis"])
        budgets = {}
        for delay in sorted({r["delay_seconds"] for r in rows}):
            subset = [r for r in rows if r["delay_seconds"] == delay]
            fields = ("kv_steps", "kv_tokens", "kv_bytes", "state_bytes")
            budgets[str(delay)] = {
                "protected": {k: float(np.mean([sum(r["streams"][f"protected{s}"][k] for s in analysis.SEEDS) / 3
                                               for r in subset])) for k in fields},
                "reservoir": {k: float(np.mean([sum(refs[r["id"]]["streams"][f"reservoir{s}"][k]
                                                   for s in analysis.SEEDS) / 3 for r in subset])) for k in fields},
                "window40": {k: float(np.mean([b0[r["id"]]["streams"]["window40"][k] for r in subset])) for k in fields},
            }
        result["mean_budgets"] = budgets
    write_json(args.out / ("smoke-ok.json" if args.smoke else "report.json"), result)


def run(args):
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("at least two distinct GPUs required")
    prepare(args)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "TRACE_WANDB": "0", "TRACE_SWANLAB": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    check = "import torch; assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count()))>24"
    run_jobs([("gpu-check", [sys.executable, "-c", check], gpus)], env, args.out / "logs")
    command = [sys.executable, "-m", "trace_vl.target_protection.run", "worker", "--out", str(args.out),
               "--reference", str(args.reference), "--b0-reference", str(args.b0_reference)]
    for phase in (("smoke",) if args.smoke_only else ("smoke", "full")):
        extra = ["--smoke"] if phase == "smoke" else []
        count = min(2, len(gpus)) if phase == "smoke" else len(gpus)
        jobs = [(f"{phase}-{i}", [*command, "--shard", str(i), "--shards", str(count), *extra], [gpu])
                for i, gpu in enumerate(gpus[:count])]
        run_jobs(jobs, env, args.out / "logs")
        args.smoke = phase == "smoke"
        report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "worker", "report", "run"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--b0-reference", type=Path, default=B0_REFERENCE)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    {"prepare": prepare, "worker": worker, "report": report, "run": run}[args.stage](args)


if __name__ == "__main__":
    main()
