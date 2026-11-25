"""B0 phase 1a: new window40 and reproduction window8; completed Test A stays read-only."""

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

REFERENCE = Path("/data/yyy/tracevl-runs/complementary-test-a-plain-fa8b205-001")
REFERENCE_ID = "a8a0a7c2aca226990e76fbd9840df92ef651a99fd130dec03b9a307fcdb5d23b"
REFERENCE_REPORT_SHA = "13b470fe2f5ac6bdb9b706c2b212f67244c27180b5a88feff384dc4454c595d9"
REPRODUCTION_TOLERANCE = .25


def read(path):
    return json.loads(Path(path).read_text())


def code_identity():
    root = Path(__file__).parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def prepare(args):
    if sha256(args.reference / "report.json") != REFERENCE_REPORT_SHA:
        raise ValueError("completed Test A report changed")
    base, report = read(args.reference / "inputs.json"), read(args.reference / "report.json")
    if (base["id"] != REFERENCE_ID or digest({k: v for k, v in base.items() if k != "id"}) != base["id"]
            or not report["completed"] or report["smoke"] or report["sources"] != 77 or report["questions"] != 385):
        raise ValueError("complete frozen Test A required")
    if base["weights"] != "plain" or base["memory_seeds"] != list(analysis.SEEDS):
        raise ValueError("wrong reference weights or seeds")
    if base["code"] != test_a.code_identity() or base["versions"] != test_a.versions():
        raise ValueError("original helper code or runtime versions changed")
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("base model changed")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("starting LoRA changed")
    hashes = {}
    for source in base["sources"]:
        for q in source["queries"]:
            path = args.reference / "questions" / f"{digest(q['id'])}.json"
            old = read(path)
            if (not old["completed"] or old["input_id"] != base["id"] or old["id"] != q["id"]
                    or old["gold"] != q["answer"] or old["source_group"] != source["source_group"]
                    or old["steps_seen"] != q["planned_encoder_calls"]):
                raise ValueError("reference question differs")
            hashes[q["id"]] = sha256(path)
    if len(base["sources"]) != 77 or len(hashes) != 385:
        raise ValueError("all frozen sources/questions required")
    inputs = {"protocol": "b0-window40-v1", "reference": base, "reference_report_sha256": REFERENCE_REPORT_SHA,
              "reference_question_sha256": hashes, "code": code_identity(),
              "versions": {**test_a.versions(), "numpy": np.__version__},
              "reproduction_tolerance": REPRODUCTION_TOLERANCE, "budgets": {"window8": 8, "window40": 40},
              "reference_reservoir_budget": 40, "primary": "reservoir_minus_window40:300:nll",
              "contrasts": analysis.CONTRASTS, "bootstrap": {"draws": 10000, "seed": 7},
              "protocol_sha256": sha256(Path(__file__).parents[3] / "docs/complementary_memory/window40_protocol.md")}
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)


def load_inputs(args):
    value = read(args.out / "inputs.json")
    if value["id"] != digest({k: v for k, v in value.items() if k != "id"}):
        raise ValueError("B0 manifest corrupted")
    if value["code"] != code_identity() or value["versions"] != {**test_a.versions(), "numpy": np.__version__}:
        raise ValueError("B0 code or versions changed")
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


@torch.no_grad()
def evaluate_source(bb, inputs, source, args, auxiliary):
    """Same decode/schedule/packing as Test A, with two budgeted live windows."""
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
    live = {"window8": NativeStream(bb, head, "window", recent=8),
            "window40": NativeStream(bb, head, "window", recent=40)}
    if auxiliary:
        live["reservoir17"] = NativeStream(bb, head, "reservoir", recent=8, capacity=32,
                                            seed=test_a.memory_seed(source["source_group"], 17))
    recent, snapshots, completed = deque(maxlen=8), {}, []
    parity_parts, parity, encoded = ([], [], []), None, 0
    seconds = {"vision": 0., "lm_step_all_streams": 0., "questions": 0., "packing_audit": 0.}
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
        if (list(live["window8"].recent) != old["recent_steps"]
                or [b.index for b in snapshots[q["target_id"]]] != old["snapshot_steps"]
                or live["window8"].counter.pos != old["streams"]["window_stream"]["next_position"]):
            raise ValueError("reference recent/snapshot indices or native coordinates differ")
        started = time.monotonic()
        tail = prompts[q["id"]][1]
        before = {name: s.snapshot() for name, s in live.items()}
        if len({s.counter.pos for s in live.values()}) != 1:
            raise ValueError("live policy positions diverged")
        def run(stream):
            value = score(bb, stream.fork(), stream.counter.pos, tail, len(q["options"]))
            if len(value["option_logits_fp32"]) != len(q["options"]) or not all(math.isfinite(v) for v in value["option_logits_fp32"]):
                raise ValueError("nonfinite or wrong option logits")
            return value
        predictions = {name: run(live[name]) for name in ("window8", "window40")}
        checks = {"window8": reproduction(predictions["window8"], old["predictions"]["window_stream"], q["answer"])}
        aux_prediction = None
        if auxiliary:
            aux_prediction = run(live["reservoir17"])
            checks["reservoir17"] = reproduction(aux_prediction, old["predictions"]["reservoir17:S1K1"], q["answer"])
        if run(live["window8"]) != predictions["window8"]:
            raise RuntimeError("question re-answer is not deterministic")
        for name, stream in live.items():
            stream.verify(before[name])
        tail_positions = PositionCounter(live["window8"].counter.pos).text(len(tail), bb.device)
        tail_hash = tensor_digest({"embeds": bb.embed_ids(tail), "positions": tail_positions})
        torch.cuda.synchronize()
        seconds["questions"] += time.monotonic() - started
        value = {"completed": True, "input_id": inputs["id"], "reference_input_id": base["id"],
                 "id": q["id"], "source_group": source["source_group"], "gold": q["answer"],
                 "task": q.get("task"), "target_id": q["target_id"], "delay_seconds": q["delay_seconds"],
                 "target_time": q["target_time"], "query_time": q["query_time"], "steps_seen": encoded,
                 "predictions": predictions, "reproduction": checks, "auxiliary_seed17_prediction": aux_prediction,
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
            audit_embeds, audit_pos = block_inputs(bb, block, copy.copy(live["window8"].counter))
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
            seconds["lm_step_all_streams"] += time.monotonic() - started
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
        "parity": parity, "live_streams": len(live), "auxiliary_seed17": auxiliary, "seconds": seconds,
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
    rows, sources, refs = [], [], {}
    for source in selected:
        audit = read(args.out / "sources" / f"{digest(source['source_group'])}.json")
        if not audit["completed"] or audit["input_id"] != inputs["id"]:
            raise ValueError("incomplete source")
        sources.append(audit)
        for q in source["queries"]:
            row = read(args.out / "questions" / f"{digest(q['id'])}.json")
            if (not row["completed"] or row["input_id"] != inputs["id"] or row["steps_seen"] != q["planned_encoder_calls"]
                    or row["id"] != q["id"] or row["gold"] != q["answer"] or row["source_group"] != source["source_group"]):
                raise ValueError("incomplete or wrong question")
            for name in ("window8", "window40"):
                if row["streams"][name]["kv_steps"] != min(q["planned_encoder_calls"], inputs["budgets"][name]):
                    raise ValueError("wrong retained budget")
            refs[q["id"]] = reference_row(args, inputs, q["id"])
            rows.append(row)
    checks = [v for row in rows for v in row["reproduction"].values()]
    by_arm = {}
    for name in sorted({name for row in rows for name in row["reproduction"]}):
        values = [row["reproduction"][name] for row in rows if name in row["reproduction"]]
        gaps = [v["max_logit_gap"] for v in values]
        by_arm[name] = {"queries": len(values), "max_gap": max(gaps),
                        "q50_q95_q99": np.quantile(gaps, [.5, .95, .99]).tolist(),
                        "changed_choices": sum(v["choice_changed"] for v in values)}
    result = {"completed": True, "smoke": args.smoke, "inputs": inputs["id"], "sources": len(sources),
              "questions": len(rows), "new_prediction_vectors": len(rows) * 2,
              "max_parity_gap": max(s["parity"]["max_letter_logit_gap"] for s in sources),
              "max_reproduction_gap": max(v["max_logit_gap"] for v in checks),
              "reproduction_by_arm": by_arm,
              "reproduction_changed_choices": sum(v["choice_changed"] for v in checks),
              "peak_allocated_gib": max(s["peak_allocated_gib"] for s in sources),
              "total_steps": sum(s["steps"] for s in sources),
              "total_seconds": {k: sum(s["seconds"][k] for s in sources) for k in sources[0]["seconds"]},
              "reference_scope": "Per-seed Test A logits reused by frozen hashes/code/weights/runtime and measured reproduction; old per-question packing digests were not recorded. Drift in reused Reservoir seeds not rerun remains unbounded."}
    if not args.smoke:
        if len(sources) != 77 or len(rows) != 385:
            raise ValueError("all 77 sources/385 questions required")
        result["analysis"] = analysis.analyze(rows, refs)
        result["outcome"] = analysis.outcome(result["analysis"])
        budget = {}
        for delay in sorted({r["delay_seconds"] for r in rows}):
            subset = [r for r in rows if r["delay_seconds"] == delay]
            arms = {name: [r["streams"][name] for r in subset] for name in ("window8", "window40")}
            # Source/seed reference budgets are averaged within source, as are outcome metrics.
            arms["reservoir"] = [{key: sum(refs[r["id"]]["streams"][f"reservoir{s}"][key]
                                               for s in analysis.SEEDS) / len(analysis.SEEDS)
                                   for key in ("kv_steps", "kv_tokens", "kv_bytes", "state_bytes")}
                                  for r in subset]
            budget[str(delay)] = {name: {key: sum(v[key] for v in values) / len(values)
                                        for key in ("kv_steps", "kv_tokens", "kv_bytes", "state_bytes")}
                                 for name, values in arms.items()}
        result["mean_budgets"] = budget
    write_json(args.out / ("smoke-ok.json" if args.smoke else "report.json"), result)


def run(args):
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("exactly two distinct GPUs required")
    prepare(args)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "TRACE_WANDB": "0", "TRACE_SWANLAB": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    check = "import torch; assert torch.cuda.device_count()==2; assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2))>24"
    run_jobs([("gpu-check", [sys.executable, "-c", check], gpus)], env, args.out / "logs")
    command = [sys.executable, "-m", "trace_vl.b0_window40.run", "worker", "--out", str(args.out),
               "--reference", str(args.reference), "--shards", "2"]
    for phase in (("smoke",) if args.smoke_only else ("smoke", "full")):
        extra = ["--smoke"] if phase == "smoke" else []
        jobs = [(f"{phase}-{i}", [*command, "--shard", str(i), *extra], [gpu]) for i, gpu in enumerate(gpus)]
        run_jobs(jobs, env, args.out / "logs")
        args.smoke = phase == "smoke"
        report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "worker", "report", "run"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--gpus", default="2,3")
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
