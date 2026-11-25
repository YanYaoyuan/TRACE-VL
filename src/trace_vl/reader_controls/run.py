"""Score frozen visual-dependence controls. Evaluation only: no training, no new video decoding."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import run_jobs
from trace_vl.core_vlm.native import NativeVideoQALoss
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import grid_pool, question_and_answer
from trace_vl.core_vlm.train import checked_checkpoint, tensor_digest
from trace_vl.longstream_training.prepare import rows
from trace_vl.longstream_training.train import sample
from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.positions import PositionCounter
from trace_vl.v3.run_io import ensure_run_manifest

from . import analysis, policies

PILOT = Path("/data/yyy/tracevl-runs/trained-reader-a898131-001")
DATA = Path("/data/yyy/tracevl-data/trained-reader-20261125-001")
AUDIT = Path("/data/yyy/tracevl-runs/reader-cpu-audit-20261125-1850/summary.json")
MODEL = Path("/data/yyy/tracevl-data/core-pool-probe-2efc7cc/model")
ADAPTER = Path("/data/yyy/tracevl-runs/core-pool-2efc7cc-001/train/plain/checkpoint-002500/adapter")
DEV_SHA256 = "a0a772caeb1f4876e65f76bcf9bd4e599a7fb8461d0885fdb89c8058f2cf150e"
SOURCES = 115
# Completed CPU audit (PR #9). Reproduced only as an input-identity gate.
AUDITED_HITS = {"bank": 111, "recent4": 0, "latest8": 34, "learned8": 66}
AUDITED_RANDOM8 = 86.53
TRAINED = {"trained_reservoir": ("reservoir", "latest8"), "trained_learned": ("learned", "learned8")}
REPRODUCTION_TOLERANCE = 0.25
GRAY = 128


def load_json(path):
    return json.loads(Path(path).read_text())


def load_entry(cache, row, cache_id):
    path = cache / f"{digest(row['id'])}.pt"
    proof = load_json(path.with_suffix(".json"))
    if sha256(path) != proof["sha256"]:
        raise ValueError("stored visual cache changed")
    entry = torch.load(path, map_location="cpu", weights_only=True)
    if entry["input_id"] != cache_id or entry["id"] != row["id"]:
        raise ValueError("wrong feature identity")
    return entry, proof["sha256"]


def freeze(args):
    """CPU only. Freeze every policy before any control is scored."""
    if sha256(args.data / "dev.jsonl") != DEV_SHA256:
        raise ValueError("development manifest differs from the audited one")
    dev = rows(args.data / "dev.jsonl")
    if len(dev) != SOURCES or len({r["source_group"] for r in dev}) != SOURCES:
        raise ValueError("expected one question from each of 115 sources")
    cache = args.pilot / "features"
    cache_id = load_json(cache / "inputs.json")["id"]
    predictions = {
        arm: load_json(args.pilot / arm / "dev-predictions.json") for arm in ("reservoir", "learned")
    }
    saved = {arm: {p["id"]: p for p in ps} for arm, ps in predictions.items()}
    if any(set(s) != {r["id"] for r in dev} or len(predictions[a]) != SOURCES for a, s in saved.items()):
        raise ValueError("saved prediction coverage differs")
    questions, hits, expected_random, features = [], dict.fromkeys(AUDITED_HITS, 0), 0.0, {}
    for row in dev:
        entry, features[row["id"]] = load_entry(cache, row, cache_id)
        reads, audit = policies.build(row, entry["times"], saved["learned"][row["id"]]["selected_bank_indices"])
        if saved["reservoir"][row["id"]]["selected_bank_indices"] != reads["latest8"]:
            raise ValueError("saved baseline did not read the latest eight steps")
        for name in hits:
            hits[name] += audit["evidence_hit"][name]
        expected_random += audit["random8_expected_hit"]
        questions.append({"id": row["id"], "source_group": row["source_group"], "reads": reads, "audit": audit})
    if hits != AUDITED_HITS or abs(expected_random - AUDITED_RANDOM8) > 0.01:
        raise ValueError(f"evidence definition differs from the CPU audit: {hits}, {expected_random}")
    here = Path(__file__).parent
    inputs = {
        "protocol": "reader-controls-v1",
        "dev_sha256": DEV_SHA256,
        "cache": cache_id,
        "feature_sha256": features,
        "model": checkpoint_identity(args.model),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "checkpoints": {w: sha256(args.pilot / arm / "checkpoint.pt") for w, (arm, _) in TRAINED.items()},
        "predictions": {a: sha256(args.pilot / a / "dev-predictions.json") for a in saved},
        "cpu_audit_summary_sha256": sha256(args.audit),
        "code": {p.name: sha256(p) for p in sorted(here.glob("*.py"))},
        "read_seeds": list(policies.SEEDS),
        "caliper_seconds": policies.CALIPER_SECONDS,
        "blank": {"uint8": GRAY, "temporal_steps": 8, "stored_step": 0},
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "questions": questions,
    }
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)
    matched = [q["audit"]["pair"] for q in questions if q["audit"]["pair"]["failure"] is None]
    gaps = [g for p in matched for g in p["age_difference_seconds"]]
    failures = {}
    for q in questions:
        reason = q["audit"]["pair"]["failure"]
        if reason:
            failures[reason] = failures.get(reason, 0) + 1
    write_json(
        args.out / "matching.json",
        {
            "inputs": inputs["id"],
            "sources_matched": len(matched),
            "failures": failures,
            "pairs_read": sum(p["pairs_read"] for p in matched),
            "unmatched_evidence_steps": sum(q["audit"]["pair"]["unmatched_evidence_steps"] for q in questions),
            "steps_per_arm_min": min((p["steps_per_arm"] for p in matched), default=None),
            "residual_age_difference_seconds": {
                "mean_signed": float(np.mean(gaps)),
                "mean_absolute": float(np.mean(np.abs(gaps))),
                "max_absolute": float(np.max(np.abs(gaps))),
                "source_mean_signed": float(np.mean([np.mean(p["age_difference_seconds"]) for p in matched])),
            }
            if gaps
            else None,
            "warning": "Matching reduces, and does not eliminate, age confounding; unmatched sources are selected.",
        },
    )


def backbone(args):
    torch.manual_seed(17)
    torch.cuda.set_per_process_memory_fraction(0.4)
    if torch.cuda.mem_get_info()[0] < 18 * 2**30:
        raise RuntimeError("18GiB free required for co-running controls")
    return Backbone(
        BackboneConfig(name=str(args.model), dtype="bfloat16", local_files_only=True),
        StreamConfig(pixels_per_frame=65536),
        device="cuda:0",
    )


@torch.no_grad()
def blank_tokens(bb, hw):
    """Pooled tokens of a constant gray clip through the real frozen vision tower."""
    images = np.full((16, *hw, 3), GRAY, dtype=np.uint8)
    prepared = bb.processor.video_processor(
        videos=[images], do_sample_frames=False, do_resize=False, return_tensors="pt"
    )
    grid = prepared["video_grid_thw"]
    if grid.shape != (1, 3) or int(grid[0, 0]) != 8:
        raise ValueError("processor changed temporal batching")
    shape = bb.merged_grid(grid)
    steps = bb.encode(prepared["pixel_values_videos"], grid).split(shape[0] * shape[1])
    pooled = torch.stack([grid_pool(s, shape, (4, 4))[0] for s in steps])
    spread = float((pooled.float() - pooled[:1].float()).abs().max())
    return pooled[0].cpu().clone(), shape, spread


def make_blanks(args):
    inputs = load_json(args.out / "inputs.json")
    path = args.out / "blank.pt"
    if path.exists():
        if load_json(args.out / "blank.json")["sha256"] != sha256(path):
            raise ValueError("blank tokens changed")
        return
    sizes = {}
    cache = args.pilot / "features"
    for row in rows(args.data / "dev.jsonl"):
        entry, _ = load_entry(cache, row, inputs["cache"])
        previous = sizes.setdefault(tuple(entry["hw"]), tuple(entry["grid"]))
        if previous != tuple(entry["grid"]):
            raise ValueError("one frame size maps to two token grids")
    bb = backbone(args)
    tokens, spreads = {}, {}
    for hw, grid in sorted(sizes.items()):
        value, shape, spreads[f"{hw[0]}x{hw[1]}"] = blank_tokens(bb, hw)
        if tuple(shape) != grid or not torch.isfinite(value.float()).all():
            raise ValueError("blank grid or values invalid")
        tokens[f"{hw[0]}x{hw[1]}"] = value
    torch.save(tokens, path)
    write_json(
        args.out / "blank.json",
        {
            "inputs": inputs["id"],
            "sha256": sha256(path),
            "sizes": sorted(tokens),
            "max_difference_between_temporal_steps": spreads,
            "rule": "uint8 gray clip, frozen vision tower, 4x4 pooled step 0 reused at every read position",
        },
    )


def fixed(indices, blank=None):
    """Feed frozen indices through the pilot's own packing; optionally replace values only."""

    def select(features, _times, _query, _query_time):
        chosen = torch.tensor(indices, device=features.device)
        values = features[chosen]
        if blank is not None:
            values = blank.to(features)[None].expand_as(values)
        return values, chosen, None

    return select


def option_logits(bb, hidden, row):
    candidates = [bb.ids(chr(65 + i)) for i in range(len(row["options"]))]
    if any(len(x) != 1 for x in candidates):
        raise ValueError("single token MC options required")
    out = (bb.lm_head.weight[[x[0] for x in candidates]].float() @ hidden[0, -1].float()).tolist()
    if not all(math.isfinite(x) for x in out):
        raise ValueError("nonfinite option logits")
    return out


@torch.no_grad()
def text_only(bb, row):
    question, _ = question_and_answer(
        {"question": row["question"], "options": row["options"], "answer": 0}
    )
    prompt = bb.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": question}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = bb.ids(prompt)
    hidden = bb.lm(bb.embed_ids(ids), PositionCounter().text(len(ids), bb.device), cache=None, use_cache=False)
    return option_logits(bb, hidden, row)


@torch.no_grad()
def score_question(bb, net, entry, row, reads, blank):
    logits, tokens = {"text_only": text_only(bb, row)}, {"text_only": 0}
    layouts = OrderedDict()
    for name, indices in reads.items():
        positions = None
        for prefix, values in (("", None), ("blank:", blank)):
            steps, record, chosen = sample(bb, fixed(indices, values), entry, row, layouts)
            if chosen != indices:
                raise ValueError("packing changed the frozen read")
            embeds, pos, _, _, info = net.prefix(steps, record)
            if positions is not None and not torch.equal(pos, positions):
                raise ValueError("blank control changed positions")
            positions = pos
            if info["visual_tokens_total"] != 16 * len(indices):
                raise ValueError("visual token budget differs")
            hidden = bb.lm(embeds, pos, cache=None, use_cache=False)
            logits[prefix + name] = option_logits(bb, hidden, row)
            tokens[prefix + name] = info["visual_tokens_total"]
    return logits, tokens


def install(args, bb, weights, initial):
    """Load exactly the named frozen LoRA; selector parameters are never used."""
    live = {n: p for n, p in bb.model.named_parameters() if p.requires_grad}
    if weights == "start":
        source = initial
    else:
        arm = args.pilot / TRAINED[weights][0]
        status, identity = load_json(arm / "status.json"), load_json(arm / "run.json")
        metrics = [json.loads(s) for s in (arm / "metrics.jsonl").read_text().splitlines()]
        if not status["completed"] or status["checkpoint_sha256"] != sha256(arm / "checkpoint.pt"):
            raise ValueError("trained reader incomplete or changed")
        state = torch.load(arm / "checkpoint.pt", map_location="cpu", weights_only=True)
        if (
            state["run"] != identity["id"]
            or state["update"] != identity["updates"]
            or metrics[-1]["update"] != state["update"]
            or tensor_digest(state["parameters"]) != metrics[-1]["parameters_sha256"]
            or identity["initial_adapter_sha256"] != sha256(args.adapter / "adapter_model.safetensors")
        ):
            raise ValueError("trained reader identity differs")
        source = {n: p for n, p in state["parameters"].items() if not n.startswith("selector.")}
    if set(source) != set(live):
        raise ValueError("LoRA parameter names differ")
    with torch.no_grad():
        for name, p in live.items():
            p.copy_(source[name].to(p.device))
    if tensor_digest(live) != tensor_digest(source):
        raise ValueError("LoRA restore is not exact")


def worker(args):
    inputs = load_json(args.out / "inputs.json")
    by_id = {q["id"]: q for q in inputs["questions"]}
    queue = rows(args.data / "dev.jsonl")[args.shard :: 2]
    if args.smoke:
        queue = queue[:1]
    if load_json(args.out / "blank.json")["sha256"] != sha256(args.out / "blank.pt"):
        raise ValueError("blank tokens changed")
    blanks = torch.load(args.out / "blank.pt", map_location="cpu", weights_only=True)
    previous = load_json(args.adapter.parent.parent / "run.json")
    checked_checkpoint(args.adapter.parent, previous["id"])
    bb = backbone(args)
    bb.add_lora(8, 16, adapter=str(args.adapter))
    bb.model.eval()
    initial = {n: p.detach().cpu().clone() for n, p in bb.model.named_parameters() if p.requires_grad}
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4), answer_end_token_id=bb.tokenizer.eos_token_id)
    net.eval()
    for weights in analysis.WEIGHTS:
        install(args, bb, weights, initial)
        saved = {}
        if weights in TRAINED:
            arm = TRAINED[weights][0]
            if sha256(args.pilot / arm / "dev-predictions.json") != inputs["predictions"][arm]:
                raise ValueError("saved predictions changed")
            saved = {p["id"]: p for p in load_json(args.pilot / arm / "dev-predictions.json")}
        for row in queue:
            path = args.out / "scores" / weights / f"{digest(row['id'])}.json"
            if path.exists():
                if load_json(path)["inputs"] != inputs["id"]:
                    raise ValueError("existing score belongs to another run")
                continue
            entry, feature = load_entry(args.pilot / "features", row, inputs["cache"])
            if feature != inputs["feature_sha256"][row["id"]]:
                raise ValueError("feature differs from frozen inputs")
            h, w = entry["hw"]
            logits, tokens = score_question(
                bb, net, entry, row, by_id[row["id"]]["reads"], blanks[f"{h}x{w}"]
            )
            gap = None
            if saved:
                own = logits[TRAINED[weights][1]]
                gap = max(abs(a - b) for a, b in zip(own, saved[row["id"]]["logits"], strict=True))
                if gap > REPRODUCTION_TOLERANCE:
                    raise ValueError(f"saved reader prediction not reproduced: {gap}")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                path,
                {
                    "inputs": inputs["id"], "id": row["id"], "source_group": row["source_group"],
                    "weights": weights, "gold": row["answer"], "logits": logits, "visual_tokens": tokens,
                    "reproduction_logit_gap": gap,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                },
            )
            print(json.dumps({"scored": row["id"], "weights": weights, "gap": gap}), flush=True)


def load_scores(args, inputs):
    records = []
    for weights in analysis.WEIGHTS:
        for q in inputs["questions"]:
            path = args.out / "scores" / weights / f"{digest(q['id'])}.json"
            if not path.exists():
                continue
            r = load_json(path)
            expected = {"text_only", *q["reads"], *(f"blank:{n}" for n in q["reads"])}
            if r["inputs"] != inputs["id"] or r["weights"] != weights or set(r["logits"]) != expected:
                raise ValueError(f"score record invalid: {path}")
            records.append(r)
    return records


def report(args):
    inputs = load_json(args.out / "inputs.json")
    records = load_scores(args, inputs)
    if len(records) != SOURCES * len(analysis.WEIGHTS):
        raise ValueError("scoring incomplete")
    result = analysis.analyze(records, [q["source_group"] for q in inputs["questions"]])
    gaps = [r["reproduction_logit_gap"] for r in records if r["reproduction_logit_gap"] is not None]
    write_json(
        args.out / "report.json",
        {
            "completed": True,
            "inputs": inputs["id"],
            "outcome": analysis.outcome(result),
            "analysis": result,
            "matching": load_json(args.out / "matching.json"),
            "max_reproduction_logit_gap": max(gaps),
            "cpu_audit_summary_sha256": inputs["cpu_audit_summary_sha256"],
            "raw_video_evidence": "not scored; bank_evidence8 is limited to retained steps",
            "scope": "Reused single-question-per-source development set; not official or untouched evaluation.",
            "automatic_training": False,
        },
    )


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("exactly two GPUs required")
    freeze(args)
    env = {
        **os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "TRACE_WANDB": "0", "TRACE_SWANLAB": "0",
    }
    base = [sys.executable, "-m", "trace_vl.reader_controls.run"]
    common = []
    for name in ("out", "pilot", "data", "audit", "model", "adapter"):
        common += [f"--{name}", str(getattr(args, name))]

    def jobs(name, extra):
        return [
            (f"{name}-{i}", [*base, "worker", *common, "--shard", str(i), *extra], [g])
            for i, g in enumerate(gpus)
        ]

    logs = args.out / "logs"
    run_jobs([("blank", [*base, "blank", *common], [gpus[0]])], env, logs)
    run_jobs(jobs("smoke", ["--smoke"]), env, logs)
    inputs = load_json(args.out / "inputs.json")
    dev = rows(args.data / "dev.jsonl")
    smoke_ids = {r["id"] for r in dev[:2]}
    smoke = [r for r in load_scores(args, inputs) if r["id"] in smoke_ids]
    if len(smoke) != 2 * len(analysis.WEIGHTS):
        raise ValueError("smoke incomplete")
    # Deliberately no accuracy or NLL here.
    write_json(
        args.out / "smoke-ok.json",
        {
            "completed": True,
            "records": len(smoke),
            "max_reproduction_logit_gap": max(
                r["reproduction_logit_gap"] for r in smoke if r["reproduction_logit_gap"] is not None
            ),
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in smoke),
            "accuracy_gate": False,
        },
    )
    if args.smoke_only:
        return
    run_jobs(jobs("full", []), env, logs)
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "freeze", "blank", "worker", "report"))
    p.add_argument("--out", type=Path, required=True)
    for name, default in (
        ("pilot", PILOT), ("data", DATA), ("audit", AUDIT), ("model", MODEL), ("adapter", ADAPTER),
    ):
        p.add_argument(f"--{name}", type=Path, default=default)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--shard", type=int, choices=(0, 1), default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    {"run": run, "freeze": freeze, "blank": make_blanks, "worker": worker, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
