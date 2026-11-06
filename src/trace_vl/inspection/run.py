"""Captioning, evaluation, scoring and LoRA training over prepared snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from .evidence import METHODS, PackConfig, Packer, clock, fit
from .meva import gap_bucket

BUDGET_FREE = {"none", "current", "textlog", "transition_meta"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_snapshot(root, question_id):
    path = Path(root) / "snapshots" / question_id / "snapshot.json"
    return json.loads(path.read_text()) if path.exists() else None


def _vlm(model_cfg, device, adapter=None):
    from .vlm import VLM

    return VLM(
        model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct"),
        device=device,
        dtype=model_cfg.get("dtype", "bfloat16"),
        unit=model_cfg.get("unit", 32),
        adapter=adapter,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        local_files_only=model_cfg.get("local_files_only", False),
    )


def caption(root, model_cfg, caption_tokens=256, shard=(0, 1), device="cuda"):
    """Online caption log for the textlog baseline; query-independent and resumable."""
    root = Path(root)
    streams = sorted(p for p in (root / "streams").iterdir() if (p / "caption_frames.json").exists())
    streams = streams[shard[0] :: shard[1]]
    vlm = _vlm(model_cfg, device)
    unit, total, started = model_cfg.get("unit", 32), 0, time.perf_counter()
    for stream in streams:
        target = stream / "captions.jsonl"
        done = {round(c["t"], 3) for c in read_jsonl(target)} if target.exists() else set()
        with target.open("a") as handle:
            for frame in json.loads((stream / "caption_frames.json").read_text()):
                if round(frame["t"], 3) in done:
                    continue
                image = fit(Image.open(root / frame["image"]).convert("RGB"), caption_tokens, unit)
                prompt = (
                    f"This is a frame from a fixed security camera at {clock(frame['t'])}. In one sentence, "
                    "describe the people, any objects being carried, put down or left behind, and vehicles, "
                    "and say where they are in the frame."
                )
                text = vlm.generate([{"type": "image", "image": image}, {"type": "text", "text": prompt}])
                handle.write(json.dumps({"t": frame["t"], "caption": text}) + "\n")
                handle.flush()
                total += 1
    return {"streams": len(streams), "captions": total, "seconds": round(time.perf_counter() - started, 1)}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_identity(adapter):
    if not adapter:
        return None
    path = Path(adapter)
    if not path.is_dir():
        return {"path": str(adapter)}
    files = sorted(f for f in path.iterdir() if f.is_file() and f.suffix in {".json", ".safetensors", ".bin"})
    return {"path": str(path.resolve()), "files": {f.name: _sha256(f) for f in files}}


def run_fingerprint(questions_path, root, model_cfg, adapter, pack_cfg, split):
    """Everything that changes a prediction; resuming into a directory requires an exact match."""
    reports = sorted(Path(root).glob("prepare_report*.json"))
    identity = {
        "model": {k: model_cfg.get(k) for k in ("name", "revision", "dtype", "unit", "attn_implementation")},
        "adapter": _adapter_identity(adapter),
        "pack": dict(pack_cfg.__dict__),
        "questions_sha256": _sha256(questions_path),
        "prepared": {r.name: _sha256(r) for r in reports},
        "split": split,
    }
    identity["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return identity


def _camera_lookup(questions_path):
    streams = Path(questions_path).parent / "streams.json"
    if not streams.exists():
        return {}
    return {name: spec.get("camera", name) for name, spec in json.loads(streams.read_text()).items()}


def evaluate(questions_path, root, model_cfg, methods, budgets, output, split="test", pack=None,
             adapter=None, shard=(0, 1), device="cuda", limit=None):  # fmt: skip
    """Score every (question, method, budget); resumable, one JSON line per prediction."""
    questions = [q for q in read_jsonl(questions_path) if q["split"] == split][shard[0] :: shard[1]]
    if limit:
        questions = questions[:limit]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"predictions.{shard[0]}of{shard[1]}.jsonl"
    base = PackConfig.from_dict({**(pack or {}), "unit": model_cfg.get("unit", 32)})
    fingerprint = run_fingerprint(questions_path, root, model_cfg, adapter, base, split)
    run_file = output / "run.json"
    if run_file.exists():
        previous = json.loads(run_file.read_text())
        if previous["id"] != fingerprint["id"]:
            changed = sorted(k for k in fingerprint if k != "id" and previous.get(k) != fingerprint[k])
            raise ValueError(
                f"{output} holds predictions from a different run (differs in {changed}); "
                "use a new --output directory"
            )
    else:
        run_file.write_text(json.dumps(fingerprint, indent=2) + "\n")
    done = set()
    if target.exists():
        for p in read_jsonl(target):
            if p.get("run") != fingerprint["id"]:
                raise ValueError(f"{target} contains predictions without this run id; use a new --output")
            done.add((p["id"], p["method"], p["budget"]))
    cameras = _camera_lookup(questions_path)
    vlm = _vlm(model_cfg, device, adapter)
    missing = 0
    with target.open("a") as handle:
        for question in questions:
            snapshot = load_snapshot(root, question["id"])
            if snapshot is None:
                missing += 1
                continue
            for method in methods:
                for budget in [0] if method in BUDGET_FREE else budgets:
                    if (question["id"], method, budget) in done:
                        continue
                    cfg = PackConfig(**{**base.__dict__, "budget_tokens": budget or base.budget_tokens})
                    segments, info = Packer(root, cfg).pack(snapshot, question, method)
                    started = time.perf_counter()
                    choice, scores, tokens = vlm.choose(segments, len(question["options"]))
                    record = {
                        "run": fingerprint["id"],
                        "id": question["id"],
                        "stream": question["stream"],
                        "camera": question.get("camera") or cameras.get(question["stream"], question["stream"]),
                        "type": question["type"],
                        "gap": question["gap"],
                        "censored": question.get("censored", False),
                        "method": method,
                        "budget": budget,
                        "answer": question["answer"],
                        "prediction": choice,
                        "scores": [round(s, 4) for s in scores],
                        "visual_tokens": tokens,
                        "seconds": round(time.perf_counter() - started, 3),
                        **info,
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
    return {
        "run": fingerprint["id"],
        "questions": len(questions),
        "missing_snapshots": missing,
        "predictions": str(target),
    }


def _balanced(rows):
    by_answer = defaultdict(list)
    for r in rows:
        by_answer[r["answer"]].append(r["prediction"] == r["answer"])
    return sum(sum(v) / len(v) for v in by_answer.values()) / len(by_answer) if by_answer else math.nan


def _bootstrap(rows, metric, samples=2000, seed=7):
    """95% interval resampling whole cameras (with all their streams), not questions or streams.

    One camera contributes several streams (other dates, gaps between clips) and many questions that
    share scene and viewpoint, so the camera is the independent unit.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("camera") or r["stream"]].append(r)
    keys, rng, values = sorted(groups), random.Random(seed), []
    if len(keys) < 2:
        return [math.nan, math.nan]
    for _ in range(samples):
        pick = [row for k in rng.choices(keys, k=len(keys)) for row in groups[k]]
        values.append(metric(pick))
    values.sort()
    return [round(values[int(0.025 * samples)], 4), round(values[int(0.975 * samples) - 1], 4)]


def score(prediction_dir, output=None, exclude_censored=False):
    rows = [r for p in sorted(Path(prediction_dir).glob("predictions.*.jsonl")) for r in read_jsonl(p)]
    runs = {r.get("run") for r in rows}
    if len(runs) > 1:
        raise ValueError(f"predictions in {prediction_dir} come from several runs: {sorted(map(str, runs))}")
    if exclude_censored:
        rows = [r for r in rows if not r["censored"]]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["budget"])].append(r)
    table = []
    for (method, budget), items in sorted(groups.items()):
        entry = {
            "method": method,
            "budget": budget,
            "n": len(items),
            "mean_visual_tokens": round(sum(r["visual_tokens"] for r in items) / len(items), 1),
            "accuracy": round(sum(r["prediction"] == r["answer"] for r in items) / len(items), 4),
            "macro_balanced_accuracy": None,
            "by_type": {},
            "by_gap": {},
        }
        types = defaultdict(list)
        for r in items:
            types[r["type"]].append(r)
        per_type = {k: _balanced(v) for k, v in types.items()}
        entry["macro_balanced_accuracy"] = round(sum(per_type.values()) / len(per_type), 4)
        entry["ci95_macro_balanced_accuracy"] = _bootstrap(
            items,
            lambda rs: sum(_balanced([r for r in rs if r["type"] == t]) for t in types) / len(types)
            if all(any(r["type"] == t for r in rs) for t in types)
            else math.nan,
        )
        for kind, values in types.items():
            entry["by_type"][kind] = {"n": len(values), "balanced_accuracy": round(per_type[kind], 4)}
            buckets = defaultdict(list)
            for r in values:
                buckets[gap_bucket(r["gap"])].append(r)
            entry["by_gap"][kind] = {
                b: {"n": len(v), "accuracy": round(sum(r["prediction"] == r["answer"] for r in v) / len(v), 4)}
                for b, v in sorted(buckets.items())
            }
        table.append(entry)
    run_file = Path(prediction_dir) / "run.json"
    report = {
        "rows": len(rows),
        "run": json.loads(run_file.read_text()) if run_file.exists() else None,
        "bootstrap_unit": "camera",
        "cameras": len({r.get("camera") or r["stream"] for r in rows}),
        "exclude_censored": exclude_censored,
        "results": table,
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def train(questions_path, root, model_cfg, method, budget, output, training=None, pack=None, init_adapter=None):
    """LoRA SFT on the answer letter. Launch with torchrun for multiple GPUs (one process per GPU)."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    settings = {
        "steps": 2000,
        "lr": 1e-4,
        "warmup": 100,
        "grad_accum": 8,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "save_every": 500,
        "seed": 7,
        "methods_mix": None,  # e.g. ["transition", "uniform"] to sample a method per example
        **(training or {}),
    }
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    try:
        rng = random.Random(settings["seed"] + rank)
        questions = [q for q in read_jsonl(questions_path) if q["split"] == "train"]
        items = [(q, s) for q in questions if (s := load_snapshot(root, q["id"])) is not None]
        if not items:
            raise ValueError("no training snapshots; run prepare on the train split first")
        vlm = _vlm(model_cfg, device, adapter=None)
        if init_adapter:
            from peft import PeftModel

            vlm.model = PeftModel.from_pretrained(vlm.model, init_adapter, is_trainable=True)
        else:
            vlm.add_lora(settings["rank"], settings["alpha"], settings["dropout"])
        vlm.model.train()
        params = [p for p in vlm.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=0.0)
        module = vlm.model
        if world > 1:
            module = DistributedDataParallel(vlm.model, device_ids=[local])
        cfg = PackConfig.from_dict({**(pack or {}), "unit": model_cfg.get("unit", 32), "budget_tokens": budget})
        packer = Packer(root, cfg)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        log = output / "metrics.jsonl"
        started = time.perf_counter()
        for step in range(settings["steps"]):
            factor = min(1.0, (step + 1) / max(1, settings["warmup"]))
            progress = max(0.0, (step - settings["warmup"]) / max(1, settings["steps"] - settings["warmup"]))
            for group in optimizer.param_groups:
                group["lr"] = settings["lr"] * factor * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for micro in range(settings["grad_accum"]):
                question, snapshot = rng.choice(items)
                chosen = rng.choice(settings["methods_mix"]) if settings["methods_mix"] else method
                segments, _ = packer.pack(snapshot, question, chosen)
                answer = "ABCDEFGH"[question["answer"]]
                sync = world == 1 or micro == settings["grad_accum"] - 1
                context = module.no_sync() if not sync else torch.enable_grad()
                with context, torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    loss, _ = vlm.loss(segments, answer) if world == 1 else _ddp_loss(module, vlm, segments, answer)
                    (loss / settings["grad_accum"]).backward()
                total += float(loss) / settings["grad_accum"]
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if rank == 0:
                record = {
                    "step": step + 1,
                    "loss": round(total, 5),
                    "grad_norm": round(float(norm), 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed": round(time.perf_counter() - started, 1),
                }
                with log.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                if (step + 1) % 10 == 0:
                    print(json.dumps(record), flush=True)
                if (step + 1) % settings["save_every"] == 0 or step + 1 == settings["steps"]:
                    vlm.model.save_pretrained(output / "adapter")
                    meta = {"step": step + 1, "method": method, "budget": budget, "settings": settings,
                            "model": model_cfg, "pack": cfg.__dict__}  # fmt: skip
                    (output / "adapter" / "trace_inspection.json").write_text(json.dumps(meta, indent=2) + "\n")
        return output / "adapter"
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _ddp_loss(module, vlm, segments, answer):
    """Route the forward pass through the DDP wrapper so gradients are all-reduced."""
    inner = vlm.model
    vlm.model = module
    try:
        return vlm.loss(segments, answer)
    finally:
        vlm.model = inner
