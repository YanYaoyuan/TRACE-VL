"""Matched LoRA training on question-blind retained long-stream observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.longstream import expand_pooled
from trace_vl.core_vlm.native import NativeVideoQALoss, prepare_native
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.train import checked_checkpoint, sample_indices, tensor_digest
from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from .prepare import rows
from .selection import QuestionSelector


def sample(bb, selector, entry, row, layouts=None):
    # Selection sees no answer or evidence annotations.
    features = entry["features"].to(bb.device)
    ends = [pair[-1] for pair in entry["times"]]
    if selector is None:
        selected = torch.arange(max(0, len(features) - 8), len(features), device=bb.device)
        values = features[selected]
    else:
        query = bb.embed_ids(bb.ids(row["question"])).detach().float().mean((0, 1))
        values, selected, _ = selector(features, ends, query, float(row["query_time"]))
    chosen = selected.detach().cpu().tolist()
    times = np.asarray([entry["times"][i] for i in chosen])
    images = np.zeros((len(chosen), 2, *entry["hw"], 3), dtype=np.uint8)
    audit = {
        "source_fps": entry["fps"],
        "source_frame_count": entry["frame_count"],
        "frame_indices": [j for i in chosen for j in entry["indices"][i]],
        "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
    }
    safe = {
        "question": row["question"],
        "options": row["options"],
        "answer": 0,
        "query_time": row["query_time"],
    }
    layout_key = digest([safe, audit, entry["hw"], times.tolist()])
    if layouts is not None and layout_key in layouts:
        layout = layouts.pop(layout_key)
    else:
        layout = prepare_native(
            bb.processor, images, times, safe, audit, allow_native_timestamp_approximation=True
        )
        # Cached visual tensors are used below; zero-pixel processor outputs
        # are needed only for native packing, never for another vision encode.
        layout.inputs.pop("pixel_values_videos", None)
    if layouts is not None:
        layouts[layout_key] = layout
        if len(layouts) > 1024:
            layouts.popitem(last=False)
    steps = [
        expand_pooled(
            SimpleNamespace(tokens=value, grid=tuple(entry["grid"]), end=pair[-1], times=tuple(pair))
        )
        for value, pair in zip(values, times.tolist(), strict=True)
    ]
    return (
        steps,
        {
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "query_time": row["query_time"],
            "native_layout": layout,
        },
        chosen,
    )


def gradient_norm(parameters):
    """One host synchronization per group; preserve optimizer/clipping math."""
    squares = [p.grad.float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(squares).double().sum().sqrt()) if squares else 0.0


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    da = json.loads((args.data / "data-audit.json").read_text())
    ci = json.loads((args.cache / "inputs.json").read_text())
    if ci["data"] != da["input_id"]:
        raise ValueError("feature cache belongs to another data queue")
    for s, h in da["manifest_sha256"].items():
        if sha256(args.data / f"{s}.jsonl") != h:
            raise ValueError("manifest changed")
    previous = json.loads((args.adapter.parent.parent / "run.json").read_text())
    checked_checkpoint(args.adapter.parent, previous["id"])
    training = rows(args.data / "train.jsonl")
    dev = rows(args.data / "dev.jsonl")
    identity = {
        "protocol": "matched-longstream-reader-pilot-v1",
        "arm": args.arm,
        "data": da["input_id"],
        "cache": ci["id"],
        "model": checkpoint_identity(args.model),
        "initial_adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "initial_adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
        "updates": args.updates,
        "accumulation": 4,
        "seed": 17,
        "lr": 5e-5,
        "read_tokens": 128,
        "bank_slots": 36,
        "code_sha256": sha256(__file__),
        "selector_code_sha256": sha256(Path(__file__).with_name("selection.py")),
    }
    identity["id"] = digest(identity)
    ensure_run_manifest(args.out / "run.json", identity)
    torch.manual_seed(17)
    torch.cuda.set_per_process_memory_fraction(0.4)
    if torch.cuda.mem_get_info()[0] < 18 * 2**30:
        raise RuntimeError("18GiB free required for co-running pilot")
    bb = Backbone(
        BackboneConfig(name=str(args.model), dtype="bfloat16", local_files_only=True),
        StreamConfig(pixels_per_frame=65536),
        device="cuda:0",
    )
    bb.add_lora(8, 16, adapter=str(args.adapter))
    bb.enable_checkpointing()
    for module in bb.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    selector = QuestionSelector(bb.hidden).to(bb.device) if args.arm == "learned" else None
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4), answer_end_token_id=bb.tokenizer.eos_token_id)
    params = {name: p for name, p in bb.model.named_parameters() if p.requires_grad}
    if selector is not None:
        params.update({"selector." + n: p for n, p in selector.named_parameters()})
    optimizer = torch.optim.AdamW(params.values(), lr=5e-5, weight_decay=0.0)
    checkpoint = args.out / "checkpoint.pt"
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["run"] != identity["id"]:
            raise ValueError("checkpoint identity changed")
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(saved["parameters"][name].to(p.device))
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        start = saved["update"]
        last = json.loads((args.out / "metrics.jsonl").read_text().splitlines()[-1])
        if last["update"] != start or tensor_digest(params) != last["parameters_sha256"]:
            raise ValueError("reloaded checkpoint did not exactly restore trained parameters")
        write_json(
            args.out / "restore-audit.json",
            {
                "completed": True,
                "update": start,
                "parameters_sha256": tensor_digest(params),
                "run": identity["id"],
            },
        )
    end = min(args.updates, args.stop_after or args.updates)
    layouts = OrderedDict()
    for update in range(start, end):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        ids = []
        losses = []
        reads = []
        lr = (
            5e-5
            * min(1.0, (update + 1) / 10)
            * (0.1 + 0.9 * (1 + math.cos(math.pi * update / args.updates)) / 2)
        )
        for g in optimizer.param_groups:
            g["lr"] = lr
        for i in sample_indices(len(training), update, 1, 4, 0, 17):
            row = training[i]
            path = args.cache / f"{digest(row['id'])}.pt"
            proof = json.loads(path.with_suffix(".json").read_text())
            if sha256(path) != proof["sha256"]:
                raise ValueError("stored visual cache changed")
            entry = torch.load(path, map_location="cpu", weights_only=True)
            if entry["input_id"] != ci["id"]:
                raise ValueError("wrong feature identity")
            steps, record, selected = sample(bb, selector, entry, row, layouts)
            loss, info = net(steps, record)
            if not torch.isfinite(loss) or info["visual_tokens_total"] > 128:
                raise ValueError("nonfinite/budget")
            (loss / 4).backward()
            ids.append(row["id"])
            losses.append(float(loss.detach()))
            reads.append(selected)
        lm_grad = gradient_norm(p for n, p in params.items() if not n.startswith("selector."))
        selector_grad = gradient_norm(p for n, p in params.items() if n.startswith("selector."))
        if not math.isfinite(lm_grad) or lm_grad <= 0:
            raise ValueError("no finite LM QA gradient")
        if selector is not None and (not math.isfinite(selector_grad) or selector_grad <= 0):
            raise ValueError("no selector QA gradient")
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        metric = {
            "update": update + 1,
            "loss": sum(losses) / 4,
            "question_ids": ids,
            "selected_bank_indices": reads,
            "lm_gradient": lm_grad,
            "selector_gradient": selector_grad,
            "lr": lr,
            "seconds": time.monotonic() - begin,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "parameters_sha256": tensor_digest(params),
        }
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if (update + 1) % 25 == 0 or update + 1 == end:
            state = {
                "run": identity["id"],
                "update": update + 1,
                "parameters": {n: p.detach().cpu() for n, p in params.items()},
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            }
            temp = checkpoint.with_suffix(".partial")
            torch.save(state, temp)
            temp.replace(checkpoint)
    write_json(
        args.out / "status.json",
        {
            "completed": end == args.updates,
            "update": end,
            "run": identity["id"],
            "checkpoint_sha256": sha256(checkpoint),
            "selector_trained": selector is not None,
        },
    )
    if end != args.updates:
        return
    bb.model.eval()
    if selector is not None:
        selector.eval()
    predictions = []
    with torch.no_grad():
        for row in dev:
            entry = torch.load(args.cache / f"{digest(row['id'])}.pt", map_location="cpu", weights_only=True)
            steps, record, selected = sample(bb, selector, entry, row, layouts)
            embeds, positions, *_ = net.prefix(steps, record)
            hidden = bb.lm(embeds, positions, cache=None, use_cache=False)[0, -1].float()
            candidates = [bb.ids(chr(65 + i)) for i in range(len(row["options"]))]
            if any(len(x) != 1 for x in candidates):
                raise ValueError("single token MC options required")
            scores = bb.lm_head.weight[[x[0] for x in candidates]].float() @ hidden
            predictions.append(
                {
                    "id": row["id"],
                    "source_group": row["source_group"],
                    "gold": row["answer"],
                    "choice": int(scores.argmax()),
                    "logits": scores.tolist(),
                    "selected_bank_indices": selected,
                    "read_times": [entry["times"][i] for i in selected],
                    "visual_tokens": 16 * len(selected),
                }
            )
    write_json(args.out / "dev-predictions.json", predictions)
    write_json(
        args.out / "report.json",
        {
            "completed": True,
            "run": identity["id"],
            "questions": len(predictions),
            "correct": sum(p["choice"] == p["gold"] for p in predictions),
            "updates": end,
            "scope": "Previously used ActivityNet development pilot; no official/held-out-final or novelty claim.",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "out", "model", "adapter"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--arm", choices=("reservoir", "learned"), required=True)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--stop-after", type=int)
    run(p.parse_args())


if __name__ == "__main__":
    main()
