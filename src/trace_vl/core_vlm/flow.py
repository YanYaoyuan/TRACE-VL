"""Real pretrained-VLM training/reload check on training videos only.

This is an engineering check, not a generalization experiment or paper score.
No model downloads, no benchmark evaluation, no alert heads. The frozen vision
encoder supplies video features; answer CE updates the language-model LoRA.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text

from .data import decode_observation, encode_observation, read_split, sha256
from .qa import VideoQALoss, VisualStep


def run(args):
    out = Path(args.out)
    # Never overwrite earlier evidence or confuse a partial attempt with success.
    out.mkdir(parents=True, exist_ok=False)
    if min(args.examples, args.updates, args.steps, args.rank) < 1:
        raise ValueError("examples/updates/steps/rank must be positive")
    torch.manual_seed(args.seed)
    rows = sorted(read_split(Path(args.manifest), "train"), key=lambda r: r["id"])[:args.examples]
    if len(rows) != args.examples:
        raise ValueError("not enough training examples")
    observations = []
    for row in rows:
        observations.append(decode_observation(row, steps=args.steps, pixels=args.pixels))
    bb_cfg = BackboneConfig(name=args.model, dtype=args.dtype, local_files_only=True)
    stream = StreamConfig(pixels_per_frame=args.pixels)
    identity = {"purpose": "training-plumbing-only", "args": vars(args),
                "manifest_sha256": sha256(args.manifest), "backbone": asdict(bb_cfg),
                "observations": [{"id": x[2]["id"], **x[3]} for x in observations]}
    atomic_text(out / "inputs.json", json.dumps(identity, indent=2) + "\n")
    bb = Backbone(bb_cfg, stream, device=args.device)
    bb.add_lora(args.rank, 2 * args.rank, dropout=0.)
    bb.enable_checkpointing()
    samples = [(encode_observation(bb, x[0], x[1]), x[2]) for x in observations]
    eos = bb.tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define an EOS token for answer termination")

    def make_model():
        return VideoQALoss(bb, mode=args.mode, pool_grid=tuple(args.pool_grid), answer_end_token_id=eos)

    net = make_model()
    params = {n: p for n, p in net.named_parameters() if p.requires_grad}
    if not params or any("lora_" not in n for n in params):
        raise ValueError("flow must train LM LoRA only")
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}

    @torch.no_grad()
    def assess():
        bb.language_model.eval()
        result = []
        for steps, row in samples:
            loss, meta = net(steps, row)
            zero = [VisualStep(torch.zeros_like(s.tokens), s.grid, s.end, s.frame_times) for s in steps]
            zero_loss, _ = net(zero, row)
            predicted = net.generate(steps, row, max_new_tokens=8)
            result.append({"id": row["id"], "loss": float(loss), "zero_video_loss": float(zero_loss),
                           "generated": bb.tokenizer.decode(predicted, skip_special_tokens=True), **meta})
        return result

    before = assess()
    atomic_text(out / "before.json", json.dumps(before, indent=2) + "\n")
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    begin = time.monotonic()
    with (out / "metrics.jsonl").open("w", buffering=1) as log:
        for update in range(args.updates):
            bb.language_model.train()
            optimizer.zero_grad(set_to_none=True)
            steps, row = samples[update % len(samples)]
            loss, meta = net(steps, row)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite answer loss")
            loss.backward()
            grads = [p.grad for p in params.values() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError("missing/nonfinite LM adapter gradients")
            norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("LM adapter received no finite learning signal")
            optimizer.step()
            record = {"update": update + 1, "id": row["id"], "loss": float(loss.detach()),
                      "gradient_norm": norm, "seconds": time.monotonic() - begin, **meta}
            log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    after = assess()
    changed = sum(float((p.detach().cpu() - initial[n]).float().square().sum()) for n, p in params.items()) ** .5
    if not changed > 0:
        raise RuntimeError("LM adapter weights did not change")
    bb.peft.save_pretrained(out / "adapter", safe_serialization=True)
    saved = {n: p.detach().cpu().clone() for n, p in params.items()}
    del optimizer, params, net
    bb.peft.unload()
    bb.peft = None
    bb.model.requires_grad_(False)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bb.add_lora(args.rank, 2 * args.rank, dropout=0., adapter=str(out / "adapter"))
    net = make_model()
    loaded = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad}
    if saved.keys() != loaded.keys() or any(not torch.equal(saved[n], loaded[n]) for n in saved):
        raise RuntimeError("saved/reloaded adapter tensors differ")
    reloaded = assess()
    max_gap = max(abs(a["loss"] - b["loss"]) for a, b in zip(after, reloaded, strict=True))
    if max_gap > 1e-5 or any(a["generated"] != b["generated"] for a, b in zip(after, reloaded, strict=True)):
        raise RuntimeError("saved/reloaded answer outputs differ")
    report = {"completed": True, "purpose": "training-plumbing-only; no accuracy/generalization claim",
              "before": before, "after": after, "reloaded": reloaded, "weight_change_l2": changed,
              "reload_max_loss_gap": max_gap, "trainable_parameters": sum(p.numel() for p in loaded.values()),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    atomic_text(out / "report.json", json.dumps(report, indent=2) + "\n")
    print(f"VERIFIED core video-QA training and reload: {out / 'report.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    p.add_argument("--mode", default="grid", choices=("full", "grid"))
    p.add_argument("--pool-grid", nargs=2, type=int, default=[4, 5])
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--pixels", type=int, default=256 * 256)
    p.add_argument("--examples", type=int, default=2)
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
