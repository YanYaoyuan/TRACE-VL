"""Diagnostics for a paired pilot run (PR #3 review of pilot-003): is a negative result a representation effect,
or a disconnected/undertrained branch? CPU parts read the run directory only; GPU parts load checkpoints.

Run directory layout (scripts/run_pilot003.sh): identity.json, train/<arm>/{metrics.jsonl, modules.pt, lora/,
complete.json}, eval/<arm>/{run.json, predictions.*.jsonl}.

CPU:
- provenance: completion records, distinct artifacts per arm, evaluated checkpoint/adapter == trained ones;
- predictions: per-pair agreement (both right, only a, only b, both wrong), identical-answer rate, exact McNemar;
- validation loss: gold-letter NLL from the logged letter log-probs (full-vocabulary and option-renormalized);
- training curves: first/last-window means of every logged loss, gradient norms and the CST gate.
GPU:
- weight changes vs the exactly replicated initialization (same seed and construction order as train());
  frozen-but-unused parameters must be bit-identical to the replica, which validates the replication;
- activation probe on validation videos: CST gated-residual / input norm, P-step spatial coverage of the tokens
  the LM sees, gist-attention entropy, slot coverage.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def _predictions(eval_dir: Path) -> dict:
    return {r["id"]: r for p in sorted(Path(eval_dir).glob("predictions.*.jsonl")) for r in _rows(p)}


# ---------------------------------------------------------------------------- CPU
def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def prediction_agreement(preds: dict, a: str, b: str) -> dict:
    pa, pb = preds[a], preds[b]
    if set(pa) != set(pb):
        raise ValueError(f"{a} and {b} cover different questions")
    counts = defaultdict(int)
    same = 0
    for key, ra in pa.items():
        rb = pb[key]
        ca, cb = ra["prediction"] == ra["answer"], rb["prediction"] == rb["answer"]
        if ca and cb:
            counts["both_correct"] += 1
        elif ca:
            counts["only_" + a] += 1
        elif cb:
            counts["only_" + b] += 1
        else:
            counts["both_wrong"] += 1
        same += int(ra["prediction"] == rb["prediction"])
    out = {k: counts[k] for k in ("both_correct", "only_" + a, "only_" + b, "both_wrong")}
    out["n"] = len(pa)
    out["same_prediction_fraction"] = same / max(1, len(pa))
    out["mcnemar_p"] = mcnemar_exact(counts["only_" + a], counts["only_" + b])
    return out


def validation_nll(preds: dict) -> dict:
    """Gold-letter NLL from the logged letter log-probs: full vocabulary, and renormalized over the options."""
    full, options = [], []
    for r in preds.values():
        scores = r.get("scores") or []
        if not scores or not isinstance(r["answer"], int) or r["answer"] >= len(scores):
            continue
        full.append(-scores[r["answer"]])
        top = max(scores)
        log_z = top + math.log(sum(math.exp(s - top) for s in scores))
        options.append(log_z - scores[r["answer"]])
    mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
    return {"n": len(full), "nll_full_vocab": mean(full), "nll_options": mean(options)}


def training_curves(metrics: list[dict], window: int = 20) -> dict:
    keys = sorted({k for r in metrics for k, v in r.items() if isinstance(v, (int, float)) and k != "step"})
    first, last = metrics[:window], metrics[-window:]

    def mean(rows, key):
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(values) / len(values) if values else None

    return {"updates": len(metrics), "window": window,
            "first": {k: mean(first, k) for k in keys}, "last": {k: mean(last, k) for k in keys}}  # fmt: skip


def provenance(runs: Path, arms: list[str]) -> dict:
    from .harness import _dir_identity, _sha256
    from .pilot import training_complete

    identity = json.loads((runs / "identity.json").read_text())
    out, hashes = {}, {}
    for arm in arms:
        train_dir, eval_dir = runs / "train" / arm, runs / "eval" / arm
        cfg = identity["configs"][arm]
        ok, reason = training_complete(train_dir, cfg, cfg["train"]["steps"])
        record = json.loads((train_dir / "complete.json").read_text()) if ok else {}
        run = json.loads((eval_dir / "run.json").read_text())
        lora = {f"lora/{k}": v for k, v in (_dir_identity(str(train_dir / "lora")) or {}).items()}
        evaluated_matches = (run.get("checkpoint") == _sha256(train_dir / "modules.pt")
                             and run.get("adapter") == _dir_identity(str(train_dir / "lora")))  # fmt: skip
        hashes[arm] = record.get("artifacts", {}).get("modules.pt")
        out[arm] = {"training_complete": ok, "reason": reason, "evaluated_checkpoint_is_trained_checkpoint": evaluated_matches,
                    "artifacts_match_completion_record": ok and lora == {k: v for k, v in record["artifacts"].items()
                                                                       if k.startswith("lora/")}}  # fmt: skip
    distinct = len({h for h in hashes.values() if h}) == len([h for h in hashes.values() if h])
    return {"arms": out, "module_checkpoints_distinct": distinct}


def cpu_report(runs: Path, window: int = 20) -> dict:
    runs = Path(runs)
    identity = json.loads((runs / "identity.json").read_text())
    arms = list(identity["configs"])
    preds = {a: _predictions(runs / "eval" / a) for a in ["native", *arms] if (runs / "eval" / a).is_dir()}
    names = list(preds)
    pairs = {f"{a}_vs_{b}": prediction_agreement(preds, a, b) for i, a in enumerate(names) for b in names[i + 1 :]}
    return {
        "provenance": provenance(runs, arms),
        "agreement": pairs,
        "validation_nll": {a: validation_nll(p) for a, p in preds.items()},
        "training": {a: training_curves(_rows(runs / "train" / a / "metrics.jsonl"), window) for a in arms},
    }


# ---------------------------------------------------------------------------- GPU
def _groups(modules) -> dict:
    """Parameter groups by sub-module (two levels), e.g. cst.gate, cst.layers, transition.slots."""
    groups = defaultdict(list)
    for name, p in modules.named_parameters():
        parts = name.split(".")
        groups[".".join(parts[:2]) if len(parts) > 1 else parts[0]].append((name, p))
    return groups


def weight_changes(runs: Path, device: str = "cuda") -> dict:
    """Relative change of every parameter group vs the replicated initialization, per arm, plus LoRA norms."""
    import torch

    from .backbone import Backbone
    from .config import load_v3_config
    from .model import TraceV3Modules

    runs = Path(runs)
    identity = json.loads((runs / "identity.json").read_text())
    arms = list(identity["configs"])
    cfg, _ = load_v3_config(runs / "configs" / f"{arms[0]}.yaml")
    tc = cfg.train
    # Replicate train(): seed, backbone, LoRA, checkpointing, then modules (same RNG consumption order).
    torch.manual_seed(tc.seed)
    bb = Backbone(cfg.backbone, cfg.stream, device)
    bb.add_lora(tc.lora_rank, tc.lora_alpha)
    hidden = bb.hidden
    init = TraceV3Modules(cfg, hidden).state_dict()
    del bb
    torch.cuda.empty_cache()
    report = {}
    for arm in arms:
        arm_cfg, _ = load_v3_config(runs / "configs" / f"{arm}.yaml")
        trained = TraceV3Modules.load(str(runs / "train" / arm / "modules.pt"), arm_cfg, hidden)
        groups = {}
        for group, params in _groups(trained).items():
            delta = sum(float((p.detach().float().cpu() - init[n].float().cpu()).pow(2).sum()) for n, p in params)
            base = sum(float(init[n].float().pow(2).sum()) for n, _ in params)
            groups[group] = {"abs_change": math.sqrt(delta), "init_norm": math.sqrt(base),
                             "rel_change": math.sqrt(delta / base) if base > 0 else None}  # fmt: skip
        gate = trained.cst.gate.detach().float()
        from safetensors.torch import load_file

        lora_files = list((runs / "train" / arm / "lora").glob("*.safetensors"))
        lora = load_file(str(lora_files[0])) if lora_files else {}
        b_norm = math.sqrt(sum(float(v.float().pow(2).sum()) for k, v in lora.items() if "lora_B" in k))
        report[arm] = {
            "groups": groups,
            "cst_gate": {"abs_mean": float(gate.abs().mean()), "abs_max": float(gate.abs().max()),
                         "fraction_abs_gt_1e-3": float((gate.abs() > 1e-3).float().mean())},
            "lora_B_norm": b_norm,  # zero at initialization
            # Never trained in Stage 1 without alert data / KV eviction: must equal the replica exactly if the
            # initialization was replicated correctly (validates every rel_change above).
            "init_replication_valid": all(g["abs_change"] == 0.0 for name, g in groups.items()
                                          if name.startswith(("alert", "retention"))),
        }  # fmt: skip
    return report


def activation_probe(runs: Path, data_dir: Path, video_root: Path, videos: int = 8, device: str = "cuda") -> dict:
    """Per arm on the first `videos` validation videos: CST residual/input norm ratio, P-step coverage of the
    tokens the LM sees, gist-attention entropy and slot coverage (means over steps)."""
    import torch

    from .backbone import Backbone
    from .config import load_v3_config
    from .data import StepReader, step_times
    from .model import TraceV3Modules, stream_step
    from .modules import CSTCache, TECache

    runs = Path(runs)
    identity = json.loads((runs / "identity.json").read_text())
    val = _rows(Path(data_dir) / "val.jsonl")
    chosen = sorted({r["video"] for r in val})[:videos]
    report = {}
    for arm in identity["configs"]:
        cfg, _ = load_v3_config(runs / "configs" / f"{arm}.yaml")
        bb = Backbone(cfg.backbone, cfg.stream, device)
        modules = TraceV3Modules.load(str(runs / "train" / arm / "modules.pt"), cfg, bb.hidden).to(device).eval()
        stats = defaultdict(list)
        gap = cfg.stream.frame_gap
        with torch.no_grad():
            for video in chosen:
                reader = StepReader({"video": video}, video_root)
                try:
                    cst, te = CSTCache(), TECache()
                    for t in [t for t in step_times(reader.duration, cfg.stream.step_seconds, gap) if t + gap / 2 <= reader.duration + 1e-6]:
                        pixel_values, grid = bb.preprocess(bb.resize_step(reader.step(t, gap)))
                        x = bb.encode(pixel_values, grid)
                        adapted, _, info = stream_step(modules, x, bb.merged_grid(grid), t, cst, te)
                        xf = x.float()
                        stats["cst_residual_ratio"].append(float((adapted.float() - xf).norm() / xf.norm().clamp_min(1e-6)))
                        if not info.is_i_step:
                            for key in ("coverage", "gist_entropy", "slot_coverage"):
                                if info.aux.get(key) is not None:
                                    stats[key].append(info.aux[key])
                finally:
                    reader.close()
        report[arm] = {k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in stats.items()}
        del bb, modules
        torch.cuda.empty_cache()
    report["videos"] = chosen
    return report


def paired_delta(preds: dict, a: str, b: str, samples: int = 4000, seed: int = 7) -> dict:
    """Accuracy of a minus b in percentage points, whole-video paired bootstrap (as in run_pilot003.sh)."""
    import random

    groups = {}
    for key, r in preds[a].items():
        o = preds[b][key]
        g = groups.setdefault(r["video_key"], [0, 0])
        g[0] += int(r["prediction"] == r["answer"]) - int(o["prediction"] == o["answer"])
        g[1] += 1
    units = list(groups.values())
    point = 100 * sum(u[0] for u in units) / sum(u[1] for u in units)
    rng, draws = random.Random(seed), []
    for _ in range(samples):
        pick = rng.choices(units, k=len(units))
        draws.append(100 * sum(u[0] for u in pick) / sum(u[1] for u in pick))
    draws.sort()
    return {"delta_pp": round(point, 3), "ci95_pp": [round(draws[int(0.025 * samples)], 3),
                                                     round(draws[int(0.975 * samples) - 1], 3)], "videos": len(units)}  # fmt: skip


def ablation_report(runs: Path) -> dict:
    """Gist-only evaluations of the trained slot arms (eval/<arm>_gistonly) against the full arms and B4."""
    runs = Path(runs)
    preds = {d.name: _predictions(d) for d in (runs / "eval").iterdir() if d.is_dir()}
    out = {}
    for arm in ("transition", "after"):
        ablated = f"{arm}_gistonly"
        if ablated not in preds or arm not in preds:
            continue
        acc = {k: sum(r["prediction"] == r["answer"] for r in preds[k].values()) / len(preds[k]) for k in (arm, ablated)}
        out[arm] = {"accuracy_full": acc[arm], "accuracy_gist_only": acc[ablated],
                    "full_minus_gist_only": paired_delta(preds, arm, ablated),
                    "agreement": prediction_agreement(preds, arm, ablated)}  # fmt: skip
        if "b4" in preds:
            out[arm]["gist_only_minus_b4"] = paired_delta(preds, ablated, "b4")
    return out
