"""Upload a completed pilot run directory to Weights & Biases (works for runs trained without live tracking,
e.g. pilot-003/004 launched from pinned snapshots).

Layout (scripts/run_pilot00{3,4}.sh): identity.json, summary.json, train/<arm>/{metrics.jsonl, run.json,
complete.json, modules.pt, lora/}, eval/<name>/{score.json, run.json}, optional selector-diagnostics.json,
diagnostics.json, *-digest.txt, *.log.

Creates, in one group (the run directory name):
  <run>-train-<arm>   training curves replayed step by step from metrics.jsonl (+ logs artifact)
  <run>-eval          summary scalars, accuracy and paired-difference tables, selector-diagnostic curves,
                      demo tables (optional, GPU), and one artifact with every report/log file
Checkpoints are not uploaded (size); their hashes are in complete.json, which is uploaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tracking import Tracker


def _rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.is_file() else []


def _flatten(prefix: str, value, out: dict, depth: int = 0):
    if isinstance(value, bool) or depth > 6:
        out[prefix] = value if isinstance(value, bool) else str(value)
    elif isinstance(value, (int, float)):
        out[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), v, out, depth + 1)


def accuracy_and_pairs(summary: dict) -> tuple[list[list], list[list]]:
    """Rows for (checkpoint, arm, accuracy, ci_low, ci_high) and (checkpoint, pair, delta_pp, ci_low, ci_high)."""
    blocks = summary.get("per_checkpoint") or {"final": summary}
    acc_rows, pair_rows = [], []
    for step, block in blocks.items():
        for arm, value in (block.get("accuracy") or {}).items():
            ci = (block.get("ci95") or {}).get(arm) or [None, None]
            acc_rows.append([str(step), arm, value, ci[0], ci[1]])
        for pair, p in (block.get("paired") or {}).items():
            lo, hi = (p.get("ci95_pp") or [None, None])
            pair_rows.append([str(step), pair, p.get("delta_pp", p.get("delta_accuracy_pp")), lo, hi])
    return acc_rows, pair_rows


def upload_run(runs: str | Path, demos: int = 0, data_dir: str | None = None, video_root: str | None = None,
               device: str = "cuda") -> dict:  # fmt: skip
    runs = Path(runs)
    group = runs.name
    identity = json.loads((runs / "identity.json").read_text()) if (runs / "identity.json").is_file() else {}
    report = {"group": group, "train_runs": [], "eval_run": None}
    for arm_dir in sorted(d for d in (runs / "train").glob("*") if d.is_dir() and ".incomplete-" not in d.name):
        rows = _rows(arm_dir / "metrics.jsonl")
        if not rows:
            continue
        tracker = Tracker.start(True, name=f"{group}-train-{arm_dir.name}", group=group, job_type="train",
                                config={"config": identity.get("configs", {}).get(arm_dir.name), "arm": arm_dir.name,
                                        "source": "replayed from metrics.jsonl"}, directory=runs / "wandb")  # fmt: skip
        if not tracker.active:
            raise RuntimeError("wandb is not installed in this environment (uv sync --extra track)")
        for row in rows:
            tracker.log(row, step=int(row["step"]))
        tracker.summary({"updates": len(rows), **{f"final/{k}": v for k, v in rows[-1].items()
                                                  if isinstance(v, (int, float)) and not isinstance(v, bool)}})
        tracker.upload(f"{group}-{arm_dir.name}-training-logs", "training-logs",
                       [arm_dir / "metrics.jsonl", arm_dir / "run.json", arm_dir / "complete.json"])  # fmt: skip
        tracker.finish()
        report["train_runs"].append(arm_dir.name)

    tracker = Tracker.start(True, name=f"{group}-eval", group=group, job_type="eval",
                            config={"identity": {k: v for k, v in identity.items() if k != "configs"}},
                            directory=runs / "wandb")  # fmt: skip
    summary = json.loads((runs / "summary.json").read_text()) if (runs / "summary.json").is_file() else {}
    flat: dict = {}
    _flatten("", {k: v for k, v in summary.items() if k != "note"}, flat)
    tracker.summary(flat)
    acc_rows, pair_rows = accuracy_and_pairs(summary)
    tracker.table("accuracy", ["checkpoint", "arm", "accuracy", "ci_low", "ci_high"], acc_rows)
    tracker.table("paired_differences_pp", ["checkpoint", "pair", "delta_pp", "ci_low", "ci_high"], pair_rows)
    selector = runs / "selector-diagnostics.json"
    if selector.is_file():  # one table per arm: step, statistic, value (plotted with wandb's line charts)
        series = json.loads(selector.read_text())
        rows = [[arm, key, step, value] for arm, stats in series.items() for key, points in stats.items()
                for step, value in points]  # fmt: skip
        tracker.table("selector_diagnostics", ["arm", "statistic", "step", "value"], rows)
    if demos > 0:
        if not (data_dir and video_root):
            raise ValueError("demos need --data-dir and --video-root")
        from .config import load_v3_config
        from .demo import log_demos, make_demos

        records = str(Path(data_dir) / "val.jsonl")
        arms = sorted(d.name for d in (runs / "train").glob("*") if d.is_dir() and ".incomplete-" not in d.name)
        for arm in ["native", *arms]:
            cfg_path = runs / "configs" / f"{arms[0] if arm == 'native' else arm}.yaml"
            cfg, _ = load_v3_config(cfg_path)
            kwargs = {} if arm == "native" else {"checkpoint": str(runs / "train" / arm / "modules.pt"),
                                                 "adapter": str(runs / "train" / arm / "lora")}  # fmt: skip
            rows = make_demos(cfg, records, video_root, str(runs / "demos" / arm), "native" if arm == "native" else "v3",
                              n=demos, device=device, **kwargs)  # fmt: skip
            log_demos(tracker, rows, arm)
    files = [runs / name for name in ("identity.json", "summary.json", "selector-diagnostics.json", "diagnostics.json",
                                      "manifest.json")]  # fmt: skip
    files += sorted(runs.glob("*-digest.txt")) + sorted(runs.glob("*.log")) + sorted(runs.glob("eval/**/score.json"))
    files += [runs / "demos"] if (runs / "demos").is_dir() else []
    tracker.upload(f"{group}-reports", "eval-reports", files)
    report["eval_run"] = tracker.run.url if tracker.active and getattr(tracker.run, "url", None) else None
    tracker.finish()
    return report
