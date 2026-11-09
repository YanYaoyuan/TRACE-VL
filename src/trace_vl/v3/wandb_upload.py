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

from .tracking import Tracker, offline_runs


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


def _start(name: str, group: str, job_type: str, config: dict, runs: Path) -> Tracker:
    tracker = Tracker.start(True, name=name, group=group, job_type=job_type, config=config, directory=runs / "wandb")
    if not tracker.active:
        raise RuntimeError("could not start a wandb run (see the warning above: wandb missing, or init failed); "
                           "nothing was uploaded and the run directory is unchanged")
    return tracker


def upload_run(runs: str | Path, demos: int = 0, data_dir: str | None = None, video_root: str | None = None,
               device: str = "cuda", extra_files: list[str] | None = None) -> dict:  # fmt: skip
    """`extra_files` (e.g. the operator log outside the run directory) are stored as operator/<file name>."""
    runs = Path(runs)
    group = runs.name
    identity = json.loads((runs / "identity.json").read_text()) if (runs / "identity.json").is_file() else {}
    report = {"group": group, "train_runs": [], "eval_run": None, "artifacts": {}, "errors": []}
    for arm_dir in sorted(d for d in (runs / "train").glob("*") if d.is_dir() and ".incomplete-" not in d.name):
        rows = _rows(arm_dir / "metrics.jsonl")
        if not rows:
            continue
        tracker = _start(f"{group}-train-{arm_dir.name}", group, "train",
                         {"config": identity.get("configs", {}).get(arm_dir.name), "arm": arm_dir.name,
                          "source": "replayed from metrics.jsonl"}, runs)  # fmt: skip
        report["mode"] = tracker.mode
        for row in rows:
            tracker.log(row, step=int(row["step"]))
        tracker.summary({"updates": len(rows), **{f"final/{k}": v for k, v in rows[-1].items()
                                                  if isinstance(v, (int, float)) and not isinstance(v, bool)}})
        artifact = tracker.upload(f"{group}-{arm_dir.name}-training-logs", "training-logs",
                                  [arm_dir / "metrics.jsonl", arm_dir / "run.json", arm_dir / "complete.json"], base=runs)
        report["artifacts"][f"train-{arm_dir.name}"] = sorted(artifact.manifest.entries) if artifact else None
        if tracker.error:
            report["errors"].append(tracker.error)
        tracker.finish()
        report["train_runs"].append(arm_dir.name)

    tracker = _start(f"{group}-eval", group, "eval", {"identity": {k: v for k, v in identity.items() if k != "configs"}},
                     runs)  # fmt: skip
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
    files += sorted(runs.glob("*-digest.txt")) + sorted(runs.glob("*.log")) + sorted(runs.glob("probe-*.json"))
    files += sorted(runs.glob("eval/**/score.json")) + sorted(runs.glob("eval/**/run.json"))
    files += sorted(runs.glob("eval/**/errors.*.json")) + sorted(runs.glob("configs/*.yaml"))
    files += [runs / "demos"] if (runs / "demos").is_dir() else []
    names = {}
    for extra in extra_files or []:  # outside the run directory: stored under operator/
        if not Path(extra).is_file():
            report["errors"].append(f"extra file not found: {extra}")
            continue
        files.append(Path(extra))
        names[str(Path(extra))] = f"operator/{Path(extra).name}"
    artifact = tracker.upload(f"{group}-reports", "eval-reports", files, base=runs, names=names)
    report["artifacts"]["reports"] = sorted(artifact.manifest.entries) if artifact else None
    report["eval_run"] = getattr(tracker.run, "url", None) if tracker.active else None
    if tracker.error:
        report["errors"].append(tracker.error)
    tracker.finish()
    report["offline_runs"] = [str(p) for p in offline_runs(runs / "wandb")]
    return report
