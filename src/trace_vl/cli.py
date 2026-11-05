from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ModelConfig, load_config
from .data.adapters import import_alert_csv, import_videomme, read_annotation_rows
from .data.llava import import_llava_video
from .data.registry import REGISTRY, fetch_annotations
from .data.schema import check_split_leakage, read_jsonl
from .data.synthetic import make_synthetic


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trace-vl")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data").add_subparsers(dest="action", required=True)
    synth = data.add_parser("synth")
    synth.add_argument("--output", default="data/synthetic")
    for split, count in (("train", 12), ("val", 4), ("test", 4)):
        synth.add_argument("--" + split, type=int, default=count)
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--video", action="store_true")
    validate = data.add_parser("validate")
    validate.add_argument("manifest")
    llava = data.add_parser("import-llava")
    llava.add_argument("input", nargs="+")
    llava.add_argument("--output", required=True)
    llava.add_argument("--video-root", required=True)
    llava.add_argument("--max-videos", type=int, default=200)
    llava.add_argument("--val-fraction", type=float, default=0.1)
    llava.add_argument("--seed", type=int, default=7)
    for name in ("import-videomme", "import-alerts"):
        adapter = data.add_parser(name)
        adapter.add_argument("input")
        adapter.add_argument("--output", required=True)
        adapter.add_argument("--video-root", required=True)
        adapter.add_argument(
            "--split",
            default="test" if name == "import-videomme" else "train",
            choices=["train", "val", "test"],
        )
    registry = commands.add_parser("datasets").add_subparsers(dest="action", required=True)
    registry.add_parser("list")
    fetch = registry.add_parser("fetch-annotations")
    fetch.add_argument("name", choices=["videomme"])
    fetch.add_argument("--output", required=True)
    for name in ("train", "evaluate", "inspect"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", default="configs/tiny.yaml")
        if name == "train":
            cmd.add_argument("--output", default="outputs/tiny-train")
            initialization = cmd.add_mutually_exclusive_group()
            initialization.add_argument("--resume")
            initialization.add_argument("--init-from")
            cmd.add_argument("--max-steps", type=int)
        if name == "evaluate":
            cmd.add_argument("--checkpoint")
            cmd.add_argument("--allow-random", action="store_true")
            cmd.add_argument("--output", default="outputs/tiny-eval")
            cmd.add_argument("--split", choices=["train", "val", "test"], default="test")
    score = commands.add_parser("score")
    score.add_argument("manifest")
    score.add_argument("predictions")
    score.add_argument("--split", choices=["train", "val", "test"], default="test")
    score.add_argument("--output", default="outputs/rescore.json")
    commands.add_parser("doctor")
    profile = commands.add_parser("profile")
    profile.add_argument("--config", default="configs/tiny.yaml")
    profile.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 256])
    profile.add_argument("--output", default="outputs/profile.json")
    args = parser.parse_args(argv)
    if args.command == "data":
        if args.action == "synth":
            print(make_synthetic(args.output, args.train, args.val, args.test, args.seed, args.video))
        elif args.action == "validate":
            rows = read_jsonl(args.manifest)
            print(json.dumps({"samples": len(rows), "splits": check_split_leakage(rows)}, indent=2))
        elif args.action == "import-videomme":
            print(import_videomme(read_annotation_rows(args.input), args.output, args.video_root, args.split))
        elif args.action == "import-llava":
            print(
                json.dumps(
                    import_llava_video(
                        args.input,
                        args.output,
                        args.video_root,
                        args.max_videos,
                        args.val_fraction,
                        args.seed,
                    ),
                    indent=2,
                )
            )
        else:
            print(import_alert_csv(args.input, args.output, args.video_root, args.split))
    elif args.command == "datasets":
        if args.action == "list":
            print(json.dumps(REGISTRY, ensure_ascii=False, indent=2))
        else:
            print(fetch_annotations(args.name, args.output))
    elif args.command in {"train", "evaluate", "inspect"}:
        config = load_config(args.config)
        if args.command == "train":
            from .training import train

            if args.max_steps is not None:
                config.setdefault("training", {})["max_steps"] = args.max_steps
            train(config, args.output, args.resume, args.init_from)
        elif args.command == "evaluate":
            from .inference import evaluate

            evaluate(config, args.checkpoint, args.output, args.split, args.allow_random)
        else:
            from .models.model import TraceVL

            print(json.dumps(TraceVL(ModelConfig.from_dict(config["model"])).parameter_report(), indent=2))
    elif args.command == "score":
        from .evaluation.metrics import score_predictions
        from .runtime import write_json

        rows = [r for r in read_jsonl(args.manifest) if r["split"] == args.split]
        predictions = [json.loads(x) for x in Path(args.predictions).read_text().splitlines() if x.strip()]
        write_json(args.output, score_predictions(rows, predictions))
        print(args.output)
    elif args.command == "profile":
        from .profiling import profile_stream

        profile_stream(load_config(args.config), args.output, args.lengths)
        print(args.output)
    else:
        import torch

        from .runtime import environment

        print(
            json.dumps(
                {
                    **environment(),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_architectures": torch.cuda.get_arch_list(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
