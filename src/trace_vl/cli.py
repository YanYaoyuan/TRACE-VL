from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

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
    _inspection_parser(commands)
    _v3_parser(commands)
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
    elif args.command == "inspection":
        _inspection(args)
    elif args.command == "v3":
        _v3(args)
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


def _shard(value):
    index, count = (int(v) for v in value.split("/"))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError("shard must be i/n with 0 <= i < n")
    return index, count


def _inspection_parser(commands):
    actions = commands.add_parser("inspection", help="fixed-camera inspection benchmark and model")
    sub = actions.add_subparsers(dest="action", required=True)
    fetch = sub.add_parser("meva-fetch", help="download MEVA put/pick annotations (no video)")
    fetch.add_argument("--output", default="data/meva/annotations")
    fetch.add_argument("--sites", nargs="*")
    fetch.add_argument("--limit", type=int)
    fetch.add_argument("--workers", type=int, default=8)
    build = sub.add_parser("meva-build", help="link events into streams and generate questions")
    build.add_argument("--annotations", default="data/meva/annotations")
    build.add_argument("--output", default="data/meva/bench")
    build.add_argument("--seed", type=int, default=7)
    videos = sub.add_parser("meva-videos", help="resolve S3 video URLs for the benchmark clips")
    videos.add_argument("--bench", default="data/meva/bench")
    videos.add_argument("--output", default="data/meva/video_urls.txt")
    videos.add_argument("--split", nargs="*", help="only streams with questions in these splits")
    prepare = sub.add_parser("prepare", help="one pass per stream: detector, snapshots, caption frames")
    prepare.add_argument("--config", default="configs/inspection/qwen3vl_2b.yaml")
    prepare.add_argument("--bench", default="data/meva/bench")
    prepare.add_argument("--video-root", default="data/raw/meva")
    prepare.add_argument("--output", default="data/meva/prepared")
    prepare.add_argument("--split", nargs="*")
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--shard", type=_shard)
    caption = sub.add_parser("caption", help="caption log for the textlog baseline")
    caption.add_argument("--config", default="configs/inspection/qwen3vl_2b.yaml")
    caption.add_argument("--root", default="data/meva/prepared")
    caption.add_argument("--shard", type=_shard, default=(0, 1))
    caption.add_argument("--device", default="cuda")
    evaluate = sub.add_parser("evaluate", help="score methods x budgets by option log-probability")
    evaluate.add_argument("--config", default="configs/inspection/qwen3vl_2b.yaml")
    evaluate.add_argument("--bench", default="data/meva/bench")
    evaluate.add_argument("--root", default="data/meva/prepared")
    evaluate.add_argument("--output", default="outputs/inspection/eval")
    evaluate.add_argument("--methods", nargs="+")
    evaluate.add_argument("--budgets", nargs="+", type=int)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--shard", type=_shard, default=(0, 1))
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--limit", type=int)
    score = sub.add_parser("score", help="aggregate predictions with camera-level bootstrap CIs")
    score.add_argument("--predictions", default="outputs/inspection/eval")
    score.add_argument("--output")
    score.add_argument("--exclude-censored", action="store_true")
    train = sub.add_parser("train", help="LoRA SFT; use torchrun for several GPUs")
    train.add_argument("--config", default="configs/inspection/qwen3vl_2b.yaml")
    train.add_argument("--bench", default="data/meva/bench")
    train.add_argument("--root", default="data/meva/prepared")
    train.add_argument("--method", default="transition")
    train.add_argument("--budget", type=int, default=1024)
    train.add_argument("--output", default="outputs/inspection/lora-transition")
    train.add_argument("--init-adapter")
    alerts = sub.add_parser("alerts", help="score duration alerts against linked put-downs")
    alerts.add_argument("--config", default="configs/inspection/qwen3vl_2b.yaml")
    alerts.add_argument("--bench", default="data/meva/bench")
    alerts.add_argument("--root", default="data/meva/prepared")
    alerts.add_argument("--min-duration", type=float, default=120.0)
    alerts.add_argument("--tolerance", type=float, default=60.0)
    alerts.add_argument("--split", default="test")
    alerts.add_argument("--verify", action="store_true", help="filter alerts with the VLM")
    alerts.add_argument("--adapter")
    alerts.add_argument("--device", default="cuda")
    alerts.add_argument("--output")


def _inspection(args):
    from .inspection import meva

    def show(value):
        print(json.dumps(value, indent=2, default=str))

    if args.action == "meva-fetch":
        return show(meva.fetch_annotations(args.output, args.sites, args.limit, args.workers))
    if args.action == "meva-build":
        return show(meva.build(args.annotations, args.output, args.seed))
    bench = Path(getattr(args, "bench", "") or ".")
    if args.action == "meva-videos":
        questions = None
        if args.split:
            from .inspection.run import read_jsonl

            keep = [q for q in read_jsonl(bench / "questions.jsonl") if q["split"] in set(args.split)]
            questions = bench / "questions.selected.jsonl"
            questions.write_text("".join(json.dumps(q) + "\n" for q in keep))
        return show(meva.video_urls(bench / "streams.json", args.output, questions))
    if args.action == "score":
        from .inspection.run import score

        report = score(args.predictions, args.output, args.exclude_censored)
        return show([{k: r[k] for k in ("method", "budget", "n", "mean_visual_tokens", "accuracy",
                                          "macro_balanced_accuracy", "ci95_macro_balanced_accuracy")}
                     for r in report["results"]])  # fmt: skip
    config = yaml.safe_load(Path(args.config).read_text()) or {}
    model = config.get("model", {})
    if args.action == "prepare":
        from .inspection.stream import prepare

        return show(
            prepare(
                bench / "questions.jsonl",
                bench / "streams.json",
                args.video_root,
                args.output,
                config.get("detector"),
                config.get("snapshot"),
                args.workers,
                args.shard,
                args.split,
            )
        )
    if args.action == "caption":
        from .inspection.run import caption

        tokens = config.get("caption", {}).get("tokens", 256)
        return show(caption(args.root, model, tokens, args.shard, args.device))
    if args.action == "evaluate":
        from .inspection.run import evaluate

        settings = config.get("evaluation", {})
        return show(
            evaluate(
                bench / "questions.jsonl",
                args.root,
                model,
                args.methods or settings.get("methods", ["none", "current", "uniform", "recent", "transition"]),
                args.budgets or settings.get("budgets", [512, 1024, 2048]),
                args.output,
                args.split,
                config.get("pack"),
                args.adapter,
                args.shard,
                args.device,
                args.limit,
            )
        )
    if args.action == "train":
        from .inspection.run import train

        path = train(
            bench / "questions.jsonl",
            args.root,
            model,
            args.method,
            args.budget,
            args.output,
            config.get("training"),
            config.get("pack"),
            args.init_adapter,
        )
        return print(path)
    if args.action == "alerts":
        from .inspection.alerts import score_alerts
        from .inspection.run import read_jsonl

        streams = {q["stream"] for q in read_jsonl(bench / "questions.jsonl") if q["split"] == args.split}
        vlm = None
        if args.verify:
            from .inspection.run import _vlm

            vlm = _vlm(model, args.device, args.adapter)
        report = score_alerts(
            bench / "events.json",
            args.root,
            args.min_duration,
            args.tolerance,
            streams,
            vlm,
            model.get("unit", 32),
        )
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        return show({k: v for k, v in report.items() if k != "per_stream"})
    raise ValueError(f"unknown inspection action {args.action}")


def _v3_parser(commands):
    sub = commands.add_parser("v3", help="TRACE-VL v3 on Qwen3.5-2B").add_subparsers(dest="action", required=True)
    for name in ("convert-ovo", "convert-videomme"):
        c = sub.add_parser(name)
        c.add_argument("input")
        c.add_argument("--output", required=True)
        c.add_argument("--video-root", help="keep only questions whose video exists here (partial download)")
        if name == "convert-ovo":
            c.add_argument("--forward", action="store_true", help="also convert forward-active REC/SSR/CRR polls")
    c = sub.add_parser("convert-llava-mc", help="LLaVA-Video-178K multiple-choice files -> Stage 1 records")
    c.add_argument("inputs", nargs="+")
    c.add_argument("--output", required=True)
    c.add_argument("--video-root", help="keep only questions whose video exists here")
    c.add_argument("--val-fraction", type=float, default=0.05)
    c = sub.add_parser("convert-tocbench", help="TOC-Bench answer_final.json -> records (local protocol)")
    c.add_argument("input")
    c.add_argument("--output", required=True)
    c.add_argument("--video-root", help="keep only questions whose video exists here")
    c = sub.add_parser("convert-meva")
    c.add_argument("--bench", default="data/meva/bench")
    c.add_argument("--split", default="test")
    c.add_argument("--output", required=True)
    c = sub.add_parser("params", help="parameter report of the new modules (no backbone download)")
    c.add_argument("--config", default="configs/v3/v3_tkm.yaml")
    c = sub.add_parser("check", help="M1 equality gate: streaming path vs HF forward, zero-gated v3 vs native")
    c.add_argument("--config", default="configs/v3/native_full.yaml")
    c.add_argument("--video")
    c.add_argument("--steps", type=int, default=4)
    c.add_argument("--device", default="cuda")
    c.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32", help="gate is float32")
    c = sub.add_parser("gradcheck", help="unclipped gradient norms on a static clip (anchored-delta regression)")
    c.add_argument("--config", default="configs/v3/stage1.yaml")
    c.add_argument("--steps", type=int, default=6)
    c.add_argument("--threshold", type=float, default=1e4)
    c.add_argument("--device", default="cuda")
    c = sub.add_parser("budget", help="matched visual-token budget audit (transition vs B4) on calibration videos")
    c.add_argument("--config", default="configs/v3/v3_tkm.yaml")
    c.add_argument("--checkpoint", help="transition-arm modules.pt")
    c.add_argument("--config-b4", default="configs/v3/v3_pooled_b4.yaml")
    c.add_argument("--checkpoint-b4", help="B4-arm modules.pt")
    c.add_argument("--records", required=True, help="calibration records (train/val split, never test)")
    c.add_argument("--video-root", required=True)
    c.add_argument("--max-videos", type=int, default=8)
    c.add_argument("--max-seconds", type=float, default=180.0)
    c.add_argument("--tolerance", type=float, default=0.05)
    c.add_argument("--device", default="cuda")
    c = sub.add_parser("profile", help="latency / KV memory on synthetic frames")
    c.add_argument("--config", default="configs/v3/native_full.yaml")
    c.add_argument("--mode", choices=["native", "v3"], default="native")
    c.add_argument("--checkpoint")
    c.add_argument("--steps", type=int, default=600)
    c.add_argument("--device", default="cuda")
    c = sub.add_parser("eval")
    c.add_argument("--config", required=True)
    c.add_argument("--records", required=True)
    c.add_argument("--video-root", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--mode", choices=["native", "v3"], default="native")
    c.add_argument("--checkpoint")
    c.add_argument("--shard", type=_shard, default=(0, 1))
    c.add_argument("--device", default="cuda")
    c.add_argument("--limit", type=int)
    c.add_argument("--adapter", help="LoRA adapter directory saved by `v3 train`")
    c = sub.add_parser("score")
    c.add_argument("predictions")
    c.add_argument("--output")
    c = sub.add_parser("train", help="Stage 1 distillation; use torchrun for several GPUs")
    c.add_argument("--config", default="configs/v3/stage1.yaml")
    c.add_argument("--records", required=True, nargs="+", help="one or more record files (QA and/or alert-only)")
    c.add_argument("--video-root", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--clip-seconds", type=float, default=120.0)
    c.add_argument("--max-steps", type=int)
    c.add_argument("--init")
    c.add_argument("--adapter")
    c = sub.add_parser("convert-anomaly", help="ShanghaiTech / Avenue frame labels -> alert-only records")
    c.add_argument("dataset", choices=["shanghaitech", "avenue"])
    c.add_argument("--root", required=True, help="extracted dataset root (also the --video-root later)")
    c.add_argument("--output", required=True)
    c.add_argument("--labels", help="folder of <video>.npy frame labels (overrides the official masks)")
    c.add_argument("--train-list", help="video names used for training (e.g. the Zhong et al. weak split)")
    c.add_argument("--train-fraction", type=float, default=0.5, help="hash split of labelled videos otherwise")
    c.add_argument("--normal-train", type=int, default=100, help="normal-only training videos added as negatives")
    c.add_argument("--fps", type=float)
    c.add_argument("--prefix", default="", help="prepended to video paths (e.g. shanghaitech/) for a shared root")
    c = sub.add_parser("eval-alerts", help="per-step abnormality AUC/AP of the alert head (or the LM Yes/No baseline)")
    c.add_argument("--config", default="configs/v3/stage1_alert.yaml")
    c.add_argument("--records", required=True)
    c.add_argument("--video-root", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--checkpoint")
    c.add_argument("--adapter")
    c.add_argument("--method", choices=["head", "lm"], default="head")
    c.add_argument("--split", default="test")
    c.add_argument("--limit", type=int)
    c.add_argument("--device", default="cuda")
    c = sub.add_parser("score-alerts")
    c.add_argument("predictions")
    c.add_argument("--output")
    c = sub.add_parser("build-stage2", help="long streams with delayed time-referenced questions (Stage 2 data)")
    c.add_argument("records", help="Stage 1 records (e.g. convert-llava-mc output) or timed records")
    c.add_argument("--output", required=True)
    c.add_argument("--video-root", required=True)
    c.add_argument("--split", choices=["train", "val"], default="train")
    c.add_argument("--stream-seconds", type=float, default=300.0)
    c.add_argument("--delay-min", type=float, default=30.0)
    c.add_argument("--delay-max", type=float, default=240.0)
    c.add_argument("--immediate-fraction", type=float, default=0.25)
    c.add_argument("--distractor-fraction", type=float, default=0.4, help="clips never asked about (negatives)")
    c.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("eval-stage2", help="held-out long streams: delayed-question accuracy and evidence kept")
    c.add_argument("--config", default="configs/v3/stage2.yaml")
    c.add_argument("--streams", required=True)
    c.add_argument("--video-root", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--mode", choices=["native", "v3"], default="v3")
    c.add_argument("--checkpoint")
    c.add_argument("--adapter")
    c.add_argument("--policy", choices=["tkm", "mass", "fifo", "window", "full"],
                   help="override memory.policy at the same budget (retention controls)")
    c.add_argument("--split", default="val")
    c.add_argument("--limit", type=int)
    c.add_argument("--device", default="cuda")
    c = sub.add_parser("train-stage2", help="Stage 2 long-stream training through the streaming cache (torchrun)")
    c.add_argument("--config", default="configs/v3/stage2.yaml")
    c.add_argument("--streams", required=True, help="build-stage2 output")
    c.add_argument("--video-root", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--init", help="Stage 1 modules.pt")
    c.add_argument("--adapter", help="Stage 1 lora/ directory")
    c.add_argument("--max-steps", type=int)


def _v3(args):
    from .v3 import data as v3data

    def show(value):
        print(json.dumps(value, indent=2, default=str))

    if args.action == "convert-ovo":
        return show(v3data.convert_ovo(args.input, args.output, args.video_root, args.forward))
    if args.action == "convert-videomme":
        return show(v3data.convert_videomme(args.input, args.output, args.video_root))
    if args.action == "convert-llava-mc":
        return show(v3data.convert_llava_mc(args.inputs, args.output, args.video_root, args.val_fraction))
    if args.action == "convert-tocbench":
        return show(v3data.convert_tocbench(args.input, args.output, args.video_root))
    if args.action == "convert-meva":
        bench = Path(args.bench)
        return show(v3data.convert_meva(bench / "questions.jsonl", bench / "streams.json", args.output, args.split))
    if args.action == "convert-anomaly":
        from .v3.anomaly import convert_anomaly

        return show(convert_anomaly(args.dataset, args.root, args.output, args.labels, args.train_list,
                                    args.train_fraction, args.normal_train, args.fps, prefix=args.prefix))  # fmt: skip
    if args.action == "score-alerts":
        from .v3.anomaly import score_alerts

        return show(score_alerts(args.predictions, args.output))
    if args.action == "build-stage2":
        return show(v3data.build_stage2_streams(args.records, args.output, args.video_root, args.split,
                                                args.stream_seconds, delay_min=args.delay_min,
                                                delay_max=args.delay_max, immediate_fraction=args.immediate_fraction,
                                                seed=args.seed, distractor_fraction=args.distractor_fraction))  # fmt: skip
    if args.action == "score":
        from .v3.harness import score

        return show(score(args.predictions, args.output))
    from .v3.config import load_v3_config

    cfg, _ = load_v3_config(args.config)
    if args.action == "params":
        from .v3.model import TraceV3Modules

        return show(TraceV3Modules(cfg).parameter_report())
    if args.action == "check":
        from .v3.harness import check_equality

        return show(check_equality(cfg, args.video, args.steps, args.device, args.dtype))
    if args.action == "gradcheck":
        from .v3.train import gradient_check

        return show(gradient_check(cfg, args.steps, args.threshold, args.device))
    if args.action == "budget":
        from .v3.config import load_v3_config as _load
        from .v3.harness import budget_audit

        cfg_b4, _ = _load(args.config_b4)
        arms = [("transition", cfg, args.checkpoint), ("b4_pooled", cfg_b4, args.checkpoint_b4)]
        return show(budget_audit(arms, args.records, args.video_root, args.device, args.max_videos,
                                 args.max_seconds, args.tolerance))  # fmt: skip
    if args.action == "profile":
        from .v3.harness import profile

        return show(profile(cfg, args.steps, args.mode, args.checkpoint, args.device))
    if args.action == "eval":
        from .v3.harness import evaluate

        return show(evaluate(cfg, args.records, args.video_root, args.output, args.mode, args.checkpoint,
                             args.shard, args.device, args.limit, args.adapter))  # fmt: skip
    if args.action == "train":
        from .v3.train import train

        return print(train(cfg, args.records, args.video_root, args.output, args.clip_seconds, args.max_steps,
                           args.init, args.adapter))  # fmt: skip
    if args.action == "eval-alerts":
        from .v3.anomaly import evaluate_alerts

        return show(evaluate_alerts(cfg, args.records, args.video_root, args.output, args.checkpoint, args.adapter,
                                    args.method, args.device, args.split, args.limit))  # fmt: skip
    if args.action == "eval-stage2":
        from .v3.train_stage2 import evaluate_streams

        return show(evaluate_streams(cfg, args.streams, args.video_root, args.output, args.mode, args.checkpoint,
                                     args.adapter, args.policy, args.device, args.split, args.limit))  # fmt: skip
    if args.action == "train-stage2":
        from .v3.train_stage2 import train_stage2

        return print(train_stage2(cfg, args.streams, args.video_root, args.output, args.init, args.adapter,
                                  args.max_steps))  # fmt: skip
    raise ValueError(f"unknown v3 action {args.action}")


if __name__ == "__main__":
    main()
