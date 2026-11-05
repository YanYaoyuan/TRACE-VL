"""Primary-source pointers. Registered does not mean integrated or downloaded."""

REGISTRY = {
    "synthetic": {
        "task": "software fixture",
        "status": "implemented",
        "media": "generated locally",
        "license": "repository code license",
        "source": "trace_vl.data.synthetic",
    },
    "custom-alerts": {
        "task": "streaming alerts + prefix QA",
        "status": "CSV adapter + local metrics",
        "media": "user supplied",
        "license": "determined by source",
        "source": "docs/data.md",
    },
    "videomme": {
        "task": "offline long-video MCQ",
        "status": "annotation fetch + import adapter + local MCQ accuracy",
        "source": "https://github.com/MME-Benchmarks/Video-MME",
        "annotations": "lmms-eval/Video-MME",
        "media": "obtain through official project",
        "license": "check upstream dataset terms; not covered by repository license",
    },
    "streamingbench": {
        "task": "streaming QA",
        "status": "registered; upstream adapter pending",
        "source": "https://streamingbench.github.io/",
        "download": "https://huggingface.co/datasets/mjuicem/StreamingBench",
        "media": "obtain upstream",
        "license": "check upstream terms",
    },
    "ovo": {
        "task": "online backward / real-time / forward QA",
        "status": "registered; upstream adapter pending",
        "source": "https://github.com/JoeLeelyf/OVO-Bench",
        "download": "https://huggingface.co/datasets/JoeLeelyf/OVO-Bench",
        "media": "obtain upstream",
        "license": "GitHub README: CC BY-NC-SA 4.0; HF tag: CC BY-SA 4.0. Conflicting declarations; clarify with authors.",
    },
    "xd-violence": {
        "task": "anomaly detection",
        "status": "registered; dense-frame evaluation adapter pending",
        "source": "https://roc-ng.github.io/XD-Violence/",
        "media": "obtain upstream",
        "license": "check upstream terms; weak labels do not imply event known_at/deadline",
    },
    "llava-video-178k": {
        "task": "general video caption / QA alignment training",
        "status": "download profiles + local conversations import adapter; ActivityNetQA pilot trained",
        "source": "https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K",
        "media": "HF archives; pilot selects one short-video shard",
        "license": "annotations and source video rights must be checked separately",
    },
    "uca": {
        "task": "surveillance temporal captions / grounding",
        "status": "annotation download profile available; training adapter pending",
        "source": "https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet",
        "media": "separate UCF-Crime videos; retain official split isolation",
        "license": "README declares Apache-2.0 and academic/research-only usage; commercial use needs clarification",
    },
    "cctv-anomaly": {
        "task": "surveillance anomaly description training / held-out evaluation",
        "status": "download profiles available; gated access and training adapter required",
        "source": "https://huggingface.co/datasets/xiaozhang79/CCTV-Anomaly",
        "media": "HF gated train/test video shards",
        "license": "card and author repository restrict annotations to CC BY-NC 4.0 research despite Apache HF tag",
    },
    "cuva": {
        "task": "video anomaly explanation evaluation",
        "status": "download profiles available; evaluator adapter pending",
        "source": "https://github.com/fesvhtr/CUVA",
        "download": "https://huggingface.co/datasets/fesvhtr/CUVA",
        "media": "HF raw/group_*.zip; full annotations contain test examples",
        "license": "official repository CC BY-NC-SA 4.0",
    },
    "fastbench": {
        "task": "high-dynamic streaming perception evaluation",
        "status": "download profile available; evaluator adapter pending",
        "source": "https://huggingface.co/datasets/Ashenone3/FastBench",
        "media": "300 MP4 files / 306 QA; viewer train split is evaluation-only",
        "license": "no dataset-wide media license declared; code MIT does not cover videos",
    },
    "streamarena": {
        "task": "long-horizon retrospective / proactive video QA evaluation",
        "status": "download profiles available; evaluator adapter pending",
        "source": "https://huggingface.co/datasets/hkuzxc/StreamArena",
        "media": "243 video tars; EN/ZH translations share the same questions",
        "license": "dataset card CC BY-NC 4.0",
    },
}


def fetch_annotations(name, output):
    if name != "videomme":
        raise ValueError("automatic annotation fetching is implemented only for videomme")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Run uv sync --extra data to fetch official annotations") from e
    from .schema import write_jsonl

    rows = load_dataset(REGISTRY[name]["annotations"], split="test")
    write_jsonl(output, list(rows))
    return len(rows)
