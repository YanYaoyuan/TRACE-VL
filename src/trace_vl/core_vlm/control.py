"""Untrained Qwen native/full/grid control on identical decoded observations.

The saved manifest defines the split. This is a development diagnostic, not an
official benchmark or training result. Free-text exact match is deliberately
named as such; it is not a semantic judge. All three arms use greedy decoding
without a cache, so timings are not production throughput measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import torch

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .data import decode_observation, read_split, sha256
from .native import NativeVideoQALoss, check_native_forward, encode_native, prepare_native


def select_records(rows, limit):
    """Stable video round-robin prevents the first video's questions dominating."""
    if limit < 0:
        raise ValueError("limit must be nonnegative; zero means all")
    groups = {}
    for row in rows:
        groups.setdefault(row.get("source_group", row["video"]), []).append(row)
    queues = [sorted(groups[key], key=lambda r: r["id"]) for key in sorted(groups)]
    ordered = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    return ordered[:limit] if limit else ordered


def score_answer(text, row):
    if "options" in row:
        # No last-letter extraction from prose; malformed answers are incorrect.
        match = re.fullmatch(r"\s*([A-Z])[.)]?\s*", text)
        choice = ord(match[1]) - ord("A") if match else None
        valid = choice is not None and 0 <= choice < len(row["options"])
        return {"metric": "strict_mc_accuracy", "parsed": choice if valid else None,
                "correct": bool(valid and choice == row["answer"])}

    def normalize(value):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).rstrip(".!?。！？，,")

    return {"metric": "normalized_text_exact_match", "parsed": text,
            "correct": normalize(text) == normalize(row["answer_text"])}


def checkpoint_identity(model):
    root = Path(model)
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model"}
                   and ".cache" not in p.relative_to(root).parts)
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("local checkpoint must contain safetensors weights")
    return {str(p.relative_to(root)): sha256(p) for p in files}


@torch.no_grad()
def run(args):
    import transformers

    if args.split == "train":
        raise ValueError("base control requires a held-out split")
    rows = select_records(read_split(Path(args.manifest), args.split), args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"protocol": "native-full-grid-control-v1", "args": vars(args),
                "manifest_sha256": sha256(args.manifest), "ids": [r["id"] for r in rows],
                "checkpoint": checkpoint_identity(args.model),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "code": {name: sha256(Path(__file__).with_name(name))
                         for name in ("control.py", "native.py", "data.py", "qa.py")},
                # Recheck media on resume, even for predictions already written.
                "videos": {r["video"]: sha256(r["video"]) for r in rows}}
    identity["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ensure_run_manifest(out / "run.json", identity)
    bb = Backbone(BackboneConfig(name=args.model, dtype=args.dtype, local_files_only=True),
                  StreamConfig(pixels_per_frame=args.pixels), device=args.device)
    bb.model.eval()
    bb.model.requires_grad_(False)
    if bb.tokenizer.eos_token_id is None:
        raise ValueError("EOS token required")
    units = out / "questions"
    units.mkdir(exist_ok=True)
    results = []
    for index, row in enumerate(rows):
        path = units / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if result["run"] != identity["id"] or result["id"] != row["id"]:
                raise ValueError("saved prediction identity differs")
            results.append(result)
            continue
        images, times, record, audit = decode_observation(row, steps=args.steps, pixels=args.pixels)
        layout = prepare_native(bb.processor, images, times, record, audit,
                                allow_native_timestamp_approximation=args.timestamp_policy == "native-index-fps")
        record = {**record, "native_layout": layout}
        steps = encode_native(bb, layout)
        parity = check_native_forward(bb, layout, steps, record)
        result = {"id": row["id"], "run": identity["id"], "video": row["video"],
                  "observation": layout.audit, "native_forward_check": parity, "arms": {}}
        for arm in ("native", "full", "grid"):
            begin = time.monotonic()
            if arm == "native":
                inputs = {k: v.to(bb.device) for k, v in layout.inputs.items()}
                output = bb.model.generate(**inputs, do_sample=False, use_cache=False,
                                           max_new_tokens=args.max_new_tokens,
                                           eos_token_id=bb.tokenizer.eos_token_id,
                                           pad_token_id=bb.tokenizer.eos_token_id)
                ids = output[0, inputs["input_ids"].shape[1]:].tolist()
                if ids and ids[-1] == bb.tokenizer.eos_token_id:
                    ids.pop()
                tokens = sum(s.tokens.shape[0] for s in steps)
            else:
                net = NativeVideoQALoss(bb, mode=arm, pool_grid=tuple(args.pool_grid),
                                       answer_end_token_id=bb.tokenizer.eos_token_id)
                ids = net.generate(steps, record, max_new_tokens=args.max_new_tokens)
                tokens = net.prefix(steps, record)[4]["visual_tokens_total"]
            text = bb.tokenizer.decode(ids, skip_special_tokens=True)
            result["arms"][arm] = {"answer": text, "token_ids": ids, "visual_tokens": tokens,
                                    "generation_seconds": time.monotonic() - begin,
                                    **score_answer(text, record)}
        result["native_full_token_equal"] = (result["arms"]["native"]["token_ids"]
                                              == result["arms"]["full"]["token_ids"])
        atomic_text(path, json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(json.dumps({"completed": index + 1, "total": len(rows), "id": row["id"],
                          "answers": {a: v["answer"] for a, v in result["arms"].items()},
                          "native_full_token_equal": result["native_full_token_equal"]}), flush=True)
    metrics = sorted({r["arms"]["native"]["metric"] for r in results})
    scores = {a: {m: {"n": len(subset := [r["arms"][a] for r in results if r["arms"][a]["metric"] == m]),
                      "score": sum(x["correct"] for x in subset) / len(subset)} for m in metrics}
              for a in ("native", "full", "grid")}
    report = {"completed": True, "run": identity["id"], "n": len(results),
              "videos": len({r["video"] for r in results}), "scores": scores,
              "native_full_token_equal": sum(r["native_full_token_equal"] for r in results),
              "timestamp_approximation_over_50ms": sum(r["observation"]["timestamp_approximation_over_50ms"]
                                                      for r in results),
              "note": "Untrained model; shared explicit frame budget. Development diagnostic only. "
                      "Exact match is not semantic QA accuracy. No-cache generation is not production throughput."}
    atomic_text(out / "predictions.jsonl", "".join(json.dumps(r) + "\n" for r in results))
    atomic_text(out / "report.json", json.dumps(report, indent=2) + "\n")
    print(f"VERIFIED original-Qwen input control: {out / 'report.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--pixels", type=int, default=256 * 256)
    p.add_argument("--pool-grid", nargs=2, type=int, default=[4, 5])
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--timestamp-policy", choices=("strict", "native-index-fps"), default="strict")
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    p.add_argument("--device", default="cuda:0")
    run(p.parse_args())


if __name__ == "__main__":
    main()
