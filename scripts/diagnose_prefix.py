"""Check whether the visual prefix carries video information before spending GPU hours on training.

For one fixed question, observe several held-out videos and compare the LLM prefixes they produce:
- a between-video variance fraction near 0 means the prefix is (almost) video-independent (collapse);
- prefix token norm far from the LLM's own token-embedding norm means a scale mismatch the frozen
  LLM may simply ignore or be swamped by;
- the same statistics for the reader input (last visual tokens, working state, capsules) show where in
  the pipeline video differences disappear.
No benchmark claim follows from these numbers; they are a sanity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from trace_vl.config import load_config
from trace_vl.runtime import autocast_for, device_for, file_hash, make_dataset, restore_model, write_json


def cross_video_cosine(values):
    """Mean off-diagonal cosine similarity between per-video flattened tensors."""
    flat = F.normalize(torch.stack([v.float().flatten() for v in values]), dim=-1)
    similarity = flat @ flat.t()
    count = len(values)
    return float((similarity.sum() - similarity.diagonal().sum()) / (count * (count - 1)))


def between_video_fraction(values):
    """Share of the tensor's energy that differs between videos (0 = identical for every video).

    Raw cosine is dominated by components shared by all inputs (biases, type/position embeddings) and
    sits near 1.0 even for an untrained model, so this centered ratio is the primary collapse signal.
    """
    stacked = torch.stack([v.float().flatten() for v in values])
    spread = (stacked - stacked.mean(0, keepdim=True)).square().sum(-1).mean()
    return float(spread / stacked.square().sum(-1).mean().clamp_min(1e-12))


def token_norm(value):
    return float(value.float().norm(dim=-1).mean())


@torch.inference_mode()
def diagnose(config, checkpoint, output, max_videos=8, question=None):
    dataset = make_dataset(config, "val")
    seen, samples = set(), []
    for index, record in enumerate(dataset.records):
        if record["source_group"] not in seen:
            seen.add(record["source_group"])
            samples.append(dataset[index])
        if len(samples) == max_videos:
            break
    if len(samples) < 2:
        raise ValueError("need at least two held-out videos")
    question = question or samples[0]["question"]
    device = device_for(config.get("evaluation", {}).get("device", "auto"))
    model = restore_model(config, checkpoint, device)
    stages = {"current_visual": [], "working_state": [], "events": [], "prefix": []}
    with autocast_for(device, config.get("evaluation", {}).get("precision", "fp32")):
        for sample in samples:
            state, _ = model.observe(
                sample["frames"][None].to(device),
                sample["timestamps"][None].to(device),
                rules=[sample["rule"]],
            )
            prefix, _ = model.evidence(state, [question])
            stages["current_visual"].append(state.last_visual[0])
            stages["working_state"].append(state.memory.working[0])
            stages["events"].append(state.memory.events[0][state.memory.valid[0]].mean(0))
            stages["prefix"].append(prefix[0])
        embedding = (
            model.language.lm.get_input_embeddings().weight
            if hasattr(model.language, "lm")
            else model.language.embedding.weight
        )
    report = {
        "videos": len(samples),
        "question": question,
        "between_video_fraction": {name: between_video_fraction(v) for name, v in stages.items()},
        "cross_video_cosine": {name: cross_video_cosine(v) for name, v in stages.items()},
        "mean_token_norm": {name: sum(map(token_norm, v)) / len(v) for name, v in stages.items()},
        "llm_embedding_token_norm": token_norm(embedding),
        "checkpoint_sha256": file_hash(checkpoint),
        "note": (
            "between_video_fraction near 0 at 'prefix' means video-independent evidence; a drop from "
            "'current_visual' to 'prefix' locates where video differences are lost. Raw cosine is "
            "reported for reference only. Compare checkpoints with the same videos and question."
        ),
    }
    report["prefix_to_embedding_norm_ratio"] = (
        report["mean_token_norm"]["prefix"] / report["llm_embedding_token_norm"]
    )
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--question", default=None)
    args = parser.parse_args()
    diagnose(load_config(args.config), args.checkpoint, args.output, args.max_videos, args.question)
