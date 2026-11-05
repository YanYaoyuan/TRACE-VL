"""Compare held-out answer loss with matched and mismatched video pixels; no official score."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from trace_vl.config import load_config
from trace_vl.data.dataset import collate_samples
from trace_vl.evaluation.metrics import normalized_text, token_f1
from trace_vl.runtime import (
    autocast_for,
    device_for,
    environment,
    file_hash,
    make_dataset,
    move_batch,
    restore_model,
    seed_everything,
    write_json,
)


@torch.inference_mode()
def diagnose(
    config,
    checkpoint,
    output,
    max_videos=20,
    seed=7,
    all_questions=False,
    generate=False,
    reuse_video_state=None,
):
    if max_videos < 2:
        raise ValueError("max_videos must be >= 2")
    seed_everything(seed, config.get("evaluation", {}).get("cpu_threads", 2))
    dataset = make_dataset(config, "val")
    groups = {}
    for index, record in enumerate(dataset.records):
        key = hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest()
        group = record["source_group"]
        if group not in groups or key < groups[group][0]:
            groups[group] = (key, index)
    selected = sorted(groups.values())[:max_videos]
    if len(selected) < 2:
        raise ValueError("need at least two held-out videos")
    representatives = [dataset[index] for _, index in selected]
    donors = {
        sample["record"]["source_group"]: representatives[(index + 1) % len(representatives)]
        for index, sample in enumerate(representatives)
    }
    samples = (
        [dataset[index] for index, record in enumerate(dataset.records) if record["source_group"] in donors]
        if all_questions
        else representatives
    )
    device = device_for(config.get("evaluation", {}).get("device", "auto"))
    model = restore_model(config, checkpoint, device)
    results = []
    reuse = (
        config.get("evaluation", {}).get("reuse_video_state", False)
        if reuse_video_state is None
        else reuse_video_state
    )
    cached = None
    prefix_pairs = 0
    with autocast_for(device, config.get("evaluation", {}).get("precision", "fp32")):
        for sample in samples:
            donor = donors[sample["record"]["source_group"]]
            assert sample["record"]["source_group"] != donor["record"]["source_group"]
            assert sample["record"]["video"] != donor["record"]["video"]
            # Keep target questions, answers, timestamps and frame counts unchanged.
            # Only replace pixels with another held-out video's uniformly spaced frames.
            indices = torch.linspace(0, len(donor["frames"]) - 1, len(sample["frames"])).round().long()
            mismatch = {**sample, "frames": donor["frames"][indices]}
            if reuse:
                key = (
                    sample["record"]["video"],
                    sample["record"]["observation"]["start"],
                    sample["cutoff"],
                    sample["rule"],
                    donor["record"]["video"],
                )
                if cached is None or cached[0] != key:
                    states = []
                    for value in (sample, mismatch):
                        state, _ = model.observe(
                            value["frames"][None].to(device),
                            value["timestamps"][None].to(device),
                            rules=[value["rule"]],
                        )
                        states.append(state)
                    cached = (key, *states)
                    prefix_pairs += 1
                outputs = []
                for state in cached[1:]:
                    prefix, _ = model.evidence(state, [sample["question"]])
                    value = model.language.loss(prefix, [sample["question"]], [sample["answer"]])
                    outputs.append({"state": state, "losses": {"answer": value}})
                matched_output, shuffled_output = outputs
            else:
                matched_output = model(move_batch(collate_samples([sample]), device))
                shuffled_output = model(move_batch(collate_samples([mismatch]), device))
                prefix_pairs += 1
            matched = matched_output["losses"]["answer"]
            shuffled = shuffled_output["losses"]["answer"]
            result = {
                "id": sample["record"]["id"],
                "video": sample["record"]["video"],
                "donor_video": donor["record"]["video"],
                "question": sample["question"],
                "answer": sample["answer"],
                "matched_answer_loss": float(matched),
                "mismatched_answer_loss": float(shuffled),
                "mismatched_minus_matched": float(shuffled - matched),
            }
            if generate:
                limit = config.get("evaluation", {}).get("max_new_tokens", 32)
                # Observation states are constructed before answer loss; no gold answer is
                # supplied to generation, and language.loss uses no persistent KV cache.
                correct, _ = model.answer(matched_output["state"], [sample["question"]], limit)
                wrong, _ = model.answer(shuffled_output["state"], [sample["question"]], limit)
                target = normalized_text(sample["answer"])
                result.update(
                    matched_prediction=correct[0],
                    mismatched_prediction=wrong[0],
                    matched_correct=normalized_text(correct[0]) == target,
                    mismatched_correct=normalized_text(wrong[0]) == target,
                    matched_token_f1=token_f1(correct[0], sample["answer"]),
                    mismatched_token_f1=token_f1(wrong[0], sample["answer"]),
                    generated_answer_changed=normalized_text(correct[0]) != normalized_text(wrong[0]),
                )
            results.append(result)
    count = len(results)
    report = {
        "videos": len(selected),
        "questions": count,
        "reuse_video_state": reuse,
        "observed_prefix_pairs": prefix_pairs,
        "seed": seed,
        "selection": (
            "All questions from selected held-out videos; video donors follow seeded representative-ID order."
            if all_questions
            else "One question per held-out source_group, selected by seeded ID hash; cyclic donor permutation."
        ),
        "matched_mean_answer_loss": sum(r["matched_answer_loss"] for r in results) / count,
        "mismatched_mean_answer_loss": sum(r["mismatched_answer_loss"] for r in results) / count,
        "mean_loss_increase": sum(r["mismatched_minus_matched"] for r in results) / count,
        "fraction_matched_lower_loss": sum(r["mismatched_minus_matched"] > 0 for r in results) / count,
        "manifest_sha256": file_hash(config["data"]["manifest"]),
        "checkpoint_sha256": file_hash(checkpoint),
        "environment": environment(),
        "config": config,
        "note": "Teacher-forced per-sample mean answer loss, optionally with greedy generation. Questions share source videos and are not independent video samples. No official benchmark or robust grounding claim from a positive gap alone.",
        "results": results,
    }
    if generate:
        report["generation"] = {
            "matched_exact_match": sum(r["matched_correct"] for r in results) / count,
            "mismatched_exact_match": sum(r["mismatched_correct"] for r in results) / count,
            "matched_token_f1": sum(r["matched_token_f1"] for r in results) / count,
            "mismatched_token_f1": sum(r["mismatched_token_f1"] for r in results) / count,
            "fraction_answer_changed": sum(r["generated_answer_changed"] for r in results) / count,
        }
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"results", "config"}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_2b_pilot_cfr.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--reuse-video-state", action="store_true", default=None)
    args = parser.parse_args()
    diagnose(
        load_config(args.config),
        args.checkpoint,
        args.output,
        args.max_videos,
        args.seed,
        args.all_questions,
        args.generate,
        args.reuse_video_state,
    )
