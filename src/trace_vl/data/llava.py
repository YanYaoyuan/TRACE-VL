"""Prepare a small, video-disjoint SFT pilot from local LLaVA-Video annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import check_split_leakage, validate_record, write_jsonl


def _dialogue_pairs(conversation):
    if not isinstance(conversation, list) or not conversation or len(conversation) % 2:
        raise ValueError("expected alternating human/gpt conversation pairs")
    history = []
    pairs = []
    for index in range(0, len(conversation), 2):
        human, assistant = conversation[index : index + 2]
        if human.get("from") != "human" or assistant.get("from") != "gpt":
            raise ValueError("expected alternating human/gpt conversation pairs")
        question, answer = human.get("value"), assistant.get("value")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("conversation values must be strings")
        question = question.replace("<image>", "").replace("<video>", "").strip()
        answer = answer.strip()
        if not question or not answer:
            raise ValueError("empty question or answer")
        prompt = question
        if history:
            prompt = "Conversation so far:\n" + "\n".join(history) + "\nUser: " + question
        pairs.append((prompt, answer))
        # Previous assistant turns are observed dialogue context; never include future turns.
        history.extend(["User: " + question, "Assistant: " + answer])
    return pairs


def import_llava_video(paths, output, video_root, max_videos=200, val_fraction=0.1, seed=7):
    """Intersect annotations with existing media; split by resolved video, not QA row."""
    if max_videos < 2 or not 0 < val_fraction < 1:
        raise ValueError("max_videos must be >= 2 and val_fraction must be between 0 and 1")
    root = Path(video_root).resolve()
    grouped = {}
    input_rows, missing_rows = 0, 0
    digests = set()
    for annotation in paths:
        annotation = Path(annotation)
        payload = annotation.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in digests:
            raise ValueError("duplicate annotation file content")
        digests.add(digest)
        rows = json.loads(payload)
        if not isinstance(rows, list):
            raise ValueError("expected a JSON list of LLaVA-Video rows")
        for row_index, row in enumerate(rows):
            input_rows += 1
            relative = row.get("video")
            if not isinstance(relative, str) or not relative.strip():
                raise ValueError(f"{annotation.name} row {row_index}: missing video path")
            path = (root / relative).resolve()
            if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_relative_to(root):
                raise ValueError("video path must stay within video_root")
            if not path.is_file():
                missing_rows += 1
                continue
            group = path.relative_to(root).as_posix()
            pairs = _dialogue_pairs(row.get("conversations"))
            records = grouped.setdefault(group, [])
            for turn, (question, answer) in enumerate(pairs):
                records.append(
                    {
                        "schema_version": 1,
                        "id": f"llava:{digest[:16]}:{row_index}:{turn}",
                        "source": "LLaVA-Video-178K",
                        "source_group": "llava-video:" + group,
                        "video": str(path),
                        "protocol": "offline",
                        "observation": {"start": 0, "end": "video_end"},
                        "question": question,
                        "answer": answer,
                        "answer_type": "text",
                        "annotation_file": annotation.name,
                        "annotation_sha256": digest,
                        "upstream_id": str(row.get("id", row_index)),
                        "dialogue_turn": turn,
                    }
                )
    ordered = sorted(grouped, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    chosen = ordered[:max_videos]
    if len(chosen) < 2:
        raise ValueError(
            "need at least two annotated local videos; check video_root matches annotation-relative paths"
        )
    val_count = max(1, min(len(chosen) - 1, round(len(chosen) * val_fraction)))
    records = []
    for index, group in enumerate(chosen):
        for record in grouped[group]:
            records.append(validate_record({**record, "split": "val" if index < val_count else "train"}))
    counts = check_split_leakage(records)
    write_jsonl(output, records)
    report = {
        "manifest": str(Path(output).resolve()),
        "annotation_rows": input_rows,
        "missing_media_rows": missing_rows,
        "available_videos": len(grouped),
        "selected_videos": len(chosen),
        "train_videos": len(chosen) - val_count,
        "val_videos": val_count,
        "samples": len(records),
        "splits": counts,
        "seed": seed,
        "purpose": "local SFT pilot; not an official benchmark split",
    }
    Path(str(output) + ".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
