from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .schema import check_split_leakage, read_jsonl
from .video import sample_video


class VideoDataset(Dataset):
    def __init__(self, manifest, split=None, image_size=32, sampling_fps=2, max_frames=8, media_root=None):
        self.manifest = Path(manifest).resolve()
        all_records = read_jsonl(self.manifest)
        check_split_leakage(all_records)
        self.records = [r for r in all_records if split is None or r["split"] == split]
        if not self.records:
            raise ValueError(f"no records for split {split!r}")
        self.media_root = Path(media_root).resolve() if media_root else self.manifest.parent
        self.image_size, self.sampling_fps, self.max_frames = image_size, sampling_fps, max_frames

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        video = Path(r["video"])
        if not video.is_absolute():
            video = self.media_root / video
        frames, times, end = sample_video(
            video,
            r["observation"]["start"],
            r["observation"]["end"],
            self.sampling_fps,
            self.max_frames,
            self.image_size,
        )
        alert = torch.full_like(times, -1.0)
        write = torch.full_like(times, -1.0)
        utility = torch.full_like(times, -1.0)
        if "events" in r:
            alert.zero_()
            write.zero_()
            for event in r["events"]:
                active = (times >= event["known_at"] - 1e-6) & (times <= event["end"] + 1e-6)
                alert[active] = 1
                indices = torch.nonzero(times >= event["known_at"] - 1e-6).flatten()
                if len(indices) and times[indices[0]] <= event["deadline"] + 1e-6:
                    write[indices[0]] = 1
        for label in r.get("utility_targets", []):
            nearest = int((times - label["time"]).abs().argmin())
            if abs(float(times[nearest]) - label["time"]) <= 1 / self.sampling_fps:
                utility[nearest] = label["value"]
        question = r["question"]
        if r.get("options"):
            question += "\n" + "\n".join(r["options"]) + "\nAnswer with the option letter only."
        return {
            "frames": frames,
            "timestamps": times,
            "alert_labels": alert,
            "write_labels": write,
            "utility_labels": utility,
            "question": question,
            "answer": r["answer"],
            "rule": r.get("rule", ""),
            "record": r,
            "cutoff": end,
        }


def collate_samples(samples):
    n, length = len(samples), max(len(s["timestamps"]) for s in samples)
    frames = samples[0]["frames"].new_zeros(n, length, *samples[0]["frames"].shape[1:])
    times = torch.zeros(n, length)
    mask = torch.zeros(n, length, dtype=torch.bool)
    labels = {k: torch.full((n, length), -1.0) for k in ("alert_labels", "write_labels", "utility_labels")}
    for i, sample in enumerate(samples):
        t = len(sample["timestamps"])
        frames[i, :t], times[i, :t], mask[i, :t] = sample["frames"], sample["timestamps"], True
        times[i, t:] = sample["timestamps"][-1]
        for k in labels:
            labels[k][i, :t] = sample[k]
    return {
        "frames": frames,
        "timestamps": times,
        "frame_mask": mask,
        **labels,
        "questions": [s["question"] for s in samples],
        "answers": [s["answer"] for s in samples],
        "rules": [s["rule"] for s in samples],
        "records": [s["record"] for s in samples],
        "cutoffs": [s["cutoff"] for s in samples],
    }
