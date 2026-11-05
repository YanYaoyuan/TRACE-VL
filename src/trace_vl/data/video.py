"""CFR OpenCV decoder and timestamped NPZ fixture decoder, with explicit prefix cutoffs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def sample_video(path, start, end, sampling_fps, max_frames, image_size):
    if sampling_fps <= 0 or max_frames <= 0:
        raise ValueError("sampling_fps and max_frames must be positive")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            images, times = z["frames"], z["timestamps"].astype(np.float64)
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("NPZ frames must be uint8 [T,H,W,3]")
        if (
            not len(times)
            or len(times) != len(images)
            or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()
        ):
            raise ValueError("invalid source timestamps")
        end = float(times[-1]) if end == "video_end" else float(end)
        if end > times[-1] + 1e-6:
            raise ValueError("observation cutoff exceeds timestamped media duration")
        eligible = np.flatnonzero((times >= start - 1e-8) & (times <= end + 1e-8))
        if not len(eligible):
            raise ValueError("no observations inside the requested video prefix")
        target = np.arange(start, end + 1e-8, 1 / sampling_fps)
        # Latest available frame at or before each target, never the next/future frame.
        idx = np.searchsorted(times, target + 1e-8, side="right") - 1
        idx = np.unique(idx[np.isin(idx, eligible)])
        if not len(idx):
            idx = eligible[:1]
        if len(idx) > max_frames:
            idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
        selected, actual = images[idx], times[idx]
    else:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count <= 0:
                raise ValueError(f"invalid video metadata: {path}")
            duration_end = (count - 1) / fps
            end = duration_end if end == "video_end" else float(end)
            if end > duration_end + 1 / fps + 1e-6:
                raise ValueError("observation cutoff exceeds video duration")
            target = np.arange(start, min(end, duration_end) + 1e-8, 1 / sampling_fps)
            idx = np.unique(np.floor(target * fps + 1e-7).astype(int))
            idx = idx[(idx >= 0) & (idx < count) & (idx / fps >= start - 1e-8) & (idx / fps <= end + 1e-8)]
            if len(idx) > max_frames:
                idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
            selected, actual = [], []
            for n in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"failed to decode frame {n} in {path}")
                decoded = round(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if decoded != n:
                    raise ValueError("decoder returned a different frame; normalize this video to CFR first")
                selected.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                actual.append(n / fps)
            if not selected:
                raise ValueError("no decodable frames inside observation window")
            selected, actual = np.stack(selected), np.array(actual)
        finally:
            cap.release()
    frames = torch.from_numpy(np.ascontiguousarray(selected)).permute(0, 3, 1, 2).float() / 255
    frames = F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return frames, torch.tensor(actual, dtype=torch.float32), float(end)
