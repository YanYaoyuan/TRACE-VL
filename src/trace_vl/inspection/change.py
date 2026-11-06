"""Causal, region-level change detection for fixed cameras.

Dual background subtraction: a fast background follows anything that stops moving, a slow one keeps
the scene as it was. Pixels that differ from the slow background but agree with the fast one are
static changes (an object put down, removed, a door left open). A change is confirmed once it has
persisted `confirm_seconds`; it is resolved when the region returns to the slow background for
`vanish_seconds`. The slow background is frozen inside active regions so a long-lasting change is not
silently absorbed; after `absorb_seconds` it is absorbed explicitly and marked as such.

Everything is causal: a transition only becomes visible at its confirmation time `t_known`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    work_width: int = 480
    short_seconds: float = 4.0
    long_seconds: float = 600.0
    threshold: float = 25.0
    confirm_seconds: float = 6.0
    vanish_seconds: float = 4.0
    min_area: float = 0.0008
    max_area: float = 0.25
    global_change_fraction: float = 0.35
    absorb_seconds: float = 1800.0
    match_iou: float = 0.3
    crop_scale: float = 1.6
    crop_size: int = 320

    @classmethod
    def from_dict(cls, values: dict | None) -> "DetectorConfig":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown detector settings: {sorted(unknown)}")
        return cls(**values)


@dataclass
class Transition:
    id: int
    box: tuple[float, float, float, float]  # normalized x1, y1, x2, y2
    t_start: float  # estimated onset of the change
    t_known: float  # confirmation time; the transition does not exist before this
    before: np.ndarray  # RGB crop of the slow background (scene before the change)
    after: np.ndarray  # RGB crop of the frame at confirmation
    kind: str = "region"  # region | global
    status: str = "active"  # active | resolved | absorbed
    t_end: float | None = None  # estimated time the region reverted
    t_end_known: float | None = None  # time the reversion was confirmed
    after_end: np.ndarray | None = None  # RGB crop after the reversion
    last_seen: float = 0.0
    gone_since: float | None = None

    def view_at(self, t: float) -> "Transition | None":
        """The transition as a causal observer knows it at time t (None if not yet confirmed)."""
        if self.t_known > t + 1e-6:
            return None
        if self.t_end_known is not None and self.t_end_known > t + 1e-6:
            # Ending (resolution or absorption) is still in the future: hide it.
            return Transition(
                self.id, self.box, self.t_start, self.t_known, self.before, self.after, self.kind
            )
        return self


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop(image: np.ndarray, box, scale: float, size: int) -> np.ndarray:
    """Context crop around a normalized box, longest side resized to `size`."""
    h, w = image.shape[:2]
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    half_w = max((box[2] - box[0]) * w * scale / 2, 16)
    half_h = max((box[3] - box[1]) * h * scale / 2, 16)
    x1, x2 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y1, y2 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        patch = image
    ratio = size / max(patch.shape[:2])
    if ratio < 1:
        patch = cv2.resize(patch, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(patch.astype(np.uint8))


@dataclass
class _Track:
    transition: Transition
    work_box: tuple[int, int, int, int]  # x1, y1, x2, y2 in work-resolution pixels


class ChangeDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self.next_id = 0
        self.tracks: list[_Track] = []
        self.initialized = False
        self.last_t = None

    def _init(self, frame: np.ndarray, small: np.ndarray, t: float):
        self.shape = small.shape[:2]
        self.bg_short = small.copy()
        self.bg_long = small.copy()
        self.reference = frame.astype(np.float32)
        self.evidence = np.zeros(self.shape, np.float32)
        # Start of each pixel's current continuous difference from the slow background: the onset of a
        # put-down is when the person/object first covered the pixel, not when it stopped moving.
        self.onset = np.full(self.shape, np.nan, np.float32)
        self.initialized, self.last_t = True, t

    def _small(self, frame):
        h, w = frame.shape[:2]
        width = min(self.cfg.work_width, w)
        height = max(1, round(h * width / w))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _new_id(self):
        self.next_id += 1
        return self.next_id

    def _normalized(self, x1, y1, x2, y2):
        h, w = self.shape
        return (x1 / w, y1 / h, x2 / w, y2 / h)

    def _crop(self, image, box):
        return crop(image, box, self.cfg.crop_scale, self.cfg.crop_size)

    def _reset_region(self, small, frame, work_box):
        x1, y1, x2, y2 = work_box
        self.bg_long[y1:y2, x1:x2] = small[y1:y2, x1:x2]
        self.evidence[y1:y2, x1:x2] = 0
        self.onset[y1:y2, x1:x2] = np.nan
        h, w = self.shape
        fh, fw = frame.shape[:2]
        fx1, fx2 = int(x1 * fw / w), int(math.ceil(x2 * fw / w))
        fy1, fy2 = int(y1 * fh / h), int(math.ceil(y2 * fh / h))
        self.reference[fy1:fy2, fx1:fx2] = frame[fy1:fy2, fx1:fx2]

    def step(self, frame: np.ndarray, t: float) -> list[tuple[str, Transition]]:
        """Consume one RGB uint8 frame at stream time t; return (event, transition) pairs.

        Events: "appear" (new confirmed change), "resolve" (region reverted), "absorb" (change
        accepted as the new background after `absorb_seconds`), "global" (scene-wide change such as
        lights switching; all backgrounds are reset).
        """
        cfg = self.cfg
        small = self._small(frame)
        if not self.initialized:
            self._init(frame, small, t)
            return []
        if t <= self.last_t:
            raise ValueError("frame timestamps must be strictly increasing")
        dt, self.last_t = t - self.last_t, t
        alpha_short = 1 - math.exp(-dt / cfg.short_seconds)
        alpha_long = 1 - math.exp(-dt / cfg.long_seconds)
        diff_long = np.abs(small - self.bg_long).mean(2) > cfg.threshold
        diff_short = np.abs(small - self.bg_short).mean(2) > cfg.threshold
        events: list[tuple[str, Transition]] = []

        if diff_long.mean() > cfg.global_change_fraction:
            transition = Transition(
                self._new_id(),
                (0.0, 0.0, 1.0, 1.0),
                t,
                t,
                self._crop(self.reference.astype(np.uint8), (0, 0, 1, 1)),
                self._crop(frame, (0, 0, 1, 1)),
                kind="global",
                status="resolved",
                t_end=t,
                t_end_known=t,
                last_seen=t,
            )
            for track in self.tracks:
                track.transition.status = "absorbed"
                track.transition.t_end_known = t
                events.append(("absorb", track.transition))
            self.tracks = []
            self._init(frame, small, t)
            events.append(("global", transition))
            return events

        static = diff_long & ~diff_short
        self.evidence = np.where(static, self.evidence + dt, np.maximum(self.evidence - 2 * dt, 0))
        self.onset = np.where(diff_long & np.isnan(self.onset), t, self.onset)
        self.onset = np.where(diff_long, self.onset, np.nan).astype(np.float32)
        # Must still differ from the slow background now: a removed object stops counting at once,
        # while a person briefly occluding a left object keeps it confirmed.
        confirmed = ((self.evidence >= cfg.confirm_seconds) & diff_long).astype(np.uint8)

        protected = np.zeros(self.shape, bool)
        for track in self.tracks:
            x1, y1, x2, y2 = track.work_box
            protected[y1:y2, x1:x2] = True
        # The slow background only learns pixels that look like it already, never active regions.
        update = ~diff_long & ~protected
        self.bg_short += alpha_short * (small - self.bg_short)
        self.bg_long[update] += alpha_long * (small[update] - self.bg_long[update])
        mask = cv2.resize(update.astype(np.uint8), frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        cv2.accumulateWeighted(frame.astype(np.float32), self.reference, alpha_long, mask)

        kernel = np.ones((3, 3), np.uint8)
        confirmed = cv2.morphologyEx(confirmed, cv2.MORPH_CLOSE, kernel, iterations=2)
        confirmed = cv2.morphologyEx(confirmed, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(confirmed, connectivity=8)
        total = self.shape[0] * self.shape[1]
        matched = set()
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if not cfg.min_area <= area / total <= cfg.max_area:
                continue
            work_box = (int(x), int(y), int(x + w), int(y + h))
            box = self._normalized(*work_box)
            best, best_iou = None, cfg.match_iou
            for index, track in enumerate(self.tracks):
                iou = box_iou(box, track.transition.box)
                if iou >= best_iou:
                    best, best_iou = index, iou
            if best is not None:
                matched.add(best)
                track = self.tracks[best]
                track.transition.last_seen, track.transition.gone_since = t, None
                continue
            onsets = self.onset[labels == label]
            onset = float(np.nanmedian(onsets)) if np.isfinite(onsets).any() else t
            transition = Transition(
                self._new_id(),
                box,
                min(onset, t),
                t,
                self._crop(self.reference.astype(np.uint8), box),
                self._crop(frame, box),
                last_seen=t,
            )
            self.tracks.append(_Track(transition, work_box))
            matched.add(len(self.tracks) - 1)
            events.append(("appear", transition))

        remaining = []
        for index, track in enumerate(self.tracks):
            transition = track.transition
            x1, y1, x2, y2 = track.work_box
            if t - transition.t_start >= cfg.absorb_seconds:
                transition.status, transition.t_end_known = "absorbed", t
                self._reset_region(small, frame, track.work_box)
                events.append(("absorb", transition))
                continue
            if index not in matched:
                # Reverted means the region looks like the slow background again (not just occluded).
                reverted = diff_long[y1:y2, x1:x2].mean() < 0.3 if y2 > y1 and x2 > x1 else True
                if reverted:
                    if transition.gone_since is None:
                        transition.gone_since = t
                    if t - transition.gone_since >= cfg.vanish_seconds:
                        transition.status = "resolved"
                        transition.t_end, transition.t_end_known = transition.gone_since, t
                        transition.after_end = self._crop(frame, transition.box)
                        self.evidence[y1:y2, x1:x2] = 0
                        self.onset[y1:y2, x1:x2] = np.nan
                        events.append(("resolve", transition))
                        continue
                else:
                    transition.gone_since = None
            remaining.append(track)
        self.tracks = remaining
        return events
