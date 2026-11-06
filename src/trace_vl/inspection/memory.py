"""Bounded store of transitions with causal snapshots and relevance ranking."""

from __future__ import annotations

from .change import Transition, box_iou


def overlap(box, region) -> float:
    """Fraction of `box` covered by `region` (both normalized x1, y1, x2, y2)."""
    x1, y1 = max(box[0], region[0]), max(box[1], region[1])
    x2, y2 = min(box[2], region[2]), min(box[3], region[3])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area if area > 0 else 0.0


class TransitionStore:
    """Keeps at most `capacity` transitions.

    Eviction order when full: resolved transitions (oldest end first), then absorbed, then global,
    and only then active ones (oldest first). Active changes are what persistence questions and
    duration alerts depend on, so they are the last to go.
    """

    def __init__(self, capacity: int = 128):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items: dict[int, Transition] = {}
        self.evicted = 0

    def update(self, events):
        for event, transition in events:
            # Later events of an evicted transition must not resurrect it.
            if event in {"appear", "global"} or transition.id in self.items:
                self.items[transition.id] = transition
        while len(self.items) > self.capacity:
            self._evict()

    def _evict(self):
        def key(item: Transition):
            rank = {"resolved": 0, "absorbed": 1}.get(item.status, 3)
            if item.kind == "global":
                rank = 2
            return rank, item.t_end_known if item.t_end_known is not None else item.t_known

        victim = min(self.items.values(), key=key)
        del self.items[victim.id]
        self.evicted += 1

    def snapshot(self, t: float) -> list[Transition]:
        views = (item.view_at(t) for item in self.items.values())
        return sorted((v for v in views if v is not None), key=lambda v: v.t_known)

    @staticmethod
    def rank(transitions, t: float, region=None, window: float | None = None) -> list[Transition]:
        """Most relevant first: overlapping the region, intersecting the window, then most recent."""

        def relevant(item: Transition):
            in_region = 1.0 if region is None else max(overlap(item.box, region), box_iou(item.box, region))
            end = item.t_end if item.t_end is not None else t
            in_window = 1.0 if window is None or end >= t - window else 0.0
            return (in_region > 0.05, in_window, in_region, item.t_known)

        return sorted(transitions, key=relevant, reverse=True)
