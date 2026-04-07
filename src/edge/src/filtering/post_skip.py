from src.scoring.tagging import VisibilityTag

class PostFrameSkipper:
    def __init__(
        self,
        rates: dict[str, int] | None = None,
        drop_floor: float = 0.15,
        stale_after: int = 30,
    ):
        self.rates = rates or {"good": 2, "mid": 3, "bad": 5}
        self.counters: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.drop_floor = drop_floor
        self.stale_after = max(1, stale_after)

    def _prune(self, frame_idx: int) -> None:
        stale_keys = [
            key
            for key, last_seen in self.last_seen.items()
            if frame_idx - last_seen > self.stale_after
        ]
        for key in stale_keys:
            self.last_seen.pop(key, None)
            self.counters.pop(key, None)

    def should_send(
        self,
        tag: VisibilityTag,
        v: float,
        spatial_key: str,
        frame_idx: int | None = None,
    ) -> bool:
        if v < self.drop_floor:
            return False

        if frame_idx is not None:
            self._prune(frame_idx)
            self.last_seen[spatial_key] = frame_idx

        rate = self.rates[tag.value]
        count = self.counters.get(spatial_key, 0) + 1
        self.counters[spatial_key] = count
        return count % rate == 0