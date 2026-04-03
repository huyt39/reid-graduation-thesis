from src.scoring.tagging import VisibilityTag


class PostFrameSkipper:
    def __init__(
        self,
        rates: dict[str, int] | None = None,
        drop_floor: float = 0.15,
    ):
        self.rates = rates or {"good": 2, "mid": 3, "bad": 5}
        self.counters: dict[str, int] = {}
        self.drop_floor = drop_floor

    def should_send(self, tag: VisibilityTag, v: float, spatial_key: str) -> bool:
        if v < self.drop_floor:
            return False
        rate = self.rates[tag.value]
        count = self.counters.get(spatial_key, 0) + 1
        self.counters[spatial_key] = count
        return count % rate == 0
