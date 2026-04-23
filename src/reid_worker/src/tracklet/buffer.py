from src.tracklet.models import Tracklet, TrackletEntry, TrackletState


class TrackletBuffer:
    def __init__(
        self,
        min_entries: int = 8,
        max_entries: int = 60,
        window_seconds: float = 3.0,
        stale_seconds: float = 5.0,
    ):
        self.tracklets: dict[int, Tracklet] = {}
        self.min_entries = min_entries
        self.max_entries = max_entries
        self.window_ns = int(window_seconds * 1e9)
        self.stale_ns = int(stale_seconds * 1e9)

    def append(self, track_id: int, entry: TrackletEntry) -> None:
        if track_id not in self.tracklets:
            self.tracklets[track_id] = Tracklet(track_id=track_id, created_at_ns=entry.timestamp_ns)
        tracklet = self.tracklets[track_id]
        tracklet.entries.append(entry)
        if len(tracklet.entries) > self.max_entries:
            tracklet.entries = tracklet.entries[-self.max_entries:]

    def get_ready_tracklets(self, current_time_ns: int) -> list[Tracklet]:
        ready = []
        for tracklet in self.tracklets.values():
            if tracklet.state != TrackletState.ACTIVE:
                continue
            has_enough = len(tracklet.entries) >= self.min_entries
            window_expired = (current_time_ns - tracklet.created_at_ns) >= self.window_ns
            if has_enough and window_expired:
                tracklet.state = TrackletState.READY
                ready.append(tracklet)
        return ready

    def evict_stale(self, current_time_ns: int) -> list[int]:
        evicted = []
        for tid, tracklet in list(self.tracklets.items()):
            last_ts = tracklet.entries[-1].timestamp_ns if tracklet.entries else 0
            if current_time_ns - last_ts > self.stale_ns:
                del self.tracklets[tid]
                evicted.append(tid)
        return evicted

    def remove(self, track_id: int) -> None:
        self.tracklets.pop(track_id, None)
