from src.tracklet.models import Tracklet, TrackletEntry, TrackletState


class TrackletBuffer:

    def __init__(
        self,
        min_entries: int = 8,
        max_entries: int = 60,
        window_frames: int = 90,
        stale_frames: int = 150,
    ):
        self.tracklets: dict[int, Tracklet] = {}
        self.min_entries = min_entries
        self.max_entries = max_entries
        self.window_frames = int(window_frames)
        self.stale_frames = int(stale_frames)

    def append(self, track_id: int, entry: TrackletEntry) -> None:
        if track_id not in self.tracklets:
            self.tracklets[track_id] = Tracklet(track_id=track_id, created_at_ns=entry.timestamp_ns)
        tracklet = self.tracklets[track_id]
        tracklet.entries.append(entry)
        if len(tracklet.entries) > self.max_entries:
            tracklet.entries = tracklet.entries[-self.max_entries:]
            tracklet.created_at_ns = tracklet.entries[0].timestamp_ns

    def _is_ready(self, tracklet: Tracklet, current_frame_idx: int) -> bool:
        if len(tracklet.entries) < self.min_entries:
            return False
        if len(tracklet.entries) >= self.max_entries:
            return True
        first_f = int(tracklet.entries[0].frame_idx)
        last_f = int(tracklet.entries[-1].frame_idx)
        observed_span = max(0, last_f - first_f)
        buffered_age = max(0, int(current_frame_idx) - first_f)
        return max(observed_span, buffered_age) >= self.window_frames

    def get_ready_tracklets(self, current_frame_idx: int) -> list[Tracklet]:
        ready = []
        for tracklet in self.tracklets.values():
            if tracklet.state != TrackletState.ACTIVE:
                continue
            if self._is_ready(tracklet, current_frame_idx):
                tracklet.state = TrackletState.READY
                ready.append(tracklet)
        return ready

    def pop_ready_tracklets(
        self,
        current_frame_idx: int,
        skip_track_ids: set[int] | None = None,
    ) -> list[Tracklet]:
        skip_track_ids = skip_track_ids or set()
        ready = []
        for tid, tracklet in list(self.tracklets.items()):
            if tid in skip_track_ids:
                continue
            if tracklet.state != TrackletState.ACTIVE:
                continue
            if not self._is_ready(tracklet, current_frame_idx):
                continue
            del self.tracklets[tid]
            tracklet.state = TrackletState.READY
            tracklet.entries = list(tracklet.entries)
            ready.append(tracklet)
        return ready

    def _last_frame(self, tracklet: Tracklet) -> int:
        return int(tracklet.entries[-1].frame_idx) if tracklet.entries else 0

    def evict_stale(self, current_frame_idx: int) -> list[int]:
        evicted = []
        for tid, tracklet in list(self.tracklets.items()):
            if int(current_frame_idx) - self._last_frame(tracklet) > self.stale_frames:
                del self.tracklets[tid]
                evicted.append(tid)
        return evicted

    def pop_stale_tracklets(
        self,
        current_frame_idx: int,
        skip_track_ids: set[int] | None = None,
    ) -> list[Tracklet]:
        skip_track_ids = skip_track_ids or set()
        stale = []
        for tid, tracklet in list(self.tracklets.items()):
            if tid in skip_track_ids:
                continue
            if int(current_frame_idx) - self._last_frame(tracklet) > self.stale_frames:
                stale.append(tracklet)
                del self.tracklets[tid]
        return stale

    def remove(self, track_id: int) -> None:
        self.tracklets.pop(track_id, None)
