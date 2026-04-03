import numpy as np
from src.tracklet.buffer import TrackletBuffer
from src.tracklet.models import TrackletEntry, TrackletState


def _make_entry(frame_idx: int, v_score: float = 0.8, ts_offset_s: float = 0.0) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.zeros((64, 32, 3), dtype=np.uint8),
        v_score=v_score,
        bbox_xyxy=[10.0, 20.0, 50.0, 120.0],
        timestamp_ns=int(ts_offset_s * 1e9),
    )


class TestTrackletBuffer:
    def test_append_creates_tracklet(self):
        buf = TrackletBuffer(min_entries=2)
        buf.append(1, _make_entry(1))
        assert 1 in buf.tracklets
        assert len(buf.tracklets[1].entries) == 1

    def test_ring_buffer_limit(self):
        buf = TrackletBuffer(max_entries=5)
        for i in range(10):
            buf.append(1, _make_entry(i))
        assert len(buf.tracklets[1].entries) == 5

    def test_ready_when_enough_entries_and_window_expired(self):
        buf = TrackletBuffer(min_entries=3, window_seconds=1.0)
        for i in range(5):
            buf.append(1, _make_entry(i, ts_offset_s=0.0))
        # Window hasn't expired yet (created_at_ns=0, need 1s)
        ready = buf.get_ready_tracklets(current_time_ns=int(0.5e9))
        assert len(ready) == 0
        # Now window expired
        ready = buf.get_ready_tracklets(current_time_ns=int(1.5e9))
        assert len(ready) == 1
        assert ready[0].state == TrackletState.READY

    def test_evict_stale(self):
        buf = TrackletBuffer(stale_seconds=2.0)
        buf.append(1, _make_entry(1, ts_offset_s=0.0))
        evicted = buf.evict_stale(current_time_ns=int(3e9))
        assert 1 in evicted
        assert 1 not in buf.tracklets
