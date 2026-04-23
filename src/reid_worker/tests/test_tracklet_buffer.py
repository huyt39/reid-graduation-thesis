import numpy as np

from src.tracklet.buffer import TrackletBuffer
from src.tracklet.models import TrackletEntry, TrackletState


def _make_entry(frame_idx: int, ts_offset_s: float = 0.0) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.zeros((64, 32, 3), dtype=np.uint8),
        v_score=0.8,
        bbox_xyxy=[10.0, 20.0, 50.0, 120.0],
        timestamp_ns=int(ts_offset_s * 1e9),
    )


def test_ready_when_enough_entries_and_window_expired():
    buf = TrackletBuffer(min_entries=3, window_seconds=1.0)
    for i in range(5):
        buf.append(1, _make_entry(i, ts_offset_s=0.0))
    assert len(buf.get_ready_tracklets(current_time_ns=int(0.5e9))) == 0
    ready = buf.get_ready_tracklets(current_time_ns=int(1.5e9))
    assert len(ready) == 1
    assert ready[0].state == TrackletState.READY
