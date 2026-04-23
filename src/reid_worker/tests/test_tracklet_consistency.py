import numpy as np

from src.tracklet.consistency import compute_tracklet_consistency
from src.tracklet.models import TrackletEntry


def _make_entry(frame_idx: int, bbox: list[float], v_score: float = 0.8) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.zeros((64, 32, 3), dtype=np.uint8),
        v_score=v_score,
        bbox_xyxy=bbox,
        timestamp_ns=0,
    )


def test_tracklet_consistency_composite():
    entries = [_make_entry(i, [100, 100, 200, 300], v_score=0.9) for i in range(10)]
    c = compute_tracklet_consistency(entries)
    assert c.overall > 0.8
    assert c.good_frame_streak == 10
