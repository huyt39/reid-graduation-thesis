import numpy as np

from src.tracklet.models import TrackletEntry
from src.tracklet.selector import TopKSelector


def _make_entry(frame_idx: int, v_score: float, overlap_ratio: float = 0.0) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.zeros((64, 32, 3), dtype=np.uint8),
        v_score=v_score,
        bbox_xyxy=[10.0, 20.0, 50.0, 120.0],
        timestamp_ns=0,
        overlap_ratio=overlap_ratio,
    )


def test_overlap_penalizes_selection():
    selector = TopKSelector(k=2, min_temporal_gap=1, overlap_lambda=0.5)
    entries = [
        _make_entry(0, v_score=0.9, overlap_ratio=0.8),
        _make_entry(1, v_score=0.7, overlap_ratio=0.0),
        _make_entry(2, v_score=0.6, overlap_ratio=0.0),
    ]
    selected = selector.select(entries)
    assert selected[0].frame_idx == 1
