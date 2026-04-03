import numpy as np
from src.tracklet.selector import TopKSelector
from src.tracklet.models import TrackletEntry


def _make_entry(frame_idx: int, v_score: float, overlap_ratio: float = 0.0) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.zeros((64, 32, 3), dtype=np.uint8),
        v_score=v_score,
        bbox_xyxy=[10.0, 20.0, 50.0, 120.0],
        timestamp_ns=0,
        overlap_ratio=overlap_ratio,
    )


class TestTopKSelector:
    def test_selects_top_k(self):
        selector = TopKSelector(k=3, min_temporal_gap=1)
        entries = [_make_entry(i, v_score=i * 0.1) for i in range(10)]
        selected = selector.select(entries)
        assert len(selected) == 3
        assert selected[0].v_score >= selected[1].v_score

    def test_temporal_diversity(self):
        selector = TopKSelector(k=3, min_temporal_gap=5)
        entries = [_make_entry(i, v_score=1.0 - i * 0.01) for i in range(20)]
        selected = selector.select(entries)
        assert len(selected) == 3
        frame_idxs = [e.frame_idx for e in selected]
        for i in range(len(frame_idxs)):
            for j in range(i + 1, len(frame_idxs)):
                assert abs(frame_idxs[i] - frame_idxs[j]) >= 5

    def test_relaxes_constraint_if_needed(self):
        selector = TopKSelector(k=5, min_temporal_gap=100)
        entries = [_make_entry(i, v_score=0.5) for i in range(3)]
        selected = selector.select(entries)
        assert len(selected) == 3

    def test_overlap_penalizes_selection(self):
        """High overlap should be penalized in selection scoring."""
        selector = TopKSelector(k=2, min_temporal_gap=1, overlap_lambda=0.5)
        entries = [
            _make_entry(0, v_score=0.9, overlap_ratio=0.8),  # score = 0.9 - 0.5*0.8 = 0.5
            _make_entry(1, v_score=0.7, overlap_ratio=0.0),  # score = 0.7 - 0 = 0.7
            _make_entry(2, v_score=0.6, overlap_ratio=0.0),  # score = 0.6 - 0 = 0.6
        ]
        selected = selector.select(entries)
        # Entry 1 (v=0.7, no overlap) should be selected first
        assert selected[0].frame_idx == 1

    def test_is_tracklet_ready_enough_quality(self):
        selector = TopKSelector(min_tracklet_len=5, min_high_quality_frames=3, high_quality_threshold=0.6)
        entries = [_make_entry(i, v_score=0.8) for i in range(10)]
        assert selector.is_tracklet_ready(entries) is True

    def test_is_tracklet_ready_too_short(self):
        selector = TopKSelector(min_tracklet_len=10, min_high_quality_frames=3, high_quality_threshold=0.6)
        entries = [_make_entry(i, v_score=0.8) for i in range(5)]
        assert selector.is_tracklet_ready(entries) is False

    def test_is_tracklet_ready_not_enough_good_frames(self):
        selector = TopKSelector(min_tracklet_len=5, min_high_quality_frames=5, high_quality_threshold=0.6)
        entries = [_make_entry(i, v_score=0.3) for i in range(10)]
        assert selector.is_tracklet_ready(entries) is False
