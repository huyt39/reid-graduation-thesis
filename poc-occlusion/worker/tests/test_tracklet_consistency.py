import numpy as np
from src.tracklet.consistency import (
    compute_bbox_size_stability,
    compute_position_stability,
    compute_good_frame_streak,
    compute_good_frame_ratio,
    compute_tracklet_consistency,
)
from src.tracklet.models import TrackletEntry


def _make_entry(frame_idx: int, bbox: list[float], v_score: float = 0.8) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.zeros((64, 32, 3), dtype=np.uint8),
        v_score=v_score,
        bbox_xyxy=bbox,
        timestamp_ns=0,
    )


class TestBboxSizeStability:
    def test_single_entry(self):
        entries = [_make_entry(0, [0, 0, 100, 200])]
        assert compute_bbox_size_stability(entries) == 1.0

    def test_stable_size(self):
        entries = [
            _make_entry(i, [0, 0, 100, 200])
            for i in range(10)
        ]
        assert compute_bbox_size_stability(entries) == 1.0

    def test_unstable_size(self):
        entries = [
            _make_entry(0, [0, 0, 100, 200]),  # area = 20000
            _make_entry(1, [0, 0, 50, 100]),    # area = 5000 (75% change)
        ]
        stab = compute_bbox_size_stability(entries)
        assert stab < 0.5

    def test_gradual_change(self):
        entries = [
            _make_entry(i, [0, 0, 100 + i * 2, 200 + i * 2])
            for i in range(10)
        ]
        stab = compute_bbox_size_stability(entries)
        assert stab > 0.8  # Gradual changes should be fairly stable


class TestPositionStability:
    def test_single_entry(self):
        assert compute_position_stability([_make_entry(0, [0, 0, 100, 200])]) == 1.0

    def test_stationary(self):
        entries = [_make_entry(i, [100, 100, 200, 300]) for i in range(10)]
        assert compute_position_stability(entries) == 1.0

    def test_smooth_movement(self):
        entries = [_make_entry(i, [100 + i * 5, 100, 200 + i * 5, 300]) for i in range(10)]
        stab = compute_position_stability(entries)
        assert stab > 0.8

    def test_erratic_jumps(self):
        entries = [
            _make_entry(0, [100, 100, 200, 300]),
            _make_entry(1, [500, 500, 600, 700]),  # Big jump
        ]
        stab = compute_position_stability(entries)
        assert stab < 0.5


class TestGoodFrameStreak:
    def test_all_good(self):
        entries = [_make_entry(i, [0, 0, 100, 200], v_score=0.9) for i in range(10)]
        assert compute_good_frame_streak(entries, 0.6) == 10

    def test_no_good(self):
        entries = [_make_entry(i, [0, 0, 100, 200], v_score=0.3) for i in range(10)]
        assert compute_good_frame_streak(entries, 0.6) == 0

    def test_mixed(self):
        scores = [0.3, 0.8, 0.9, 0.7, 0.3, 0.8, 0.9, 0.85, 0.7, 0.3]
        entries = [_make_entry(i, [0, 0, 100, 200], v_score=s) for i, s in enumerate(scores)]
        assert compute_good_frame_streak(entries, 0.6) == 4  # indices 5-8

    def test_ratio(self):
        scores = [0.3, 0.8, 0.9, 0.3, 0.3]
        entries = [_make_entry(i, [0, 0, 100, 200], v_score=s) for i, s in enumerate(scores)]
        assert compute_good_frame_ratio(entries, 0.6) == 0.4


class TestTrackletConsistency:
    def test_overall_composite(self):
        entries = [_make_entry(i, [100, 100, 200, 300], v_score=0.9) for i in range(10)]
        c = compute_tracklet_consistency(entries)
        assert c.overall > 0.8
        assert c.bbox_size_stability == 1.0
        assert c.position_stability == 1.0
        assert c.good_frame_streak == 10
        assert c.good_frame_ratio == 1.0
