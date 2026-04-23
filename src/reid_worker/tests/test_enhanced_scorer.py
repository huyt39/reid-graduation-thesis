import numpy as np

from src.scoring.enhanced_visibility import compute_iou_prev, compute_vel_smooth, compute_v_worker


class TestIouPrev:
    def test_no_previous(self):
        assert compute_iou_prev(np.array([0, 0, 100, 100]), None) == 0.5

    def test_high_overlap(self):
        bbox_curr = np.array([10, 10, 110, 110])
        bbox_prev = np.array([10, 10, 110, 110])
        assert compute_iou_prev(bbox_curr, bbox_prev) == 1.0

    def test_low_overlap(self):
        bbox_curr = np.array([0, 0, 100, 100])
        bbox_prev = np.array([200, 200, 300, 300])
        assert compute_iou_prev(bbox_curr, bbox_prev) == 0.2


class TestVelSmooth:
    def test_no_previous(self):
        assert compute_vel_smooth(np.array([50, 50]), None, None, 100) == 0.5

    def test_stable_movement(self):
        score = compute_vel_smooth(
            np.array([51, 51]),
            np.array([50, 50]),
            np.array([49, 49]),
            100,
        )
        assert score > 0.8


class TestVWorker:
    def test_all_good(self):
        assert compute_v_worker(1.0, 1.0, 1.0) == 1.0

    def test_all_bad(self):
        assert compute_v_worker(0.0, 0.0, 0.0) == 0.0
