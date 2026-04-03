import pytest
from src.scoring.visibility import (
    compute_cutoff,
    compute_area_ratio,
    compute_aspect_ratio,
    compute_det_conf_score,
    compute_person_overlap,
    compute_overlap_ratio,
    compute_subscores,
    compute_visibility_score,
)


class TestCutoff:
    def test_no_edges_touched(self):
        bbox = [100, 100, 200, 300]
        assert compute_cutoff(bbox, 1920, 1080) == 1.0

    def test_one_edge_touched(self):
        bbox = [0, 100, 200, 300]
        assert compute_cutoff(bbox, 1920, 1080) == 0.6

    def test_two_edges_touched(self):
        bbox = [0, 0, 200, 300]
        assert compute_cutoff(bbox, 1920, 1080) == 0.3

    def test_three_edges_touched(self):
        bbox = [0, 0, 1920, 300]
        assert compute_cutoff(bbox, 1920, 1080) == 0.1

    def test_all_edges_touched(self):
        bbox = [0, 0, 1920, 1080]
        assert compute_cutoff(bbox, 1920, 1080) == 0.05


class TestAreaRatio:
    def test_ideal_range(self):
        # 100x200 = 20000 in 1920x1080 = 2073600 => ratio ~0.0096 (ideal)
        bbox = [100, 100, 200, 300]
        score = compute_area_ratio(bbox, 1920, 1080)
        assert score == 0.7  # 0.0096 is in [0.005, 0.01)

    def test_tiny_bbox(self):
        bbox = [100, 100, 110, 115]  # 10x15 = 150
        score = compute_area_ratio(bbox, 1920, 1080)
        assert score == 0.1

    def test_large_bbox(self):
        # 500x400 = 200000 / 2073600 ~= 0.096 (ideal range)
        bbox = [100, 100, 600, 500]
        score = compute_area_ratio(bbox, 1920, 1080)
        assert score == 1.0


class TestAspectRatio:
    def test_standing_person(self):
        bbox = [100, 100, 200, 400]  # w=100, h=300, ratio=3.0
        assert compute_aspect_ratio(bbox) == 1.0

    def test_crouching(self):
        bbox = [100, 100, 300, 350]  # w=200, h=250, ratio=1.25
        assert compute_aspect_ratio(bbox) == 0.7

    def test_lying_down(self):
        bbox = [100, 100, 400, 170]  # w=300, h=70, ratio=0.23
        assert compute_aspect_ratio(bbox) == 0.2


class TestDetConf:
    def test_high_confidence(self):
        assert compute_det_conf_score(0.95) > 0.9

    def test_low_confidence(self):
        assert compute_det_conf_score(0.25) == 0.3

    def test_mid_confidence(self):
        score = compute_det_conf_score(0.6)
        assert 0.3 < score < 1.0


class TestPersonOverlap:
    def test_single_detection_no_overlap(self):
        bbox = [100, 100, 200, 300]
        assert compute_person_overlap(bbox, [bbox]) == 1.0

    def test_no_other_bboxes(self):
        bbox = [100, 100, 200, 300]
        assert compute_person_overlap(bbox, []) == 1.0

    def test_no_overlap_between_persons(self):
        bbox = [100, 100, 200, 300]
        other = [500, 500, 600, 700]
        assert compute_person_overlap(bbox, [bbox, other]) == 1.0

    def test_partial_overlap(self):
        bbox = [100, 100, 200, 300]
        other = [150, 100, 250, 300]  # 50px overlap in x, 200px in y
        score = compute_person_overlap(bbox, [bbox, other])
        assert score < 1.0

    def test_heavy_overlap(self):
        bbox = [100, 100, 200, 300]
        other = [110, 110, 210, 310]  # Almost fully overlapping
        score = compute_person_overlap(bbox, [bbox, other])
        assert score <= 0.4

    def test_overlap_ratio_raw(self):
        bbox = [100, 100, 200, 300]
        other = [500, 500, 600, 700]
        assert compute_overlap_ratio(bbox, [bbox, other]) == 0.0

    def test_overlap_ratio_partial(self):
        bbox = [100, 100, 200, 300]
        other = [150, 100, 250, 300]
        ratio = compute_overlap_ratio(bbox, [bbox, other])
        assert 0.0 < ratio < 1.0


class TestCompositeScore:
    def test_perfect_detection(self):
        subscores = {"cut_off": 1.0, "area_ratio": 1.0, "aspect_ratio": 1.0, "det_conf": 1.0, "person_overlap": 1.0}
        assert compute_visibility_score(subscores) == 1.0

    def test_worst_detection(self):
        subscores = {"cut_off": 0.05, "area_ratio": 0.1, "aspect_ratio": 0.2, "det_conf": 0.3, "person_overlap": 0.2}
        score = compute_visibility_score(subscores)
        assert score < 0.2

    def test_mixed_detection(self):
        subscores = {"cut_off": 1.0, "area_ratio": 0.4, "aspect_ratio": 0.7, "det_conf": 0.8, "person_overlap": 0.6}
        score = compute_visibility_score(subscores)
        assert 0.3 < score < 0.9

    def test_person_overlap_has_highest_impact(self):
        """person_overlap has weight 0.30, should have the most impact."""
        base = {"cut_off": 1.0, "area_ratio": 1.0, "aspect_ratio": 1.0, "det_conf": 1.0, "person_overlap": 1.0}
        occluded = {**base, "person_overlap": 0.2}
        score_clean = compute_visibility_score(base)
        score_occluded = compute_visibility_score(occluded)
        assert score_clean - score_occluded > 0.2  # 0.30 * 0.8 = 0.24 impact
