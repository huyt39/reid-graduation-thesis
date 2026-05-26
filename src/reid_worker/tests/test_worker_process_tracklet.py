import asyncio
from types import SimpleNamespace

import numpy as np

from src.attributes.attribute_voter import AttributeVoter
from src.matching.reid_matcher import PersonIdAllocationError
from src.tracklet.models import Tracklet, TrackletEntry, TrackletState
from src.workers.main import (
    WorkerPipeline,
    _build_attribute_crop,
    _compute_person_snapshot_score,
    _choose_person_snapshot_entry,
    _select_embedding_consensus_indices,
    _tracklet_motion_summary,
)


def _make_entry(frame_idx: int, v_score: float = 0.8, overlap_ratio: float = 0.1) -> TrackletEntry:
    return TrackletEntry(
        frame_idx=frame_idx,
        crop=np.ones((32, 16, 3), dtype=np.uint8),
        v_score=v_score,
        bbox_xyxy=[10.0, 20.0, 50.0, 120.0],
        timestamp_ns=frame_idx * int(1e9),
        overlap_ratio=overlap_ratio,
    )


def _make_pipeline() -> WorkerPipeline:
    pipeline = WorkerPipeline.__new__(WorkerPipeline)
    pipeline.settings = SimpleNamespace(
        recent_person_reuse_enabled=False,
        recent_person_reuse_seconds=2.5,
        recent_person_reuse_min_iou=0.2,
        recent_person_reuse_max_center_distance_ratio=0.75,
        gender_person_threshold=0.7,
        gender_ambiguous_conflict_enabled=True,
        gender_ambiguous_conflict_tracklet_confidence=0.70,
        gender_ambiguous_conflict_max_person_confidence=0.80,
        gender_match_blocking_enabled=False,
        glasses_best_frame_override_threshold=0.6,
        attribute_crop_top_padding_ratio=0.22,
        attribute_crop_side_padding_ratio=0.08,
        attribute_crop_bottom_padding_ratio=0.04,
        cooccurrence_guard_enabled=True,
        cooccurrence_guard_min_shared_frames=1,
        cooccurrence_guard_max_iou=0.15,
        cooccurrence_guard_min_center_distance_ratio=0.55,
        attribute_conflict_guard_enabled=True,
        attribute_conflict_person_confidence=0.88,
        attribute_conflict_tracklet_confidence=0.88,
        attribute_conflict_person_min_support=3,
        pretrack_static_filter_enabled=False,
        pretrack_static_filter_min_frames=4,
        pretrack_static_filter_max_width_px=130.0,
        pretrack_static_filter_max_height_px=260.0,
        pretrack_static_filter_max_center_drift_px=6.0,
        tracklet_identity_shift_guard_enabled=True,
        tracklet_identity_shift_min_entries=12,
        tracklet_identity_shift_min_endpoint_displacement_ratio=0.55,
        tracklet_identity_shift_min_size_ratio=1.60,
        tracklet_identity_shift_min_area_ratio=2.20,
        tracklet_identity_shift_anchor_min_frame_gap=24,
        tracklet_identity_shift_anchor_min_endpoint_displacement_ratio=0.40,
        tracklet_identity_shift_anchor_min_size_ratio=1.45,
        tracklet_identity_shift_anchor_min_area_ratio=1.90,
        static_artifact_filter_enabled=True,
        static_artifact_max_mean_width_px=130.0,
        static_artifact_max_mean_height_px=260.0,
        static_artifact_max_path_displacement_ratio=0.05,
        static_artifact_max_endpoint_displacement_ratio=0.02,
        static_artifact_min_bbox_stability=0.97,
        static_artifact_min_position_stability=0.97,
        static_artifact_min_entries=6,
        occlusion_provisional_match_enabled=True,
        occlusion_provisional_match_threshold=0.60,
        occlusion_provisional_min_margin=0.03,
        occlusion_provisional_short_reentry_enabled=True,
        occlusion_provisional_reentry_min_similarity=0.58,
        occlusion_provisional_reentry_max_entries=8,
        occlusion_provisional_reentry_max_gap_frames=180,
        occlusion_provisional_reentry_max_center_distance_ratio=2.0,
        recent_match_guard_enabled=True,
        recent_match_guard_seconds=4.0,
        recent_match_guard_min_iou=0.1,
        recent_match_guard_max_center_distance_ratio=0.9,
        tracklet_idle_flush_enabled=True,
        tracklet_idle_flush_seconds=1.5,
        tracklet_min_entries=4,
        max_person_identities=0,
        duplicate_merge_enabled=False,
        duplicate_merge_min_score=0.54,
        duplicate_merge_weak_max_tracklets=2,
        duplicate_merge_singleton_min_score=0.49,
        duplicate_merge_singleton_min_target_tracklets=3,
        duplicate_merge_min_margin=0.08,
        duplicate_merge_temporal_continuity_enabled=False,
        duplicate_merge_temporal_continuity_min_score=0.85,
        duplicate_merge_temporal_continuity_max_gap_frames=15,
        duplicate_merge_adjacent_fragment_enabled=False,
        duplicate_merge_adjacent_fragment_min_score=0.70,
        duplicate_merge_adjacent_fragment_max_gap_frames=3,
        duplicate_merge_occlusion_reentry_enabled=False,
        duplicate_merge_occlusion_reentry_min_score=0.58,
        duplicate_merge_occlusion_reentry_max_gap_frames=180,
        duplicate_merge_occlusion_reentry_max_center_distance_ratio=2.0,
        duplicate_merge_same_gender_singleton_enabled=False,
        duplicate_merge_same_gender_singleton_min_score=0.80,
        duplicate_merge_same_gender_singleton_gender_confidence=0.80,
        duplicate_merge_soft_split_override_threshold=0.75,
        duplicate_merge_soft_split_max_weak_tracklets=4,
        duplicate_merge_soft_split_max_center_distance_ratio=0.35,
        duplicate_merge_soft_split_duplicate_iou_threshold=0.45,
        duplicate_merge_soft_split_duplicate_box_multitrack_min_score=0.58,
        duplicate_merge_soft_split_spatial_only_min_score=0.50,
        duplicate_merge_soft_split_spatial_only_multitrack_min_score=0.60,
        duplicate_merge_soft_split_spatial_only_max_center_distance_ratio=0.30,
        duplicate_merge_overlap_spatial_duplicate_enabled=True,
        duplicate_merge_overlap_spatial_duplicate_min_score=0.58,
        duplicate_merge_overlap_spatial_duplicate_max_gap_frames=4,
        duplicate_merge_overlap_spatial_duplicate_max_tracklets=24,
        duplicate_merge_overlap_spatial_duplicate_max_center_distance_ratio=0.08,
        duplicate_merge_overlap_spatial_duplicate_max_size_ratio=1.25,
        duplicate_merge_overlap_spatial_duplicate_max_area_ratio=1.60,
        duplicate_merge_trajectory_reentry_enabled=True,
        duplicate_merge_trajectory_reentry_min_score=0.60,
        duplicate_merge_trajectory_reentry_max_gap_frames=240,
        duplicate_merge_trajectory_reentry_max_tracklets=24,
        duplicate_merge_trajectory_reentry_max_center_distance_ratio=0.06,
        duplicate_merge_trajectory_reentry_max_size_ratio=1.30,
        duplicate_merge_trajectory_reentry_max_area_ratio=2.00,
        duplicate_merge_singleton_spatial_continuation_min_score=0.30,
        duplicate_merge_singleton_spatial_continuation_max_gap_frames=15,
        duplicate_merge_singleton_spatial_continuation_max_center_distance_ratio=0.30,
        duplicate_merge_singleton_spatial_continuation_max_size_ratio=1.80,
        duplicate_merge_singleton_spatial_continuation_max_area_ratio=2.20,
        duplicate_merge_adjacent_tight_continuation_enabled=True,
        duplicate_merge_adjacent_tight_continuation_min_score=0.50,
        duplicate_merge_adjacent_tight_continuation_max_gap_frames=4,
        duplicate_merge_adjacent_tight_continuation_max_tracklets=8,
        duplicate_merge_adjacent_tight_continuation_max_center_distance_ratio=0.06,
        duplicate_merge_adjacent_tight_continuation_max_size_ratio=1.10,
        duplicate_merge_adjacent_tight_continuation_max_area_ratio=1.15,
        duplicate_merge_boundary_weak_continuation_enabled=True,
        duplicate_merge_boundary_weak_continuation_min_score=0.50,
        duplicate_merge_boundary_weak_continuation_max_gap_frames=12,
        duplicate_merge_boundary_weak_continuation_max_weak_tracklets=2,
        duplicate_merge_boundary_weak_continuation_max_supported_tracklets=8,
        duplicate_merge_boundary_weak_continuation_min_center_distance_ratio=0.10,
        duplicate_merge_boundary_weak_continuation_max_center_distance_ratio=0.32,
        duplicate_merge_boundary_weak_continuation_max_size_ratio=1.80,
        duplicate_merge_boundary_weak_continuation_max_area_ratio=2.30,
        duplicate_merge_boundary_weak_continuation_max_bottom_delta_ratio=0.06,
        duplicate_merge_boundary_duplicate_min_score=0.68,
        duplicate_merge_boundary_duplicate_min_iou=0.10,
        duplicate_merge_boundary_duplicate_max_center_distance_ratio=0.45,
        duplicate_merge_ultra_continuity_min_score=0.50,
        duplicate_merge_ultra_continuity_max_gap_frames=6,
        duplicate_merge_ultra_continuity_max_center_distance_ratio=0.12,
        duplicate_merge_ultra_continuity_max_weak_tracklets=2,
        duplicate_merge_ultra_continuity_max_supported_tracklets=8,
        duplicate_merge_reentry_bridge_enabled=True,
        duplicate_merge_reentry_bridge_min_score=0.535,
        duplicate_merge_reentry_bridge_max_tracklets=4,
        duplicate_merge_reentry_bridge_max_supported_tracklets=8,
        duplicate_merge_reentry_bridge_min_gap_frames=30,
        duplicate_merge_reentry_bridge_max_gap_frames=180,
        duplicate_merge_reentry_bridge_max_center_distance_ratio=0.85,
        duplicate_merge_reentry_bridge_gender_confidence=0.70,
        duplicate_merge_reentry_bridge_min_attr_matches=2,
        duplicate_merge_reentry_bridge_supported_min_score=0.70,
        duplicate_merge_reentry_bridge_supported_min_margin=0.12,
        duplicate_merge_supported_spatial_reentry_enabled=True,
        duplicate_merge_supported_spatial_reentry_min_score=0.53,
        duplicate_merge_supported_spatial_reentry_max_tracklets=8,
        duplicate_merge_supported_spatial_reentry_min_gap_frames=24,
        duplicate_merge_supported_spatial_reentry_max_gap_frames=90,
        duplicate_merge_supported_spatial_reentry_max_center_distance_ratio=0.18,
        duplicate_merge_supported_spatial_reentry_max_size_ratio=1.20,
        duplicate_merge_supported_spatial_reentry_max_area_ratio=1.80,
        duplicate_merge_clothing_reentry_enabled=False,
        duplicate_merge_clothing_reentry_min_score=0.515,
        duplicate_merge_clothing_reentry_max_weak_tracklets=2,
        duplicate_merge_clothing_reentry_max_supported_tracklets=8,
        duplicate_merge_clothing_reentry_min_gap_frames=30,
        duplicate_merge_clothing_reentry_max_gap_frames=240,
        duplicate_merge_clothing_reentry_min_attr_matches=3,
        duplicate_merge_clothing_reentry_attr_confidence=0.70,
        duplicate_merge_weak_to_supported_guard_enabled=True,
        duplicate_merge_weak_to_supported_min_target_tracklets=5,
        duplicate_merge_weak_to_supported_max_target_tracklets=8,
        duplicate_merge_weak_to_supported_min_score=0.78,
        duplicate_merge_weak_to_supported_min_margin=0.08,
        duplicate_merge_weak_to_supported_strong_score=0.89,
        duplicate_merge_weak_to_supported_strong_margin=0.18,
        duplicate_merge_occlusion_spatial_rejoin_enabled=False,
        duplicate_merge_occlusion_spatial_rejoin_min_score=0.53,
        duplicate_merge_occlusion_spatial_rejoin_strong_min_score=0.59,
        duplicate_merge_occlusion_spatial_rejoin_max_gap_frames=180,
        duplicate_merge_occlusion_spatial_rejoin_max_center_distance_ratio=0.50,
        duplicate_merge_occlusion_spatial_rejoin_tight_center_distance_ratio=0.42,
        duplicate_merge_occlusion_spatial_rejoin_max_size_ratio=1.55,
        duplicate_merge_occlusion_spatial_rejoin_max_area_ratio=2.10,
        duplicate_merge_occlusion_spatial_rejoin_tight_size_ratio=1.10,
        duplicate_merge_occlusion_spatial_rejoin_tight_area_ratio=2.00,
        duplicate_merge_spatial_continuation_enabled=False,
        duplicate_merge_spatial_continuation_min_score=0.20,
        duplicate_merge_spatial_continuation_max_gap_frames=60,
        duplicate_merge_spatial_continuation_max_center_distance_ratio=0.30,
        fragment_recovery_enabled=True,
        fragment_recovery_min_fragments=2,
        fragment_recovery_min_total_entries=5,
        fragment_recovery_min_visibility=0.72,
        fragment_recovery_min_similarity=0.62,
        fragment_recovery_max_gap_frames=180,
        fragment_recovery_max_center_distance_ratio=1.8,
        fragment_recovery_near_gallery_threshold=0.52,
    )
    pipeline.track_id_to_person_id = {}
    pipeline.track_metadata = {}
    pipeline.track_last_seen_ns = {}
    pipeline.person_last_observation = {}
    pipeline.person_confirmed_gender = {}
    pipeline.prev_bboxes = {}
    pipeline.track_forbidden_person_ids = {}
    pipeline.track_cooccurrence_counts = {}
    pipeline.fragment_recovery_clusters = []
    pipeline.processing_tracklet_ids = set()
    pipeline._current_device_id = "cam-1"
    pipeline.last_message_time_ns = 0
    pipeline.last_idle_flush_ns = 0
    pipeline.attribute_voter = AttributeVoter(person_threshold=0.7)
    pipeline.processed_messages = 0
    pipeline.ready_tracklets = 0
    pipeline.embedded_tracklets = 0
    pipeline.matched_tracklets = 0
    pipeline.worker_started_at = 0.0
    return pipeline


def test_duplicate_merge_collapses_weak_non_cooccurring_identity():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.track_id_to_person_id = {10: 1, 20: 6}
    pipeline.person_last_observation = {
        1: {"timestamp_ns": 100, "bbox_xyxy": [0, 0, 20, 60]},
        6: {"timestamp_ns": 200, "bbox_xyxy": [1, 0, 21, 60]},
    }
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 6
            assert min_score == 0.49
            return 1, 0.57, 0.42

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 4, 6: 1}[person_id]

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_strong_attribute_conflict(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def persons_have_strong_gender_conflict(self, source_person_id, target_person_id):
            return False

        async def persons_have_moderate_attribute_conflict(self, source_person_id, target_person_id):
            return False

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(6))

    assert result == 1
    assert calls["gallery_merge"] == (6, 1)
    assert calls["mongo_merge"][0:2] == (6, 1)
    assert sorted(calls["invalidated"]) == [1, 6]
    assert pipeline.track_id_to_person_id == {10: 1, 20: 1}
    assert 6 not in pipeline.person_last_observation


def test_duplicate_merge_blocks_cooccurring_identity():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    calls = {"gallery_merge": 0, "mongo_merge": 0}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            return 1, 0.62, 0.41

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] += 1

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 4, 6: 1}[person_id]

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_strong_attribute_conflict(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def persons_have_strong_gender_conflict(self, source_person_id, target_person_id):
            return False

        async def persons_have_moderate_attribute_conflict(self, source_person_id, target_person_id):
            return False

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] += 1

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(6))

    assert result == 6
    assert calls == {"gallery_merge": 0, "mongo_merge": 0}


def test_duplicate_merge_rejects_supported_singleton_with_low_margin():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.535
    pipeline.settings.duplicate_merge_singleton_min_score = 0.52
    pipeline.settings.duplicate_merge_singleton_min_target_tracklets = 3
    pipeline.settings.duplicate_merge_min_margin = 0.10
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 7
            assert min_score == 0.52
            return 4, 0.525, 0.50

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {4: 5, 7: 1}[person_id]

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_strong_attribute_conflict(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def persons_have_strong_gender_conflict(self, source_person_id, target_person_id):
            return False

        async def persons_have_moderate_attribute_conflict(self, source_person_id, target_person_id):
            return False

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(7))

    assert result == 7
    assert calls == {}


def test_duplicate_merge_allows_short_temporal_continuity_split():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_temporal_continuity_enabled = True
    pipeline.settings.duplicate_merge_temporal_continuity_min_score = 0.85
    pipeline.settings.duplicate_merge_temporal_continuity_max_gap_frames = 15
    pipeline.track_id_to_person_id = {13: 4, -9000006: 6}
    pipeline.person_last_observation = {
        4: {"timestamp_ns": 42, "bbox_xyxy": [0, 0, 20, 80]},
        6: {"timestamp_ns": 50, "bbox_xyxy": [2, 0, 22, 80]},
    }
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 6
            assert min_score == 0.75
            return 4, 0.8589, 0.70

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {4: 2, 6: 4}[person_id]

        async def persons_min_frame_gap(self, person_a, person_b):
            return 8

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(6))

    assert result == 6
    assert calls["gallery_merge"] == (4, 6)
    assert calls["mongo_merge"][0:2] == (4, 6)
    assert calls["mongo_merge"][2]["method"] == "temporal_continuity_gallery_merge"
    assert sorted(calls["invalidated"]) == [4, 6]
    assert pipeline.track_id_to_person_id == {13: 6, -9000006: 6}


def test_duplicate_merge_allows_occlusion_reentry_split_with_weak_similarity():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_occlusion_reentry_enabled = True
    pipeline.settings.duplicate_merge_occlusion_reentry_min_score = 0.58
    pipeline.settings.duplicate_merge_occlusion_reentry_max_gap_frames = 180
    pipeline.track_id_to_person_id = {-9000038: 9, -9000043: 12}
    pipeline.person_last_observation = {
        9: {"timestamp_ns": 986, "frame_idx": 986, "bbox_xyxy": [100, 50, 150, 180]},
        12: {"timestamp_ns": 1154, "frame_idx": 1154, "bbox_xyxy": [112, 52, 162, 182]},
    }
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 12
            assert min_score == 0.58
            return 9, 0.5844, 0.41

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {9: 1, 12: 1}[person_id]

        async def persons_min_frame_gap(self, person_a, person_b):
            return 168

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(12))

    assert result == 9
    assert calls["gallery_merge"] == (12, 9)
    assert calls["mongo_merge"][0:2] == (12, 9)
    assert calls["mongo_merge"][2]["method"] == "occlusion_reentry_gallery_merge"
    assert pipeline.track_id_to_person_id == {-9000038: 9, -9000043: 9}


def test_duplicate_merge_occlusion_reentry_uses_persisted_tracklet_geometry():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_occlusion_reentry_enabled = True
    pipeline.settings.duplicate_merge_occlusion_reentry_min_score = 0.58
    pipeline.settings.duplicate_merge_occlusion_reentry_max_gap_frames = 180
    pipeline.person_last_observation = {}
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 12
            return 9, 0.5844, 0.41

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {9: 1, 12: 1}[person_id]

        async def persons_min_frame_gap(self, person_a, person_b):
            return 168

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 168,
                "bbox_a": [100.0, 50.0, 150.0, 180.0],
                "bbox_b": [112.0, 52.0, 162.0, 182.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(12))

    assert result == 9
    assert calls["mongo_merge"][2]["method"] == "occlusion_reentry_gallery_merge"


def test_duplicate_merge_retries_after_guard_blocked_top_candidate():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_same_gender_singleton_enabled = True
    pipeline.settings.duplicate_merge_same_gender_singleton_min_score = 0.80
    pipeline.settings.duplicate_merge_occlusion_reentry_enabled = True
    pipeline.settings.duplicate_merge_occlusion_reentry_min_score = 0.58
    pipeline.settings.duplicate_merge_occlusion_reentry_max_gap_frames = 180
    pipeline.person_last_observation = {}
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            excluded = set(exclude_person_ids or set())
            if 3 not in excluded:
                return 3, 0.8929, 0.5844
            return 9, 0.5844, 0.41

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {3: 19, 9: 1, 12: 1}[person_id]

        async def persons_min_frame_gap(self, person_a, person_b):
            return 168

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 168,
                "bbox_a": [100.0, 50.0, 150.0, 180.0],
                "bbox_b": [112.0, 52.0, 162.0, 182.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            attrs = {
                3: {"gender": "female", "gender_confidence": 0.83},
                9: {"gender": "male", "gender_confidence": 0.74},
                12: {"gender": "male", "gender_confidence": 0.74},
            }
            return attrs.get(source_person_id, {}), attrs.get(target_person_id, {})

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(12))

    assert result == 9
    assert calls["gallery_merge"] == (12, 9)
    assert calls["mongo_merge"][2]["method"] == "occlusion_reentry_gallery_merge"


def test_final_reconciler_uses_mongo_persons_not_only_live_observations():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.person_last_observation = {}
    calls = []

    class DummyMongo:
        async def list_recent_person_ids(self, limit=50):
            return [12, 9]

    async def fake_merge(person_id):
        calls.append(person_id)
        return 9 if person_id == 12 else person_id

    pipeline.mongo = DummyMongo()
    pipeline._maybe_merge_duplicate_person = fake_merge

    asyncio.run(pipeline._reconcile_recent_persons(max_persons=50, passes=1, reason="end_of_stream"))

    assert calls == [12, 9]


def test_duplicate_merge_blocks_temporal_continuity_when_frame_gap_is_large():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_temporal_continuity_enabled = True
    calls = {"gallery_merge": 0, "mongo_merge": 0}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            return 4, 0.8589, 0.70

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] += 1

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {4: 2, 6: 4}[person_id]

        async def persons_min_frame_gap(self, person_a, person_b):
            return 60

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] += 1

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(6))

    assert result == 6
    assert calls == {"gallery_merge": 0, "mongo_merge": 0}


def test_duplicate_merge_allows_adjacent_singleton_fragment():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_adjacent_fragment_enabled = True
    pipeline.settings.duplicate_merge_adjacent_fragment_min_score = 0.70
    pipeline.settings.duplicate_merge_adjacent_fragment_max_gap_frames = 3
    pipeline.track_id_to_person_id = {-9000016: 6, 95: 7}
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 7
            assert min_score == 0.70
            return 6, 0.7255, 0.68

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {6: 4, 7: 1}[person_id]

        async def persons_min_frame_gap(self, person_a, person_b):
            return 2

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {"gender": "female", "gender_confidence": 0.65}, {"gender": "male", "gender_confidence": 0.96}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(7))

    assert result == 6
    assert calls["gallery_merge"] == (7, 6)
    assert calls["mongo_merge"][0:2] == (7, 6)
    assert calls["mongo_merge"][2]["method"] == "adjacent_fragment_gallery_merge"
    assert pipeline.track_id_to_person_id == {-9000016: 6, 95: 6}


def test_duplicate_merge_blocks_adjacent_fragment_when_similarity_is_low():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_adjacent_fragment_enabled = True
    calls = {"gallery_merge": 0, "mongo_merge": 0}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            return 6, 0.69, 0.60

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] += 1

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {6: 4, 7: 1}[person_id]

        async def persons_min_frame_gap(self, person_a, person_b):
            return 2

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] += 1

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(7))

    assert result == 7
    assert calls == {"gallery_merge": 0, "mongo_merge": 0}


def test_duplicate_merge_allows_same_gender_supported_singleton_with_clear_margin():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_same_gender_singleton_enabled = True
    pipeline.settings.duplicate_merge_same_gender_singleton_min_score = 0.80
    pipeline.settings.duplicate_merge_same_gender_singleton_gender_confidence = 0.80
    pipeline.track_id_to_person_id = {130: 5, 128: 8}
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 8
            assert min_score == 0.75
            return 5, 0.8129, 0.58

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {5: 8, 8: 1}[person_id]

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.905},
                {"gender": "male", "gender_confidence": 0.863},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(8))

    assert result == 5
    assert calls["gallery_merge"] == (8, 5)
    assert calls["mongo_merge"][0:2] == (8, 5)
    assert calls["mongo_merge"][2]["method"] == "same_gender_singleton_gallery_merge"
    assert pipeline.track_id_to_person_id == {130: 5, 128: 5}


def test_duplicate_merge_blocks_same_gender_singleton_without_confident_gender():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_same_gender_singleton_enabled = True
    calls = {"gallery_merge": 0, "mongo_merge": 0}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            return 5, 0.8129, 0.58

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] += 1

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {5: 8, 8: 1}[person_id]

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.79},
                {"gender": "male", "gender_confidence": 0.91},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] += 1

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(8))

    assert result == 8
    assert calls == {"gallery_merge": 0, "mongo_merge": 0}


def test_duplicate_merge_allows_high_similarity_singleton_with_missing_attributes():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_same_gender_singleton_enabled = True
    pipeline.settings.duplicate_merge_same_gender_singleton_min_score = 0.80
    pipeline.settings.duplicate_merge_singleton_unknown_attr_min_score = 0.88
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 11
            assert min_score == 0.75
            return 2, 0.8926, 0.70

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 12, 11: 1}[person_id]

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {},
                {"gender": "female", "gender_confidence": 0.83},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(11))

    assert result == 2
    assert calls["gallery_merge"] == (11, 2)
    assert calls["mongo_merge"][0:2] == (11, 2)
    assert calls["mongo_merge"][2]["method"] == "same_gender_singleton_gallery_merge"


def test_duplicate_merge_allows_cooccurring_soft_split_when_bboxes_are_close():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.75
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 4
    pipeline.settings.duplicate_merge_soft_split_max_center_distance_ratio = 0.35
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 7
            assert min_score == 0.75
            return 1, 0.761, 0.61

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 11, 7: 1}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [674.0, 531.0, 976.0, 1080.0],
                "bbox_b": [724.0, 576.0, 1003.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(7))

    assert result == 1
    assert calls["gallery_merge"] == (7, 1)
    assert calls["mongo_merge"][0:2] == (7, 1)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"
    assert calls["mongo_merge"][2]["soft_split_reason"] == "duplicate_box"


def test_duplicate_merge_rejects_accumulated_cooccurring_soft_split_below_established_threshold():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_established_min_score = 0.78
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    calls = {}

    class DummyQdrant:
        def find_duplicate_candidate(self, person_id, min_score, exclude_person_ids=None):
            assert person_id == 2
            assert min_score == 0.70
            return 3, 0.7187, 0.51

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 6, 3: 11}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [744.0, 264.0, 861.0, 579.0],
                "bbox_b": [748.0, 250.0, 868.0, 570.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    result = asyncio.run(pipeline._maybe_merge_duplicate_person(2))

    assert result == 2
    assert calls == {}


def test_spatial_reconciler_rejects_low_similarity_multitrack_duplicate_box_split():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (1, 6)
            return 0.4018

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 9, 6: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [724.0, 576.0, 1003.0, 1080.0],
                "bbox_b": [674.0, 531.0, 976.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 6]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_short_weak_reentry_into_supported_person():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (5, 14)
            return 0.61

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {5: 9, 14: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 8,
                "bbox_a": [1221.0, 489.0, 1476.0, 1082.0],
                "bbox_b": [1244.0, 540.0, 1658.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "female", "gender_confidence": 0.82},
                {"gender": "female", "gender_confidence": 0.92},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([5, 14]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_low_score_two_tracklet_spatial_fragment():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (8, 15)
            return 0.5298

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {8: 6, 15: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 8,
                "bbox_a": [1048.0, 438.0, 1266.0, 1079.0],
                "bbox_b": [1054.0, 446.0, 1273.0, 1079.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([8, 15]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_allows_short_boundary_weak_continuation():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_boundary_weak_continuation_enabled = True
    pipeline.settings.duplicate_merge_boundary_weak_continuation_min_score = 0.52
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_gap_frames = 12
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_weak_tracklets = 2
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_boundary_weak_continuation_min_center_distance_ratio = 0.10
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_center_distance_ratio = 0.32
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_bottom_delta_ratio = 0.03
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (10, 16)
            return 0.530

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {10: 6, 16: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 8,
                "bbox_a": [1221.76, 489.21, 1476.33, 1082.32],
                "bbox_b": [1244.60, 540.45, 1658.50, 1080.00],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.7549,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8508,
                    "hat": "no_hat",
                    "hat_confidence": 0.8527,
                    "sleeve": "long_sleeve",
                    "sleeve_confidence": 0.9895,
                },
                {
                    "gender": "female",
                    "gender_confidence": 0.9237,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8849,
                    "hat": "no_hat",
                    "hat_confidence": 0.83,
                    "sleeve": "long_sleeve",
                    "sleeve_confidence": 0.9532,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([10, 16]))

    assert merged == 1
    assert calls["gallery_merge"] == (16, 10)
    assert calls["mongo_merge"][0:2] == (16, 10)
    assert calls["mongo_merge"][2]["soft_split_reason"] == "boundary_weak_continuation"


def test_spatial_reconciler_allows_adjacent_tight_continuation_for_split_track():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_adjacent_tight_continuation_enabled = True
    pipeline.settings.duplicate_merge_adjacent_tight_continuation_min_score = 0.50
    pipeline.settings.duplicate_merge_adjacent_tight_continuation_max_gap_frames = 4
    pipeline.settings.duplicate_merge_adjacent_tight_continuation_max_tracklets = 8
    pipeline.settings.duplicate_merge_adjacent_tight_continuation_max_center_distance_ratio = 0.06
    pipeline.settings.duplicate_merge_adjacent_tight_continuation_max_size_ratio = 1.10
    pipeline.settings.duplicate_merge_adjacent_tight_continuation_max_area_ratio = 1.15
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (5, 8)
            return 0.518

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {5: 5, 8: 6}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [743.31, 489.31, 928.31, 894.72],
                "bbox_b": [746.75, 490.15, 937.08, 907.17],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {},
                {
                    "gender": "male",
                    "gender_confidence": 0.941,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8557,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.7625,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([5, 8]))

    assert merged == 1
    assert calls["gallery_merge"] == (5, 8)
    assert calls["mongo_merge"][0:2] == (5, 8)
    assert calls["mongo_merge"][2]["soft_split_reason"] == "adjacent_tight_continuation"


def test_spatial_reconciler_allows_boundary_weak_continuation_with_low_score():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_boundary_weak_continuation_enabled = True
    pipeline.settings.duplicate_merge_boundary_weak_continuation_min_score = 0.50
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_gap_frames = 12
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_weak_tracklets = 2
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_boundary_weak_continuation_min_center_distance_ratio = 0.10
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_center_distance_ratio = 0.32
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_bottom_delta_ratio = 0.06
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (6, 9)
            return 0.506

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {6: 7, 9: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 12,
                "bbox_a": [677.85, 568.12, 865.23, 1056.33],
                "bbox_b": [599.16, 599.68, 795.69, 1080.45],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.88,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8463,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.923,
                },
                {},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([6, 9]))

    assert merged == 1
    assert calls["gallery_merge"] == (9, 6)
    assert calls["mongo_merge"][0:2] == (9, 6)
    assert calls["mongo_merge"][2]["soft_split_reason"] == "boundary_weak_continuation"


def test_spatial_reconciler_allows_supported_boundary_continuation():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_boundary_weak_continuation_enabled = True
    pipeline.settings.duplicate_merge_boundary_weak_continuation_min_score = 0.50
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_gap_frames = 12
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_weak_tracklets = 2
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_boundary_weak_continuation_min_center_distance_ratio = 0.10
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_center_distance_ratio = 0.32
    pipeline.settings.duplicate_merge_boundary_weak_continuation_max_bottom_delta_ratio = 0.06
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (7, 11)
            return 0.510

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {7: 6, 11: 6}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 6,
                "bbox_a": [1222.52, 509.95, 1319.18, 748.38],
                "bbox_b": [1202.94, 508.66, 1280.93, 763.57],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.9925,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.9034,
                    "hat": "no_hat",
                    "hat_confidence": 0.8114,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9754,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.9506,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8761,
                    "hat": "no_hat",
                    "hat_confidence": 0.8126,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.8499,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([7, 11]))

    assert merged == 1
    assert calls["gallery_merge"] == (11, 7)
    assert calls["mongo_merge"][0:2] == (11, 7)
    assert calls["mongo_merge"][2]["soft_split_reason"] == "boundary_weak_continuation"


def test_spatial_reconciler_allows_supported_spatial_reentry():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_reentry_bridge_enabled = True
    pipeline.settings.duplicate_merge_reentry_bridge_min_score = 0.535
    pipeline.settings.duplicate_merge_reentry_bridge_max_tracklets = 4
    pipeline.settings.duplicate_merge_reentry_bridge_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_reentry_bridge_supported_min_score = 0.70
    pipeline.settings.duplicate_merge_reentry_bridge_supported_min_margin = 0.12
    pipeline.settings.duplicate_merge_supported_spatial_reentry_enabled = True
    pipeline.settings.duplicate_merge_supported_spatial_reentry_min_score = 0.53
    pipeline.settings.duplicate_merge_supported_spatial_reentry_max_tracklets = 8
    pipeline.settings.duplicate_merge_supported_spatial_reentry_min_gap_frames = 24
    pipeline.settings.duplicate_merge_supported_spatial_reentry_max_gap_frames = 90
    pipeline.settings.duplicate_merge_supported_spatial_reentry_max_center_distance_ratio = 0.18
    pipeline.settings.duplicate_merge_supported_spatial_reentry_max_size_ratio = 1.20
    pipeline.settings.duplicate_merge_supported_spatial_reentry_max_area_ratio = 1.80
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (7, 11)
            return 0.536

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {7: 7, 11: 3}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 30,
                "bbox_a": [1063.51, 522.75, 1191.79, 823.24],
                "bbox_b": [1112.59, 504.67, 1200.79, 767.46],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.9924,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8749,
                    "hat": "no_hat",
                    "hat_confidence": 0.8058,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9716,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.9429,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8735,
                    "hat": "no_hat",
                    "hat_confidence": 0.8091,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.7782,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([7, 11]))

    assert merged == 1
    assert calls["gallery_merge"] == (11, 7)
    assert calls["mongo_merge"][0:2] == (11, 7)
    assert calls["mongo_merge"][2]["soft_split_reason"] == "supported_spatial_reentry"


def test_spatial_reconciler_rejects_clothing_supported_long_reentry_as_hard_merge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_clothing_reentry_enabled = True
    pipeline.settings.duplicate_merge_clothing_reentry_min_score = 0.515
    pipeline.settings.duplicate_merge_clothing_reentry_max_weak_tracklets = 2
    pipeline.settings.duplicate_merge_clothing_reentry_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_clothing_reentry_min_gap_frames = 30
    pipeline.settings.duplicate_merge_clothing_reentry_max_gap_frames = 240
    pipeline.settings.duplicate_merge_clothing_reentry_min_attr_matches = 3
    pipeline.settings.duplicate_merge_clothing_reentry_attr_confidence = 0.70
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (2, 9)
            return 0.571

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 7, 9: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 84,
                "bbox_a": [760.89, 230.26, 865.16, 519.14],
                "bbox_b": [777.76, 327.98, 910.88, 724.14],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.7681,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8186,
                    "hat": "no_hat",
                    "hat_confidence": 0.7986,
                    "lower": "trousers",
                    "lower_confidence": 0.749,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9102,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.7869,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.7575,
                    "hat": "no_hat",
                    "hat_confidence": 0.8316,
                    "lower": "trousers",
                    "lower_confidence": 0.7742,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.6804,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([2, 9]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_clothing_reentry_without_weak_side():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_clothing_reentry_enabled = True
    pipeline.settings.duplicate_merge_clothing_reentry_min_score = 0.515
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (1, 7)
            return 0.570

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 9, 7: 7}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 84,
                "bbox_a": [760.89, 230.26, 865.16, 519.14],
                "bbox_b": [777.76, 327.98, 910.88, 724.14],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.90},
                {"gender": "male", "gender_confidence": 0.90},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 7]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_singleton_clothing_reentry_as_hard_merge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_clothing_reentry_enabled = True
    pipeline.settings.duplicate_merge_clothing_reentry_min_score = 0.515
    pipeline.settings.duplicate_merge_clothing_reentry_max_weak_tracklets = 2
    pipeline.settings.duplicate_merge_clothing_reentry_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_clothing_reentry_min_gap_frames = 30
    pipeline.settings.duplicate_merge_clothing_reentry_max_gap_frames = 240
    pipeline.settings.duplicate_merge_clothing_reentry_min_attr_matches = 3
    pipeline.settings.duplicate_merge_clothing_reentry_attr_confidence = 0.70
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (2, 10)
            return 0.517

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 7, 10: 1}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 202,
                "bbox_a": [760.89, 230.26, 865.16, 519.14],
                "bbox_b": [1262.05, 688.23, 1666.23, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.7681,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8186,
                    "hat": "no_hat",
                    "hat_confidence": 0.7986,
                    "lower": "trousers",
                    "lower_confidence": 0.749,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9102,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.8238,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8994,
                    "hat": "no_hat",
                    "hat_confidence": 0.7776,
                    "lower": "skirt_dress",
                    "lower_confidence": 0.8661,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.7107,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([2, 10]))

    assert merged == 0
    assert calls == {}


def test_ambiguous_gender_conflict_masks_low_confidence_person_gender():
    pipeline = _make_pipeline()

    out = pipeline._mask_ambiguous_gender_conflict(
        {"gender": ("male", 0.7875), "sleeve": ("long_sleeve", 0.913)},
        {"gender": ("female", 0.7434)},
    )

    assert out["gender"] == ("unknown", 0.0)
    assert out["sleeve"] == ("long_sleeve", 0.913)


def test_ambiguous_gender_conflict_keeps_strong_person_gender():
    pipeline = _make_pipeline()

    out = pipeline._mask_ambiguous_gender_conflict(
        {"gender": ("male", 0.91)},
        {"gender": ("female", 0.7434)},
    )

    assert out["gender"] == ("male", 0.91)


def test_spatial_reconciler_rejects_accumulated_multitrack_spatial_only_pair():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (5, 9)
            return 0.5785

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {5: 13, 9: 5}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 8,
                "bbox_a": [1221.0, 489.0, 1476.0, 1082.0],
                "bbox_b": [1244.0, 540.0, 1658.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "female", "gender_confidence": 0.81},
                {"gender": "female", "gender_confidence": 0.92},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([5, 9]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_low_score_multitrack_fragment():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (1, 11)
            return 0.5243

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 12, 11: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 8,
                "bbox_a": [721.0, 175.0, 812.0, 425.0],
                "bbox_b": [730.0, 180.0, 823.0, 440.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 11]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_ultra_tight_multitrack_continuity():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_ultra_continuity_min_score = 0.50
    pipeline.settings.duplicate_merge_ultra_continuity_max_gap_frames = 6
    pipeline.settings.duplicate_merge_ultra_continuity_max_center_distance_ratio = 0.12
    pipeline.settings.duplicate_merge_ultra_continuity_max_weak_tracklets = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (3, 6)
            return 0.5393

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {3: 5, 6: 7}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [605.0, 590.0, 804.0, 1080.0],
                "bbox_b": [599.0, 599.0, 795.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "female", "gender_confidence": 0.93},
                {"gender": "female", "gender_confidence": 0.83},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([3, 6]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_allows_ultra_tight_weak_continuity():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_ultra_continuity_min_score = 0.50
    pipeline.settings.duplicate_merge_ultra_continuity_max_gap_frames = 6
    pipeline.settings.duplicate_merge_ultra_continuity_max_center_distance_ratio = 0.12
    pipeline.settings.duplicate_merge_ultra_continuity_max_weak_tracklets = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (3, 6)
            return 0.5393

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {3: 2, 6: 7}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [605.0, 590.0, 804.0, 1080.0],
                "bbox_b": [599.0, 599.0, 795.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "female", "gender_confidence": 0.93},
                {"gender": "female", "gender_confidence": 0.83},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([3, 6]))

    assert merged == 1
    assert calls["gallery_merge"] == (3, 6)
    assert calls["mongo_merge"][0:2] == (3, 6)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"


def test_spatial_reconciler_rejects_ultra_continuity_into_over_supported_identity():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_ultra_continuity_min_score = 0.50
    pipeline.settings.duplicate_merge_ultra_continuity_max_gap_frames = 6
    pipeline.settings.duplicate_merge_ultra_continuity_max_center_distance_ratio = 0.12
    pipeline.settings.duplicate_merge_ultra_continuity_max_weak_tracklets = 2
    pipeline.settings.duplicate_merge_ultra_continuity_max_supported_tracklets = 8
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (1, 10)
            return 0.5243

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 10, 10: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [690.0, 507.0, 930.0, 1080.0],
                "bbox_b": [692.0, 510.0, 932.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 10]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_allows_attribute_supported_tight_spatial_reentry():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_tight_spatial_reentry_enabled = True
    pipeline.settings.duplicate_merge_tight_spatial_reentry_min_score = 0.50
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_gap_frames = 6
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_center_distance_ratio = 0.12
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_size_ratio = 1.15
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_area_ratio = 1.25
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_weak_tracklets = 2
    pipeline.settings.duplicate_merge_ultra_continuity_max_weak_tracklets = 1
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (5, 9)
            return 0.518

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {5: 7, 9: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [743.31, 489.31, 928.31, 894.72],
                "bbox_b": [746.75, 490.15, 937.08, 907.17],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.99,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.80,
                    "hat": "no_hat",
                    "hat_confidence": 0.74,
                    "lower": "trousers",
                    "lower_confidence": 0.78,
                    "sleeve": "long_sleeve",
                    "sleeve_confidence": 0.76,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.94,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.86,
                    "hat": "no_hat",
                    "hat_confidence": 0.79,
                    "lower": "trousers",
                    "lower_confidence": 0.78,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.76,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([5, 9]))

    assert merged == 1
    assert calls["gallery_merge"] == (9, 5)
    assert calls["mongo_merge"][0:2] == (9, 5)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"
    assert calls["mongo_merge"][2]["soft_split_reason"] == "tight_spatial_reentry"


def test_spatial_reconciler_rejects_tight_spatial_reentry_without_attribute_support():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_tight_spatial_reentry_enabled = True
    pipeline.settings.duplicate_merge_tight_spatial_reentry_min_score = 0.50
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_gap_frames = 6
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_center_distance_ratio = 0.12
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_size_ratio = 1.15
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_area_ratio = 1.25
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_weak_tracklets = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.533

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 7, 6: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [674.0, 531.0, 975.0, 1079.0],
                "bbox_b": [681.0, 539.0, 978.0, 1079.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.80,
                    "hat": "no_hat",
                    "hat_confidence": 0.70,
                    "lower": "trousers",
                    "lower_confidence": 0.82,
                },
                {
                    "gender": "female",
                    "gender_confidence": 0.92,
                    "backpack": "backpack",
                    "backpack_confidence": 0.83,
                    "hat": "no_hat",
                    "hat_confidence": 0.74,
                    "lower": "skirt",
                    "lower_confidence": 0.84,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 6]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_tight_spatial_reentry_between_established_ids():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_tight_spatial_reentry_enabled = True
    pipeline.settings.duplicate_merge_tight_spatial_reentry_min_score = 0.50
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_gap_frames = 6
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_center_distance_ratio = 0.12
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_size_ratio = 1.15
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_area_ratio = 1.25
    pipeline.settings.duplicate_merge_tight_spatial_reentry_max_weak_tracklets = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.5591

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 8, 5: 10}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [743.31, 489.31, 928.31, 894.72],
                "bbox_b": [746.75, 490.15, 937.08, 907.17],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.82,
                    "lower": "trousers",
                    "lower_confidence": 0.80,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.94,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.84,
                    "lower": "trousers",
                    "lower_confidence": 0.81,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 5]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_low_score_multitrack_duplicate_box():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_duplicate_iou_threshold = 0.45
    pipeline.settings.duplicate_merge_soft_split_duplicate_box_multitrack_min_score = 0.58
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.4964

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {5: 18, 10: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [743.31, 489.31, 928.31, 894.72],
                "bbox_b": [746.75, 490.15, 937.08, 907.17],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return ({}, {})

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([5, 10]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_allows_occlusion_spatial_rejoin_with_strong_score():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_occlusion_spatial_rejoin_enabled = True
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (2, 9)
            return 0.597

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 8, 9: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 84,
                "bbox_a": [760.89, 230.26, 865.16, 519.14],
                "bbox_b": [777.76, 327.98, 910.88, 724.14],
            }

        async def persons_closest_spatial_transition_with_bboxes(self, person_a, person_b, max_gap_frames):
            return {
                "gap": 84,
                "bbox_a": [760.89, 230.26, 865.16, 519.14],
                "bbox_b": [777.76, 327.98, 910.88, 724.14],
                "center_distance_ratio": 0.3902,
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "female", "gender_confidence": 0.79, "lower": "trousers", "lower_confidence": 0.73},
                {"gender": "male", "gender_confidence": 0.80, "lower": "trousers", "lower_confidence": 0.74},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([2, 9]))

    assert merged == 1
    assert calls["gallery_merge"] == (9, 2)
    assert calls["mongo_merge"][0:2] == (9, 2)
    assert calls["mongo_merge"][2]["soft_split_reason"] == "occlusion_spatial_rejoin"


def test_spatial_reconciler_allows_precise_occlusion_spatial_rejoin_with_lower_score():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_occlusion_spatial_rejoin_enabled = True
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (1, 5)
            return 0.533

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 6, 5: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 66,
                "bbox_a": [611.90, 255.60, 735.41, 577.13],
                "bbox_b": [674.28, 531.11, 975.86, 1079.80],
            }

        async def persons_closest_spatial_transition_with_bboxes(self, person_a, person_b, max_gap_frames):
            return {
                "gap": 148,
                "bbox_a": [624.25, 239.45, 714.44, 480.00],
                "bbox_b": [703.73, 149.90, 792.74, 395.88],
                "center_distance_ratio": 0.4769,
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.92, "glasses": "glasses", "glasses_confidence": 0.87},
                {"gender": "male", "gender_confidence": 0.97, "glasses": "glasses", "glasses_confidence": 0.80},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 5]))

    assert merged == 1
    assert calls["gallery_merge"] == (5, 1)
    assert calls["mongo_merge"][0:2] == (5, 1)
    assert calls["mongo_merge"][2]["soft_split_reason"] == "occlusion_spatial_rejoin"


def test_spatial_reconciler_rejects_occlusion_spatial_rejoin_when_cooccurred():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_occlusion_spatial_rejoin_enabled = True
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.60

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {4: 8, 9: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 20,
                "bbox_a": [760.89, 230.26, 865.16, 519.14],
                "bbox_b": [777.76, 327.98, 910.88, 724.14],
            }

        async def persons_closest_spatial_transition_with_bboxes(self, person_a, person_b, max_gap_frames):
            return {
                "gap": 20,
                "bbox_a": [760.89, 230.26, 865.16, 519.14],
                "bbox_b": [777.76, 327.98, 910.88, 724.14],
                "center_distance_ratio": 0.39,
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return ({}, {})

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([4, 9]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_allows_same_frame_established_duplicate_with_tight_overlap():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_enabled = True
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_min_score = 0.50
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_min_iou = 0.75
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_max_center_distance_ratio = 0.05
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_max_size_ratio = 1.15
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_max_area_ratio = 1.25
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (7, 10)
            return 0.507

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {7: 7, 10: 6}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [1063.51, 522.75, 1191.79, 823.24],
                "bbox_b": [1057.08, 523.28, 1188.36, 830.57],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.99,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.87,
                    "hat": "no_hat",
                    "hat_confidence": 0.80,
                    "lower": "trousers",
                    "lower_confidence": 0.71,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.97,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.87,
                    "hat": "no_hat",
                    "hat_confidence": 0.81,
                    "lower": "trousers",
                    "lower_confidence": 0.85,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.85,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([7, 10]))

    assert merged == 1
    assert calls["gallery_merge"] == (10, 7)
    assert calls["mongo_merge"][0:2] == (10, 7)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"
    assert calls["mongo_merge"][2]["soft_split_reason"] == "same_frame_established_duplicate"


def test_spatial_reconciler_allows_overlap_spatial_duplicate_despite_cooccurrence():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_overlap_spatial_duplicate_enabled = True
    pipeline.settings.duplicate_merge_overlap_spatial_duplicate_min_score = 0.58
    pipeline.settings.duplicate_merge_overlap_spatial_duplicate_max_gap_frames = 4
    pipeline.settings.duplicate_merge_overlap_spatial_duplicate_max_tracklets = 24
    pipeline.settings.duplicate_merge_overlap_spatial_duplicate_max_center_distance_ratio = 0.08
    pipeline.settings.duplicate_merge_overlap_spatial_duplicate_max_size_ratio = 1.25
    pipeline.settings.duplicate_merge_overlap_spatial_duplicate_max_area_ratio = 1.60
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (2, 9)
            return 0.798

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 15, 9: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [1162.7, 467.6, 1415.3, 1083.5],
                "bbox_b": [1262.0, 688.2, 1666.2, 1080.0],
            }

        async def persons_closest_spatial_transition_with_bboxes(self, person_a, person_b, max_gap_frames):
            assert max_gap_frames == 4
            return {
                "gap": 2,
                "bbox_a": [776.37, 327.60, 906.67, 716.40],
                "bbox_b": [777.76, 327.98, 910.88, 724.14],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(
            self,
            source_person_id,
            target_person_id,
            sighting_confidence_threshold=0.90,
            min_consecutive=2,
        ):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.7949,
                    "sidebag": "no_sidebag",
                    "sidebag_confidence": 0.8102,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9122,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.8067,
                    "sidebag": "sidebag",
                    "sidebag_confidence": 0.7622,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.7801,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([2, 9]))

    assert merged == 1
    assert calls["gallery_merge"] == (9, 2)
    assert calls["mongo_merge"][0:2] == (9, 2)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"
    assert calls["mongo_merge"][2]["soft_split_reason"] == "overlap_spatial_duplicate"


def test_spatial_reconciler_allows_trajectory_reentry_with_strong_spatial_support():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_trajectory_reentry_enabled = True
    pipeline.settings.duplicate_merge_trajectory_reentry_min_score = 0.60
    pipeline.settings.duplicate_merge_trajectory_reentry_max_gap_frames = 240
    pipeline.settings.duplicate_merge_trajectory_reentry_max_tracklets = 24
    pipeline.settings.duplicate_merge_trajectory_reentry_max_center_distance_ratio = 0.06
    pipeline.settings.duplicate_merge_trajectory_reentry_max_size_ratio = 1.30
    pipeline.settings.duplicate_merge_trajectory_reentry_max_area_ratio = 2.00
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (2, 4)
            return 0.626

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 20, 4: 6}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [750.15, 242.27, 863.17, 544.10],
                "bbox_b": [807.53, 167.49, 894.67, 419.74],
            }

        async def persons_closest_spatial_transition_with_bboxes(self, person_a, person_b, max_gap_frames):
            if max_gap_frames == 4:
                return {
                    "gap": 0,
                    "bbox_a": [746.83, 245.34, 859.37, 546.39],
                    "bbox_b": [835.94, 213.91, 939.73, 501.76],
                }
            assert max_gap_frames == 240
            return {
                "gap": 190,
                "bbox_a": [829.57, 186.49, 888.22, 382.88],
                "bbox_b": [807.53, 167.49, 894.67, 419.74],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(
            self,
            source_person_id,
            target_person_id,
            sighting_confidence_threshold=0.90,
            min_consecutive=2,
        ):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.79,
                    "sleeve": "long_sleeve",
                    "sleeve_confidence": 0.98,
                },
                {},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([2, 4]))

    assert merged == 1
    assert calls["gallery_merge"] == (4, 2)
    assert calls["mongo_merge"][0:2] == (4, 2)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"
    assert calls["mongo_merge"][2]["soft_split_reason"] == "trajectory_reentry"


def test_spatial_reconciler_rejects_trajectory_reentry_with_attribute_conflict():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_trajectory_reentry_enabled = True
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (2, 6)
            return 0.626

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 20, 6: 12}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {"gap": 0, "bbox_a": [0, 0, 100, 300], "bbox_b": [0, 0, 100, 300]}

        async def persons_closest_spatial_transition_with_bboxes(self, person_a, person_b, max_gap_frames):
            return {"gap": 136, "bbox_a": [0, 0, 100, 300], "bbox_b": [1, 2, 102, 302]}

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(
            self,
            source_person_id,
            target_person_id,
            sighting_confidence_threshold=0.90,
            min_consecutive=2,
        ):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.94,
                    "sleeve": "long_sleeve",
                    "sleeve_confidence": 0.98,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.92,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([2, 6]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_same_frame_duplicate_without_tight_overlap():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_enabled = True
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_min_score = 0.50
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_min_iou = 0.75
    pipeline.settings.duplicate_merge_same_frame_established_duplicate_max_center_distance_ratio = 0.05
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.55

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {4: 7, 7: 7}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [1386.30, 517.41, 1536.42, 877.32],
                "bbox_b": [1372.17, 513.26, 1453.76, 739.68],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "female", "gender_confidence": 0.90},
                {"gender": "male", "gender_confidence": 0.95},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([4, 7]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_ultra_tight_continuity_when_cooccurred():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_ultra_continuity_min_score = 0.50
    pipeline.settings.duplicate_merge_ultra_continuity_max_gap_frames = 6
    pipeline.settings.duplicate_merge_ultra_continuity_max_center_distance_ratio = 0.12
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.5393

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {3: 5, 6: 7}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 2,
                "bbox_a": [605.0, 590.0, 804.0, 1080.0],
                "bbox_b": [599.0, 599.0, 795.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return {}, {}

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([3, 6]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_allows_attribute_supported_reentry_bridge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_reentry_bridge_enabled = True
    pipeline.settings.duplicate_merge_reentry_bridge_min_score = 0.535
    pipeline.settings.duplicate_merge_reentry_bridge_max_tracklets = 4
    pipeline.settings.duplicate_merge_reentry_bridge_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_reentry_bridge_max_gap_frames = 180
    pipeline.settings.duplicate_merge_reentry_bridge_max_center_distance_ratio = 0.85
    pipeline.settings.duplicate_merge_reentry_bridge_gender_confidence = 0.70
    pipeline.settings.duplicate_merge_reentry_bridge_min_attr_matches = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (3, 6)
            return 0.5393

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {3: 4, 6: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 116,
                "bbox_a": [1428.4, 533.8, 1504.6, 749.9],
                "bbox_b": [1204.1, 538.4, 1305.3, 836.4],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.9333,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8632,
                    "hat": "no_hat",
                    "hat_confidence": 0.7802,
                    "lower": "trousers",
                    "lower_confidence": 0.8355,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9433,
                },
                {
                    "gender": "female",
                    "gender_confidence": 0.748,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8563,
                    "hat": "no_hat",
                    "hat_confidence": 0.7925,
                    "lower": "trousers",
                    "lower_confidence": 0.7469,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9422,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([3, 6]))

    assert merged == 1
    assert calls["gallery_merge"] == (6, 3)
    assert calls["mongo_merge"][0:2] == (6, 3)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"
    assert calls["mongo_merge"][2]["soft_split_reason"] == "reentry_bridge"


def test_spatial_reconciler_allows_one_sided_supported_reentry_bridge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_reentry_bridge_enabled = True
    pipeline.settings.duplicate_merge_reentry_bridge_min_score = 0.535
    pipeline.settings.duplicate_merge_reentry_bridge_max_tracklets = 4
    pipeline.settings.duplicate_merge_reentry_bridge_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_reentry_bridge_min_gap_frames = 30
    pipeline.settings.duplicate_merge_reentry_bridge_max_gap_frames = 180
    pipeline.settings.duplicate_merge_reentry_bridge_max_center_distance_ratio = 0.85
    pipeline.settings.duplicate_merge_reentry_bridge_gender_confidence = 0.70
    pipeline.settings.duplicate_merge_reentry_bridge_min_attr_matches = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (4, 7)
            return 0.72

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {4: 4, 7: 7}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 116,
                "bbox_a": [1428.4, 533.8, 1504.6, 749.9],
                "bbox_b": [1204.1, 538.4, 1305.3, 836.4],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "female",
                    "gender_confidence": 0.9333,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8632,
                    "hat": "no_hat",
                    "hat_confidence": 0.7802,
                    "lower": "trousers",
                    "lower_confidence": 0.8355,
                    "sidebag": "no_sidebag",
                    "sidebag_confidence": 0.8332,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.9433,
                },
                {
                    "gender": "female",
                    "gender_confidence": 0.8271,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.8248,
                    "hat": "no_hat",
                    "hat_confidence": 0.7897,
                    "lower": "trousers",
                    "lower_confidence": 0.6947,
                    "sidebag": "sidebag",
                    "sidebag_confidence": 0.7008,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.941,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([4, 7]))

    assert merged == 1
    assert calls["gallery_merge"] == (4, 7)
    assert calls["mongo_merge"][0:2] == (4, 7)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"
    assert calls["mongo_merge"][2]["soft_split_reason"] == "reentry_bridge"


def test_spatial_reconciler_rejects_ambiguous_one_sided_reentry_bridge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_reentry_bridge_enabled = True
    pipeline.settings.duplicate_merge_reentry_bridge_min_score = 0.535
    pipeline.settings.duplicate_merge_reentry_bridge_max_tracklets = 4
    pipeline.settings.duplicate_merge_reentry_bridge_max_supported_tracklets = 8
    pipeline.settings.duplicate_merge_reentry_bridge_min_gap_frames = 30
    pipeline.settings.duplicate_merge_reentry_bridge_max_gap_frames = 180
    pipeline.settings.duplicate_merge_reentry_bridge_max_center_distance_ratio = 0.85
    pipeline.settings.duplicate_merge_reentry_bridge_supported_min_score = 0.70
    pipeline.settings.duplicate_merge_reentry_bridge_supported_min_margin = 0.12
    calls = {}

    class DummyQdrant:
        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {8: 8, 16: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 160,
                "bbox_a": [1054.0, 438.0, 1267.0, 1079.0],
                "bbox_b": [1049.0, 454.0, 1273.0, 1079.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.91,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.86,
                    "hat": "no_hat",
                    "hat_confidence": 0.82,
                    "lower": "trousers",
                    "lower_confidence": 0.82,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.88,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.88,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.84,
                    "hat": "no_hat",
                    "hat_confidence": 0.80,
                    "lower": "trousers",
                    "lower_confidence": 0.79,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.86,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    result = asyncio.run(pipeline._try_merge_candidate(8, (16, 0.6503, 0.5865)))

    assert result.merged is False
    assert result.retryable_blocked is True
    assert calls == {}


def test_duplicate_merge_rejects_weak_fragment_into_supported_person_with_tiny_margin():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_min_score = 0.30
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_gap_frames = 15
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_size_ratio = 1.80
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_area_ratio = 2.20
    pipeline.settings.duplicate_merge_weak_to_supported_guard_enabled = True
    pipeline.settings.duplicate_merge_weak_to_supported_min_target_tracklets = 5
    pipeline.settings.duplicate_merge_weak_to_supported_max_target_tracklets = 8
    pipeline.settings.duplicate_merge_weak_to_supported_min_score = 0.82
    pipeline.settings.duplicate_merge_weak_to_supported_min_margin = 0.12
    calls = {}

    class DummyQdrant:
        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 11, 13: 1}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 12,
                "bbox_a": [755.0, 225.0, 879.0, 565.0],
                "bbox_b": [824.0, 129.0, 885.0, 323.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(
            self,
            source_person_id,
            target_person_id,
            sighting_confidence_threshold=0.90,
            min_consecutive=2,
        ):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return ({}, {})

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()

    result = asyncio.run(pipeline._try_merge_candidate(2, (13, 0.91, 0.90)))

    assert result.merged is False
    assert result.retryable_blocked is True
    assert calls == {}


def test_spatial_reconciler_rejects_short_gap_reentry_bridge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_reentry_bridge_enabled = True
    pipeline.settings.duplicate_merge_reentry_bridge_min_score = 0.535
    pipeline.settings.duplicate_merge_reentry_bridge_max_tracklets = 4
    pipeline.settings.duplicate_merge_reentry_bridge_min_gap_frames = 30
    pipeline.settings.duplicate_merge_reentry_bridge_max_gap_frames = 180
    pipeline.settings.duplicate_merge_reentry_bridge_max_center_distance_ratio = 0.85
    pipeline.settings.duplicate_merge_reentry_bridge_gender_confidence = 0.70
    pipeline.settings.duplicate_merge_reentry_bridge_min_attr_matches = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.5591

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 8, 7: 11}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 4,
                "bbox_a": [643.0, 339.0, 813.0, 795.0],
                "bbox_b": [681.0, 540.0, 978.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.80,
                    "hat": "no_hat",
                    "hat_confidence": 0.80,
                    "lower": "trousers",
                    "lower_confidence": 0.80,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.80,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.80,
                    "hat": "no_hat",
                    "hat_confidence": 0.80,
                    "lower": "trousers",
                    "lower_confidence": 0.80,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.80,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 7]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_established_low_score_reentry_bridge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_reentry_bridge_enabled = True
    pipeline.settings.duplicate_merge_reentry_bridge_min_score = 0.535
    pipeline.settings.duplicate_merge_reentry_bridge_max_tracklets = 4
    pipeline.settings.duplicate_merge_reentry_bridge_min_gap_frames = 30
    pipeline.settings.duplicate_merge_reentry_bridge_max_gap_frames = 180
    pipeline.settings.duplicate_merge_reentry_bridge_max_center_distance_ratio = 0.85
    pipeline.settings.duplicate_merge_reentry_bridge_gender_confidence = 0.70
    pipeline.settings.duplicate_merge_reentry_bridge_min_attr_matches = 2
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.5591

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 8, 6: 9}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 116,
                "bbox_a": [1301.2, 663.6, 1591.9, 1080.0],
                "bbox_b": [824.1, 128.8, 884.7, 323.2],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.80,
                    "hat": "no_hat",
                    "hat_confidence": 0.80,
                    "lower": "trousers",
                    "lower_confidence": 0.80,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.80,
                },
                {
                    "gender": "male",
                    "gender_confidence": 0.95,
                    "backpack": "no_backpack",
                    "backpack_confidence": 0.80,
                    "hat": "no_hat",
                    "hat_confidence": 0.80,
                    "lower": "trousers",
                    "lower_confidence": 0.80,
                    "sleeve": "short_sleeve",
                    "sleeve_confidence": 0.80,
                },
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 6]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_refreshes_counts_after_in_pass_merge():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    calls = {"merged": []}
    counts = {1: 8, 7: 1, 9: 1}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return {
                (1, 7): 0.5591,
                (1, 9): 0.30,
                (7, 9): 0.8433,
            }.get((person_a, person_b), 0.0)

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return counts[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            if {person_a, person_b} == {7, 9}:
                return {
                    "gap": 0,
                    "bbox_a": [703.7, 149.9, 792.7, 395.9],
                    "bbox_b": [715.0, 158.6, 804.1, 404.5],
                }
            return {
                "gap": 80,
                "bbox_a": [755.4, 224.5, 879.0, 565.1],
                "bbox_b": [824.1, 128.8, 884.7, 323.2],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.95},
                {"gender": "male", "gender_confidence": 0.95},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["merged"].append((source_person_id, target_person_id, reason))
            counts[target_person_id] += counts[source_person_id]
            counts[source_person_id] = 0

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 7, 9]))

    assert merged == 1
    assert calls["merged"][0][0:2] == (9, 7)
    assert all(pair[0:2] != (1, 7) for pair in calls["merged"])


def test_spatial_reconciler_allows_boundary_singleton_duplicate_box():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_duplicate_iou_threshold = 0.45
    pipeline.settings.duplicate_merge_boundary_duplicate_min_score = 0.68
    pipeline.settings.duplicate_merge_boundary_duplicate_min_iou = 0.10
    pipeline.settings.duplicate_merge_boundary_duplicate_max_center_distance_ratio = 0.45
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (7, 17)
            return 0.7095

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {7: 22, 17: 1}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [1301.2, 663.6, 1591.9, 1080.0],
                "bbox_b": [1078.5, 509.9, 1383.2, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return ({}, {})

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([7, 17]))

    assert merged == 1
    assert calls["gallery_merge"] == (17, 7)
    assert calls["mongo_merge"][0:2] == (17, 7)
    assert calls["mongo_merge"][2]["method"] == "soft_split_gallery_merge"


def test_spatial_reconciler_rejects_reentry_bridge_below_score_floor():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_reentry_bridge_enabled = True
    pipeline.settings.duplicate_merge_reentry_bridge_min_score = 0.535
    pipeline.settings.duplicate_merge_reentry_bridge_max_tracklets = 4
    pipeline.settings.duplicate_merge_reentry_bridge_max_gap_frames = 180
    pipeline.settings.duplicate_merge_reentry_bridge_max_center_distance_ratio = 0.85
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            return 0.524

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {1: 4, 11: 4}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 90,
                "bbox_a": [100.0, 500.0, 220.0, 900.0],
                "bbox_b": [280.0, 505.0, 405.0, 910.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.90},
                {"gender": "male", "gender_confidence": 0.88},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([1, 11]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_attempts_low_score_occlusion_reentry_with_no_cooccurrence():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_multitrack_min_score = 0.60
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (2, 8)
            return 0.5171

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {2: 22, 8: 2}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 12,
                "bbox_a": [1218.0, 601.0, 1439.0, 1080.0],
                "bbox_b": [1262.0, 688.0, 1666.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.82},
                {"gender": "female", "gender_confidence": 0.79},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([2, 8]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_low_embedding_singleton_into_supported_person():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_min_score = 0.30
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_gap_frames = 15
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_size_ratio = 1.80
    pipeline.settings.duplicate_merge_singleton_spatial_continuation_max_area_ratio = 2.20
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (11, 14)
            return 0.3604

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {11: 8, 14: 1}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 12,
                "bbox_a": [1218.1765, 601.0527, 1439.5460, 1080.1427],
                "bbox_b": [1262.0481, 688.2322, 1666.2317, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return False

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.7432},
                {"gender": "male", "gender_confidence": 0.8238},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([11, 14]))

    assert merged == 0
    assert calls == {}


def test_spatial_reconciler_rejects_continuation_when_persons_cooccur():
    pipeline = _make_pipeline()
    pipeline.settings.duplicate_merge_enabled = True
    pipeline.settings.duplicate_merge_min_score = 0.90
    pipeline.settings.duplicate_merge_singleton_min_score = 0.90
    pipeline.settings.duplicate_merge_weak_max_tracklets = 2
    pipeline.settings.duplicate_merge_soft_split_override_threshold = 0.70
    pipeline.settings.duplicate_merge_soft_split_max_weak_tracklets = 8
    pipeline.settings.duplicate_merge_soft_split_spatial_only_min_score = 0.50
    pipeline.settings.duplicate_merge_soft_split_spatial_only_max_center_distance_ratio = 0.30
    pipeline.settings.duplicate_merge_spatial_continuation_enabled = True
    pipeline.settings.duplicate_merge_spatial_continuation_min_score = 0.20
    pipeline.settings.duplicate_merge_spatial_continuation_max_gap_frames = 60
    pipeline.settings.duplicate_merge_spatial_continuation_max_center_distance_ratio = 0.30
    calls = {}

    class DummyQdrant:
        def person_pair_similarity(self, person_a, person_b):
            assert (person_a, person_b) == (4, 10)
            return 0.2465

        def merge_person_gallery(self, source_person_id, target_person_id):
            calls["gallery_merge"] = (source_person_id, target_person_id)

    class DummyMongo:
        async def count_tracklets(self, person_id):
            return {4: 12, 10: 1}[person_id]

        async def persons_min_frame_gap_with_bboxes(self, person_a, person_b):
            return {
                "gap": 0,
                "bbox_a": [1128.0, 430.0, 1391.0, 1085.0],
                "bbox_b": [1262.0, 688.0, 1666.0, 1080.0],
            }

        async def persons_closest_spatial_transition_with_bboxes(self, person_a, person_b, max_gap_frames):
            assert max_gap_frames == 60
            return {
                "gap": 44,
                "bbox_a": [1244.0, 540.0, 1658.0, 1080.0],
                "bbox_b": [1301.0, 726.0, 1666.0, 1080.0],
            }

        async def persons_cooccur(self, source_person_id, target_person_id):
            return True

        async def persons_have_clear_gender_disagreement(self, source_person_id, target_person_id, sighting_confidence_threshold=0.90, min_consecutive=2):
            return False

        async def fetch_two_persons_attributes(self, source_person_id, target_person_id):
            return (
                {"gender": "male", "gender_confidence": 0.82},
                {"gender": "female", "gender_confidence": 0.83},
            )

        async def merge_person(self, source_person_id, target_person_id, reason):
            calls["mongo_merge"] = (source_person_id, target_person_id, reason)

    class DummyRedis:
        async def invalidate(self, person_id):
            calls.setdefault("invalidated", []).append(person_id)

    pipeline.qdrant_store = DummyQdrant()
    pipeline.mongo = DummyMongo()
    pipeline.redis_cache = DummyRedis()

    merged = asyncio.run(pipeline._reconcile_spatial_split_persons([4, 10]))

    assert merged == 0
    assert calls == {}


def test_untracked_detection_candidate_persists_only_unmatched_detection():
    pipeline = _make_pipeline()
    pipeline.settings.untracked_detection_candidates_enabled = True
    pipeline.settings.untracked_detection_raw_candidates_enabled = True
    pipeline.settings.untracked_detection_cluster_enabled = False
    pipeline.settings.untracked_detection_min_confidence = 0.25
    pipeline.settings.untracked_detection_min_visibility = 0.35
    pipeline.settings.untracked_detection_max_track_iou = 0.2
    calls = []

    async def fake_persist(tracklet, **kwargs):
        calls.append((tracklet, kwargs))

    pipeline._persist_occlusion_candidate = fake_persist
    frame = np.ones((120, 160, 3), dtype=np.uint8)
    detections = [
        {
            "bbox": [10.0, 10.0, 50.0, 80.0],
            "confidence": 0.9,
            "class_id": 0,
            "visibility_score": 0.8,
            "overlap_ratio": 0.0,
        },
        {
            "bbox": [100.0, 10.0, 140.0, 80.0],
            "confidence": 0.9,
            "class_id": 0,
            "visibility_score": 0.8,
            "overlap_ratio": 0.0,
        },
    ]
    track_results = np.asarray([[10.0, 10.0, 50.0, 80.0, 1.0, 0.9]], dtype=np.float32)

    asyncio.run(
        pipeline._persist_untracked_detection_candidates(
            detections=detections,
            track_results=track_results,
            frame=frame,
            frame_number=42,
            timestamp_ns=42_000,
        )
    )

    assert len(calls) == 1
    tracklet, kwargs = calls[0]
    assert tracklet.track_id == -42002
    assert len(tracklet.entries) == 1
    assert kwargs["reason"] == "untracked_detection"
    assert kwargs["min_entries"] == 1


def test_untracked_detection_cluster_persists_after_temporal_support():
    pipeline = _make_pipeline()
    pipeline.settings.untracked_detection_candidates_enabled = True
    pipeline.settings.untracked_detection_raw_candidates_enabled = False
    pipeline.settings.untracked_detection_cluster_enabled = True
    pipeline.settings.untracked_detection_cluster_min_entries = 2
    pipeline.settings.untracked_detection_cluster_max_gap_frames = 18
    pipeline.settings.untracked_detection_cluster_max_center_distance_ratio = 1.25
    pipeline.settings.untracked_detection_cluster_flush_after_frames = 36
    pipeline.settings.untracked_detection_min_confidence = 0.25
    pipeline.settings.untracked_detection_min_visibility = 0.35
    pipeline.settings.untracked_detection_max_track_iou = 0.2
    calls = []

    async def fake_persist(tracklet, **kwargs):
        calls.append((tracklet, kwargs))

    pipeline._persist_occlusion_candidate = fake_persist
    frame = np.ones((120, 160, 3), dtype=np.uint8)
    track_results = np.empty((0, 6), dtype=np.float32)

    for frame_number, x_shift in [(42, 0.0), (44, 2.0)]:
        asyncio.run(
            pipeline._persist_untracked_detection_candidates(
                detections=[
                    {
                        "bbox": [100.0 + x_shift, 10.0, 140.0 + x_shift, 80.0],
                        "confidence": 0.9,
                        "class_id": 0,
                        "visibility_score": 0.8,
                        "overlap_ratio": 0.0,
                    }
                ],
                track_results=track_results,
                frame=frame,
                frame_number=frame_number,
                timestamp_ns=frame_number * 1000,
            )
        )

    assert len(calls) == 1
    tracklet, kwargs = calls[0]
    assert tracklet.track_id < -9_000_000
    assert len(tracklet.entries) == 2
    assert kwargs["reason"] == "untracked_detection_cluster"
    assert kwargs["min_entries"] == 2
    assert kwargs["candidate_id_override"].startswith("cam-1:untracked_cluster:")


def test_synthetic_fast_tracklet_ready_requires_five_clean_frames():
    pipeline = _make_pipeline()
    pipeline.settings.untracked_cluster_promote_min_entries_fast = 5
    pipeline.settings.untracked_cluster_promote_min_visibility_fast = 0.85
    pipeline.settings.untracked_cluster_promote_fast_min_overall_consistency = 0.88
    pipeline.settings.high_quality_threshold = 0.55
    pipeline.settings.min_high_quality_frames = 3

    four_frame_tracklet = Tracklet(
        track_id=-9001,
        entries=[_make_entry(i, v_score=0.92, overlap_ratio=0.0) for i in range(4)],
    )
    five_frame_tracklet = Tracklet(
        track_id=-9002,
        entries=[_make_entry(i, v_score=0.92, overlap_ratio=0.0) for i in range(5)],
    )
    regular_tracklet = Tracklet(
        track_id=12,
        entries=[_make_entry(i, v_score=0.92, overlap_ratio=0.0) for i in range(5)],
    )

    assert not pipeline._is_synthetic_fast_tracklet_ready(four_frame_tracklet)
    assert pipeline._is_synthetic_fast_tracklet_ready(five_frame_tracklet)
    assert not pipeline._is_synthetic_fast_tracklet_ready(regular_tracklet)


def test_synthetic_fast_tracklet_rejects_incoherent_cluster():
    pipeline = _make_pipeline()
    pipeline.settings.untracked_cluster_promote_min_entries_fast = 5
    pipeline.settings.untracked_cluster_promote_min_visibility_fast = 0.85
    pipeline.settings.untracked_cluster_promote_fast_min_overall_consistency = 0.88
    pipeline.settings.high_quality_threshold = 0.55
    pipeline.settings.min_high_quality_frames = 3
    entries = [_make_entry(i, v_score=0.92, overlap_ratio=0.0) for i in range(5)]
    entries[-1].bbox_xyxy = [180.0, 20.0, 220.0, 120.0]
    tracklet = Tracklet(track_id=-9003, entries=entries)

    assert not pipeline._is_synthetic_fast_tracklet_ready(tracklet)


def test_process_tracklet_match_success_updates_state_and_persists(monkeypatch):
    pipeline = _make_pipeline()
    removed_track_ids = []
    persisted = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95,
                               "probabilities": {"male": 0.95, "female": 0.05}}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            persisted["matcher_kwargs"] = kwargs
            return 101

    class DummyBuffer:
        def remove(self, track_id):
            removed_track_ids.append(track_id)

    async def fake_persist_tracklet(**kwargs):
        persisted["persist_kwargs"] = kwargs

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline.tracklet_buffer = DummyBuffer()
    pipeline._persist_tracklet = fake_persist_tracklet

    monkeypatch.setattr("src.workers.main.cv2.imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)))
    monkeypatch.setattr("src.workers.main.compute_tracklet_consistency", lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75))
    monkeypatch.setattr("src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency", lambda embeddings: 0.91)
    monkeypatch.setattr("src.workers.main.uuid.uuid4", lambda: "tracklet-123")

    tracklet = Tracklet(track_id=7, entries=[_make_entry(1), _make_entry(2, v_score=0.9)])

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert tracklet.state == TrackletState.MATCHED
    assert tracklet.person_id == 101
    assert pipeline.track_id_to_person_id == {7: 101}
    assert pipeline.track_metadata[7]["tracklet_id"] == "tracklet-123"
    assert pipeline.track_metadata[7]["tracklet_state"] == "matched"
    assert pipeline.track_metadata[7]["attributes"] == {"gender": "male"}
    assert pipeline.track_metadata[7]["quality"] == {
        "v_avg": 0.85,
        "embedding_consistency": 0.91,
        "overall_consistency": 0.88,
        "good_frame_ratio": 0.75,
    }
    assert removed_track_ids == []
    assert persisted["matcher_kwargs"]["track_id"] == 7
    assert persisted["persist_kwargs"]["tracklet_id"] == "tracklet-123"
    assert persisted["persist_kwargs"]["person_id"] == 101
    p_attrs = persisted["persist_kwargs"]["person_attrs"]
    assert p_attrs["gender"][0] == "male"


def test_process_tracklet_without_match_does_not_assign_person_or_persist(monkeypatch):
    pipeline = _make_pipeline()
    removed_track_ids = []
    persist_called = {"value": False}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95,
                               "probabilities": {"male": 0.95, "female": 0.05}}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            return None

    class DummyBuffer:
        def remove(self, track_id):
            removed_track_ids.append(track_id)

    async def fake_persist_tracklet(**kwargs):
        persist_called["value"] = True

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline.tracklet_buffer = DummyBuffer()
    pipeline._persist_tracklet = fake_persist_tracklet

    monkeypatch.setattr("src.workers.main.cv2.imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)))
    monkeypatch.setattr("src.workers.main.compute_tracklet_consistency", lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75))
    monkeypatch.setattr("src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency", lambda embeddings: 0.91)
    monkeypatch.setattr("src.workers.main.uuid.uuid4", lambda: "tracklet-456")

    tracklet = Tracklet(track_id=8, entries=[_make_entry(1), _make_entry(2, v_score=0.9)])

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert tracklet.person_id is None
    assert tracklet.state == TrackletState.ACTIVE
    assert pipeline.track_id_to_person_id == {}
    assert pipeline.track_metadata == {}
    assert removed_track_ids == []
    assert persist_called["value"] is False
def test_process_tracklet_prefers_best_frame_glasses_signal(monkeypatch):
    pipeline = _make_pipeline()
    persisted = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    classify_calls = {"count": 0}

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            classify_calls["count"] += 1
            if classify_calls["count"] == 1:
                return {
                    "gender": {"label": "male", "confidence": 0.95},
                    "glasses": {"label": "no_glasses", "confidence": 0.76},
                }
            return {
                "gender": {"label": "male", "confidence": 0.95},
                "glasses": {"label": "glasses", "confidence": 0.91},
            }

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            return 202

    async def fake_persist_tracklet(**kwargs):
        persisted["persist_kwargs"] = kwargs

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline._persist_tracklet = fake_persist_tracklet
    pipeline.settings.glasses_best_frame_override_threshold = 0.6

    monkeypatch.setattr("src.workers.main.cv2.imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)))
    monkeypatch.setattr("src.workers.main.compute_tracklet_consistency", lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75))
    monkeypatch.setattr("src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency", lambda embeddings: 0.91)
    monkeypatch.setattr("src.workers.main.uuid.uuid4", lambda: "tracklet-glasses")

    tracklet = Tracklet(track_id=11, entries=[_make_entry(1, v_score=0.7), _make_entry(2, v_score=0.95)])

    asyncio.run(pipeline._process_tracklet(tracklet))

    p_attrs = persisted["persist_kwargs"]["person_attrs"]
    assert p_attrs["glasses"] == ("glasses", 0.91)


def test_build_attribute_crop_expands_upward_and_sideways():
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    bbox = np.array([20.0, 30.0, 40.0, 90.0], dtype=np.float32)

    crop = _build_attribute_crop(
        frame,
        bbox,
        top_padding_ratio=0.25,
        side_padding_ratio=0.1,
        bottom_padding_ratio=0.05,
    )

    # Original bbox height/width = 60x20. Expanded crop should include extra context.
    assert crop.shape[:2] == (78, 24)


def test_compute_person_snapshot_score_prefers_clearer_tracklet():
    strong = _compute_person_snapshot_score(
        v_avg=0.93,
        overall_consistency=0.98,
        embedding_consistency=0.98,
    )
    weaker = _compute_person_snapshot_score(
        v_avg=0.81,
        overall_consistency=0.96,
        embedding_consistency=0.92,
    )

    assert strong > weaker


def test_compute_person_snapshot_score_penalizes_overlapped_crop():
    clean = _compute_person_snapshot_score(
        v_avg=0.90,
        overall_consistency=0.90,
        embedding_consistency=0.90,
        overlap_ratio=0.05,
    )
    overlapped = _compute_person_snapshot_score(
        v_avg=0.90,
        overall_consistency=0.90,
        embedding_consistency=0.90,
        overlap_ratio=0.70,
    )

    assert clean > overlapped


def test_choose_person_snapshot_entry_rejects_all_overlapped_crops():
    entries = [
        _make_entry(10, v_score=0.94, overlap_ratio=0.62),
        _make_entry(12, v_score=0.88, overlap_ratio=0.48),
    ]

    assert _choose_person_snapshot_entry(entries, entries, max_overlap_ratio=0.35) is None


def test_choose_person_snapshot_entry_prefers_clean_crop_over_high_overlap():
    overlapped = _make_entry(10, v_score=0.96, overlap_ratio=0.55)
    clean = _make_entry(12, v_score=0.82, overlap_ratio=0.05)

    selected = _choose_person_snapshot_entry(
        [overlapped, clean],
        [overlapped],
        max_overlap_ratio=0.35,
    )

    assert selected is clean


def test_select_embedding_consensus_rejects_outlier():
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.98, 0.2], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]

    indices = _select_embedding_consensus_indices(
        embeddings,
        [0.8, 0.7, 0.95],
        similarity_threshold=0.72,
    )

    assert indices == [0, 1]


def test_tracklet_motion_summary_identifies_tiny_static_track():
    entries = [
        TrackletEntry(
            frame_idx=1,
            crop=np.ones((100, 40, 3), dtype=np.uint8),
            v_score=0.8,
            bbox_xyxy=[10.0, 10.0, 50.0, 110.0],
            timestamp_ns=int(1e9),
        ),
        TrackletEntry(
            frame_idx=2,
            crop=np.ones((100, 40, 3), dtype=np.uint8),
            v_score=0.82,
            bbox_xyxy=[10.5, 10.0, 50.5, 110.0],
            timestamp_ns=int(2e9),
        ),
        TrackletEntry(
            frame_idx=3,
            crop=np.ones((100, 40, 3), dtype=np.uint8),
            v_score=0.84,
            bbox_xyxy=[11.0, 10.0, 51.0, 110.0],
            timestamp_ns=int(3e9),
        ),
        TrackletEntry(
            frame_idx=4,
            crop=np.ones((100, 40, 3), dtype=np.uint8),
            v_score=0.85,
            bbox_xyxy=[11.5, 10.0, 51.5, 110.0],
            timestamp_ns=int(4e9),
        ),
    ]

    summary = _tracklet_motion_summary(entries)

    assert summary["mean_width_px"] == 40.0
    assert summary["mean_height_px"] == 100.0
    assert summary["path_displacement_ratio"] < 0.05
    assert summary["endpoint_displacement_ratio"] < 0.02


def test_should_suppress_static_artifact_with_bbox_jitter():
    pipeline = _make_pipeline()
    entries = []
    for idx, x in enumerate([10.0, 10.6, 11.2, 11.8, 12.4, 13.0], start=1):
        entries.append(
            TrackletEntry(
                frame_idx=idx,
                crop=np.ones((100, 40, 3), dtype=np.uint8),
                v_score=0.9,
                bbox_xyxy=[x, 20.0, x + 40.0, 120.0],
                timestamp_ns=idx * int(1e9),
            )
        )
    tracklet = Tracklet(track_id=-9000006, entries=entries)

    assert _tracklet_motion_summary(entries)["endpoint_displacement_ratio"] > 0.02
    assert pipeline._should_suppress_new_identity(tracklet) is True


def test_should_not_ignore_small_stationary_person_when_pretrack_filter_disabled():
    pipeline = _make_pipeline()
    pipeline.prev_bboxes[55] = [
        np.array([10.0, 20.0, 46.0, 121.0], dtype=np.float32),
        np.array([10.8, 20.2, 46.8, 121.2], dtype=np.float32),
        np.array([11.1, 20.1, 47.1, 121.1], dtype=np.float32),
        np.array([11.4, 20.0, 47.4, 121.0], dtype=np.float32),
    ]

    assert pipeline._should_ignore_pretrack_static_artifact(55, [11.4, 20.0, 47.4, 121.0]) is False


def test_should_ignore_pretrack_static_artifact_when_filter_explicitly_enabled():
    pipeline = _make_pipeline()
    pipeline.settings.pretrack_static_filter_enabled = True
    pipeline.prev_bboxes[55] = [
        np.array([10.0, 20.0, 46.0, 121.0], dtype=np.float32),
        np.array([10.8, 20.2, 46.8, 121.2], dtype=np.float32),
        np.array([11.1, 20.1, 47.1, 121.1], dtype=np.float32),
        np.array([11.4, 20.0, 47.4, 121.0], dtype=np.float32),
    ]

    assert pipeline._should_ignore_pretrack_static_artifact(55, [11.4, 20.0, 47.4, 121.0]) is True


def test_should_ignore_extinguisher_scale_static_artifact_when_filter_enabled():
    pipeline = _make_pipeline()
    pipeline.settings.pretrack_static_filter_enabled = True
    pipeline.prev_bboxes[57] = [
        np.array([20.0, 30.0, 131.0, 261.0], dtype=np.float32),
        np.array([20.2, 30.0, 131.2, 261.0], dtype=np.float32),
        np.array([20.1, 30.1, 131.1, 261.1], dtype=np.float32),
        np.array([20.0, 30.0, 131.0, 261.0], dtype=np.float32),
    ]

    assert pipeline._should_ignore_pretrack_static_artifact(
        57,
        [20.0, 30.0, 131.0, 261.0],
    ) is True


def test_pretrack_static_artifact_filter_preserves_boundary_occlusion():
    pipeline = _make_pipeline()
    pipeline.settings.pretrack_static_filter_enabled = True
    pipeline.prev_bboxes[55] = [
        np.array([0.0, 20.0, 36.0, 121.0], dtype=np.float32),
        np.array([0.0, 20.2, 36.0, 121.2], dtype=np.float32),
        np.array([0.0, 20.1, 36.0, 121.1], dtype=np.float32),
        np.array([0.0, 20.0, 36.0, 121.0], dtype=np.float32),
    ]

    assert pipeline._should_ignore_pretrack_static_artifact(
        55,
        [0.0, 20.0, 36.0, 121.0],
        frame_w=640,
        frame_h=480,
    ) is False


def test_should_not_ignore_pretrack_static_artifact_for_regular_person_scale():
    pipeline = _make_pipeline()
    pipeline.prev_bboxes[56] = [
        np.array([10.0, 20.0, 120.0, 320.0], dtype=np.float32),
        np.array([16.0, 21.0, 126.0, 321.0], dtype=np.float32),
        np.array([24.0, 22.0, 134.0, 322.0], dtype=np.float32),
        np.array([34.0, 24.0, 144.0, 324.0], dtype=np.float32),
    ]

    assert pipeline._should_ignore_pretrack_static_artifact(56, [34.0, 24.0, 144.0, 324.0]) is False


def test_process_tracklet_uses_attribute_crop_for_attribute_classification(monkeypatch):
    pipeline = _make_pipeline()
    seen = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            seen["attribute_bytes"] = img_bytes
            return {"gender": {"label": "male", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            return 303

    async def fake_persist_tracklet(**kwargs):
        seen["persisted"] = kwargs

    def fake_imencode(_ext, image, _params):
        payload = f"{image.shape[0]}x{image.shape[1]}".encode()
        return True, np.frombuffer(payload, dtype=np.uint8)

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline._persist_tracklet = fake_persist_tracklet

    monkeypatch.setattr("src.workers.main.cv2.imencode", fake_imencode)
    monkeypatch.setattr("src.workers.main.compute_tracklet_consistency", lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75))
    monkeypatch.setattr("src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency", lambda embeddings: 0.91)
    monkeypatch.setattr("src.workers.main.uuid.uuid4", lambda: "tracklet-attr-crop")

    crop = np.ones((40, 20, 3), dtype=np.uint8)
    attribute_crop = np.ones((52, 24, 3), dtype=np.uint8)
    tracklet = Tracklet(
        track_id=12,
        entries=[
            TrackletEntry(
                frame_idx=1,
                crop=crop,
                v_score=0.9,
                bbox_xyxy=[10.0, 20.0, 30.0, 60.0],
                timestamp_ns=int(1e9),
                attribute_crop=attribute_crop,
                overlap_ratio=0.0,
            )
        ],
    )

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert seen["attribute_bytes"] == b"52x24"


def test_process_tracklet_not_ready_marks_tentative_and_returns():
    pipeline = _make_pipeline()
    removed_track_ids = []

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return False

    class DummyBuffer:
        def remove(self, track_id):
            removed_track_ids.append(track_id)

    pipeline.topk_selector = DummySelector()
    pipeline.tracklet_buffer = DummyBuffer()

    tracklet = Tracklet(track_id=9, entries=[_make_entry(1)])

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert tracklet.state == TrackletState.ACTIVE
    assert tracklet.person_id is None
    assert pipeline.track_id_to_person_id == {}
    assert pipeline.track_metadata == {}
    assert removed_track_ids == []


def test_schedule_tracklet_processing_skips_duplicate_inflight_task():
    pipeline = _make_pipeline()
    tracklet = Tracklet(track_id=91, entries=[_make_entry(1)])
    calls = {"count": 0}

    async def fake_process_tracklet(
        _tracklet,
        reserved_person_ids=None,
        allow_tentative_fallback=True,
    ):
        calls["count"] += 1
        await asyncio.sleep(0)
        return None

    async def run_schedule():
        pipeline._process_tracklet = fake_process_tracklet
        first = pipeline._schedule_tracklet_processing(tracklet, reserved_person_ids=set())
        second = pipeline._schedule_tracklet_processing(tracklet, reserved_person_ids=set())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        third = pipeline._schedule_tracklet_processing(tracklet, reserved_person_ids=set())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return first, second, third

    first, second, third = asyncio.run(run_schedule())

    assert (first, second, third) == (True, False, True)
    assert calls["count"] == 2
    assert pipeline.processing_tracklet_ids == set()


def test_fragment_recovery_promotes_after_consistent_short_fragments():
    pipeline = _make_pipeline()
    allocated = {"next": 900}
    calls = {}

    class DummyAllocator:
        def allocate(self):
            pid = allocated["next"]
            allocated["next"] += 1
            return pid

    class DummyQdrant:
        def search(self, embedding, top_k=1, score_threshold=None):
            return []

        def add_person(self, person_id, embedding, metadata):
            calls["add_person"] = (person_id, embedding, metadata)

    pipeline.person_id_allocator = DummyAllocator()
    pipeline.qdrant_store = DummyQdrant()

    emb = np.array([1.0, 0.0], dtype=np.float32)
    first = Tracklet(
        track_id=201,
        entries=[
            _make_entry(10, v_score=0.9),
            _make_entry(12, v_score=0.88),
        ],
    )
    second = Tracklet(
        track_id=202,
        entries=[
            _make_entry(20, v_score=0.92),
            _make_entry(22, v_score=0.91),
            _make_entry(24, v_score=0.89),
        ],
    )

    first_pid, _ = pipeline._add_fragment_recovery_candidate(
        tracklet=first,
        embedding=emb,
        v_avg=0.89,
        emb_consistency=1.0,
    )
    second_pid, matching = pipeline._add_fragment_recovery_candidate(
        tracklet=second,
        embedding=emb,
        v_avg=0.90,
        emb_consistency=1.0,
    )

    assert first_pid is None
    assert second_pid == 900
    assert matching["source"] == "fragment_recovery"
    assert matching["fragment_count"] == 2
    assert matching["total_entries"] == 5
    assert calls["add_person"][0] == 900
    assert calls["add_person"][2]["source"] == "fragment_recovery"


def test_fragment_recovery_defers_when_cluster_is_near_existing_gallery():
    pipeline = _make_pipeline()
    calls = {"allocated": 0}

    class DummyAllocator:
        def allocate(self):
            calls["allocated"] += 1
            return 900

    class DummyQdrant:
        def search(self, embedding, top_k=1, score_threshold=None):
            return [(42, 0.55)]

        def add_person(self, person_id, embedding, metadata):
            calls["add_person"] = True

    pipeline.person_id_allocator = DummyAllocator()
    pipeline.qdrant_store = DummyQdrant()

    emb = np.array([1.0, 0.0], dtype=np.float32)
    first = Tracklet(track_id=301, entries=[_make_entry(10, v_score=0.9), _make_entry(12, v_score=0.88)])
    second = Tracklet(track_id=302, entries=[_make_entry(20, v_score=0.9), _make_entry(22, v_score=0.9), _make_entry(24, v_score=0.9)])

    pipeline._add_fragment_recovery_candidate(
        tracklet=first,
        embedding=emb,
        v_avg=0.89,
        emb_consistency=1.0,
    )
    pid, matching = pipeline._add_fragment_recovery_candidate(
        tracklet=second,
        embedding=emb,
        v_avg=0.9,
        emb_consistency=1.0,
    )

    assert pid is None
    assert matching["method"] == "fragment_recovery_deferred_near_gallery"
    assert matching["reuse_person_id"] == 42
    assert calls["allocated"] == 0
    assert "add_person" not in calls


def test_short_fragment_near_gallery_defer_attaches_as_provisional(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.settings.fragment_recovery_enabled = True
    pipeline.settings.occlusion_provisional_match_enabled = True
    pipeline.settings.occlusion_provisional_match_threshold = 0.60
    pipeline.settings.occlusion_provisional_min_margin = 0.03
    persisted = {}

    class DummyQdrant:
        def search(self, embedding, top_k=1, score_threshold=None):
            return [(42, 0.80)]

        def add_person(self, person_id, embedding, metadata):
            persisted["add_person"] = (person_id, metadata)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [1.0, 0.0]}

        async def classify_attributes(self, img_bytes):
            return {}

    async def fake_persist_tracklet(**kwargs):
        persisted["tracklet"] = kwargs

    async def fake_persist_occlusion_candidate(tracklet, **kwargs):
        persisted["candidate"] = {"tracklet": tracklet, **kwargs}

    pipeline.qdrant_store = DummyQdrant()
    pipeline.model_client = DummyModelClient()
    pipeline._persist_tracklet = fake_persist_tracklet
    pipeline._persist_occlusion_candidate = fake_persist_occlusion_candidate
    pipeline.attribute_voter.resolve_person(42, {"gender": ("male", 0.95)})
    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )

    first = Tracklet(
        track_id=-901,
        entries=[_make_entry(10, v_score=0.9), _make_entry(12, v_score=0.9)],
    )
    second = Tracklet(
        track_id=-902,
        entries=[
            _make_entry(20, v_score=0.9),
            _make_entry(22, v_score=0.9),
            _make_entry(24, v_score=0.9),
        ],
    )

    pipeline._add_fragment_recovery_candidate(
        tracklet=first,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        v_avg=0.9,
        emb_consistency=1.0,
    )
    person_id = asyncio.run(
        pipeline._process_short_fragment_tracklet(
            second,
            reason="short_stale_tracklet",
        )
    )

    assert person_id == 42
    assert "add_person" not in persisted
    assert "tracklet" not in persisted
    assert persisted["candidate"]["tracklet"].person_id == 42
    assert persisted["candidate"]["matching"]["method"] == "occlusion_provisional_match"
    assert persisted["candidate"]["matching"]["provisional"] is True


def test_new_identity_allocation_blocks_stale_backlog_tracklet(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.settings.max_new_identity_lag_seconds = 30.0
    pipeline.settings.stream_quiescence_seconds = 0.0
    monkeypatch.setattr("src.workers.main.time.time_ns", lambda: int(100e9))

    fresh = Tracklet(track_id=501, entries=[_make_entry(80)])
    stale = Tracklet(track_id=502, entries=[_make_entry(10)])

    assert pipeline._can_allocate_new_identity(fresh) is True
    assert pipeline._can_allocate_new_identity(stale) is False


def test_new_identity_allocation_allows_stale_tracklet_during_stream_finalization(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.settings.max_new_identity_lag_seconds = 30.0
    pipeline.settings.stream_quiescence_seconds = 0.0
    pipeline._stream_finalizing = True
    monkeypatch.setattr("src.workers.main.time.time_ns", lambda: int(100e9))

    stale = Tracklet(track_id=502, entries=[_make_entry(10)])

    assert pipeline._can_allocate_new_identity(stale) is True


def test_admission_gate_throttles_embedding_checks_after_cache_is_armed():
    pipeline = _make_pipeline()
    pipeline.settings.tracklet_appearance_gate_enabled = True
    pipeline.settings.tracklet_appearance_gate_min_v = 0.6
    pipeline.settings.tracklet_appearance_gate_check_interval_frames = 6
    pipeline.settings.tracklet_split_threshold = 0.1
    pipeline.settings.embedding_model = "osnet"
    pipeline._tracklet_embedding_cache = {}
    pipeline._track_id_split_counts = {}
    pipeline._tracklet_gate_last_check_frame = {}
    calls = {"count": 0}

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            calls["count"] += 1
            return None, {"embedding": [1.0, 0.0]}

    pipeline.model_client = DummyModelClient()
    crop = np.ones((16, 8, 3), dtype=np.uint8)

    async def run_gate():
        assert await pipeline._admission_gate_or_split(7, crop, 0.9, frame_idx=1) == 7
        assert await pipeline._admission_gate_or_split(7, crop, 0.9, frame_idx=2) == 7
        assert await pipeline._admission_gate_or_split(7, crop, 0.9, frame_idx=3) == 7
        assert await pipeline._admission_gate_or_split(7, crop, 0.9, frame_idx=8) == 7

    asyncio.run(run_gate())

    assert calls["count"] == 3


def test_current_identity_shift_risk_blocks_low_threshold_continuity(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_id_to_person_id = {11: 1}
    pipeline.settings.duplicate_merge_enabled = False
    persisted_occlusion = {}
    removed = []

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return [entries[0], entries[-1]]

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {}

    class DummyMatcher:
        tentative = {}

        def match_tracklet(self, **kwargs):
            assert kwargs["current_person_id"] is None
            assert 1 in kwargs["forbidden_person_ids"]
            assert kwargs["allow_new_identity"] is False
            return None

        def pop_last_decision(self, track_id):
            return {"method": "unconfirmed", "source": "identity_shift_risk"}

    class DummyBuffer:
        tracklets = {}

        def remove(self, track_id):
            removed.append(track_id)

    async def fake_persist_occlusion_candidate(tracklet, **kwargs):
        persisted_occlusion["track_id"] = tracklet.track_id
        persisted_occlusion["reason"] = kwargs.get("reason")

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline.tracklet_buffer = DummyBuffer()
    pipeline._persist_occlusion_candidate = fake_persist_occlusion_candidate
    pipeline._inflight = set()

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(
            overall=0.88,
            good_frame_ratio=0.75,
            bbox_size_stability=0.72,
            position_stability=0.82,
        ),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    entries = []
    for idx in range(16):
        t = idx / 15.0
        entry = _make_entry(1770 + idx * 4, v_score=0.82)
        entry.bbox_xyxy = [
            618.0 + 50.0 * t,
            262.0 + 194.0 * t,
            739.0 + 165.0 * t,
            579.0 + 509.0 * t,
        ]
        entries.append(entry)
    tracklet = Tracklet(track_id=11, entries=entries)

    async def run_and_drain():
        await pipeline._process_tracklet(tracklet)
        await pipeline._drain_inflight_tasks(timeout_s=1.0)

    asyncio.run(run_and_drain())

    assert pipeline.track_id_to_person_id == {}
    assert 1 in pipeline.track_forbidden_person_ids[11]
    assert persisted_occlusion == {
        "track_id": 11,
        "reason": "identity_shift_risk",
    }


def test_current_identity_anchor_shift_risk_blocks_gradual_tracker_swap(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_id_to_person_id = {11: 1}
    pipeline.track_identity_memory = {
        11: {
            "person_id": 1,
            "anchor_frame_idx": 1762,
            "anchor_bbox_xyxy": [613.16, 249.94, 721.81, 531.66],
            "last_frame_idx": 1768,
            "last_bbox_xyxy": [611.90, 255.60, 735.41, 577.13],
        }
    }
    pipeline.settings.duplicate_merge_enabled = False
    persisted_occlusion = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return [entries[0], entries[-1]]

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {}

    class DummyMatcher:
        tentative = {}

        def match_tracklet(self, **kwargs):
            assert kwargs["current_person_id"] is None
            assert 1 in kwargs["forbidden_person_ids"]
            assert kwargs["allow_new_identity"] is False
            return None

        def pop_last_decision(self, track_id):
            return {"method": "unconfirmed", "source": "identity_shift_risk"}

    async def fake_persist_occlusion_candidate(tracklet, **kwargs):
        persisted_occlusion["track_id"] = tracklet.track_id
        persisted_occlusion["reason"] = kwargs.get("reason")

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline._persist_occlusion_candidate = fake_persist_occlusion_candidate
    pipeline._inflight = set()

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(
            overall=0.90,
            good_frame_ratio=0.80,
            bbox_size_stability=0.80,
            position_stability=0.85,
        ),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    entries = []
    for idx in range(12):
        t = idx / 11.0
        entry = _make_entry(1770 + idx * 3, v_score=0.92)
        entry.bbox_xyxy = [
            618.03 + (643.64 - 618.03) * t,
            262.29 + (340.47 - 262.29) * t,
            739.51 + (825.13 - 739.51) * t,
            579.26 + (833.34 - 579.26) * t,
        ]
        entries.append(entry)
    tracklet = Tracklet(track_id=11, entries=entries)

    async def run_and_drain():
        await pipeline._process_tracklet(tracklet)
        await pipeline._drain_inflight_tasks(timeout_s=1.0)

    asyncio.run(run_and_drain())

    assert pipeline.track_id_to_person_id == {}
    assert 1 in pipeline.track_forbidden_person_ids[11]
    assert persisted_occlusion == {
        "track_id": 11,
        "reason": "identity_shift_risk",
    }


def test_idle_flush_processes_ready_and_short_unmatched_tracklets(monkeypatch):
    pipeline = _make_pipeline()
    ready = Tracklet(track_id=401, entries=[_make_entry(i) for i in range(1, 5)])
    short = Tracklet(track_id=402, entries=[_make_entry(1), _make_entry(2)])
    pipeline.last_message_time_ns = int(1e9)
    pipeline.tracklet_buffer = SimpleNamespace(
        tracklets={401: ready, 402: short},
        remove=lambda track_id: pipeline.tracklet_buffer.tracklets.pop(track_id, None),
    )
    scheduled = []
    short_processed = []

    def fake_schedule(tracklet, *, reserved_person_ids, allow_tentative_fallback=True):
        scheduled.append((tracklet.track_id, reserved_person_ids))
        return True

    async def fake_short(tracklet, *, reason):
        short_processed.append((tracklet.track_id, reason))
        return None

    pipeline._schedule_tracklet_processing = fake_schedule
    pipeline._process_short_fragment_tracklet = fake_short

    asyncio.run(pipeline._flush_idle_tracklets_if_needed(int(3e9)))

    assert scheduled == [(401, set())]
    assert short_processed == [(402, "idle_flush_short_tracklet")]
    assert pipeline.tracklet_buffer.tracklets == {}


def test_process_tracklet_with_no_embeddings_returns_without_match_or_persist(monkeypatch):
    pipeline = _make_pipeline()
    removed_track_ids = []
    persist_called = {"value": False}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            raise RuntimeError("embedding failed")

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95,
                               "probabilities": {"male": 0.95, "female": 0.05}}}

    class DummyBuffer:
        def remove(self, track_id):
            removed_track_ids.append(track_id)

    async def fake_persist_tracklet(**kwargs):
        persist_called["value"] = True

    pipeline.topk_selector = DummySelector()
    pipeline.model_client = DummyModelClient()
    pipeline.tracklet_buffer = DummyBuffer()
    pipeline._persist_tracklet = fake_persist_tracklet

    monkeypatch.setattr("src.workers.main.cv2.imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)))

    tracklet = Tracklet(track_id=10, entries=[_make_entry(1), _make_entry(2)])

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert tracklet.state == TrackletState.ACTIVE
    assert tracklet.person_id is None
    assert pipeline.track_id_to_person_id == {}
    assert pipeline.track_metadata == {}
    assert removed_track_ids == []
    assert persist_called["value"] is False


def test_process_tracklet_mixed_embeddings_graceful_degrade_match_only(monkeypatch):
    # Consensus-failure now degrades gracefully: it still calls the matcher
    # with the top-v_score embedding but forces allow_new_identity=False so
    # noisy fragments can only match an existing gallery entry, never mint a
    # new person.
    pipeline = _make_pipeline()
    persist_called = {"value": False}
    match_kwargs: dict = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array(embeddings[0], dtype=np.float32)

    class DummyModelClient:
        calls = 0

        async def extract_features(self, img_bytes, model="osnet"):
            self.calls += 1
            if self.calls == 1:
                return None, {"embedding": [1.0, 0.0]}
            return None, {"embedding": [0.0, 1.0]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            match_kwargs.update(kwargs)
            return None  # no gallery match → tracklet stays unconfirmed

        def pop_last_decision(self, track_id):
            return None

    async def fake_persist_tracklet(**kwargs):
        persist_called["value"] = True

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline._persist_tracklet = fake_persist_tracklet

    monkeypatch.setattr("src.workers.main.cv2.imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)))
    monkeypatch.setattr("src.workers.main.compute_tracklet_consistency", lambda entries: SimpleNamespace(overall=0.5, good_frame_ratio=0.5))
    monkeypatch.setattr("src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency", lambda embeddings: 0.0)

    tracklet = Tracklet(track_id=10, entries=[_make_entry(1), _make_entry(2)])

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert match_kwargs.get("allow_new_identity") is False
    assert pipeline.track_id_to_person_id == {}
    assert tracklet.person_id is None
    assert persist_called["value"] is False


def test_process_tracklet_persists_only_consensus_selected_frames(monkeypatch):
    pipeline = _make_pipeline()
    persisted = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            assert len(embeddings) == 2
            return np.array([1.0, 0.0], dtype=np.float32)

    class DummyModelClient:
        calls = 0

        async def extract_features(self, img_bytes, model="osnet"):
            self.calls += 1
            if self.calls == 3:
                return None, {"embedding": [0.0, 1.0]}
            return None, {"embedding": [1.0, 0.0]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            return 505

    async def fake_persist_tracklet(**kwargs):
        persisted["kwargs"] = kwargs

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline._persist_tracklet = fake_persist_tracklet

    monkeypatch.setattr("src.workers.main.cv2.imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)))
    monkeypatch.setattr("src.workers.main.compute_tracklet_consistency", lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75))
    monkeypatch.setattr("src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency", lambda embeddings: 0.91)
    monkeypatch.setattr("src.workers.main.uuid.uuid4", lambda: "tracklet-consensus")

    tracklet = Tracklet(
        track_id=13,
        entries=[_make_entry(1), _make_entry(2), _make_entry(3, v_score=0.99)],
    )

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert [entry.frame_idx for entry in persisted["kwargs"]["selected"]] == [1, 2]
    assert persisted["kwargs"]["best_entry"].frame_idx in {1, 2}
    assert pipeline.track_id_to_person_id == {13: 505}


def test_process_tracklet_allocation_failure_removes_tracklet_and_skips_persist(monkeypatch):
    pipeline = _make_pipeline()
    removed_track_ids = []
    persist_called = {"value": False}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95,
                               "probabilities": {"male": 0.95, "female": 0.05}}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            raise PersonIdAllocationError("alloc failed")

    class DummyBuffer:
        def remove(self, track_id):
            removed_track_ids.append(track_id)

    async def fake_persist_tracklet(**kwargs):
        persist_called["value"] = True

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline.tracklet_buffer = DummyBuffer()
    pipeline._persist_tracklet = fake_persist_tracklet

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    tracklet = Tracklet(track_id=77, entries=[_make_entry(1), _make_entry(2, v_score=0.9)])

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert pipeline.track_id_to_person_id == {}
    assert pipeline.track_metadata == {}
    assert removed_track_ids == [77]
    assert persist_called["value"] is False


def test_update_temporal_exclusions_blocks_cooccurring_person_after_repeated_frames():
    pipeline = _make_pipeline()
    pipeline.track_id_to_person_id = {10: 501}

    visible_tracks = np.array([
        [10.0, 20.0, 50.0, 120.0, 10.0, 0.9],
        [90.0, 18.0, 130.0, 118.0, 11.0, 0.88],
    ], dtype=np.float32)

    pipeline._update_temporal_exclusions(visible_tracks)
    assert pipeline.track_forbidden_person_ids[11] == {501}


def test_update_temporal_exclusions_ignores_likely_duplicate_tracks():
    pipeline = _make_pipeline()
    pipeline.track_id_to_person_id = {10: 501}

    visible_tracks = np.array([
        [10.0, 20.0, 50.0, 120.0, 10.0, 0.9],
        [12.0, 22.0, 52.0, 122.0, 11.0, 0.88],
    ], dtype=np.float32)

    pipeline._update_temporal_exclusions(visible_tracks)
    pipeline._update_temporal_exclusions(visible_tracks)

    assert pipeline.track_forbidden_person_ids == {}


def test_process_tracklet_passes_forbidden_person_ids_to_matcher(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_forbidden_person_ids = {7: {303}}
    seen = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["kwargs"] = kwargs
            return None

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    tracklet = Tracklet(track_id=7, entries=[_make_entry(1), _make_entry(2, v_score=0.9)])
    asyncio.run(pipeline._process_tracklet(tracklet))

    assert seen["kwargs"]["forbidden_person_ids"] == {303}


def test_find_attribute_incompatible_person_ids_blocks_stable_gender_conflict():
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.95)})

    conflicts = pipeline._find_attribute_incompatible_person_ids(
        {"gender": ("female", 0.94)},
        current_person_id=None,
    )

    assert conflicts == {501}


def test_current_identity_attribute_conflict_blocks_track_continuity():
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.95)})

    assert pipeline._has_current_identity_attribute_conflict(
        {"gender": ("female", 0.94)},
        current_person_id=501,
    )
    assert not pipeline._has_current_identity_attribute_conflict(
        {"gender": ("female", 0.87)},
        current_person_id=501,
    )


def test_find_attribute_incompatible_person_ids_blocks_low_confidence_but_stable_gender_conflict():
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.76)})
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.77)})
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.75)})

    conflicts = pipeline._find_attribute_incompatible_person_ids(
        {"gender": ("female", 0.94)},
        current_person_id=None,
    )

    assert conflicts == {501}


def test_find_attribute_incompatible_person_ids_ignores_noise_floor_tracklet():
    """Tracklet below the noise floor (< 0.55) has no usable gender — never blocks."""
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.95)})

    conflicts = pipeline._find_attribute_incompatible_person_ids(
        {"gender": ("female", 0.40)},
        current_person_id=None,
    )

    assert conflicts == set()


def test_find_attribute_incompatible_person_ids_weak_tracklet_blocks_strong_person():
    """V23 L1: a weak-but-above-noise opposite-gender tracklet is still blocked
    when the person side has any established gender opinion."""
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.95)})

    conflicts = pipeline._find_attribute_incompatible_person_ids(
        {"gender": ("female", 0.65)},
        current_person_id=None,
    )

    assert conflicts == {501}


def test_singleton_weak_person_gender_does_not_block_stronger_tracklet():
    pipeline = _make_pipeline()
    pipeline.settings.attribute_conflict_person_confidence = 0.60
    pipeline.settings.attribute_conflict_person_min_support = 1
    pipeline.settings.attribute_conflict_tracklet_confidence = 0.70
    pipeline.attribute_voter = AttributeVoter(person_threshold=0.60)
    pipeline.attribute_voter.resolve_person(701, {"gender": ("female", 0.65)})

    tracklet_attrs = {"gender": ("male", 0.86)}

    assert not pipeline._has_current_identity_attribute_conflict(
        tracklet_attrs,
        current_person_id=701,
    )
    assert not pipeline._has_person_attribute_conflict(tracklet_attrs, 701)
    assert pipeline._find_attribute_incompatible_person_ids(
        tracklet_attrs,
        current_person_id=None,
    ) == set()


def test_repeated_weak_person_gender_blocks_opposite_tracklet():
    pipeline = _make_pipeline()
    pipeline.settings.attribute_conflict_person_confidence = 0.60
    pipeline.settings.attribute_conflict_person_min_support = 1
    pipeline.settings.attribute_conflict_tracklet_confidence = 0.70
    pipeline.attribute_voter = AttributeVoter(person_threshold=0.60)
    pipeline.attribute_voter.resolve_person(701, {"gender": ("female", 0.65)})
    pipeline.attribute_voter.resolve_person(701, {"gender": ("female", 0.66)})

    tracklet_attrs = {"gender": ("male", 0.86)}

    assert pipeline._has_current_identity_attribute_conflict(
        tracklet_attrs,
        current_person_id=701,
    )
    assert pipeline._has_person_attribute_conflict(tracklet_attrs, 701)
    assert pipeline._find_attribute_incompatible_person_ids(
        tracklet_attrs,
        current_person_id=None,
    ) == {701}


def test_process_tracklet_merges_temporal_and_attribute_forbidden_person_ids(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_forbidden_person_ids = {7: {303}}
    pipeline.attribute_voter.resolve_person(404, {"gender": ("male", 0.95)})
    seen = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "female", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["kwargs"] = kwargs
            return None

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    tracklet = Tracklet(track_id=7, entries=[_make_entry(1), _make_entry(2, v_score=0.9)])
    asyncio.run(pipeline._process_tracklet(tracklet))

    assert seen["kwargs"]["forbidden_person_ids"] == {303, 404}


def test_process_tracklet_skips_attribute_forbidden_for_occlusion_like_tracklet(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_forbidden_person_ids = {-9000008: {303}}
    pipeline.attribute_voter.resolve_person(404, {"gender": ("male", 0.95)})
    seen = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "female", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["kwargs"] = kwargs
            return None

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    tracklet = Tracklet(track_id=-9000008, entries=[
        _make_entry(1, v_score=0.9),
        _make_entry(2, v_score=0.9),
    ])
    asyncio.run(pipeline._process_tracklet(tracklet))

    assert seen["kwargs"]["forbidden_person_ids"] == {303}


def test_process_tracklet_rejects_current_person_on_strong_attribute_conflict(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_id_to_person_id = {7: 501}
    pipeline.attribute_voter.resolve_person(501, {"gender": ("male", 0.95)})
    seen = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "female", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["kwargs"] = kwargs
            return None

        def pop_last_decision(self, track_id):
            return None

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    tracklet = Tracklet(track_id=7, entries=[_make_entry(1), _make_entry(2, v_score=0.9)])
    asyncio.run(pipeline._process_tracklet(tracklet))

    assert seen["kwargs"]["current_person_id"] is None
    assert seen["kwargs"]["forbidden_person_ids"] == {501}
    assert pipeline.track_id_to_person_id == {}
    assert pipeline.track_forbidden_person_ids[7] == {501}


def test_occlusion_provisional_match_accepts_near_gallery_fragment():
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(5, {"gender": ("female", 0.91)})
    tracklet = Tracklet(track_id=-9000008, entries=[
        _make_entry(102, v_score=0.50, overlap_ratio=0.45),
        _make_entry(103, v_score=0.52, overlap_ratio=0.40),
    ])

    person_id, matching = pipeline._maybe_accept_occlusion_provisional_match(
        tracklet=tracklet,
        matching={
            "method": "near_gallery_deferred",
            "source": "tentative_promoted",
            "reuse_person_id": 5,
            "similarity_score": 0.67,
            "runner_up_score": 0.60,
        },
        v_avg=0.51,
        tracklet_attrs={"gender": ("female", 0.86)},
        forbidden_person_ids=set(),
        recent_incompatible_person_ids=set(),
        blocked_person_ids=set(),
    )

    assert person_id == 5
    assert matching["method"] == "occlusion_provisional_match"
    assert matching["provisional"] is True
    assert matching["canonical_update_applied"] is False


def test_occlusion_provisional_match_accepts_short_positive_reentry_fragment():
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(12, {"gender": ("male", 0.91)})
    pipeline.person_last_observation[12] = {
        "bbox_xyxy": [100.0, 50.0, 160.0, 190.0],
        "timestamp_ns": 1202,
        "device_id": "cam-1",
        "frame_idx": 1202,
    }
    tracklet = Tracklet(track_id=90, entries=[
        _make_entry(1290, v_score=0.94, overlap_ratio=0.05),
        _make_entry(1292, v_score=0.95, overlap_ratio=0.05),
        _make_entry(1294, v_score=0.94, overlap_ratio=0.05),
        _make_entry(1296, v_score=0.93, overlap_ratio=0.05),
    ])
    for entry in tracklet.entries:
        entry.bbox_xyxy = [110.0, 55.0, 170.0, 195.0]

    person_id, matching = pipeline._maybe_accept_occlusion_provisional_match(
        tracklet=tracklet,
        matching={
            "method": "near_gallery_deferred",
            "reuse_person_id": 12,
            "similarity_score": 0.61,
        },
        v_avg=0.94,
        tracklet_attrs={"gender": ("male", 0.86)},
        forbidden_person_ids=set(),
        recent_incompatible_person_ids=set(),
        blocked_person_ids=set(),
    )

    assert person_id == 12
    assert matching["method"] == "occlusion_provisional_match"
    assert matching["provisional"] is True


def test_occlusion_provisional_match_rejects_short_reentry_when_spatially_far():
    pipeline = _make_pipeline()
    pipeline.person_last_observation[12] = {
        "bbox_xyxy": [100.0, 50.0, 160.0, 190.0],
        "timestamp_ns": 1202,
        "device_id": "cam-1",
        "frame_idx": 1202,
    }
    tracklet = Tracklet(track_id=90, entries=[
        _make_entry(1290, v_score=0.94, overlap_ratio=0.05),
        _make_entry(1292, v_score=0.95, overlap_ratio=0.05),
        _make_entry(1294, v_score=0.94, overlap_ratio=0.05),
        _make_entry(1296, v_score=0.93, overlap_ratio=0.05),
    ])
    for entry in tracklet.entries:
        entry.bbox_xyxy = [420.0, 55.0, 480.0, 195.0]

    person_id, matching = pipeline._maybe_accept_occlusion_provisional_match(
        tracklet=tracklet,
        matching={
            "method": "near_gallery_deferred",
            "reuse_person_id": 12,
            "similarity_score": 0.61,
        },
        v_avg=0.94,
        tracklet_attrs={"gender": ("unknown", 0.0)},
        forbidden_person_ids=set(),
        recent_incompatible_person_ids=set(),
        blocked_person_ids=set(),
    )

    assert person_id is None
    assert matching["method"] == "near_gallery_deferred"


def test_occlusion_provisional_match_rejects_attribute_conflict():
    pipeline = _make_pipeline()
    pipeline.attribute_voter.resolve_person(4, {"gender": ("male", 0.92)})
    tracklet = Tracklet(track_id=-9000008, entries=[
        _make_entry(102, v_score=0.50, overlap_ratio=0.45),
        _make_entry(103, v_score=0.52, overlap_ratio=0.40),
    ])

    person_id, matching = pipeline._maybe_accept_occlusion_provisional_match(
        tracklet=tracklet,
        matching={
            "method": "near_gallery_deferred",
            "source": "tentative_promoted",
            "reuse_person_id": 4,
            "similarity_score": 0.70,
        },
        v_avg=0.51,
        tracklet_attrs={"gender": ("female", 0.89)},
        forbidden_person_ids=set(),
        recent_incompatible_person_ids=set(),
        blocked_person_ids=set(),
    )

    assert person_id is None
    assert matching["method"] == "near_gallery_deferred"


def test_occlusion_like_tracklet_attributes_do_not_register_new_person(monkeypatch):
    pipeline = _make_pipeline()
    registered = []

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "female", "confidence": 0.91}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            kwargs["on_new_identity"](8)
            return 8

        def pop_last_decision(self, track_id):
            return {"method": "new_identity", "source": "new_detection"}

    class DummyVoter(AttributeVoter):
        def resolve_person(self, person_id, tracklet_attrs):
            registered.append((person_id, tracklet_attrs))
            return super().resolve_person(person_id, tracklet_attrs)

    async def fake_persist_tracklet(**kwargs):
        return None

    async def fake_merge_duplicate_person(person_id):
        return person_id

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline.attribute_voter = DummyVoter(person_threshold=0.7)
    pipeline._persist_tracklet = fake_persist_tracklet
    pipeline._maybe_merge_duplicate_person = fake_merge_duplicate_person

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )
    monkeypatch.setattr("src.workers.main.uuid.uuid4", lambda: "tracklet-occluded")

    tracklet = Tracklet(track_id=-9000008, entries=[
        _make_entry(276, v_score=0.82, overlap_ratio=0.1),
        _make_entry(278, v_score=0.80, overlap_ratio=0.1),
    ])
    asyncio.run(pipeline._process_tracklet(tracklet))

    assert registered == []
    assert pipeline.track_metadata[-9000008]["attributes"] == {}


def test_process_tracklet_suppresses_new_identity_for_tiny_static_track(monkeypatch):
    pipeline = _make_pipeline()
    seen = {"called": False}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries[:3]

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {"gender": {"label": "male", "confidence": 0.95}}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["called"] = True
            seen["kwargs"] = kwargs
            return None

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(
            overall=0.99,
            good_frame_ratio=1.0,
            bbox_size_stability=0.99,
            position_stability=0.99,
        ),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.99,
    )

    tracklet = Tracklet(
        track_id=88,
        entries=[
            TrackletEntry(
                frame_idx=1,
                crop=np.ones((100, 40, 3), dtype=np.uint8),
                v_score=0.8,
                bbox_xyxy=[10.0, 10.0, 50.0, 110.0],
                timestamp_ns=int(1e9),
            ),
            TrackletEntry(
                frame_idx=2,
                crop=np.ones((100, 40, 3), dtype=np.uint8),
                v_score=0.82,
                bbox_xyxy=[10.1, 10.0, 50.1, 110.0],
                timestamp_ns=int(2e9),
            ),
            TrackletEntry(
                frame_idx=3,
                crop=np.ones((100, 40, 3), dtype=np.uint8),
                v_score=0.84,
                bbox_xyxy=[10.0, 10.1, 50.0, 110.1],
                timestamp_ns=int(3e9),
            ),
            TrackletEntry(
                frame_idx=4,
                crop=np.ones((100, 40, 3), dtype=np.uint8),
                v_score=0.85,
                bbox_xyxy=[10.1, 10.0, 50.1, 110.0],
                timestamp_ns=int(4e9),
            ),
            TrackletEntry(
                frame_idx=5,
                crop=np.ones((100, 40, 3), dtype=np.uint8),
                v_score=0.85,
                bbox_xyxy=[10.0, 10.1, 50.0, 110.1],
                timestamp_ns=int(5e9),
            ),
            TrackletEntry(
                frame_idx=6,
                crop=np.ones((100, 40, 3), dtype=np.uint8),
                v_score=0.85,
                bbox_xyxy=[10.1, 10.0, 50.1, 110.0],
                timestamp_ns=int(6e9),
            ),
        ],
    )

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert seen["called"] is False


def test_process_tracklet_suppresses_new_identity_when_identity_cap_reached(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.settings.max_person_identities = 2
    pipeline.track_id_to_person_id = {1: 101, 2: 202}
    seen = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["kwargs"] = kwargs
            return None

        def pop_last_decision(self, track_id):
            return {"method": "new_identity_suppressed"}

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()

    tracklet = Tracklet(track_id=7)
    tracklet.entries = [
        TrackletEntry(
            frame_idx=i,
            crop=np.ones((400, 160, 3), dtype=np.uint8),
            v_score=0.9,
            bbox_xyxy=[20.0 + (i * 8.0), 30.0, 180.0 + (i * 8.0), 430.0],
            timestamp_ns=i * int(1e9),
            overlap_ratio=0.1,
        )
        for i in range(1, 7)
    ]

    asyncio.run(pipeline._process_tracklet(tracklet))

    assert seen["kwargs"]["allow_new_identity"] is False


def test_process_tracklet_disables_new_identity_for_existing_track_id(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_id_to_person_id = {7: 101}
    seen = {}
    persisted_occlusion = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["kwargs"] = kwargs
            return None

        def pop_last_decision(self, track_id):
            return {"method": "new_identity_suppressed", "source": "current_track_deferred"}

    async def fake_persist_occlusion_candidate(tracklet, **kwargs):
        persisted_occlusion["track_id"] = tracklet.track_id
        persisted_occlusion["reason"] = kwargs.get("reason")

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline._persist_occlusion_candidate = fake_persist_occlusion_candidate

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    tracklet = Tracklet(
        track_id=7,
        entries=[
            _make_entry(1, v_score=0.72, overlap_ratio=0.22),
            _make_entry(2, v_score=0.70, overlap_ratio=0.24),
            _make_entry(3, v_score=0.68, overlap_ratio=0.26),
            _make_entry(4, v_score=0.66, overlap_ratio=0.28),
        ],
    )

    async def run_and_drain():
        await pipeline._process_tracklet(tracklet)
        await pipeline._drain_inflight_tasks(timeout_s=1.0)

    asyncio.run(run_and_drain())

    assert seen["kwargs"]["current_person_id"] == 101
    assert seen["kwargs"]["allow_new_identity"] is False
    assert pipeline.track_id_to_person_id == {7: 101}
    assert tracklet.person_id is None
    assert tracklet.state == TrackletState.ACTIVE
    assert persisted_occlusion == {
        "track_id": 7,
        "reason": "current_track_deferred",
    }


def test_process_tracklet_restores_current_identity_from_frame_memory(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.track_identity_memory = {
        7: {
            "person_id": 101,
            "last_frame_idx": 10,
            "last_bbox_xyxy": [10.0, 20.0, 50.0, 120.0],
        }
    }
    seen = {}
    persisted_occlusion = {}

    class DummySelector:
        def is_tracklet_ready(self, entries):
            return True

        def select(self, entries):
            return entries

    class DummyAggregator:
        def aggregate(self, embeddings, v_scores, overlap_ratios):
            return np.array([0.6, 0.8], dtype=np.float32)

    class DummyModelClient:
        async def extract_features(self, img_bytes, model="osnet"):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_attributes(self, img_bytes):
            return {}

    class DummyMatcher:
        def match_tracklet(self, **kwargs):
            seen["kwargs"] = kwargs
            return None

        def pop_last_decision(self, track_id):
            return {"method": "new_identity_suppressed", "source": "memory_restored"}

    async def fake_persist_occlusion_candidate(tracklet, **kwargs):
        persisted_occlusion["track_id"] = tracklet.track_id
        persisted_occlusion["reason"] = kwargs.get("reason")

    pipeline.topk_selector = DummySelector()
    pipeline.aggregator = DummyAggregator()
    pipeline.model_client = DummyModelClient()
    pipeline.matcher = DummyMatcher()
    pipeline._persist_occlusion_candidate = fake_persist_occlusion_candidate

    monkeypatch.setattr(
        "src.workers.main.cv2.imencode",
        lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "src.workers.main.compute_tracklet_consistency",
        lambda entries: SimpleNamespace(overall=0.88, good_frame_ratio=0.75),
    )
    monkeypatch.setattr(
        "src.workers.main.WeightedEmbeddingAggregator.compute_embedding_consistency",
        lambda embeddings: 0.91,
    )

    tracklet = Tracklet(
        track_id=7,
        entries=[
            _make_entry(12, v_score=0.72, overlap_ratio=0.22),
            _make_entry(14, v_score=0.70, overlap_ratio=0.24),
            _make_entry(16, v_score=0.68, overlap_ratio=0.26),
            _make_entry(18, v_score=0.66, overlap_ratio=0.28),
        ],
    )

    async def run_and_drain():
        await pipeline._process_tracklet(tracklet)
        await pipeline._drain_inflight_tasks(timeout_s=1.0)

    asyncio.run(run_and_drain())

    assert seen["kwargs"]["current_person_id"] == 101
    assert seen["kwargs"]["allow_new_identity"] is False
    assert pipeline.track_id_to_person_id == {7: 101}
    assert persisted_occlusion == {
        "track_id": 7,
        "reason": "memory_restored",
    }
