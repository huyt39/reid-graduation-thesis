from types import SimpleNamespace

import asyncio
import numpy as np

from src.tracklet.buffer import TrackletBuffer
from src.tracklet.models import TrackletEntry
from src.workers.main import WorkerPipeline


def _make_pipeline(stale_seconds: float = 5.0) -> WorkerPipeline:
    pipeline = WorkerPipeline.__new__(WorkerPipeline)
    pipeline.settings = SimpleNamespace(tracklet_stale_seconds=stale_seconds)
    pipeline.prev_bboxes = {1: ["bbox-1"], 2: ["bbox-2"]}
    pipeline.track_id_to_person_id = {1: 101, 2: 202}
    pipeline.track_metadata = {
        1: {"tracklet_id": "tracklet-1"},
        2: {"tracklet_id": "tracklet-2"},
    }
    pipeline.track_last_seen_ns = {
        1: int(1e9),
        2: int(4e9),
    }
    pipeline.person_last_observation = {}
    pipeline.current_track_metrics = {}
    pipeline.track_forbidden_person_ids = {}
    pipeline.track_cooccurrence_counts = {}
    pipeline.processed_messages = 0
    pipeline.ready_tracklets = 0
    pipeline.embedded_tracklets = 0
    pipeline.matched_tracklets = 0
    pipeline.worker_started_at = 0.0
    return pipeline


def test_cleanup_inactive_tracks_keeps_recent_tracks():
    pipeline = _make_pipeline(stale_seconds=5.0)

    pipeline._cleanup_inactive_tracks(current_time_ns=int(5.5e9))

    assert pipeline.prev_bboxes == {1: ["bbox-1"], 2: ["bbox-2"]}
    assert pipeline.track_id_to_person_id == {1: 101, 2: 202}
    assert pipeline.track_metadata == {
        1: {"tracklet_id": "tracklet-1"},
        2: {"tracklet_id": "tracklet-2"},
    }
    assert pipeline.track_last_seen_ns == {
        1: int(1e9),
        2: int(4e9),
    }
def test_cleanup_inactive_tracks_removes_only_stale_tracks():
    pipeline = _make_pipeline(stale_seconds=5.0)

    pipeline._cleanup_inactive_tracks(current_time_ns=int(6.5e9))

    assert pipeline.prev_bboxes == {2: ["bbox-2"]}
    assert pipeline.track_id_to_person_id == {2: 202}
    assert pipeline.track_metadata == {2: {"tracklet_id": "tracklet-2"}}
    assert pipeline.track_last_seen_ns == {2: int(4e9)}
def test_process_message_with_empty_track_results_keeps_recent_state(monkeypatch):
    pipeline = WorkerPipeline.__new__(WorkerPipeline)
    pipeline.settings = SimpleNamespace(
        tracklet_stale_seconds=5.0,
        untracked_detection_candidates_enabled=False,
    )
    pipeline.prev_bboxes = {1: [np.array([0.0, 0.0, 10.0, 10.0])]}
    pipeline.track_id_to_person_id = {1: 101}
    pipeline.track_metadata = {1: {"tracklet_id": "tracklet-1"}}
    pipeline.track_last_seen_ns = {1: int(1e9)}
    pipeline.person_last_observation = {}
    pipeline.current_track_metrics = {}
    pipeline.track_forbidden_person_ids = {}
    pipeline.track_cooccurrence_counts = {}
    pipeline._current_device_id = ""
    pipeline.processed_messages = 0
    pipeline.ready_tracklets = 0
    pipeline.embedded_tracklets = 0
    pipeline.matched_tracklets = 0
    pipeline.worker_started_at = 0.0

    class DummyTracker:
        def update(self, *args, **kwargs):
            return []

    pipeline.tracker = DummyTracker()

    monkeypatch.setattr("src.workers.main.time.time_ns", lambda: int(5.5e9))

    msg = {
        "device_id": "cam-1",
        "frame_number": 1,
        "detections": [
            {
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "confidence": 0.9,
                "class_id": 0,
                "visibility_score": 0.8,
            }
        ],
        "image_data": b"not-a-real-image",
        "created_at": 123456789,
    }

    monkeypatch.setattr("src.workers.main.cv2.imdecode", lambda *args, **kwargs: np.zeros((32, 32, 3), dtype=np.uint8))

    asyncio.run(pipeline._process_message(msg))

    assert 1 in pipeline.prev_bboxes
    assert pipeline.track_id_to_person_id == {1: 101}
    assert pipeline.track_metadata == {1: {"tracklet_id": "tracklet-1"}}
    assert pipeline.track_last_seen_ns == {1: int(1e9)}
def test_process_message_with_empty_track_results_removes_stale_state(monkeypatch):
    pipeline = WorkerPipeline.__new__(WorkerPipeline)
    pipeline.settings = SimpleNamespace(
        tracklet_stale_seconds=5.0,
        untracked_detection_candidates_enabled=False,
    )
    pipeline.prev_bboxes = {1: [np.array([0.0, 0.0, 10.0, 10.0])]}
    pipeline.track_id_to_person_id = {1: 101}
    pipeline.track_metadata = {1: {"tracklet_id": "tracklet-1"}}
    pipeline.track_last_seen_ns = {1: int(1e9)}
    pipeline.person_last_observation = {}
    pipeline.current_track_metrics = {}
    pipeline.track_forbidden_person_ids = {}
    pipeline.track_cooccurrence_counts = {}
    pipeline._current_device_id = ""
    pipeline.processed_messages = 0
    pipeline.ready_tracklets = 0
    pipeline.embedded_tracklets = 0
    pipeline.matched_tracklets = 0
    pipeline.worker_started_at = 0.0

    class DummyTracker:
        def update(self, *args, **kwargs):
            return []

    pipeline.tracker = DummyTracker()

    monkeypatch.setattr("src.workers.main.time.time_ns", lambda: int(6.5e9))
    monkeypatch.setattr(
        "src.workers.main.cv2.imdecode",
        lambda *args, **kwargs: np.zeros((32, 32, 3), dtype=np.uint8),
    )

    msg = {
        "device_id": "cam-1",
        "frame_number": 1,
        "detections": [
            {
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "confidence": 0.9,
                "class_id": 0,
                "visibility_score": 0.8,
            }
        ],
        "image_data": b"not-a-real-image",
        "created_at": 123456789,
    }

    asyncio.run(pipeline._process_message(msg))

    assert pipeline.prev_bboxes == {}
    assert pipeline.track_id_to_person_id == {}
    assert pipeline.track_metadata == {}
    assert pipeline.track_last_seen_ns == {}


def test_end_of_stream_sentinel_is_detected():
    assert WorkerPipeline._is_end_of_stream_message(
        {
            "device_id": "cam-1",
            "frame_number": -1,
            "detections": [],
            "image_data": b"",
        }
    )
    assert not WorkerPipeline._is_end_of_stream_message(
        {
            "device_id": "cam-1",
            "frame_number": 10,
            "detections": [],
            "image_data": b"",
        }
    )


def test_finalize_stream_flushes_buffered_tracklets(monkeypatch):
    pipeline = WorkerPipeline.__new__(WorkerPipeline)
    pipeline.settings = SimpleNamespace(
        tracklet_min_entries=4,
        stream_finalization_timeout_seconds=1.0,
        track_high_thresh=0.7,
        track_low_thresh=0.35,
        match_thresh=0.3,
        new_track_thresh=0.65,
        track_buffer=30,
        fuse_score=True,
    )
    pipeline.tracklet_buffer = TrackletBuffer(min_entries=4)
    pipeline.processing_tracklet_ids = set()
    pipeline._inflight = set()
    pipeline._stream_finalizing = False
    pipeline.untracked_detection_clusters = [{"cluster_id": -1}]
    pipeline.fragment_recovery_clusters = [{"fragments": 1}]
    pipeline.prev_bboxes = {7: []}
    pipeline.track_id_to_person_id = {7: 101}
    pipeline.track_metadata = {7: {"tracklet_id": "old"}}
    pipeline.track_last_seen_ns = {7: 1}
    pipeline.person_last_observation = {101: {"timestamp_ns": 1}}
    pipeline.current_track_metrics = {7: {"live_visibility_score": 0.8}}
    pipeline.track_forbidden_person_ids = {7: {101}}
    pipeline.track_cooccurrence_counts = {7: {101: 1}}
    pipeline.occlusion_candidate_track_ids = {7}
    pipeline._tracklet_embedding_cache = {7: []}
    pipeline._track_id_split_counts = {7: 1}
    pipeline._tracklet_gate_last_check_frame = {7: 1}

    entry = TrackletEntry(
        frame_idx=1,
        crop=np.zeros((8, 8, 3), dtype=np.uint8),
        v_score=0.8,
        bbox_xyxy=[0.0, 0.0, 8.0, 8.0],
        timestamp_ns=1,
    )
    for idx in range(4):
        pipeline.tracklet_buffer.append(7, entry.__class__(
            frame_idx=idx + 1,
            crop=entry.crop,
            v_score=entry.v_score,
            bbox_xyxy=entry.bbox_xyxy,
            timestamp_ns=idx + 1,
        ))

    scheduled = []

    def fake_schedule(tracklet, *, reserved_person_ids, allow_tentative_fallback=True):
        scheduled.append((tracklet.track_id, allow_tentative_fallback))
        return True

    monkeypatch.setattr(pipeline, "_schedule_tracklet_processing", fake_schedule)

    asyncio.run(pipeline._finalize_stream(device_id="cam-1"))

    assert scheduled == [(7, True)]
    assert pipeline.tracklet_buffer.tracklets == {}
    assert pipeline.untracked_detection_clusters == []
    assert pipeline.fragment_recovery_clusters == []
    assert pipeline.track_id_to_person_id == {}
    assert pipeline.person_last_observation == {}
    assert pipeline._stream_finalizing is False


def test_finalize_stream_drains_processing_before_flushing_followup_window(monkeypatch):
    pipeline = WorkerPipeline.__new__(WorkerPipeline)
    pipeline.settings = SimpleNamespace(
        tracklet_min_entries=4,
        stream_finalization_timeout_seconds=1.0,
        track_high_thresh=0.7,
        track_low_thresh=0.35,
        match_thresh=0.3,
        new_track_thresh=0.65,
        track_buffer=30,
        fuse_score=True,
    )
    pipeline.tracklet_buffer = TrackletBuffer(min_entries=4)
    pipeline.processing_tracklet_ids = {7}
    pipeline._inflight = set()
    pipeline._stream_finalizing = False
    pipeline.untracked_detection_clusters = []
    pipeline.fragment_recovery_clusters = []
    pipeline.prev_bboxes = {}
    pipeline.track_id_to_person_id = {}
    pipeline.track_metadata = {}
    pipeline.track_last_seen_ns = {}
    pipeline.person_last_observation = {}
    pipeline.current_track_metrics = {}
    pipeline.track_forbidden_person_ids = {}
    pipeline.track_cooccurrence_counts = {}
    pipeline.occlusion_candidate_track_ids = set()
    pipeline._tracklet_embedding_cache = {}
    pipeline._track_id_split_counts = {}
    pipeline._tracklet_gate_last_check_frame = {}

    for idx in range(4):
        pipeline.tracklet_buffer.append(
            7,
            TrackletEntry(
                frame_idx=idx + 10,
                crop=np.zeros((8, 8, 3), dtype=np.uint8),
                v_score=0.8,
                bbox_xyxy=[0.0, 0.0, 8.0, 8.0],
                timestamp_ns=idx + 10,
            ),
        )

    scheduled = []

    def fake_schedule(tracklet, *, reserved_person_ids, allow_tentative_fallback=True):
        scheduled.append((tracklet.track_id, [entry.frame_idx for entry in tracklet.entries]))
        return True

    async def release_processing():
        await asyncio.sleep(0)
        pipeline.processing_tracklet_ids.discard(7)

    async def run_finalize():
        pipeline._track_inflight(asyncio.create_task(release_processing()))
        monkeypatch.setattr(pipeline, "_schedule_tracklet_processing", fake_schedule)
        await pipeline._finalize_stream(device_id="cam-1")

    asyncio.run(run_finalize())

    assert scheduled == [(7, [10, 11, 12, 13])]
    assert pipeline.tracklet_buffer.tracklets == {}
    assert pipeline.processing_tracklet_ids == set()


def test_finalize_stream_drains_inflight_even_when_buffer_is_empty(monkeypatch):
    pipeline = WorkerPipeline.__new__(WorkerPipeline)
    pipeline.settings = SimpleNamespace(
        tracklet_min_entries=4,
        stream_finalization_timeout_seconds=1.0,
        final_reconciler_passes=1,
        background_reconciler_max_persons=50,
        track_high_thresh=0.7,
        track_low_thresh=0.35,
        match_thresh=0.3,
        new_track_thresh=0.65,
        track_buffer=30,
        fuse_score=True,
    )
    pipeline.tracklet_buffer = TrackletBuffer(min_entries=4)
    pipeline.processing_tracklet_ids = set()
    pipeline._inflight = {"pending"}
    pipeline._stream_finalizing = False
    pipeline.untracked_detection_clusters = []
    pipeline.fragment_recovery_clusters = []
    pipeline.prev_bboxes = {}
    pipeline.track_id_to_person_id = {}
    pipeline.track_metadata = {}
    pipeline.track_last_seen_ns = {}
    pipeline.person_last_observation = {}
    pipeline.current_track_metrics = {}
    pipeline.track_forbidden_person_ids = {}
    pipeline.track_cooccurrence_counts = {}
    pipeline.occlusion_candidate_track_ids = set()
    pipeline._tracklet_embedding_cache = {}
    pipeline._track_id_split_counts = {}
    pipeline._tracklet_gate_last_check_frame = {}
    calls = []

    async def fake_drain(*, timeout_s):
        calls.append("drain")
        pipeline._inflight.clear()
        return True

    async def fake_reconcile(*, max_persons, passes, reason):
        calls.append("reconcile")

    monkeypatch.setattr(pipeline, "_drain_inflight_tasks", fake_drain)
    monkeypatch.setattr(pipeline, "_reconcile_recent_persons", fake_reconcile)

    asyncio.run(pipeline._finalize_stream(device_id="cam-1"))

    assert calls == ["drain", "reconcile"]
    assert pipeline._inflight == set()
