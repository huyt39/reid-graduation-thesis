from types import SimpleNamespace

import asyncio
import numpy as np

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
    pipeline.settings = SimpleNamespace(tracklet_stale_seconds=5.0)
    pipeline.prev_bboxes = {1: [np.array([0.0, 0.0, 10.0, 10.0])]}
    pipeline.track_id_to_person_id = {1: 101}
    pipeline.track_metadata = {1: {"tracklet_id": "tracklet-1"}}
    pipeline.track_last_seen_ns = {1: int(1e9)}
    pipeline._current_device_id = ""

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
