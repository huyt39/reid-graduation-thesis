import asyncio
from types import SimpleNamespace

import numpy as np

from src.attributes.gender_voter import GenderVoter
from src.tracklet.models import Tracklet, TrackletEntry, TrackletState
from src.workers.main import WorkerPipeline


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
    pipeline.settings = SimpleNamespace()
    pipeline.track_id_to_person_id = {}
    pipeline.track_metadata = {}
    pipeline.track_last_seen_ns = {}
    pipeline.prev_bboxes = {}
    pipeline._current_device_id = "cam-1"
    pipeline.gender_voter = GenderVoter(person_threshold=0.7)
    return pipeline


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
        async def extract_features(self, img_bytes):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_gender(self, img_bytes):
            return {"gender": "male", "confidence": 0.95}

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
    assert removed_track_ids == [7]
    assert persisted["matcher_kwargs"]["track_id"] == 7
    assert persisted["persist_kwargs"]["tracklet_id"] == "tracklet-123"
    assert persisted["persist_kwargs"]["person_id"] == 101
    assert persisted["persist_kwargs"]["gender"] == "male"


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
        async def extract_features(self, img_bytes):
            return None, {"embedding": [0.6, 0.8]}

        async def classify_gender(self, img_bytes):
            return {"gender": "male", "confidence": 0.95}

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
    assert removed_track_ids == [8]
    assert persist_called["value"] is False


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

    assert tracklet.state == TrackletState.TENTATIVE
    assert tracklet.person_id is None
    assert pipeline.track_id_to_person_id == {}
    assert pipeline.track_metadata == {}
    assert removed_track_ids == []


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
        async def extract_features(self, img_bytes):
            raise RuntimeError("embedding failed")

        async def classify_gender(self, img_bytes):
            return {"gender": "male", "confidence": 0.95}

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
