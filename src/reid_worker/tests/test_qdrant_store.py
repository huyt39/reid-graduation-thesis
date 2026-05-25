import numpy as np

from src.matching.qdrant_store import QdrantPersonStore


class _Hit:
    def __init__(self, hit_id, score, payload):
        self.id = hit_id
        self.score = score
        self.payload = payload


def test_gated_momentum_update_skips_gallery_pollution_when_no_gallery_match(monkeypatch):
    store = QdrantPersonStore.__new__(QdrantPersonStore)
    calls = {"upsert": 0, "prune": 0}

    # Gate now checks best-of-gallery, not the frozen anchor.
    monkeypatch.setattr(store, "search_person", lambda person_id, embedding, min_score=0.0: 0.65)
    monkeypatch.setattr(store, "_upsert_gallery_point", lambda *args, **kwargs: calls.__setitem__("upsert", calls["upsert"] + 1))
    monkeypatch.setattr(store, "_prune_gallery", lambda *args, **kwargs: calls.__setitem__("prune", calls["prune"] + 1))

    updated = store.gated_momentum_update(
        person_id=7,
        new_embedding=np.ones(4, dtype=np.float32),
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        update_anchor_min_score=0.72,
    )

    assert updated is False
    assert calls["upsert"] == 0
    assert calls["prune"] == 0


def test_gated_momentum_update_allows_gallery_compatible_update(monkeypatch):
    store = QdrantPersonStore.__new__(QdrantPersonStore)
    calls = {"upsert": 0, "prune": 0}

    monkeypatch.setattr(store, "search_person", lambda person_id, embedding, min_score=0.0: 0.84)
    monkeypatch.setattr(store, "_upsert_gallery_point", lambda *args, **kwargs: calls.__setitem__("upsert", calls["upsert"] + 1))
    monkeypatch.setattr(store, "_prune_gallery", lambda *args, **kwargs: calls.__setitem__("prune", calls["prune"] + 1))

    updated = store.gated_momentum_update(
        person_id=7,
        new_embedding=np.ones(4, dtype=np.float32),
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        update_anchor_min_score=0.72,
    )

    assert updated is True
    assert calls["upsert"] == 1
    assert calls["prune"] == 1


def test_search_uses_consensus_of_top_two_gallery_points(monkeypatch):
    store = QdrantPersonStore.__new__(QdrantPersonStore)
    store.similarity_threshold = 0.70
    store.max_gallery_size = 8
    store.consensus_weight = 0.5

    class DummyClient:
        def search(self, **kwargs):
            return [
                _Hit("a1", 0.96, {"person_id": 1}),
                _Hit("a2", 0.44, {"person_id": 1}),
                _Hit("b1", 0.88, {"person_id": 2}),
                _Hit("b2", 0.86, {"person_id": 2}),
            ]

    store.client = DummyClient()
    monkeypatch.setattr(store, "_log_below_threshold", lambda *args, **kwargs: None)

    out = store.search(np.ones(4, dtype=np.float32), top_k=5)

    assert out[0][0] == 2
    assert round(out[0][1], 4) == 0.87
    assert out[1][0] == 1
    assert round(out[1][1], 4) == 0.70
