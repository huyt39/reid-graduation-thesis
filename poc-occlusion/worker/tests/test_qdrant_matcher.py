import numpy as np
import pytest
from src.matching.reid_matcher import ReIDMatcher


class MockQdrantStore:
    def __init__(self):
        self.persons = {}
        self._next_id = 1

    def search(self, embedding, top_k=5):
        return []  # No matches by default

    def add_person(self, person_id, embedding, metadata):
        self.persons[person_id] = {"embedding": embedding, "metadata": metadata}

    def get_embedding(self, person_id):
        if person_id in self.persons:
            return self.persons[person_id]["embedding"]
        return None

    def gated_momentum_update(self, person_id, new_embedding, v_avg,
                               embedding_consistency, tracklet_len,
                               update_v_threshold=0.6, update_consistency_threshold=0.7,
                               update_min_tracklet_len=5, update_sim_threshold=0.5):
        # Simplified: just check if gates pass
        if v_avg < update_v_threshold:
            return False
        if embedding_consistency < update_consistency_threshold:
            return False
        if tracklet_len < update_min_tracklet_len:
            return False
        return True


class TestReIDMatcher:
    def test_good_tracklet_no_match_creates_person(self):
        """Good v_avg + good consistency → new person created."""
        store = MockQdrantStore()
        matcher = ReIDMatcher(store, promote_v_threshold=0.6, promote_consistency_threshold=0.7)
        emb = np.random.randn(512).astype(np.float32)
        pid = matcher.match_tracklet(
            track_id=1, embedding=emb, v_avg=0.8,
            embedding_consistency=0.9, tracklet_len=10,
        )
        assert pid == 1
        assert 1 in store.persons

    def test_bad_tracklet_goes_tentative(self):
        """Low v_avg → tentative, not created."""
        store = MockQdrantStore()
        matcher = ReIDMatcher(store, promote_v_threshold=0.6, promote_consistency_threshold=0.7)
        emb = np.random.randn(512).astype(np.float32)
        pid = matcher.match_tracklet(
            track_id=1, embedding=emb, v_avg=0.3,
            embedding_consistency=0.9, tracklet_len=10,
        )
        assert pid is None
        assert 1 in matcher.tentative

    def test_low_consistency_goes_tentative(self):
        """Good v_avg but low embedding consistency → tentative."""
        store = MockQdrantStore()
        matcher = ReIDMatcher(store, promote_v_threshold=0.6, promote_consistency_threshold=0.7)
        emb = np.random.randn(512).astype(np.float32)
        pid = matcher.match_tracklet(
            track_id=1, embedding=emb, v_avg=0.8,
            embedding_consistency=0.3, tracklet_len=10,
        )
        assert pid is None
        assert 1 in matcher.tentative

    def test_tentative_promotes_when_quality_improves(self):
        """Tentative with accumulated good quality promotes to new person."""
        store = MockQdrantStore()
        matcher = ReIDMatcher(store, promote_v_threshold=0.6, promote_consistency_threshold=0.7)
        emb = np.random.randn(512).astype(np.float32)

        # First attempt: bad
        matcher.match_tracklet(track_id=1, embedding=emb, v_avg=0.3, embedding_consistency=0.5, tracklet_len=10)
        assert 1 in matcher.tentative

        # Second attempt: good quality → should promote
        good_emb = np.random.randn(512).astype(np.float32)
        pid = matcher.match_tracklet(track_id=1, embedding=good_emb, v_avg=0.8, embedding_consistency=0.9, tracklet_len=10)
        assert pid is not None
        assert 1 not in matcher.tentative

    def test_tentative_fallback_after_many_attempts(self):
        """After 5 attempts, create person as fallback."""
        store = MockQdrantStore()
        matcher = ReIDMatcher(store, promote_v_threshold=0.6, promote_consistency_threshold=0.7)
        emb = np.random.randn(512).astype(np.float32)

        for i in range(4):
            pid = matcher.match_tracklet(track_id=1, embedding=emb, v_avg=0.3, embedding_consistency=0.3, tracklet_len=10)
            assert pid is None

        pid = matcher.match_tracklet(track_id=1, embedding=emb, v_avg=0.3, embedding_consistency=0.3, tracklet_len=10)
        assert pid is not None
        assert 1 not in matcher.tentative

    def test_match_found_returns_existing_pid(self):
        """When a match is found in Qdrant, return the existing person_id."""
        store = MockQdrantStore()
        store.search = lambda emb, top_k=5: [(42, 0.85)]
        matcher = ReIDMatcher(store)
        emb = np.random.randn(512).astype(np.float32)
        pid = matcher.match_tracklet(
            track_id=1, embedding=emb, v_avg=0.9,
            embedding_consistency=0.9, tracklet_len=10,
        )
        assert pid == 42

    def test_match_with_gated_update(self):
        """When matched, gated update should be attempted."""
        store = MockQdrantStore()
        update_called = {"called": False, "result": None}

        original_update = store.gated_momentum_update
        def tracking_update(*args, **kwargs):
            update_called["called"] = True
            update_called["result"] = original_update(*args, **kwargs)
            return update_called["result"]

        store.gated_momentum_update = tracking_update
        store.search = lambda emb, top_k=5: [(42, 0.85)]
        store.persons[42] = {"embedding": np.random.randn(512), "metadata": {}}

        matcher = ReIDMatcher(store, update_v_threshold=0.6)
        emb = np.random.randn(512).astype(np.float32)
        pid = matcher.match_tracklet(
            track_id=1, embedding=emb, v_avg=0.8,
            embedding_consistency=0.9, tracklet_len=10,
        )
        assert pid == 42
        assert update_called["called"]
