import numpy as np

from src.matching.reid_matcher import ReIDMatcher


class MockQdrantStore:
    def __init__(self):
        self.persons = {}

    def search(self, embedding, top_k=5):
        return []

    def add_person(self, person_id, embedding, metadata):
        self.persons[person_id] = {"embedding": embedding, "metadata": metadata}

    def gated_momentum_update(self, *args, **kwargs):
        return True


def test_good_tracklet_no_match_creates_person():
    store = MockQdrantStore()
    matcher = ReIDMatcher(store, promote_v_threshold=0.6, promote_consistency_threshold=0.7)
    emb = np.random.randn(512).astype(np.float32)
    pid = matcher.match_tracklet(
        track_id=1,
        embedding=emb,
        v_avg=0.8,
        embedding_consistency=0.9,
        tracklet_len=10,
    )
    assert pid == 1
    assert 1 in store.persons
