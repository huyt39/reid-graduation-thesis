import numpy as np

from src.matching.reid_matcher import PersonIdAllocationError, ReIDMatcher


def _make_id_allocator(start: int = 1):
    next_id = start

    def alloc() -> int:
        nonlocal next_id
        pid = next_id
        next_id += 1
        return pid

    return alloc


class MockQdrantStore:
    def __init__(self):
        self.persons = {}
        self.search_results = []
        self.gated_update_result = True
        self.gated_update_calls = []

    def search(self, embedding, top_k=5):
        return self.search_results

    def add_person(self, person_id, embedding, metadata):
        self.persons[person_id] = {"embedding": embedding, "metadata": metadata}

    def gated_momentum_update(self, *args, **kwargs):
        self.gated_update_calls.append((args, kwargs))
        return self.gated_update_result


def test_good_tracklet_no_match_creates_person():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
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


def test_low_quality_first_attempt_stays_tentative():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=2,
        embedding=emb,
        v_avg=0.4,
        embedding_consistency=0.5,
        tracklet_len=3,
    )

    assert pid is None
    assert 2 in matcher.tentative
    assert matcher.tentative[2]["attempts"] == 1
    assert store.persons == {}


def test_tentative_tracklet_promotes_when_later_quality_is_good():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)

    first = matcher.match_tracklet(
        track_id=3,
        embedding=emb1,
        v_avg=0.4,
        embedding_consistency=0.5,
        tracklet_len=3,
    )
    second = matcher.match_tracklet(
        track_id=3,
        embedding=emb2,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=8,
    )

    assert first is None
    assert second == 1
    assert 3 not in matcher.tentative
    assert store.persons[1]["metadata"]["source"] == "tentative_promoted"
    np.testing.assert_allclose(store.persons[1]["embedding"], emb2)


def test_tentative_tracklet_falls_back_after_five_attempts():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
    emb = np.random.randn(512).astype(np.float32)

    result = None
    for _ in range(5):
        result = matcher.match_tracklet(
            track_id=4,
            embedding=emb,
            v_avg=0.3,
            embedding_consistency=0.4,
            tracklet_len=2,
        )

    assert result == 1
    assert 4 not in matcher.tentative
    assert store.persons[1]["metadata"]["source"] == "tentative_fallback"


def test_existing_match_returns_person_even_when_canonical_update_is_skipped():
    store = MockQdrantStore()
    store.search_results = [(42, 0.93)]
    store.gated_update_result = False
    alloc_calls = {"n": 0}

    def alloc() -> int:
        alloc_calls["n"] += 1
        return 999

    matcher = ReIDMatcher(store, id_allocator=alloc)
    emb = np.random.randn(512).astype(np.float32)
    matcher.tentative[5] = {
        "embedding": emb,
        "v_avg": 0.4,
        "consistency": 0.5,
        "attempts": 2,
    }

    pid = matcher.match_tracklet(
        track_id=5,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
    )

    assert pid == 42
    assert 5 not in matcher.tentative
    assert len(store.gated_update_calls) == 1
    assert store.gated_update_calls[0][1]["person_id"] == 42
    assert alloc_calls["n"] == 0


def test_id_allocator_failure_raises_person_id_allocation_error_and_does_not_create_person():
    store = MockQdrantStore()

    def alloc() -> int:
        raise RuntimeError("redis down")

    matcher = ReIDMatcher(
        store,
        id_allocator=alloc,
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
    emb = np.random.randn(512).astype(np.float32)

    try:
        matcher.match_tracklet(
            track_id=99,
            embedding=emb,
            v_avg=0.9,
            embedding_consistency=0.95,
            tracklet_len=10,
        )
        assert False, "Expected PersonIdAllocationError"
    except PersonIdAllocationError:
        pass

    assert store.persons == {}


def test_tentative_state_is_preserved_when_id_allocation_fails_during_promote():
    store = MockQdrantStore()

    def alloc() -> int:
        raise RuntimeError("redis down")

    matcher = ReIDMatcher(
        store,
        id_allocator=alloc,
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)

    first = matcher.match_tracklet(
        track_id=55,
        embedding=emb1,
        v_avg=0.4,
        embedding_consistency=0.5,
        tracklet_len=3,
    )

    assert first is None
    assert 55 in matcher.tentative

    try:
        matcher.match_tracklet(
            track_id=55,
            embedding=emb2,
            v_avg=0.95,
            embedding_consistency=0.98,
            tracklet_len=8,
        )
        assert False, "Expected PersonIdAllocationError"
    except PersonIdAllocationError:
        pass

    assert 55 in matcher.tentative
    assert matcher.tentative[55]["attempts"] == 1
    np.testing.assert_allclose(matcher.tentative[55]["embedding"], emb1)
    assert store.persons == {}
