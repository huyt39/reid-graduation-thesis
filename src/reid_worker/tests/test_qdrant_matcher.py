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
        self.anchor_scores = {}
        self.person_scores = {}

    def search(self, embedding, top_k=5, score_threshold=None):
        results = self.search_results[:top_k]
        threshold = 0.70 if score_threshold is None else score_threshold
        return [(pid, score) for pid, score in results if score >= threshold]

    def search_person(self, person_id, embedding, min_score=0.70):
        score = self.person_scores.get(person_id)
        if score is None:
            return None
        return score if score >= min_score else None

    def search_person_anchor(self, person_id, embedding, min_score=0.0):
        score = self.anchor_scores.get(person_id)
        if score is None:
            return None
        return score if score >= min_score else None

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


def test_short_good_tracklet_no_match_stays_provisional():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
        new_identity_min_tracklet_len=6,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=10,
        embedding=emb,
        v_avg=0.95,
        embedding_consistency=0.9,
        tracklet_len=4,
    )

    assert pid is None
    assert 10 in matcher.tentative
    assert store.persons == {}
    decision = matcher.pop_last_decision(10)
    assert decision["method"] == "tentative_pending"
    assert decision["source"] == "provisional_short_tracklet"


def test_short_tracklet_can_promote_after_more_observations():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
        new_identity_min_tracklet_len=6,
    )
    emb = np.random.randn(512).astype(np.float32)

    first = matcher.match_tracklet(
        track_id=11,
        embedding=emb,
        v_avg=0.95,
        embedding_consistency=0.9,
        tracklet_len=4,
    )
    second = matcher.match_tracklet(
        track_id=11,
        embedding=emb,
        v_avg=0.95,
        embedding_consistency=0.9,
        tracklet_len=6,
    )

    assert first is None
    assert second == 1
    assert 11 not in matcher.tentative


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
        embedding_consistency=0.6,
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
        embedding_consistency=0.6,
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


def test_low_quality_tentative_tracklet_does_not_fallback_to_new_identity():
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
            embedding_consistency=0.6,
            tracklet_len=2,
        )

    assert result is None
    assert 4 in matcher.tentative
    assert store.persons == {}
    decision = matcher.pop_last_decision(4)
    assert decision["method"] == "tentative_pending"
    assert decision["source"] == "quality_gate_blocked_fallback"


def test_static_artifact_suppression_never_mints_new_identity():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
    emb = np.random.randn(512).astype(np.float32)

    result = None
    for _ in range(6):
        result = matcher.match_tracklet(
            track_id=44,
            embedding=emb,
            v_avg=0.9,
            embedding_consistency=0.95,
            tracklet_len=10,
            allow_new_identity=False,
        )

    assert result is None
    assert 44 in matcher.tentative
    assert store.persons == {}
    decision = matcher.pop_last_decision(44)
    assert decision["method"] == "new_identity_suppressed"
    assert decision["source"] == "new_identity_disabled"


def test_identity_cap_soft_matches_near_gallery_instead_of_creating_person():
    store = MockQdrantStore()
    store.search_results = [(42, 0.68), (7, 0.55)]
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
        soft_match_threshold=0.72,
        capped_identity_soft_match_threshold=0.66,
        match_margin=0.05,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=45,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=8,
        allow_new_identity=False,
    )

    assert pid == 42
    assert store.persons == {}
    decision = matcher.pop_last_decision(45)
    assert decision["method"] == "capped_soft_match"
    assert decision["source"] == "identity_cap"


def test_near_gallery_tracklet_is_deferred_instead_of_minting_duplicate_identity():
    store = MockQdrantStore()
    store.search_results = [(42, 0.56), (7, 0.49)]
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
        soft_match_threshold=0.62,
        near_gallery_defer_threshold=0.50,
        match_margin=0.05,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=145,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=8,
    )

    assert pid is None
    assert store.persons == {}
    decision = matcher.pop_last_decision(145)
    assert decision["method"] == "near_gallery_deferred"
    assert decision["source"] == "new_detection"
    assert decision["reuse_person_id"] == 42


def test_identity_cap_keeps_uncertain_tracklet_tentative():
    store = MockQdrantStore()
    store.search_results = [(42, 0.54), (7, 0.52)]
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
        capped_identity_soft_match_threshold=0.50,
        match_margin=0.05,
    )
    emb = np.random.randn(512).astype(np.float32)

    result = None
    for _ in range(5):
        result = matcher.match_tracklet(
            track_id=46,
            embedding=emb,
            v_avg=0.9,
            embedding_consistency=0.95,
            tracklet_len=8,
            allow_new_identity=False,
        )

    assert result is None
    assert store.persons == {}
    decision = matcher.pop_last_decision(46)
    assert decision["method"] == "new_identity_suppressed"


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


def test_existing_match_can_block_canonical_update_for_unclean_evidence():
    store = MockQdrantStore()
    store.search_results = [(42, 0.93)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=50,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        allow_gallery_update=False,
    )

    assert pid == 42
    assert store.gated_update_calls == []
    decision = matcher.pop_last_decision(50)
    assert decision["method"] == "gallery_match"
    assert decision["canonical_update_applied"] is False


def test_blocked_person_is_not_reused_by_score_only():
    store = MockQdrantStore()
    store.search_results = [(42, 0.98)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=6,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        blocked_person_ids={42},
    )

    assert pid == 1
    assert store.persons[1]["metadata"]["source"] == "new_detection"
    decision = matcher.pop_last_decision(6)
    assert decision["method"] == "new_identity"


def test_blocked_duplicate_person_can_still_match_when_explicitly_allowed():
    store = MockQdrantStore()
    store.search_results = [(42, 0.93)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=7,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        blocked_person_ids={42},
        blocked_duplicate_person_ids={42},
    )

    assert pid == 42
    assert len(store.gated_update_calls) == 1


def test_blocked_duplicate_person_still_requires_high_similarity():
    store = MockQdrantStore()
    store.search_results = [(42, 0.88)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=8,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        blocked_person_ids={42},
        blocked_duplicate_person_ids={42},
    )

    assert pid == 1


def test_recent_incompatible_person_is_not_reused_by_gallery_match():
    store = MockQdrantStore()
    store.search_results = [(42, 0.98)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=9,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        recent_incompatible_person_ids={42},
    )

    assert pid == 1
    assert store.persons[1]["metadata"]["source"] == "new_detection"


def test_recent_incompatible_person_does_not_block_current_identity():
    store = MockQdrantStore()
    store.search_results = [(42, 0.98)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=10,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        current_person_id=42,
        recent_incompatible_person_ids={42},
    )

    assert pid == 42


def test_current_identity_continuity_wins_over_nearby_gallery_match():
    store = MockQdrantStore()
    store.search_results = [(99, 0.86), (42, 0.78)]
    store.person_scores[42] = 0.78
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=10,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        current_person_id=42,
    )

    assert pid == 42
    assert matcher.pop_last_decision(10)["method"] == "current_identity_maintained"


def test_very_strong_gallery_match_can_override_current_identity():
    store = MockQdrantStore()
    store.search_results = [(99, 0.95), (42, 0.60)]
    store.person_scores[42] = 0.60
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=10,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        current_person_id=42,
    )

    assert pid == 99
    assert matcher.pop_last_decision(10)["method"] == "gallery_match"


def test_stronger_gallery_match_overrides_weak_current_identity():
    store = MockQdrantStore()
    store.search_results = [(99, 0.7936)]
    store.person_scores[42] = 0.5819
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=10,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        current_person_id=42,
    )

    assert pid == 99
    decision = matcher.pop_last_decision(10)
    assert decision["method"] == "gallery_match"


def test_recent_incompatible_person_is_excluded_from_soft_match():
    store = MockQdrantStore()
    store.search_results = [(42, 0.75)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator(), soft_match_threshold=0.72)
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=11,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        recent_incompatible_person_ids={42},
    )

    assert pid == 1


def test_forbidden_person_is_not_reused_by_gallery_match():
    store = MockQdrantStore()
    store.search_results = [(42, 0.98)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=12,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        forbidden_person_ids={42},
    )

    assert pid == 1
    assert store.persons[1]["metadata"]["source"] == "new_detection"


def test_forbidden_person_is_excluded_from_spatial_reuse():
    store = MockQdrantStore()
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator())
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=13,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        reuse_person_id=99,
        forbidden_person_ids={99},
    )

    assert pid == 1
    assert store.persons[1]["metadata"]["source"] == "new_detection"


def test_forbidden_person_is_excluded_from_soft_match():
    store = MockQdrantStore()
    store.search_results = [(42, 0.75)]
    matcher = ReIDMatcher(store, id_allocator=_make_id_allocator(), soft_match_threshold=0.72)
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=14,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        forbidden_person_ids={42},
    )

    assert pid == 1


def test_recent_reuse_hint_requires_appearance_agreement():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=15,
        embedding=emb,
        v_avg=0.8,
        embedding_consistency=0.9,
        tracklet_len=10,
        reuse_person_id=99,
    )

    assert pid == 1
    assert store.persons[1]["metadata"]["source"] == "new_detection"


def test_recent_reuse_hint_reuses_when_appearance_agrees():
    store = MockQdrantStore()
    store.person_scores[99] = 0.74
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
        spatial_reuse_threshold=0.62,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=16,
        embedding=emb,
        v_avg=0.8,
        embedding_consistency=0.9,
        tracklet_len=10,
        reuse_person_id=99,
    )

    assert pid == 99
    decision = matcher.pop_last_decision(16)
    assert decision["method"] == "spatial_appearance_reuse"
    assert decision["similarity_score"] == 0.74


def test_recent_reuse_hint_rejects_borderline_appearance():
    store = MockQdrantStore()
    store.person_scores[99] = 0.54
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        promote_v_threshold=0.6,
        promote_consistency_threshold=0.7,
        spatial_reuse_threshold=0.62,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=116,
        embedding=emb,
        v_avg=0.8,
        embedding_consistency=0.9,
        tracklet_len=10,
        reuse_person_id=99,
    )

    assert pid == 1
    assert store.persons[1]["metadata"]["source"] == "new_detection"


def test_current_identity_requires_continuity_appearance_agreement():
    store = MockQdrantStore()
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        current_identity_min_score=0.55,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=17,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        current_person_id=42,
    )

    assert pid == 1
    assert store.persons[1]["metadata"]["source"] == "new_detection"


def test_current_identity_is_maintained_with_continuity_score():
    store = MockQdrantStore()
    store.person_scores[42] = 0.59
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        current_identity_min_score=0.55,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=18,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        current_person_id=42,
    )

    assert pid == 42
    decision = matcher.pop_last_decision(18)
    assert decision["method"] == "current_identity_maintained"
    assert decision["similarity_score"] == 0.59


def test_synthetic_tracklet_requires_strong_current_identity_continuity():
    store = MockQdrantStore()
    store.person_scores[42] = 0.59
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        current_identity_min_score=0.55,
        low_visibility_match_threshold=0.75,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=-9000008,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
        current_person_id=42,
    )

    assert pid == 1
    decision = matcher.pop_last_decision(-9000008)
    assert decision["method"] == "new_identity"


def test_synthetic_gallery_match_does_not_update_canonical():
    store = MockQdrantStore()
    store.search_results = [(42, 0.82)]
    matcher = ReIDMatcher(
        store,
        id_allocator=_make_id_allocator(),
        low_visibility_match_threshold=0.75,
    )
    emb = np.random.randn(512).astype(np.float32)

    pid = matcher.match_tracklet(
        track_id=-9000008,
        embedding=emb,
        v_avg=0.9,
        embedding_consistency=0.95,
        tracklet_len=10,
    )

    assert pid == 42
    assert store.gated_update_calls == []
    decision = matcher.pop_last_decision(-9000008)
    assert decision["canonical_update_applied"] is False


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
        embedding_consistency=0.6,
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
