from __future__ import annotations

from collections.abc import Callable

import numpy as np
from typing import TYPE_CHECKING

from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.matching.qdrant_store import QdrantPersonStore

logger = Logger("reid_matcher")


class PersonIdAllocationError(RuntimeError):
    pass


class ReIDMatcher:
    def __init__(
        self,
        qdrant_store: QdrantPersonStore,
        id_allocator: Callable[[], int],
        promote_v_threshold: float = 0.6,
        promote_consistency_threshold: float = 0.7,
        new_identity_min_tracklet_len: int = 6,
        # Default to 0 so legacy unit tests that don't model
        # high-quality-frame counting still pass through the gate.
        # Production wiring in workers/main.py passes the configured
        # settings.min_high_quality_frames value, enforcing the PDF Bước 5
        # condition in runtime.
        min_high_quality_frames: int = 0,
        update_v_threshold: float = 0.6,
        update_consistency_threshold: float = 0.7,
        update_min_tracklet_len: int = 5,
        update_sim_threshold: float = 0.5,
        match_margin: float = 0.0,
        tentative_max_attempts: int = 5,
        tentative_fallback_enabled: bool = True,
        spatial_reuse_threshold: float = 0.70,
        soft_match_threshold: float = 0.72,
        eager_soft_match_threshold: float = 0.50,
        match_consistency_threshold: float = 0.55,
        low_visibility_threshold: float = 0.65,
        low_visibility_match_threshold: float = 0.75,
        blocked_match_score_threshold: float = 0.90,
        update_anchor_min_score: float = 0.72,
        current_identity_min_score: float = 0.55,
        current_identity_switch_min_score: float = 0.78,
        current_identity_switch_min_margin: float = 0.18,
        current_identity_switch_max_current_score: float = 0.70,
        capped_identity_soft_match_threshold: float = 0.72,
        near_gallery_defer_threshold: float = 0.50,
        good_streak_min_consecutive: int = 4,
        good_streak_promotion_enabled: bool = True,
        scale_aux_gallery_enabled: bool = True,
        scale_aux_match_threshold: float = 0.70,
        scale_aux_match_margin: float = 0.03,
        scale_aux_full_gallery_min_score: float = 0.0,
    ):
        self.store = qdrant_store
        self.id_allocator = id_allocator
        self.tentative: dict[int, dict] = {}
        self._last_decisions: dict[int, dict] = {}
        self.promote_v_threshold = promote_v_threshold
        self.promote_consistency_threshold = promote_consistency_threshold
        self.new_identity_min_tracklet_len = new_identity_min_tracklet_len
        self.min_high_quality_frames = int(min_high_quality_frames)
        self.update_v_threshold = update_v_threshold
        self.update_consistency_threshold = update_consistency_threshold
        self.update_min_tracklet_len = update_min_tracklet_len
        self.update_sim_threshold = update_sim_threshold
        self.update_anchor_min_score = update_anchor_min_score
        self.match_margin = match_margin
        self.tentative_max_attempts = tentative_max_attempts
        self.tentative_fallback_enabled = tentative_fallback_enabled
        self.spatial_reuse_threshold = spatial_reuse_threshold
        self.soft_match_threshold = soft_match_threshold
        self.eager_soft_match_threshold = eager_soft_match_threshold
        self.match_consistency_threshold = match_consistency_threshold
        self.low_visibility_threshold = low_visibility_threshold
        self.low_visibility_match_threshold = low_visibility_match_threshold
        self.blocked_match_score_threshold = blocked_match_score_threshold
        self.current_identity_min_score = current_identity_min_score
        self.current_identity_switch_min_score = current_identity_switch_min_score
        self.current_identity_switch_min_margin = current_identity_switch_min_margin
        self.current_identity_switch_max_current_score = current_identity_switch_max_current_score
        self.capped_identity_soft_match_threshold = capped_identity_soft_match_threshold
        self.near_gallery_defer_threshold = near_gallery_defer_threshold
        self.good_streak_min_consecutive = int(good_streak_min_consecutive)
        self.good_streak_promotion_enabled = bool(good_streak_promotion_enabled)
        self.scale_aux_gallery_enabled = bool(scale_aux_gallery_enabled)
        self.scale_aux_match_threshold = float(scale_aux_match_threshold)
        self.scale_aux_match_margin = float(scale_aux_match_margin)
        self.scale_aux_full_gallery_min_score = float(scale_aux_full_gallery_min_score)

    def _has_enough_new_identity_evidence(self, tracklet_len: int) -> bool:
        return int(tracklet_len or 0) >= self.new_identity_min_tracklet_len

    @staticmethod
    def _composite_quality(v_avg: float, embedding_consistency: float) -> float:
        # PDF Bước 5 promote-tentative gate uses BOTH v_avg and
        # embedding_consistency conjunctively. We rank attempts by their
        # product so an attempt that improves on either dimension can
        # replace a stale-but-marginally-higher-v_avg attempt.
        return float(max(0.0, v_avg) * max(0.0, embedding_consistency))

    def _defer_short_new_identity(
        self,
        *,
        track_id: int,
        embedding: np.ndarray,
        v_avg: float,
        embedding_consistency: float,
        tracklet_len: int,
        num_high_quality_frames: int,
        ambiguous: bool,
        tentative_attempts: int | None,
    ) -> None:
        composite = self._composite_quality(v_avg, embedding_consistency)
        tent = self.tentative.setdefault(
            track_id,
            {
                "embedding": embedding,
                "v_avg": v_avg,
                "consistency": embedding_consistency,
                "composite": composite,
                "num_high_quality_frames": int(num_high_quality_frames or 0),
                "attempts": tentative_attempts or 1,
                "ambiguous": ambiguous,
                "tracklet_len": tracklet_len,
            },
        )
        tent["tracklet_len"] = max(int(tent.get("tracklet_len", 0) or 0), int(tracklet_len or 0))
        tent["num_high_quality_frames"] = max(
            int(tent.get("num_high_quality_frames", 0) or 0),
            int(num_high_quality_frames or 0),
        )
        # Rank attempts by composite (v_avg * consistency) so a later
        # attempt with better consistency replaces a stale first-attempt
        # snapshot whose consistency was 0.40 from a partially-occluded
        # frame. The previous v_avg-only criterion stranded the matcher
        # with attempt-1 metadata even when attempt 5 was clearly cleaner.
        if composite > float(tent.get("composite", -1.0)):
            tent["embedding"] = embedding
            tent["v_avg"] = v_avg
            tent["consistency"] = embedding_consistency
            tent["composite"] = composite
        if ambiguous:
            tent["ambiguous"] = True
        self._record_decision(
            track_id,
            method="tentative_pending",
            source="provisional_short_tracklet",
            similarity_score=None,
            runner_up_score=None,
            margin_to_runner_up=None,
            reuse_person_id=None,
            tentative_attempts=tentative_attempts,
            canonical_update_applied=None,
        )

    def _mint_new_identity_if_allowed(
        self,
        *,
        track_id: int,
        embedding: np.ndarray,
        v_avg: float,
        embedding_consistency: float,
        tracklet_len: int,
        num_high_quality_frames: int,
        source: str,
        tentative_attempts: int | None,
        on_new_identity: Callable[[int], None] | None,
    ) -> int | None:
        """Single bottleneck for minting a new person_id.

        Enforces the four conjunctive gates from PDF Bước 5
        (promote-tentative policy):

          1. tracklet_len >= new_identity_min_tracklet_len
          2. num_high_quality_frames >= min_high_quality_frames
          3. v_avg >= promote_v_threshold
          4. embedding_consistency >= promote_consistency_threshold

        The fifth design condition — "no candidate above match threshold"
        — is satisfied implicitly by the call sites: every code path that
        funnels into this helper has already searched the gallery and
        failed to find an eligible match. Re-running the search here
        would double the Qdrant cost without adding safety.

        Returns the new person_id if all gates pass, else None (the
        caller is responsible for keeping the tracklet tentative).
        """
        if tracklet_len < self.new_identity_min_tracklet_len:
            self._record_decision(
                track_id,
                method="new_identity_blocked",
                source=f"{source}__gate_tracklet_len",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=None,
                tentative_attempts=tentative_attempts,
                canonical_update_applied=None,
            )
            return None
        if num_high_quality_frames < self.min_high_quality_frames:
            self._record_decision(
                track_id,
                method="new_identity_blocked",
                source=f"{source}__gate_high_quality_frames",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=None,
                tentative_attempts=tentative_attempts,
                canonical_update_applied=None,
            )
            return None
        if v_avg < self.promote_v_threshold:
            self._record_decision(
                track_id,
                method="new_identity_blocked",
                source=f"{source}__gate_v_avg",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=None,
                tentative_attempts=tentative_attempts,
                canonical_update_applied=None,
            )
            return None
        if embedding_consistency < self.promote_consistency_threshold:
            self._record_decision(
                track_id,
                method="new_identity_blocked",
                source=f"{source}__gate_consistency",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=None,
                tentative_attempts=tentative_attempts,
                canonical_update_applied=None,
            )
            return None
        pid = self._create_person(
            embedding,
            {
                "source": source,
                "v_avg": round(float(v_avg), 4),
                "consistency": round(float(embedding_consistency), 4),
                "tracklet_len": int(tracklet_len),
                "num_high_quality_frames": int(num_high_quality_frames),
            },
            v_avg=v_avg,
            embedding_consistency=embedding_consistency,
            tracklet_len=tracklet_len,
        )
        if pid is None:
            # Initial-anchor quality gate (Fix 3) rejected the write.
            self._record_decision(
                track_id,
                method="new_identity_blocked",
                source=f"{source}__gate_initial_anchor",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=None,
                tentative_attempts=tentative_attempts,
                canonical_update_applied=None,
            )
            return None
        if on_new_identity is not None:
            try:
                on_new_identity(pid)
            except Exception:
                logger.warning(f"on_new_identity callback failed for pid {pid}")
        self._record_decision(
            track_id,
            method="new_identity",
            source=source,
            similarity_score=None,
            runner_up_score=None,
            margin_to_runner_up=None,
            reuse_person_id=None,
            tentative_attempts=tentative_attempts,
            canonical_update_applied=None,
        )
        return pid

    def _record_decision(self, track_id: int, **decision: object) -> None:
        self._last_decisions[track_id] = dict(decision)

    def pop_last_decision(self, track_id: int) -> dict | None:
        return self._last_decisions.pop(track_id, None)

    def _soft_match_existing(
        self,
        *,
        track_id: int,
        embedding: np.ndarray,
        threshold: float,
        blocked_person_ids: set[int],
        forbidden_person_ids: set[int],
        recent_incompatible_person_ids: set[int],
        source: str,
        method: str,
        tentative_attempts: int | None,
    ) -> int | None:
        if threshold >= 1.0:
            return None
        hits = self.store.search(embedding, top_k=2, score_threshold=threshold)
        eligible = [
            (pid, score) for pid, score in (hits or [])
            if pid not in blocked_person_ids
            and pid not in forbidden_person_ids
            and pid not in recent_incompatible_person_ids
        ]
        if not eligible:
            return None
        pid, score = eligible[0]
        runner_up = eligible[1][1] if len(eligible) > 1 else None
        gap = (score - runner_up) if runner_up is not None else float("inf")
        if gap < self.match_margin:
            return None
        self._record_decision(
            track_id,
            method=method,
            source=source,
            similarity_score=float(score),
            runner_up_score=None if runner_up is None else float(runner_up),
            margin_to_runner_up=None if runner_up is None else float(gap),
            reuse_person_id=None,
            tentative_attempts=tentative_attempts,
            canonical_update_applied=None,
        )
        logger.info(
            f"Track {track_id} soft-matched to existing person {pid} "
            f"(method={method}, sim={score:.3f}, gap={gap:.3f})"
        )
        self.tentative.pop(track_id, None)
        return pid

    def _defer_if_near_existing(
        self,
        *,
        track_id: int,
        embedding: np.ndarray,
        blocked_person_ids: set[int],
        forbidden_person_ids: set[int],
        recent_incompatible_person_ids: set[int],
        source: str,
        tentative_attempts: int | None,
    ) -> bool:
        if self.near_gallery_defer_threshold >= self.soft_match_threshold:
            return False
        hits = self.store.search(
            embedding,
            top_k=2,
            score_threshold=self.near_gallery_defer_threshold,
        )
        eligible = [
            (pid, score) for pid, score in (hits or [])
            if pid not in blocked_person_ids
            and pid not in forbidden_person_ids
            and pid not in recent_incompatible_person_ids
        ]
        if not eligible:
            return False
        pid, score = eligible[0]
        runner_up = eligible[1][1] if len(eligible) > 1 else None
        gap = (score - runner_up) if runner_up is not None else None
        self._record_decision(
            track_id,
            method="near_gallery_deferred",
            source=source,
            similarity_score=float(score),
            runner_up_score=None if runner_up is None else float(runner_up),
            margin_to_runner_up=None if gap is None else float(gap),
            reuse_person_id=pid,
            tentative_attempts=tentative_attempts,
            canonical_update_applied=None,
        )
        logger.info(
            f"Track {track_id} deferred near-gallery candidate person {pid} "
            f"(sim={score:.3f}, threshold={self.soft_match_threshold:.3f})"
        )
        return True

    def _scale_aux_gallery_match(
        self,
        *,
        track_id: int,
        embedding: np.ndarray,
        blocked_person_ids: set[int],
        forbidden_person_ids: set[int],
        recent_incompatible_person_ids: set[int],
        tentative_attempts: int | None,
    ) -> int | None:
        search_upper_body = getattr(self.store, "search_upper_body", None)
        if not callable(search_upper_body):
            return None
        hits = search_upper_body(
            embedding,
            top_k=3,
            score_threshold=self.scale_aux_match_threshold,
        )
        eligible = [
            (pid, score) for pid, score in (hits or [])
            if pid not in blocked_person_ids
            and pid not in forbidden_person_ids
            and pid not in recent_incompatible_person_ids
        ]
        if not eligible:
            return None
        pid, score = eligible[0]
        runner_up = eligible[1][1] if len(eligible) > 1 else None
        gap = (score - runner_up) if runner_up is not None else float("inf")
        full_gallery_score = None
        if self.scale_aux_full_gallery_min_score > 0.0:
            full_gallery_score = self.store.search_person(
                pid,
                embedding,
                min_score=self.scale_aux_full_gallery_min_score,
            )
            if full_gallery_score is None:
                self._record_decision(
                    track_id,
                    method="scale_aux_rejected_low_full_gallery_support",
                    source="upper_body_gallery",
                    similarity_score=float(score),
                    runner_up_score=None if runner_up is None else float(runner_up),
                    margin_to_runner_up=float(gap),
                    reuse_person_id=pid,
                    tentative_attempts=tentative_attempts,
                    canonical_update_applied=None,
                )
                return None
        if gap < self.scale_aux_match_margin:
            self._record_decision(
                track_id,
                method="scale_aux_ambiguous_rejected",
                source="upper_body_gallery",
                similarity_score=float(score),
                runner_up_score=None if runner_up is None else float(runner_up),
                margin_to_runner_up=float(gap),
                reuse_person_id=None,
                tentative_attempts=tentative_attempts,
                canonical_update_applied=None,
            )
            return None
        self._record_decision(
            track_id,
            method="scale_aux_gallery_match",
            source="upper_body_gallery",
            similarity_score=float(score),
            full_gallery_score=None if full_gallery_score is None else float(full_gallery_score),
            runner_up_score=None if runner_up is None else float(runner_up),
            margin_to_runner_up=None if runner_up is None else float(gap),
            reuse_person_id=None,
            tentative_attempts=tentative_attempts,
            canonical_update_applied=False,
        )
        logger.info(
            f"Track {track_id} matched via upper-body auxiliary gallery to person "
            f"{pid} (sim={score:.3f}, gap={gap:.3f})"
        )
        return pid

    def _create_new_identity_decision(
        self,
        *,
        track_id: int,
        embedding: np.ndarray,
        v_avg: float,
        embedding_consistency: float,
        tracklet_len: int,
        num_high_quality_frames: int,
        source: str,
        tentative_attempts: int | None,
        on_new_identity: Callable[[int], None] | None = None,
    ) -> int | None:
        # Thin delegator to the gated mint helper. Every create site in the
        # matcher must funnel through here so the promote-tentative policy
        # (PDF Bước 5) is enforced uniformly. on_new_identity fires
        # synchronously inside the helper so the attribute-conflict guard
        # sees the new person_id immediately.
        return self._mint_new_identity_if_allowed(
            track_id=track_id,
            embedding=embedding,
            v_avg=v_avg,
            embedding_consistency=embedding_consistency,
            tracklet_len=tracklet_len,
            num_high_quality_frames=num_high_quality_frames,
            source=source,
            tentative_attempts=tentative_attempts,
            on_new_identity=on_new_identity,
        )

    def match_tracklet(
        self,
        track_id: int,
        embedding: np.ndarray,
        v_avg: float,
        embedding_consistency: float = 1.0,
        tracklet_len: int = 0,
        num_high_quality_frames: int = 0,
        blocked_person_ids: set[int] | None = None,
        current_person_id: int | None = None,
        reuse_person_id: int | None = None,
        blocked_duplicate_person_ids: set[int] | None = None,
        forbidden_person_ids: set[int] | None = None,
        recent_incompatible_person_ids: set[int] | None = None,
        allow_new_identity: bool = True,
        # When allow_new_identity=False, capped_soft_match is the fallback
        # path that absorbs the tracklet into the nearest existing person
        # at a deliberately-lower threshold (capped_identity_soft_match_threshold,
        # default 0.57). That behaviour is appropriate when minting is
        # denied by the MAX_PERSON_IDENTITIES cap (we can't make a new
        # person, so absorb at low confidence). It is the WRONG behaviour
        # when minting is denied because the tracklet itself is
        # unreliable (e.g., embedding consensus failed) — those tracklets
        # should defer to occlusion candidates, not get force-merged into
        # the nearest existing person at sim 0.57. The caller signals the
        # distinction via this flag: True for the identity-cap case,
        # False for the unreliable-tracklet case.
        allow_capped_soft_match: bool = True,
        on_new_identity: Callable[[int], None] | None = None,
        good_streak: int = 0,
        allow_tentative_fallback: bool = True,
        allow_gallery_update: bool = True,
        scale_aux_embedding: np.ndarray | None = None,
        allow_scale_aux_match: bool = False,
    ) -> int | None:
        # on_new_identity fires synchronously after a new pid is allocated,
        # before match_tracklet returns. Used by the worker to register the
        # new person in attribute_voter immediately, so concurrent tracklet
        # tasks see it in the conflict-guard check.
        ambiguous_skipped = False
        blocked_person_ids = blocked_person_ids or set()
        blocked_duplicate_person_ids = blocked_duplicate_person_ids or set()
        forbidden_person_ids = forbidden_person_ids or set()
        recent_incompatible_person_ids = recent_incompatible_person_ids or set()

        # Visibility-aware match threshold. Partial-body crops (low visibility
        # from cut_off, area_ratio, person_overlap penalties; OR untracked-
        # detection-cluster origin where ByteTrack couldn't track the person)
        # produce noisier embeddings whose cosine similarity to *different
        # persons'* canonicals naturally drifts into the 0.60-0.75 range.
        # Without this gate, those tracklets get wrongly absorbed into existing
        # identities at the default 0.57 threshold — the failure mode this
        # project (occlusion-focused ReID) most needs to avoid.
        #
        # Two triggers for the stricter threshold:
        #   1) v_avg < low_visibility_threshold (top-K-averaged visibility low).
        #   2) track_id < 0 (synthetic id from untracked_detection_cluster_promoted
        #      — these are by definition the partial / boundary cases that
        #      ByteTrack itself couldn't lock onto).
        effective_search_threshold: float | None = None
        is_low_visibility = (
            float(v_avg) < self.low_visibility_threshold
            or int(track_id) < 0
        )
        if is_low_visibility:
            effective_search_threshold = max(
                getattr(self.store, "similarity_threshold", 0.57),
                self.low_visibility_match_threshold,
            )
        current_identity_checked = False
        current_identity_score: float | None = None
        current_continuity_min_score = self.current_identity_min_score
        if current_person_id is not None and is_low_visibility:
            current_continuity_min_score = max(
                current_continuity_min_score,
                self.low_visibility_match_threshold,
            )

        # Incoherent-tracklet guard. If the selected embeddings disagree with
        # each other below the noise band, the aggregated embedding is a
        # mixture of multiple people and cannot be safely matched or minted.
        # Refuse to return a person_id — the tracklet stays an occlusion
        # candidate, and its snapshots are never persisted under a confirmed
        # identity. Continuity for already-bound tracks (current_person_id)
        # is preserved so this doesn't drop matched persons mid-stream.
        if (
            current_person_id is None
            and float(embedding_consistency) < self.match_consistency_threshold
        ):
            self._record_decision(
                track_id,
                method="consistency_rejected",
                source="incoherent_tracklet",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=None,
                tentative_attempts=None,
                canonical_update_applied=None,
            )
            logger.info(
                f"Track {track_id} rejected: embedding_consistency="
                f"{embedding_consistency:.3f} < {self.match_consistency_threshold:.3f}"
            )
            return None

        matches = self.store.search(embedding, score_threshold=effective_search_threshold)
        if matches:
            # A very high score (≥ 0.90) means near-certain identity even when the
            # person_id is currently held by another track, but we only allow that
            # bypass when the worker has already marked the blocked person as a likely
            # duplicate track in the same spatial region.
            eligible_matches = [
                (pid, score)
                for pid, score in matches
                if pid == current_person_id
                or pid not in forbidden_person_ids
            ]
            eligible_matches = [
                (pid, score)
                for pid, score in eligible_matches
                if pid == current_person_id
                or pid not in recent_incompatible_person_ids
            ]
            eligible_matches = [
                (pid, score)
                for pid, score in eligible_matches
                if pid == current_person_id
                or pid not in blocked_person_ids
                or (
                    pid in blocked_duplicate_person_ids
                    and score >= self.blocked_match_score_threshold
                )
            ]
            if eligible_matches:
                if current_person_id is not None:
                    current_identity_score = self.store.search_person(
                        current_person_id,
                        embedding,
                        min_score=current_continuity_min_score,
                    )
                    current_identity_checked = True
                    if current_identity_score is not None:
                        best_other = next(
                            (
                                (pid, score)
                                for pid, score in eligible_matches
                                if pid != current_person_id
                            ),
                            None,
                        )
                        # Track continuity is a primary signal in the pipeline,
                        # but a low current score plus a clearly stronger gallery
                        # hit is the signature of a ByteTrack ID-swap fragment.
                        # Let normal gallery matching take over in that case
                        # instead of contaminating the bound person's snapshots.
                        gallery_margin = (
                            None
                            if best_other is None
                            else float(best_other[1] - current_identity_score)
                        )
                        allow_gallery_switch = (
                            best_other is not None
                            and (
                                (
                                    best_other[1] >= 0.92
                                    and gallery_margin is not None
                                    and gallery_margin >= 0.22
                                )
                                or (
                                    current_identity_score
                                    <= self.current_identity_switch_max_current_score
                                    and best_other[1]
                                    >= self.current_identity_switch_min_score
                                    and gallery_margin is not None
                                    and gallery_margin
                                    >= self.current_identity_switch_min_margin
                                )
                            )
                        )
                        if not allow_gallery_switch:
                            self._record_decision(
                                track_id,
                                method="current_identity_maintained",
                                source="track_continuity",
                                similarity_score=float(current_identity_score),
                                runner_up_score=None if best_other is None else float(best_other[1]),
                                margin_to_runner_up=(
                                    None
                                    if best_other is None
                                    else float(best_other[1] - current_identity_score)
                                ),
                                reuse_person_id=current_person_id,
                                tentative_attempts=None,
                                canonical_update_applied=None,
                            )
                            self.tentative.pop(track_id, None)
                            return current_person_id
                best_pid, best_score = eligible_matches[0]
                runner_up_score = eligible_matches[1][1] if len(eligible_matches) > 1 else None
                # Skip ambiguity check when the best score is very high — at ≥ 0.92 the
                # match is reliable enough regardless of runner-up proximity.
                is_ambiguous = (
                    runner_up_score is not None
                    and best_pid != current_person_id
                    and best_score < 0.92
                    and (best_score - runner_up_score) < self.match_margin
                )
                if is_ambiguous:
                    margin = (
                        (best_score - runner_up_score)
                        if runner_up_score is not None
                        else None
                    )
                    self._record_decision(
                        track_id,
                        method="ambiguous_rejected",
                        source="gallery_search",
                        similarity_score=float(best_score),
                        runner_up_score=None if runner_up_score is None else float(runner_up_score),
                        margin_to_runner_up=None if margin is None else float(margin),
                        reuse_person_id=None,
                        tentative_attempts=None,
                        canonical_update_applied=None,
                    )
                    logger.info(
                        f"Track {track_id} skipped ambiguous match to person {best_pid} "
                        f"(sim={best_score:.3f}, runner_up={runner_up_score:.3f}, "
                        f"margin={self.match_margin:.3f})"
                    )
                    ambiguous_skipped = True
                else:
                    logger.info(
                        f"Track {track_id} matched to person {best_pid} (sim={best_score:.3f})"
                    )
                    updated = False
                    if not is_low_visibility and allow_gallery_update:
                        updated = self.store.gated_momentum_update(
                            person_id=best_pid,
                            new_embedding=embedding,
                            v_avg=v_avg,
                            embedding_consistency=embedding_consistency,
                            tracklet_len=tracklet_len,
                            update_v_threshold=self.update_v_threshold,
                            update_consistency_threshold=self.update_consistency_threshold,
                            update_min_tracklet_len=self.update_min_tracklet_len,
                            update_sim_threshold=self.update_sim_threshold,
                            update_anchor_min_score=self.update_anchor_min_score,
                        )
                    if not updated:
                        logger.info("Canonical update skipped")
                    margin = (
                        (best_score - runner_up_score)
                        if runner_up_score is not None
                        else None
                    )
                    self._record_decision(
                        track_id,
                        method="gallery_match",
                        source="gallery_search",
                        similarity_score=float(best_score),
                        runner_up_score=None if runner_up_score is None else float(runner_up_score),
                        margin_to_runner_up=None if margin is None else float(margin),
                        reuse_person_id=None,
                        tentative_attempts=None,
                        canonical_update_applied=bool(updated),
                    )
                    self.tentative.pop(track_id, None)
                    return best_pid

            blocked_match_pids = [pid for pid, _ in matches if pid in blocked_person_ids]
            if blocked_match_pids:
                logger.info(
                    f"Track {track_id} ignored blocked person_ids {blocked_match_pids} "
                    "in current frame"
                )

        # If already confirmed but Qdrant returned nothing above the hard threshold,
        # keep the identity only when the person's own gallery still agrees at a
        # lower continuity threshold. ByteTrack can switch a track_id after occlusion;
        # blindly preserving current_person_id contaminates that identity with
        # snapshots from another physical person.
        if current_person_id is not None:
            if not current_identity_checked:
                current_identity_score = self.store.search_person(
                    current_person_id,
                    embedding,
                    min_score=current_continuity_min_score,
                )
                current_identity_checked = True
            current_score = current_identity_score
            if current_score is not None:
                self._record_decision(
                    track_id,
                    method="current_identity_maintained",
                    source="track_continuity",
                    similarity_score=float(current_score),
                    runner_up_score=None,
                    margin_to_runner_up=None,
                    reuse_person_id=current_person_id,
                    tentative_attempts=None,
                    canonical_update_applied=None,
                )
                return current_person_id

            forbidden_person_ids = set(forbidden_person_ids)
            forbidden_person_ids.add(current_person_id)
            if reuse_person_id == current_person_id:
                reuse_person_id = None
            self._record_decision(
                track_id,
                method="current_identity_rejected",
                source="track_continuity",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=current_person_id,
                tentative_attempts=None,
                canonical_update_applied=None,
            )
            logger.info(
                f"Track {track_id} rejected current person {current_person_id}; "
                f"continuity score below {current_continuity_min_score:.3f}"
            )

        # Spatial + appearance combined check: if a spatial hint names a candidate person
        # AND their gallery also agrees at a lower threshold, accept the match. Two
        # independent signals agreeing is much more reliable than either alone, so we
        # can afford a looser appearance threshold here (spatial_reuse_threshold < similarity_threshold).
        if reuse_person_id is not None:
            if forbidden_person_ids and reuse_person_id in forbidden_person_ids:
                reuse_person_id = None
            elif recent_incompatible_person_ids and reuse_person_id in recent_incompatible_person_ids:
                reuse_person_id = None
        if reuse_person_id is not None:
            hint_score = self.store.search_person(
                reuse_person_id, embedding, min_score=self.spatial_reuse_threshold
            )
            if hint_score is not None:
                self._record_decision(
                    track_id,
                    method="spatial_appearance_reuse",
                    source="spatial_hint",
                    similarity_score=float(hint_score),
                    runner_up_score=None,
                    margin_to_runner_up=None,
                    reuse_person_id=reuse_person_id,
                    tentative_attempts=None,
                    canonical_update_applied=None,
                )
                logger.info(
                    f"Track {track_id} matched via spatial+appearance to person "
                    f"{reuse_person_id} (sim={hint_score:.3f})"
                )
                self.tentative.pop(track_id, None)
                return reuse_person_id
            logger.info(
                f"Track {track_id} rejected spatial reuse of person {reuse_person_id}; "
                f"appearance score below {self.spatial_reuse_threshold:.3f}"
            )
            reuse_person_id = None

        if (
            self.scale_aux_gallery_enabled
            and allow_scale_aux_match
            and scale_aux_embedding is not None
            and current_person_id is None
        ):
            aux_pid = self._scale_aux_gallery_match(
                track_id=track_id,
                embedding=scale_aux_embedding,
                blocked_person_ids=blocked_person_ids,
                forbidden_person_ids=forbidden_person_ids,
                recent_incompatible_person_ids=recent_incompatible_person_ids,
                tentative_attempts=None,
            )
            if aux_pid is not None:
                self.tentative.pop(track_id, None)
                return aux_pid

        # PDF Bước 5: creating a new identity remains a conjunctive decision.
        # The good-frame streak is useful evidence recorded by the worker, but
        # it must not bypass the visibility + embedding-consistency gates.
        can_promote = (
            v_avg >= self.promote_v_threshold
            and embedding_consistency >= self.promote_consistency_threshold
        )
        if can_promote and not allow_new_identity and allow_capped_soft_match:
            capped_pid = self._soft_match_existing(
                track_id=track_id,
                embedding=embedding,
                threshold=self.capped_identity_soft_match_threshold,
                blocked_person_ids=blocked_person_ids,
                forbidden_person_ids=forbidden_person_ids,
                recent_incompatible_person_ids=recent_incompatible_person_ids,
                source="identity_cap",
                method="capped_soft_match",
                tentative_attempts=None,
            )
            if capped_pid is not None:
                return capped_pid

        if can_promote and not ambiguous_skipped and allow_new_identity:
            # If this track has been tentative before, treat promotion as a tentative upgrade.
            tent = self.tentative.get(track_id)
            best_embedding = embedding
            source = "tentative_promoted" if tent is not None else "new_detection"
            if tent is not None and tent.get("v_avg", -1.0) > v_avg:
                best_embedding = tent["embedding"]
            effective_tracklet_len = max(
                int(tracklet_len or 0),
                int((tent or {}).get("tracklet_len", 0) or 0),
            )
            if not self._has_enough_new_identity_evidence(effective_tracklet_len):
                self._defer_short_new_identity(
                    track_id=track_id,
                    embedding=best_embedding,
                    v_avg=v_avg,
                    embedding_consistency=embedding_consistency,
                    tracklet_len=effective_tracklet_len,
                    num_high_quality_frames=num_high_quality_frames,
                    ambiguous=ambiguous_skipped,
                    tentative_attempts=tent.get("attempts") if tent is not None else None,
                )
                return None

            # Soft-match gate: try a looser threshold before minting a new ID.
            # Catches re-entries where the person's appearance shifted enough to fall
            # below the hard threshold but is still clearly closest to one known person.
            # Respects blocked_person_ids so we don't steal a person currently owned by
            # another active track (the same constraint as the hard match above).
            if self.soft_match_threshold < 1.0:
                sp = self._soft_match_existing(
                    track_id=track_id,
                    embedding=best_embedding,
                    threshold=self.soft_match_threshold,
                    blocked_person_ids=blocked_person_ids,
                    forbidden_person_ids=forbidden_person_ids,
                    recent_incompatible_person_ids=recent_incompatible_person_ids,
                    source=source,
                    method="soft_match_at_promotion",
                    tentative_attempts=tent.get("attempts") if tent is not None else None,
                )
                if sp is not None:
                    return sp
            if self._defer_if_near_existing(
                track_id=track_id,
                embedding=best_embedding,
                blocked_person_ids=blocked_person_ids,
                forbidden_person_ids=forbidden_person_ids,
                recent_incompatible_person_ids=recent_incompatible_person_ids,
                source=source,
                tentative_attempts=tent.get("attempts") if tent is not None else None,
            ):
                return None

            # Resolve the best-seen quality metrics for the gate. The tent
            # dict (when present) carries the running max-composite from
            # earlier attempts; combine with the current attempt so a clean
            # final attempt isn't rejected because an earlier dirty attempt
            # had a worse composite.
            gate_v_avg = v_avg
            gate_consistency = embedding_consistency
            gate_high_q = num_high_quality_frames
            if tent is not None:
                current_composite = self._composite_quality(v_avg, embedding_consistency)
                if float(tent.get("composite", -1.0)) > current_composite:
                    gate_v_avg = float(tent.get("v_avg", v_avg))
                    gate_consistency = float(tent.get("consistency", embedding_consistency))
                gate_high_q = max(gate_high_q, int(tent.get("num_high_quality_frames", 0) or 0))
            pid = self._create_new_identity_decision(
                track_id=track_id,
                embedding=best_embedding,
                v_avg=gate_v_avg,
                embedding_consistency=gate_consistency,
                tracklet_len=effective_tracklet_len,
                num_high_quality_frames=gate_high_q,
                source=source,
                tentative_attempts=tent.get("attempts") if tent is not None else None,
                on_new_identity=on_new_identity,
            )
            # Only clear the tentative state on a successful mint. If the
            # gate rejected the create, we must keep tentative so retries
            # can accumulate more evidence — popping here would drop the
            # work the deferred path had already done.
            if pid is not None:
                self.tentative.pop(track_id, None)
            return pid

        if track_id not in self.tentative:
            # Eager gallery soft-match at a deliberately lower threshold before
            # parking as tentative. Catches noisy re-entries (small/occluded crop
            # of an already-known person) whose embedding scores below the hard
            # similarity threshold. Respects all filter sets and the match_margin
            # guard so a non-unambiguous hit can't steal an identity.
            if self.eager_soft_match_threshold < self.soft_match_threshold:
                eager_pid = self._soft_match_existing(
                    track_id=track_id,
                    embedding=embedding,
                    threshold=self.eager_soft_match_threshold,
                    blocked_person_ids=blocked_person_ids,
                    forbidden_person_ids=forbidden_person_ids,
                    recent_incompatible_person_ids=recent_incompatible_person_ids,
                    source="eager_pre_tentative",
                    method="eager_soft_match",
                    tentative_attempts=None,
                )
                if eager_pid is not None:
                    return eager_pid
            if (
                self.scale_aux_gallery_enabled
                and allow_scale_aux_match
                and scale_aux_embedding is not None
            ):
                aux_pid = self._scale_aux_gallery_match(
                    track_id=track_id,
                    embedding=scale_aux_embedding,
                    blocked_person_ids=blocked_person_ids,
                    forbidden_person_ids=forbidden_person_ids,
                    recent_incompatible_person_ids=recent_incompatible_person_ids,
                    tentative_attempts=None,
                )
                if aux_pid is not None:
                    return aux_pid
            self.tentative[track_id] = {
                "embedding": embedding,
                "scale_aux_embedding": scale_aux_embedding,
                "v_avg": v_avg,
                "consistency": embedding_consistency,
                "composite": self._composite_quality(v_avg, embedding_consistency),
                "num_high_quality_frames": int(num_high_quality_frames or 0),
                "attempts": 1,
                "ambiguous": ambiguous_skipped,
                "tracklet_len": tracklet_len,
            }
            self._record_decision(
                track_id,
                method="tentative_pending",
                source="quality_gate",
                similarity_score=None,
                runner_up_score=None,
                margin_to_runner_up=None,
                reuse_person_id=reuse_person_id,
                tentative_attempts=1,
                canonical_update_applied=None,
            )
            return None

        tent = self.tentative[track_id]
        tent["attempts"] += 1
        if ambiguous_skipped:
            tent["ambiguous"] = True
        # Composite-based replacement: keep the best (v_avg * consistency)
        # snapshot across attempts rather than the highest-v_avg one.
        # Avoids being stuck with a high-v_avg-but-incoherent attempt.
        current_composite = self._composite_quality(v_avg, embedding_consistency)
        if current_composite > float(tent.get("composite", -1.0)):
            tent["embedding"] = embedding
            tent["scale_aux_embedding"] = scale_aux_embedding
            tent["v_avg"] = v_avg
            tent["consistency"] = embedding_consistency
            tent["composite"] = current_composite
        tent["tracklet_len"] = max(int(tent.get("tracklet_len", 0) or 0), int(tracklet_len or 0))
        tent["num_high_quality_frames"] = max(
            int(tent.get("num_high_quality_frames", 0) or 0),
            int(num_high_quality_frames or 0),
        )
        if (
            tent["v_avg"] >= self.promote_v_threshold
            and tent["consistency"] >= self.promote_consistency_threshold
            and not tent.get("ambiguous")
            and allow_new_identity
        ):
            if not self._has_enough_new_identity_evidence(int(tent.get("tracklet_len", 0) or 0)):
                self._defer_short_new_identity(
                    track_id=track_id,
                    embedding=tent["embedding"],
                    v_avg=tent["v_avg"],
                    embedding_consistency=tent["consistency"],
                    tracklet_len=int(tent.get("tracklet_len", 0) or 0),
                    num_high_quality_frames=int(tent.get("num_high_quality_frames", 0) or 0),
                    ambiguous=bool(tent.get("ambiguous")),
                    tentative_attempts=tent["attempts"],
                )
                return None
            pid = self._create_new_identity_decision(
                track_id=track_id,
                embedding=tent["embedding"],
                v_avg=float(tent["v_avg"]),
                embedding_consistency=float(tent["consistency"]),
                tracklet_len=int(tent.get("tracklet_len", 0) or 0),
                num_high_quality_frames=int(tent.get("num_high_quality_frames", 0) or 0),
                source="tentative_promoted",
                tentative_attempts=tent["attempts"],
                on_new_identity=on_new_identity,
            )
            if pid is not None:
                del self.tentative[track_id]
            return pid
        if tent["attempts"] >= self.tentative_max_attempts:
            if not allow_new_identity:
                if allow_capped_soft_match:
                    capped_pid = self._soft_match_existing(
                        track_id=track_id,
                        embedding=tent["embedding"],
                        threshold=self.capped_identity_soft_match_threshold,
                        blocked_person_ids=blocked_person_ids,
                        forbidden_person_ids=forbidden_person_ids,
                        recent_incompatible_person_ids=recent_incompatible_person_ids,
                        source="identity_cap_fallback",
                        method="capped_soft_match",
                        tentative_attempts=tent["attempts"],
                    )
                    if capped_pid is not None:
                        return capped_pid
                self._record_decision(
                    track_id,
                    method="new_identity_suppressed",
                    source="new_identity_disabled",
                    similarity_score=None,
                    runner_up_score=None,
                    margin_to_runner_up=None,
                    reuse_person_id=reuse_person_id,
                    tentative_attempts=tent["attempts"],
                    canonical_update_applied=None,
                )
                return None
            if self.tentative_fallback_enabled and allow_tentative_fallback:
                if tent.get("ambiguous"):
                    # Re-search after retries — accept best match (filtered by safety
                    # guards) to prevent cascade: if two gallery entries for the same
                    # person exist, the gap between them will always be small, so a
                    # gap guard creates a third ID forever. We still respect
                    # blocked/forbidden/recent-incompatible so an attribute-conflicting
                    # or co-active candidate can't steal this tracklet.
                    _blocked_a = blocked_person_ids or set()
                    _forbidden_a = forbidden_person_ids or set()
                    _recent_incompatible_a = recent_incompatible_person_ids or set()
                    fresh = self.store.search(tent["embedding"])
                    eligible_fresh = [
                        (pid, score) for pid, score in (fresh or [])
                        if pid not in _blocked_a
                        and pid not in _forbidden_a
                        and pid not in _recent_incompatible_a
                    ]
                    if eligible_fresh:
                        fp, fs = eligible_fresh[0]
                        runner_up_score = eligible_fresh[1][1] if len(eligible_fresh) > 1 else None
                        logger.info(
                            f"Track {track_id} accepting best match to {fp} after "
                            f"{tent['attempts']} ambiguous retries (sim={fs:.3f})"
                        )
                        self._record_decision(
                            track_id,
                            method="ambiguous_resolved",
                            source="tentative_fallback",
                            similarity_score=float(fs),
                            runner_up_score=None if runner_up_score is None else float(runner_up_score),
                            margin_to_runner_up=None if runner_up_score is None else float(fs - runner_up_score),
                            reuse_person_id=None,
                            tentative_attempts=tent["attempts"],
                            canonical_update_applied=None,
                        )
                        del self.tentative[track_id]
                        return fp
                    if not self._has_enough_new_identity_evidence(int(tent.get("tracklet_len", 0) or 0)):
                        self._defer_short_new_identity(
                            track_id=track_id,
                            embedding=tent["embedding"],
                            v_avg=tent["v_avg"],
                            embedding_consistency=tent["consistency"],
                            tracklet_len=int(tent.get("tracklet_len", 0) or 0),
                            num_high_quality_frames=int(tent.get("num_high_quality_frames", 0) or 0),
                            ambiguous=True,
                            tentative_attempts=tent["attempts"],
                        )
                        return None
                    # PDF Bước 5 promote-tentative gate must hold here as
                    # well — this was the prime fragmentation site, where
                    # tracklets that didn't match anything in Qdrant after
                    # tentative_max_attempts retries minted a brand-new
                    # person regardless of v_avg / consistency / good
                    # frame count. The _mint_new_identity_if_allowed helper
                    # enforces all four conditions; if any fails the
                    # tracklet stays tentative for the next attempt.
                    pid = self._mint_new_identity_if_allowed(
                        track_id=track_id,
                        embedding=tent["embedding"],
                        v_avg=float(tent["v_avg"]),
                        embedding_consistency=float(tent["consistency"]),
                        tracklet_len=int(tent.get("tracklet_len", 0) or 0),
                        num_high_quality_frames=int(tent.get("num_high_quality_frames", 0) or 0),
                        source="ambiguous_fallback",
                        tentative_attempts=tent["attempts"],
                        on_new_identity=on_new_identity,
                    )
                    if pid is not None:
                        del self.tentative[track_id]
                    return pid
                # Soft-match: try a lower threshold before minting a new ID.
                # Filter blocked_person_ids the same way the can_promote soft_match does
                # so tentative tracks can't steal identities from currently active tracks.
                _blocked_t = blocked_person_ids or set()
                _forbidden_t = forbidden_person_ids or set()
                _recent_incompatible_t = recent_incompatible_person_ids or set()
                raw_soft = self.store.search(tent["embedding"], top_k=2, score_threshold=self.soft_match_threshold)
                eligible_soft = [
                    (pid, score) for pid, score in (raw_soft or [])
                    if pid not in _blocked_t and pid not in _forbidden_t and pid not in _recent_incompatible_t
                ]
                if eligible_soft:
                    best_soft_score = eligible_soft[0][1]
                    pid = eligible_soft[0][0]
                    runner_up = eligible_soft[1][1] if len(eligible_soft) > 1 else None
                    gap = (best_soft_score - runner_up) if runner_up is not None else float("inf")
                    if gap >= self.match_margin:
                        self._record_decision(
                            track_id,
                            method="tentative_soft_match",
                            source="tentative_fallback",
                            similarity_score=float(best_soft_score),
                            runner_up_score=None if runner_up is None else float(runner_up),
                            margin_to_runner_up=float(gap),
                            reuse_person_id=None,
                            tentative_attempts=tent["attempts"],
                            canonical_update_applied=None,
                        )
                        logger.info(
                            f"Track {track_id} tentative_fallback soft-matched to person "
                            f"{pid} (sim={best_soft_score:.3f}, gap={gap:.3f})"
                        )
                        del self.tentative[track_id]
                        return pid
                tent_aux_embedding = tent.get("scale_aux_embedding")
                if (
                    self.scale_aux_gallery_enabled
                    and allow_scale_aux_match
                    and tent_aux_embedding is not None
                ):
                    aux_pid = self._scale_aux_gallery_match(
                        track_id=track_id,
                        embedding=tent_aux_embedding,
                        blocked_person_ids=_blocked_t,
                        forbidden_person_ids=_forbidden_t,
                        recent_incompatible_person_ids=_recent_incompatible_t,
                        tentative_attempts=tent["attempts"],
                    )
                    if aux_pid is not None:
                        del self.tentative[track_id]
                        return aux_pid
                if self._defer_if_near_existing(
                    track_id=track_id,
                    embedding=tent["embedding"],
                    blocked_person_ids=_blocked_t,
                    forbidden_person_ids=_forbidden_t,
                    recent_incompatible_person_ids=_recent_incompatible_t,
                    source="tentative_near_gallery",
                    tentative_attempts=tent["attempts"],
                ):
                    return None
                # Pipeline design (promote-tentative policy) requires the same quality
                # gates as immediate promotion before minting a new ID. Without this,
                # noisy/low-consistency tracklets that don't match any existing person
                # silently create duplicate IDs for re-entered people whose embeddings
                # are temporarily poor.
                fallback_quality_ok = (
                    tent["v_avg"] < self.promote_v_threshold
                    or tent["consistency"] < self.promote_consistency_threshold
                )
                if fallback_quality_ok:
                    self._record_decision(
                        track_id,
                        method="tentative_pending",
                        source="quality_gate_blocked_fallback",
                        similarity_score=None,
                        runner_up_score=None,
                        margin_to_runner_up=None,
                        reuse_person_id=reuse_person_id,
                        tentative_attempts=tent["attempts"],
                        canonical_update_applied=None,
                    )
                    return None
                if not self._has_enough_new_identity_evidence(int(tent.get("tracklet_len", 0) or 0)):
                    self._defer_short_new_identity(
                        track_id=track_id,
                        embedding=tent["embedding"],
                        v_avg=tent["v_avg"],
                        embedding_consistency=tent["consistency"],
                        tracklet_len=int(tent.get("tracklet_len", 0) or 0),
                        num_high_quality_frames=int(tent.get("num_high_quality_frames", 0) or 0),
                        ambiguous=bool(tent.get("ambiguous")),
                        tentative_attempts=tent["attempts"],
                    )
                    return None
                pid = self._create_new_identity_decision(
                    track_id=track_id,
                    embedding=tent["embedding"],
                    v_avg=float(tent["v_avg"]),
                    embedding_consistency=float(tent["consistency"]),
                    tracklet_len=int(tent.get("tracklet_len", 0) or 0),
                    num_high_quality_frames=int(tent.get("num_high_quality_frames", 0) or 0),
                    source="tentative_fallback",
                    tentative_attempts=tent["attempts"],
                    on_new_identity=on_new_identity,
                )
                if pid is not None:
                    del self.tentative[track_id]
                return pid
        self._record_decision(
            track_id,
            method="tentative_pending",
            source="quality_gate",
            similarity_score=None,
            runner_up_score=None,
            margin_to_runner_up=None,
            reuse_person_id=reuse_person_id,
            tentative_attempts=tent["attempts"],
            canonical_update_applied=None,
        )
        return None

    def _create_person(
        self,
        embedding: np.ndarray,
        metadata: dict,
        *,
        v_avg: float,
        embedding_consistency: float,
        tracklet_len: int,
    ) -> int | None:
        # PDF Bước 5 — the initial anchor must meet the promote-tentative
        # quality bar (the same policy that lets the matcher attempt a
        # create at all). Re-checking here is a defence-in-depth in case
        # a future caller bypasses _mint_new_identity_if_allowed: the
        # mint helper enforces promote_v / promote_consistency /
        # new_identity_min_tracklet_len, and so do we here, with the
        # SAME thresholds rather than the canonical-update thresholds.
        # Using the stricter update_* thresholds here was the previous
        # over-strict variant: it deferred legitimate first IDs whose
        # consistency sat in the (promote=0.65, update=0.70) band — the
        # exact band where a partially occluded but real person lives.
        if v_avg < self.promote_v_threshold:
            return None
        if embedding_consistency < self.promote_consistency_threshold:
            return None
        if tracklet_len < self.new_identity_min_tracklet_len:
            return None
        try:
            pid = self.id_allocator()
        except Exception as e:
            logger.error(f"Person ID allocation failed: {e}")
            raise PersonIdAllocationError(str(e)) from e

        self.store.add_person(pid, embedding, metadata)
        return pid
