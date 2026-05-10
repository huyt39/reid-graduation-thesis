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
        update_v_threshold: float = 0.6,
        update_consistency_threshold: float = 0.7,
        update_min_tracklet_len: int = 5,
        update_sim_threshold: float = 0.5,
        match_margin: float = 0.0,
        tentative_max_attempts: int = 5,
        tentative_fallback_enabled: bool = True,
        spatial_reuse_threshold: float = 0.70,
        soft_match_threshold: float = 0.72,
    ):
        self.store = qdrant_store
        self.id_allocator = id_allocator
        self.tentative: dict[int, dict] = {}
        self.promote_v_threshold = promote_v_threshold
        self.promote_consistency_threshold = promote_consistency_threshold
        self.update_v_threshold = update_v_threshold
        self.update_consistency_threshold = update_consistency_threshold
        self.update_min_tracklet_len = update_min_tracklet_len
        self.update_sim_threshold = update_sim_threshold
        self.match_margin = match_margin
        self.tentative_max_attempts = tentative_max_attempts
        self.tentative_fallback_enabled = tentative_fallback_enabled
        self.spatial_reuse_threshold = spatial_reuse_threshold
        self.soft_match_threshold = soft_match_threshold

    def match_tracklet(
        self,
        track_id: int,
        embedding: np.ndarray,
        v_avg: float,
        embedding_consistency: float = 1.0,
        tracklet_len: int = 0,
        blocked_person_ids: set[int] | None = None,
        current_person_id: int | None = None,
        reuse_person_id: int | None = None,
    ) -> int | None:
        matches = self.store.search(embedding)
        if matches:
            blocked_person_ids = blocked_person_ids or set()
            # A very high score (≥ 0.90) means near-certain identity even when the
            # person_id is currently held by another track (ByteTracker duplicate).
            # Allow it through; the output loop suppresses the duplicate bbox.
            eligible_matches = [
                (pid, score)
                for pid, score in matches
                if pid == current_person_id
                or pid not in blocked_person_ids
                or score >= 0.90
            ]
            if eligible_matches:
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
                    logger.info(
                        f"Track {track_id} skipped ambiguous match to person {best_pid} "
                        f"(sim={best_score:.3f}, runner_up={runner_up_score:.3f}, "
                        f"margin={self.match_margin:.3f})"
                    )
                else:
                    logger.info(
                        f"Track {track_id} matched to person {best_pid} (sim={best_score:.3f})"
                    )
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
                    )
                    if not updated:
                        logger.info("Canonical update skipped")
                    self.tentative.pop(track_id, None)
                    return best_pid

            blocked_match_pids = [pid for pid, _ in matches if pid in blocked_person_ids]
            if blocked_match_pids:
                logger.info(
                    f"Track {track_id} ignored blocked person_ids {blocked_match_pids} "
                    "in current frame"
                )

        # If already confirmed but Qdrant returned nothing above threshold, keep the
        # existing identity rather than minting a new one (prevents embedding drift
        # from creating duplicate IDs for a continuously-tracked person).
        if current_person_id is not None:
            return current_person_id

        # Spatial + appearance combined check: if a spatial hint names a candidate person
        # AND their gallery also agrees at a lower threshold, accept the match. Two
        # independent signals agreeing is much more reliable than either alone, so we
        # can afford a looser appearance threshold here (spatial_reuse_threshold < similarity_threshold).
        if reuse_person_id is not None:
            hint_score = self.store.search_person(
                reuse_person_id, embedding, min_score=self.spatial_reuse_threshold
            )
            if hint_score is not None:
                logger.info(
                    f"Track {track_id} matched via spatial+appearance to person "
                    f"{reuse_person_id} (sim={hint_score:.3f})"
                )
                self.tentative.pop(track_id, None)
                return reuse_person_id

        can_promote = (
            v_avg >= self.promote_v_threshold
            and embedding_consistency >= self.promote_consistency_threshold
        )
        if can_promote:
            # If this track has been tentative before, treat promotion as a tentative upgrade.
            tent = self.tentative.get(track_id)
            best_embedding = embedding
            if tent is not None and tent.get("v_avg", -1.0) > v_avg:
                best_embedding = tent["embedding"]
            if reuse_person_id is not None:
                logger.info(
                    f"Track {track_id} reusing recent person {reuse_person_id} "
                    "instead of creating a new identity"
                )
                self.tentative.pop(track_id, None)
                return reuse_person_id
            source = "tentative_promoted" if tent is not None else "new_detection"

            pid = self._create_person(best_embedding, {"source": source})
            self.tentative.pop(track_id, None)
            return pid

        if track_id not in self.tentative:
            self.tentative[track_id] = {
                "embedding": embedding,
                "v_avg": v_avg,
                "consistency": embedding_consistency,
                "attempts": 1,
            }
            return None

        tent = self.tentative[track_id]
        tent["attempts"] += 1
        if v_avg > tent["v_avg"]:
            tent["embedding"] = embedding
            tent["v_avg"] = v_avg
            tent["consistency"] = embedding_consistency
        if (
            tent["v_avg"] >= self.promote_v_threshold
            and tent["consistency"] >= self.promote_consistency_threshold
        ):
            pid = self._create_person(tent["embedding"], {"source": "tentative_promoted"})
            del self.tentative[track_id]
            return pid
        if tent["attempts"] >= self.tentative_max_attempts:
            if self.tentative_fallback_enabled:
                if reuse_person_id is not None:
                    logger.info(
                        f"Track {track_id} reusing recent person {reuse_person_id} "
                        "after tentative fallback"
                    )
                    del self.tentative[track_id]
                    return reuse_person_id
                # Soft-match: try a lower threshold before minting a new ID.
                soft_hits = self.store.search(tent["embedding"], top_k=2, score_threshold=self.soft_match_threshold)
                if soft_hits:
                    best_soft_score = soft_hits[0][1]
                    runner_up = soft_hits[1][1] if len(soft_hits) > 1 else None
                    gap = (best_soft_score - runner_up) if runner_up is not None else float("inf")
                    if gap >= self.match_margin:
                        pid = soft_hits[0][0]
                        logger.info(
                            f"Track {track_id} tentative_fallback soft-matched to person "
                            f"{pid} (sim={best_soft_score:.3f}, gap={gap:.3f})"
                        )
                        del self.tentative[track_id]
                        return pid
                pid = self._create_person(tent["embedding"], {"source": "tentative_fallback"})
                del self.tentative[track_id]
                return pid
        return None

    def _create_person(self, embedding: np.ndarray, metadata: dict) -> int:
        try:
            pid = self.id_allocator()
        except Exception as e:
            logger.error(f"Person ID allocation failed: {e}")
            raise PersonIdAllocationError(str(e)) from e

        self.store.add_person(pid, embedding, metadata)
        return pid
