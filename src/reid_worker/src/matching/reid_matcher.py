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

    def match_tracklet(
        self,
        track_id: int,
        embedding: np.ndarray,
        v_avg: float,
        embedding_consistency: float = 1.0,
        tracklet_len: int = 0,
    ) -> int | None:
        matches = self.store.search(embedding)
        if matches:
            best_pid, best_score = matches[0]
            logger.info(f"Track {track_id} matched to person {best_pid} (sim={best_score:.3f})")
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
        if tent["attempts"] >= 5:
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
