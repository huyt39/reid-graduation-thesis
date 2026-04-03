import numpy as np

from src.matching.qdrant_store import QdrantPersonStore
from src.utils.logger import Logger

logger = Logger("reid_matcher")


class ReIDMatcher:
    """ReID matching with promote-tentative policy and gated canonical updates."""

    def __init__(
        self,
        qdrant_store: QdrantPersonStore,
        next_id_counter: int = 1,
        promote_v_threshold: float = 0.6,
        promote_consistency_threshold: float = 0.7,
        update_v_threshold: float = 0.6,
        update_consistency_threshold: float = 0.7,
        update_min_tracklet_len: int = 5,
        update_sim_threshold: float = 0.5,
    ):
        self.store = qdrant_store
        self.next_id = next_id_counter
        self.tentative: dict[int, dict] = {}

        # Promote tentative → new person thresholds
        self.promote_v_threshold = promote_v_threshold
        self.promote_consistency_threshold = promote_consistency_threshold

        # Gated canonical update thresholds
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
        """Match a tracklet embedding against known persons.

        Returns person_id if matched/created, None if still tentative.

        Uses:
        - Gated canonical update when match is found
        - Promote-tentative policy for new person creation
        """
        matches = self.store.search(embedding)

        if matches:
            best_pid, best_score = matches[0]
            logger.info(f"Track {track_id} matched to person {best_pid} (sim={best_score:.3f})")

            # Gated canonical update — only if tracklet quality is high enough
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
                logger.info(f"  Canonical update SKIPPED (gates not met)")

            self.tentative.pop(track_id, None)
            return best_pid

        # No match found — decide between new person or tentative
        can_promote = (
            v_avg >= self.promote_v_threshold
            and embedding_consistency >= self.promote_consistency_threshold
        )

        if can_promote:
            # Good tracklet, no match → create new person
            pid = self._create_person(embedding, {"source": "new_detection"})
            self.tentative.pop(track_id, None)
            logger.info(
                f"Track {track_id} created new person {pid} "
                f"(v_avg={v_avg:.3f}, consistency={embedding_consistency:.3f})"
            )
            return pid
        else:
            # Not good enough → tentative
            if track_id not in self.tentative:
                self.tentative[track_id] = {
                    "embedding": embedding,
                    "v_avg": v_avg,
                    "consistency": embedding_consistency,
                    "attempts": 1,
                }
                logger.info(
                    f"Track {track_id} tentative "
                    f"(v_avg={v_avg:.3f}, consistency={embedding_consistency:.3f})"
                )
            else:
                tent = self.tentative[track_id]
                tent["attempts"] += 1
                # Keep best embedding seen so far
                if v_avg > tent["v_avg"]:
                    tent["embedding"] = embedding
                    tent["v_avg"] = v_avg
                    tent["consistency"] = embedding_consistency

                # Check if accumulated tentative is now strong enough to promote
                if (
                    tent["v_avg"] >= self.promote_v_threshold
                    and tent["consistency"] >= self.promote_consistency_threshold
                ):
                    pid = self._create_person(tent["embedding"], {"source": "tentative_promoted"})
                    del self.tentative[track_id]
                    logger.info(f"Track {track_id} promoted tentative → person {pid}")
                    return pid

                # Fallback: after many attempts, create anyway to avoid stuck tentatives
                if tent["attempts"] >= 5:
                    pid = self._create_person(tent["embedding"], {"source": "tentative_fallback"})
                    del self.tentative[track_id]
                    logger.info(f"Track {track_id} fallback → person {pid} after {tent['attempts']} attempts")
                    return pid

            return None

    def _create_person(self, embedding: np.ndarray, metadata: dict) -> int:
        pid = self.next_id
        self.next_id += 1
        self.store.add_person(pid, embedding, metadata)
        return pid
