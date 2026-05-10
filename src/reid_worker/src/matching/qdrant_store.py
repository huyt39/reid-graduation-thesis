import logging
import uuid
from datetime import datetime, timezone

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


class QdrantPersonStore:
    COLLECTION_NAME = "persons"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        embedding_dim: int = 512,
        similarity_threshold: float = 0.70,
        momentum: float = 0.8,   # kept for API compat, unused
        max_gallery_size: int = 8,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, embedding: np.ndarray, top_k: int = 5, score_threshold: float | None = None) -> list[tuple[int, float]]:
        """Return up to top_k (person_id, score) pairs above the threshold.

        Multiple gallery vectors can exist per person; we return the best score
        per person so the caller always sees deduplicated person_id entries.
        Pass score_threshold to override the default similarity_threshold.
        """
        threshold = score_threshold if score_threshold is not None else self.similarity_threshold
        query_vector = embedding.tolist()
        # Fetch enough candidates to cover all persons even with a large gallery.
        limit = max(top_k * self.max_gallery_size * 2, 100)

        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
            )
        else:
            result = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
            )
            hits = getattr(result, "points", result)

        # Deduplicate: best score per person_id.
        best: dict[int, float] = {}
        for hit in hits:
            pid = self._person_id_from_hit(hit)
            if pid is not None and (pid not in best or hit.score > best[pid]):
                best[pid] = hit.score

        filtered = sorted(best.items(), key=lambda x: -x[1])[:top_k]

        if not filtered:
            self._log_below_threshold(query_vector, top_k)

        return filtered

    def _person_id_from_hit(self, hit) -> int | None:
        """Extract person_id from a search hit, supporting old and new formats."""
        if hit.payload and "person_id" in hit.payload:
            return hit.payload["person_id"]
        # Legacy format: point id was the person_id (integer).
        try:
            return int(hit.id)
        except (ValueError, TypeError):
            return None

    def _log_below_threshold(self, query_vector: list, top_k: int) -> None:
        try:
            if hasattr(self.client, "search"):
                raw_hits = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
            else:
                result = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                raw_hits = getattr(result, "points", result)

            if raw_hits:
                raw_best: dict[int, float] = {}
                for h in raw_hits:
                    pid = self._person_id_from_hit(h)
                    if pid is not None and (pid not in raw_best or h.score > raw_best[pid]):
                        raw_best[pid] = round(h.score, 4)
                top = sorted(raw_best.items(), key=lambda x: -x[1])[:top_k]
                logger.warning(
                    "qdrant_no_match_above_threshold threshold=%.2f top_k=%s",
                    self.similarity_threshold,
                    top,
                )
        except Exception:
            pass

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_person(self, person_id: int, embedding: np.ndarray, metadata: dict) -> None:
        """Add the first gallery entry for a newly created person."""
        self._upsert_gallery_point(person_id, embedding, metadata)

    def gated_momentum_update(
        self,
        person_id: int,
        new_embedding: np.ndarray,
        v_avg: float,
        embedding_consistency: float,
        tracklet_len: int,
        update_v_threshold: float = 0.6,
        update_consistency_threshold: float = 0.7,
        update_min_tracklet_len: int = 5,
        update_sim_threshold: float = 0.5,   # kept for API compat, unused
    ) -> bool:
        """Add new_embedding to this person's gallery when quality gates pass.

        Instead of updating a single canonical vector (momentum approach), we
        accumulate diverse viewpoints in a gallery so future searches match
        whichever stored angle is closest to the query.
        """
        if v_avg < update_v_threshold:
            return False
        if embedding_consistency < update_consistency_threshold:
            return False
        if tracklet_len < update_min_tracklet_len:
            return False

        self._upsert_gallery_point(
            person_id,
            new_embedding,
            {"source": "gallery_update", "v_avg": round(v_avg, 4), "consistency": round(embedding_consistency, 4)},
        )
        self._prune_gallery(person_id)
        return True

    def _upsert_gallery_point(self, person_id: int, embedding: np.ndarray, metadata: dict) -> None:
        point_id = str(uuid.uuid4())
        payload = {"person_id": person_id, "added_at": datetime.now(timezone.utc).isoformat(), **metadata}
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)],
        )

    def _prune_gallery(self, person_id: int) -> None:
        """Delete oldest gallery entries when a person exceeds max_gallery_size."""
        try:
            records, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="person_id", match=MatchValue(value=person_id)),
                ]),
                limit=self.max_gallery_size + 20,
                with_vectors=False,
                with_payload=True,
            )
            if len(records) <= self.max_gallery_size:
                return

            # Sort by insertion order (UUID v4 is random, so use added_at if present,
            # otherwise just delete the tail of the returned list).
            sorted_records = sorted(
                records,
                key=lambda r: (r.payload or {}).get("added_at", ""),
            )
            to_delete = [r.id for r in sorted_records[:len(records) - self.max_gallery_size]]
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=PointIdsList(points=to_delete),
            )
        except Exception as exc:
            logger.warning("gallery_prune_failed person_id=%d error=%s", person_id, exc)

    def search_person(self, person_id: int, embedding: np.ndarray, min_score: float = 0.70) -> float | None:
        """Return the best gallery score for a specific person_id, or None if below min_score.

        Uses scroll+manual cosine similarity so the person_id filter is guaranteed correct
        (scroll_filter is the same path used by _prune_gallery which is known to work).
        """
        try:
            records, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="person_id", match=MatchValue(value=person_id)),
                ]),
                limit=self.max_gallery_size,
                with_vectors=True,
                with_payload=False,
            )
            if not records:
                return None
            query = embedding / (np.linalg.norm(embedding) + 1e-8)
            best = 0.0
            for r in records:
                if not r.vector:
                    continue
                vec = np.array(r.vector, dtype=np.float32)
                n = np.linalg.norm(vec)
                if n < 1e-8:
                    continue
                score = float(np.dot(query, vec / n))
                if score > best:
                    best = score
            return best if best >= min_score else None
        except Exception:
            return None

    # ── Misc ──────────────────────────────────────────────────────────────────

    def get_embedding(self, person_id: int) -> np.ndarray | None:
        """Return one gallery embedding for this person (first found)."""
        try:
            records, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="person_id", match=MatchValue(value=person_id)),
                ]),
                limit=1,
                with_vectors=True,
                with_payload=False,
            )
            if not records:
                return None
            return np.array(records[0].vector)
        except Exception:
            return None

    def count(self) -> int:
        return self.client.get_collection(self.COLLECTION_NAME).points_count
