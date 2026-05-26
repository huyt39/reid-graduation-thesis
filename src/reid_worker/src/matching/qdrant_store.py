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
        consensus_weight: float = 0.5,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        self.consensus_weight = consensus_weight
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

        # Deduplicate with support: combine the top two scores per person when available.
        per_person_scores: dict[int, list[float]] = {}
        for hit in hits:
            pid = self._person_id_from_hit(hit)
            if pid is None:
                continue
            per_person_scores.setdefault(pid, []).append(float(hit.score))

        ranked: list[tuple[int, float]] = []
        for pid, scores in per_person_scores.items():
            scores.sort(reverse=True)
            best = scores[0]
            if len(scores) >= 2:
                robust_score = ((1.0 - self.consensus_weight) * best) + (self.consensus_weight * scores[1])
            else:
                robust_score = best
            if robust_score >= threshold:
                ranked.append((pid, robust_score))

        filtered = sorted(ranked, key=lambda x: -x[1])[:top_k]

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
                raw_best: dict[int, list[float]] = {}
                for h in raw_hits:
                    pid = self._person_id_from_hit(h)
                    if pid is not None:
                        raw_best.setdefault(pid, []).append(float(h.score))
                top = []
                for pid, scores in raw_best.items():
                    scores.sort(reverse=True)
                    best = scores[0]
                    robust = ((1.0 - self.consensus_weight) * best) + (self.consensus_weight * scores[1]) if len(scores) >= 2 else best
                    top.append((pid, round(robust, 4)))
                top = sorted(top, key=lambda x: -x[1])[:top_k]
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
        self._upsert_gallery_point(person_id, embedding, {**metadata, "is_anchor": True})

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
        update_anchor_min_score: float = 0.72,
    ) -> bool:
        """Add new_embedding to this person's gallery when quality gates pass.

        Quality gate is "matches any existing gallery view" (max-of-gallery),
        not "matches the original anchor". A frozen anchor permanently rejects
        legitimate appearance variation; comparing against the whole gallery
        lets the representation expand naturally with each accepted update.
        """
        if v_avg < update_v_threshold:
            return False
        if embedding_consistency < update_consistency_threshold:
            return False
        if tracklet_len < update_min_tracklet_len:
            return False
        # Use max-of-gallery rather than anchor-only so canonical evidence can
        # grow across valid pose and lighting changes.
        best_score = self.search_person(person_id, new_embedding, min_score=0.0)
        if best_score is not None and best_score < update_anchor_min_score:
            return False

        self._upsert_gallery_point(
            person_id,
            new_embedding,
            {
                "source": "gallery_update",
                "v_avg": round(v_avg, 4),
                "consistency": round(embedding_consistency, 4),
                "is_anchor": False,
            },
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

    def _scroll_gallery_records(self, person_id: int | None = None):
        scroll_filter = None
        if person_id is not None:
            scroll_filter = Filter(must=[
                FieldCondition(key="person_id", match=MatchValue(value=person_id)),
            ])
        records, _ = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=1000,
            with_vectors=True,
            with_payload=True,
        )
        return records

    @staticmethod
    def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vec_a / norm_a, vec_b / norm_b))

    def find_duplicate_candidate(
        self,
        person_id: int,
        *,
        min_score: float,
        exclude_person_ids: set[int] | None = None,
    ) -> tuple[int, float, float | None] | None:
        """Return the most similar other person by gallery max similarity.

        The third return value is the runner-up score. Callers use it as a
        confidence margin for lower-score singleton merges under occlusion.

        ``exclude_person_ids`` lets the caller retry after a gender-conflict
        rejection without re-picking the same person — the runner-up becomes
        the new best candidate.
        """
        exclude_person_ids = exclude_person_ids or set()
        source_records = self._scroll_gallery_records(person_id)
        source_vectors = [
            np.array(record.vector, dtype=np.float32)
            for record in source_records
            if record.vector is not None
        ]
        if not source_vectors:
            return None

        per_person_best: dict[int, float] = {}
        for record in self._scroll_gallery_records():
            payload = record.payload or {}
            other_pid = payload.get("person_id")
            if other_pid is None or int(other_pid) == person_id or record.vector is None:
                continue
            other_pid = int(other_pid)
            if other_pid in exclude_person_ids:
                continue
            other_vec = np.array(record.vector, dtype=np.float32)
            score = max(self._cosine(src_vec, other_vec) for src_vec in source_vectors)
            if score > per_person_best.get(other_pid, -1.0):
                per_person_best[other_pid] = score

        ranked = sorted(per_person_best.items(), key=lambda item: -item[1])
        if not ranked:
            return None
        best_pid, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else None
        if best_score < min_score:
            return None
        return best_pid, float(best_score), None if runner_up_score is None else float(runner_up_score)

    def person_pair_similarity(self, person_a: int, person_b: int) -> float:
        """Return max gallery cosine between two persisted person identities."""
        records_a = self._scroll_gallery_records(person_a)
        records_b = self._scroll_gallery_records(person_b)
        vectors_a = [
            np.array(record.vector, dtype=np.float32)
            for record in records_a
            if record.vector is not None
        ]
        vectors_b = [
            np.array(record.vector, dtype=np.float32)
            for record in records_b
            if record.vector is not None
        ]
        if not vectors_a or not vectors_b:
            return 0.0
        return float(max(self._cosine(vec_a, vec_b) for vec_a in vectors_a for vec_b in vectors_b))

    def merge_person_gallery(self, source_person_id: int, target_person_id: int) -> None:
        """Move all source gallery points into target person's gallery."""
        records = self._scroll_gallery_records(source_person_id)
        point_ids = [record.id for record in records]
        if not point_ids:
            return
        self.client.set_payload(
            collection_name=self.COLLECTION_NAME,
            payload={"person_id": target_person_id, "is_anchor": False},
            points=point_ids,
        )
        self._prune_gallery(target_person_id)

    def _prune_gallery(self, person_id: int) -> None:
        """Keep max_gallery_size most-diverse gallery entries.

        Uses farthest-point sampling (FPS) seeded on the newest record so the
        gallery preserves appearance diversity (different poses/lighting) rather
        than just the original anchor + most recent additions. The anchor has
        no special status here — it is kept only if it remains one of the most
        diverse points.
        """
        try:
            records, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="person_id", match=MatchValue(value=person_id)),
                ]),
                limit=self.max_gallery_size + 20,
                with_vectors=True,
                with_payload=True,
            )
            if len(records) <= self.max_gallery_size:
                return

            sorted_records = sorted(
                records,
                key=lambda r: (r.payload or {}).get("added_at", ""),
            )
            with_vec = [r for r in sorted_records if r.vector is not None]
            without_vec = [r for r in sorted_records if r.vector is None]

            keep_ids = self._select_diverse_records(with_vec, self.max_gallery_size)
            keep_ids.update(r.id for r in without_vec[-self.max_gallery_size:])

            to_delete = [r.id for r in records if r.id not in keep_ids]
            if not to_delete:
                return
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=PointIdsList(points=to_delete),
            )
        except Exception as exc:
            logger.warning("gallery_prune_failed person_id=%d error=%s", person_id, exc)

    @staticmethod
    def _select_diverse_records(records: list, k: int) -> set:
        """Farthest-point sampling on cosine distance. Seeds on the newest record."""
        if len(records) <= k:
            return {r.id for r in records}
        vectors = [np.array(r.vector, dtype=np.float32) for r in records]
        norms = [v / (np.linalg.norm(v) + 1e-8) for v in vectors]
        # Records are sorted oldest→newest by added_at; seed with the newest.
        selected_idx = [len(records) - 1]
        min_dist = [1.0 - float(np.dot(norms[i], norms[selected_idx[0]])) for i in range(len(records))]
        min_dist[selected_idx[0]] = -1.0  # mark seed as taken
        while len(selected_idx) < k:
            nxt = max(range(len(records)), key=lambda i: min_dist[i])
            if min_dist[nxt] < 0:
                break
            selected_idx.append(nxt)
            sel_norm = norms[nxt]
            min_dist[nxt] = -1.0
            for i in range(len(records)):
                if min_dist[i] < 0:
                    continue
                d = 1.0 - float(np.dot(norms[i], sel_norm))
                if d < min_dist[i]:
                    min_dist[i] = d
        return {records[i].id for i in selected_idx}

    def search_person_anchor(self, person_id: int, embedding: np.ndarray, min_score: float = 0.0) -> float | None:
        """Return anchor similarity for one person, or None if no anchor exists."""
        try:
            records, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="person_id", match=MatchValue(value=person_id)),
                    FieldCondition(key="is_anchor", match=MatchValue(value=True)),
                ]),
                limit=1,
                with_vectors=True,
                with_payload=False,
            )
            if not records:
                return None
            query = embedding / (np.linalg.norm(embedding) + 1e-8)
            vec = np.array(records[0].vector, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                return None
            score = float(np.dot(query, vec / norm))
            return score if score >= min_score else None
        except Exception:
            return None

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
