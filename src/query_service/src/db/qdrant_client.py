"""Qdrant vector search client for similarity queries."""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint


COLLECTION = "persons"


class QdrantQueryClient:
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self._client = QdrantClient(host=host, port=port)

    def _search_hits(self, query_vector: list[float], limit: int, min_score: float):
        if hasattr(self._client, "search"):
            return self._client.search(
                collection_name=COLLECTION,
                query_vector=query_vector,
                limit=limit,
                score_threshold=min_score,
            )

        results = self._client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=limit,
            score_threshold=min_score,
        )
        return getattr(results, "points", results)

    def search_similar(
        self, person_id: int, top_k: int = 10, min_score: float = 0.5,
    ) -> list[dict]:
        """Find persons similar to the given person_id."""
        # Retrieve the person's embedding
        points = self._client.retrieve(COLLECTION, ids=[person_id], with_vectors=True)
        if not points or points[0].vector is None:
            return []

        embedding = points[0].vector
        results: list[ScoredPoint] = self._search_hits(
            query_vector=embedding,
            limit=top_k + 1,
            min_score=min_score,
        )
        # Exclude self
        return [
            {"person_id": int(r.id), "score": round(r.score, 4)}
            for r in results
            if int(r.id) != person_id
        ][:top_k]

    def search_by_embedding(
        self, embedding: list[float], top_k: int = 10, min_score: float = 0.5,
    ) -> list[dict]:
        results = self._search_hits(
            query_vector=embedding,
            limit=top_k,
            min_score=min_score,
        )
        return [
            {"person_id": int(r.id), "score": round(r.score, 4)}
            for r in results
        ]

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
