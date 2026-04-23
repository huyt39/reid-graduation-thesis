import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantPersonStore:
    COLLECTION_NAME = "persons"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        embedding_dim: int = 512,
        similarity_threshold: float = 0.70,
        momentum: float = 0.8,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.momentum = momentum
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=embedding.tolist(),
            limit=top_k,
            score_threshold=self.similarity_threshold,
        )
        return [(hit.id, hit.score) for hit in results]

    def add_person(self, person_id: int, embedding: np.ndarray, metadata: dict) -> None:
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[PointStruct(id=person_id, vector=embedding.tolist(), payload=metadata)],
        )

    def get_embedding(self, person_id: int) -> np.ndarray | None:
        points = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[person_id],
            with_vectors=True,
        )
        if not points:
            return None
        return np.array(points[0].vector)

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
        update_sim_threshold: float = 0.5,
    ) -> bool:
        if v_avg < update_v_threshold:
            return False
        if embedding_consistency < update_consistency_threshold:
            return False
        if tracklet_len < update_min_tracklet_len:
            return False

        canonical = self.get_embedding(person_id)
        if canonical is None:
            return False
        if float(np.dot(canonical, new_embedding)) < update_sim_threshold:
            return False

        updated = self.momentum * canonical + (1 - self.momentum) * new_embedding
        norm = np.linalg.norm(updated)
        if norm > 1e-8:
            updated = updated / norm

        points = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[person_id],
            with_vectors=False,
        )
        payload = points[0].payload if points else {}
        payload["update_count"] = payload.get("update_count", 1) + 1
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[PointStruct(id=person_id, vector=updated.tolist(), payload=payload)],
        )
        return True

    def count(self) -> int:
        return self.client.get_collection(self.COLLECTION_NAME).points_count
