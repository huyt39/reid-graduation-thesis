import numpy as np


class WeightedEmbeddingAggregator:
    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma

    def aggregate(
        self,
        embeddings: list[np.ndarray],
        v_scores: list[float],
        overlap_ratios: list[float] | None = None,
    ) -> np.ndarray:
        assert len(embeddings) == len(v_scores)
        assert len(embeddings) > 0
        if overlap_ratios is None:
            overlap_ratios = [0.0] * len(embeddings)

        raw_weights = np.array(
            [v * (1 - self.gamma * o) for v, o in zip(v_scores, overlap_ratios)],
            dtype=np.float64,
        )
        weights = raw_weights / (raw_weights.sum() + 1e-8)
        stacked = np.stack(embeddings, axis=0)
        weighted_sum = (stacked * weights[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(weighted_sum)
        if norm > 1e-8:
            weighted_sum = weighted_sum / norm
        return weighted_sum

    @staticmethod
    def compute_embedding_consistency(embeddings: list[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return 1.0
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(float(np.dot(embeddings[i], embeddings[j])))
        return max(0.0, min(1.0, sum(sims) / len(sims)))
