import numpy as np


class WeightedEmbeddingAggregator:
    """Aggregate multiple embeddings from top-K frames into one tracklet embedding.

    Uses overlap-aware weights: w_i = v_score_i * (1 - gamma * overlap_ratio_i)
    This down-weights frames where the person is occluded by another person.
    """

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma

    def aggregate(
        self,
        embeddings: list[np.ndarray],
        v_scores: list[float],
        overlap_ratios: list[float] | None = None,
    ) -> np.ndarray:
        """Compute weighted average of embeddings, L2-normalized.

        Args:
            embeddings: list of K embedding vectors, each [D]
            v_scores: corresponding visibility scores
            overlap_ratios: per-frame overlap ratios (0=no overlap, 1=full overlap)
        """
        assert len(embeddings) == len(v_scores)
        assert len(embeddings) > 0

        if overlap_ratios is None:
            overlap_ratios = [0.0] * len(embeddings)

        # w_i = v_score_i * (1 - gamma * overlap_ratio_i)
        raw_weights = np.array([
            v * (1 - self.gamma * o)
            for v, o in zip(v_scores, overlap_ratios)
        ], dtype=np.float64)

        # Normalize weights to sum to 1
        weights = raw_weights / (raw_weights.sum() + 1e-8)

        stacked = np.stack(embeddings, axis=0)
        weighted_sum = (stacked * weights[:, np.newaxis]).sum(axis=0)

        norm = np.linalg.norm(weighted_sum)
        if norm > 1e-8:
            weighted_sum = weighted_sum / norm

        return weighted_sum

    @staticmethod
    def compute_embedding_consistency(embeddings: list[np.ndarray]) -> float:
        """Compute mean pairwise cosine similarity between embeddings.

        High consistency (close to 1.0) means all selected frames produce
        similar embeddings → likely the same person with good crops.
        Low consistency suggests mixed identities or poor crops.

        Returns value in [0, 1].
        """
        if len(embeddings) < 2:
            return 1.0

        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity (embeddings should already be L2-normalized)
                sim = float(np.dot(embeddings[i], embeddings[j]))
                sims.append(sim)

        return max(0.0, min(1.0, sum(sims) / len(sims)))
