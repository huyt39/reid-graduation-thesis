import numpy as np


class WeightedEmbeddingAggregator:
    def __init__(
        self,
        gamma: float = 0.5,
        outlier_threshold: float = 0.5,
        exclude_overlap_ratio: float = 1.0,
    ):
        self.gamma = gamma
        self.outlier_threshold = outlier_threshold
        # Frames whose bbox overlaps another person at or above this ratio are
        # crop-contaminated (the crop contains the other person's pixels) and
        # are dropped before aggregation so the matching embedding represents
        # this person's clean appearance. 1.0 disables the exclusion. Falls
        # back to all frames when too few clean ones survive.
        self.exclude_overlap_ratio = exclude_overlap_ratio

    def aggregate(
        self,
        embeddings: list[np.ndarray],
        v_scores: list[float],
        overlap_ratios: list[float] | None = None,
    ) -> np.ndarray:
        """Visibility-weighted mean with one round of outlier-aware reweighting.

        First pass computes a standard weighted mean; second pass drops frames
        whose cosine similarity to that provisional mean falls below
        ``outlier_threshold`` and recomputes on the survivors. This keeps a
        partially-contaminated tracklet (e.g., one stray crop from another
        person) from skewing the final representation toward the outlier.
        Falls back to the first-pass mean if too few frames survive.

        Frames whose overlap_ratio is at or above ``exclude_overlap_ratio`` are
        dropped up front (crop-contamination guard), unless fewer than two
        clean frames remain, in which case all frames are kept.
        """
        assert len(embeddings) == len(v_scores)
        assert len(embeddings) > 0
        if overlap_ratios is None:
            overlap_ratios = [0.0] * len(embeddings)

        if self.exclude_overlap_ratio < 1.0:
            clean = [
                i for i, o in enumerate(overlap_ratios)
                if float(o or 0.0) < self.exclude_overlap_ratio
            ]
            if 2 <= len(clean) < len(embeddings):
                embeddings = [embeddings[i] for i in clean]
                v_scores = [v_scores[i] for i in clean]
                overlap_ratios = [overlap_ratios[i] for i in clean]

        stacked = np.stack(embeddings, axis=0)
        raw_weights = np.array(
            [v * (1 - self.gamma * o) for v, o in zip(v_scores, overlap_ratios)],
            dtype=np.float64,
        )
        provisional = self._weighted_mean(stacked, raw_weights)

        if len(embeddings) >= 3 and self.outlier_threshold > 0.0:
            sims = stacked @ provisional
            keep = sims >= self.outlier_threshold
            if int(keep.sum()) >= 2:
                refined = self._weighted_mean(stacked[keep], raw_weights[keep])
                return refined
        return provisional

    @staticmethod
    def _weighted_mean(stacked: np.ndarray, raw_weights: np.ndarray) -> np.ndarray:
        weights = raw_weights / (raw_weights.sum() + 1e-8)
        weighted_sum = (stacked * weights[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(weighted_sum)
        if norm > 1e-8:
            weighted_sum = weighted_sum / norm
        return weighted_sum

    @staticmethod
    def compute_embedding_consistency(embeddings: list[np.ndarray]) -> float:
        # PDF Bước 4: embedding_consistency = mean cosine(e_i, e_mean).
        # The previous all-pairs formulation systematically under-scores
        # tracklets where one strong frame dominates a weighted mean —
        # exactly the well-curated top-K case the design wants to reward.
        # Inputs are assumed L2-normalized (workers/main.py normalizes
        # before pushing into this aggregator).
        if len(embeddings) < 2:
            return 1.0
        stacked = np.stack(embeddings, axis=0)
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-8:
            return 0.0
        mean_unit = mean / norm
        sims = stacked @ mean_unit
        return max(0.0, min(1.0, float(sims.mean())))
