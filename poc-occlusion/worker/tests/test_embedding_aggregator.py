import numpy as np
from src.embedding.aggregator import WeightedEmbeddingAggregator


class TestAggregator:
    def test_single_embedding_passthrough(self):
        agg = WeightedEmbeddingAggregator()
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        result = agg.aggregate([emb], [1.0])
        np.testing.assert_allclose(result, emb, atol=1e-6)

    def test_equal_weights_no_overlap(self):
        agg = WeightedEmbeddingAggregator()
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        result = agg.aggregate([e1, e2], [1.0, 1.0])
        expected = np.array([1.0, 1.0, 0.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_l2_normalized(self):
        agg = WeightedEmbeddingAggregator()
        embeddings = [np.random.randn(512) for _ in range(5)]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = agg.aggregate(embeddings, scores)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_higher_weight_dominates(self):
        agg = WeightedEmbeddingAggregator()
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        result = agg.aggregate([e1, e2], [0.99, 0.01])
        assert result[0] > result[1]

    def test_overlap_reduces_weight(self):
        """High overlap ratio should reduce a frame's contribution."""
        agg = WeightedEmbeddingAggregator(gamma=1.0)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        # e1 has full overlap (weight becomes v*(1-1.0*1.0) = 0)
        # e2 has no overlap (weight stays v*1.0 = 0.8)
        result = agg.aggregate([e1, e2], [0.8, 0.8], overlap_ratios=[1.0, 0.0])
        # e2 should completely dominate since e1's weight is 0
        assert result[1] > 0.99

    def test_overlap_partial_penalty(self):
        """Partial overlap should partially reduce weight."""
        agg = WeightedEmbeddingAggregator(gamma=0.5)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        # e1: weight = 0.8 * (1 - 0.5*0.6) = 0.8 * 0.7 = 0.56
        # e2: weight = 0.8 * (1 - 0.5*0.0) = 0.8 * 1.0 = 0.80
        result = agg.aggregate([e1, e2], [0.8, 0.8], overlap_ratios=[0.6, 0.0])
        assert result[1] > result[0]  # e2 should dominate


class TestEmbeddingConsistency:
    def test_identical_embeddings(self):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        consistency = WeightedEmbeddingAggregator.compute_embedding_consistency([emb, emb, emb])
        assert consistency > 0.99

    def test_orthogonal_embeddings(self):
        e1 = np.zeros(512)
        e2 = np.zeros(512)
        e1[0] = 1.0
        e2[1] = 1.0
        consistency = WeightedEmbeddingAggregator.compute_embedding_consistency([e1, e2])
        assert consistency < 0.1

    def test_single_embedding(self):
        emb = np.random.randn(512).astype(np.float32)
        assert WeightedEmbeddingAggregator.compute_embedding_consistency([emb]) == 1.0

    def test_similar_embeddings(self):
        np.random.seed(42)
        base = np.random.randn(512).astype(np.float64)
        base = base / np.linalg.norm(base)
        # Add very small noise to keep embeddings similar
        embeddings = [base + np.random.randn(512) * 0.01 for _ in range(5)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        consistency = WeightedEmbeddingAggregator.compute_embedding_consistency(embeddings)
        assert consistency > 0.9
