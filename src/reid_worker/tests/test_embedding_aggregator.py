import numpy as np

from src.embedding.aggregator import WeightedEmbeddingAggregator


def test_overlap_reduces_weight():
    agg = WeightedEmbeddingAggregator(gamma=1.0)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    result = agg.aggregate([e1, e2], [0.8, 0.8], overlap_ratios=[1.0, 0.0])
    assert result[1] > 0.99
