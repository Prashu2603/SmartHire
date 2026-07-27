"""Basic tests for reusable match features and evaluation helpers."""

import pytest

from src.evaluate import precision_at_k
from src.features.match_features import calculate_fit_score


def test_precision_at_k():
    assert precision_at_k({"a", "c"}, ["a", "b", "c"], k=2) == 0.5


def test_precision_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k({"a"}, ["a"], k=0)


def test_interpretable_fit_feature_output():
    result = calculate_fit_score("Python SQL", "Python SQL", similarity=0.8)
    assert result["fit_score"] == 86.0
