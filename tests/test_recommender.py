"""Tests for the recommender module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocess import clean_text


class TestRecommenderLogic:
    """Tests for recommendation-related logic (clean_text)."""

    def test_recommendation_cleaning(self):
        """Should clean resume text identically to the recommender's preprocessing."""
        sample = "Python 3.9, Machine Learning, SQL (Advanced)"
        cleaned = clean_text(sample)
        assert cleaned == "python machine learning sql advanced"

    def test_empty_text(self):
        """Should handle empty resume text."""
        cleaned = clean_text("")
        assert cleaned == ""
