"""Tests for the preprocessor module."""

import sys
import os
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocess import (
    clean_text,
    build_resume_text,
    TEXT_COLUMNS,
)


class TestCleanText:
    """Tests for the clean_text function."""

    def test_lowercase_conversion(self):
        """Should convert to lowercase."""
        result = clean_text("Python Machine Learning")
        assert result == "python machine learning"

    def test_url_removal(self):
        """Should remove URLs."""
        result = clean_text("check http://example.com for details")
        assert "http" not in result

    def test_special_chars_removal(self):
        """Should remove non-alphabetic characters."""
        result = clean_text("Python 3.9 + TensorFlow 2.x!")
        assert "3.9" not in result
        assert "+" not in result
        assert "!" not in result

    def test_extra_whitespace_stripped(self):
        """Should collapse multiple spaces and trim."""
        result = clean_text("  python   machine   learning  ")
        assert result == "python machine learning"

    def test_empty_string(self):
        """Should handle empty string."""
        assert clean_text("") == ""

    def test_none_input(self):
        """Should handle None input."""
        assert clean_text(None) == ""

    def test_non_string_input(self):
        """Should handle non-string input gracefully."""
        assert clean_text(123) == ""


class TestBuildResumeText:
    """Tests for the build_resume_text function."""

    def test_combines_text_columns(self):
        """Should combine text from specified columns."""
        row = pd.Series({
            "career_objective": "Looking for ML role",
            "skills": "Python, ML",
            "positions": "Data Scientist",
        })
        result = build_resume_text(row, ["career_objective", "skills"])
        assert "Looking for ML role" in result
        assert "Python, ML" in result

    def test_handles_nan(self):
        """Should handle NaN values gracefully."""
        row = pd.Series({
            "career_objective": None,
            "skills": "Python",
        })
        result = build_resume_text(row, TEXT_COLUMNS)
        assert "Python" in result

    def test_default_columns_used(self):
        """Should use TEXT_COLUMNS if no columns specified."""
        row = pd.Series(dict.fromkeys(TEXT_COLUMNS, ""))
        row["skills"] = "Python"
        row["positions"] = "Engineer"
        result = build_resume_text(row)
        assert "Python" in result
        assert "Engineer" in result
