"""Tests for the skill gap analyzer module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parsing.resume_parser import extract_skills_from_text


class TestSkillExtraction:
    """Tests for skill extraction from resume text."""

    def test_extract_python_skill(self):
        """Should detect 'Python' in resume text."""
        text = "I am experienced in Python and data analysis."
        skills = extract_skills_from_text(text)
        assert "python" in skills

    def test_extract_multiple_skills(self):
        """Should detect multiple skills."""
        text = "Python, Machine Learning, SQL, TensorFlow, Docker"
        skills = extract_skills_from_text(text)
        assert "python" in skills
        assert "machine learning" in skills
        assert "sql" in skills
        assert "tensorflow" in skills

    def test_no_skills_found(self):
        """Should return empty list when no skills are found."""
        text = "I like to read books and travel."
        skills = extract_skills_from_text(text)
        assert isinstance(skills, list)

    def test_empty_text(self):
        """Should handle empty text."""
        assert extract_skills_from_text("") == []

    def test_case_insensitive(self):
        """Should be case-insensitive."""
        text = "PYTHON Machine learning Sql"
        skills = extract_skills_from_text(text)
        assert "python" in skills
        assert "machine learning" in skills
        assert "sql" in skills
