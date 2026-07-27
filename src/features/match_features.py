"""Interpretable resume-to-job fit features."""

from src.parsing.resume_parser import extract_skills_from_text


def calculate_fit_score(resume_text: str, job_text: str, similarity: float) -> dict:
    """Blend text similarity and skill coverage into a transparent 0–100 score."""
    resume_skills = set(extract_skills_from_text(resume_text))
    job_skills = set(extract_skills_from_text(job_text))
    matched = sorted(resume_skills & job_skills)
    coverage = len(matched) / len(job_skills) if job_skills else 0.0
    score = min(100.0, 100 * ((0.7 * float(similarity)) + (0.3 * coverage)))
    return {
        "fit_score": round(score, 1),
        "skill_coverage": round(coverage * 100, 1),
        "matched_skills": matched,
    }
