"""Skill gap analysis module.

Compares skills extracted from a resume against the required skills
for a target job position, identifying matched and missing skills.
"""

import pandas as pd
from typing import Dict, List, Optional

from src.parsing.resume_parser import extract_skills_from_text


def build_job_skills_mapping(csv_path: str) -> Dict[str, List[str]]:
    """Build a mapping of job positions to their required skills from the dataset.

    Parameters
    ----------
    csv_path : str
        Path to the resume dataset CSV file.

    Returns
    -------
    dict
        Mapping of job position name (lowercase) -> list of required skills.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.lstrip("\ufeffï»¿") for col in df.columns]

    # Extract job_position_name and skills_required
    if "job_position_name" not in df.columns or "skills_required" not in df.columns:
        return {}

    job_skills = {}

    for _, row in df.iterrows():
        position = row.get("job_position_name", None)
        skills = row.get("skills_required", None)

        if pd.isna(position) or pd.isna(skills):
            continue

        position_key = str(position).strip().lower()

        # Split skills by newline or comma
        skill_list = []
        for skill in str(skills).replace("\\n", "\n").split("\n"):
            skill = skill.strip().strip("'\"[]")
            if skill and skill != "N/A":
                skill_list.append(skill.lower())

        if skill_list:
            if position_key not in job_skills:
                job_skills[position_key] = []
            job_skills[position_key].extend(skill_list)

    # Deduplicate skills per position
    for key in job_skills:
        job_skills[key] = sorted(set(job_skills[key]))

    return job_skills


def find_best_matching_job(
    target_job: str, job_skills_mapping: Dict[str, List[str]]
) -> Optional[str]:
    """Find the best matching job position in the dataset for the target job.

    Uses fuzzy substring matching to find the closest job title.

    Parameters
    ----------
    target_job : str
        User-specified target job title.
    job_skills_mapping : dict
        Mapping of job positions to skills.

    Returns
    -------
    str or None
        Best matching job position key, or None if no match found.
    """
    target_lower = target_job.strip().lower()

    # Exact match
    if target_lower in job_skills_mapping:
        return target_lower

    # Partial match - check if target is contained in any job title
    for job_key in job_skills_mapping:
        if target_lower in job_key or job_key in target_lower:
            return job_key

    # Word-level matching
    target_words = set(target_lower.split())
    best_match = None
    best_score = 0

    for job_key in job_skills_mapping:
        job_words = set(job_key.split())
        overlap = len(target_words & job_words)
        if overlap > best_score:
            best_score = overlap
            best_match = job_key

    return best_match if best_score > 0 else None


def analyze_skill_gap(
    resume_text: str,
    target_job: str,
    csv_path: str = "data/raw/resume_data.csv",
) -> Dict:
    """Analyze the skill gap between a resume and a target job position.

    Parameters
    ----------
    resume_text : str
        The resume text content.
    target_job : str
        The target job position title.
    csv_path : str
        Path to the resume dataset CSV. Defaults to "data/raw/resume_data.csv".

    Returns
    -------
    dict
        {
            "resume_skills": list of str - skills found in the resume,
            "required_skills": list of str - skills required for the target job,
            "matched": list of str - skills the candidate has,
            "missing": list of str - skills the candidate lacks,
            "gap_percentage": float - percentage of required skills missing (0-100),
            "target_job_matched": str - actual job title found in dataset (or None),
        }
    """
    # Extract skills from resume
    resume_skills = extract_skills_from_text(resume_text)

    # Load job skills mapping
    job_skills_mapping = build_job_skills_mapping(csv_path)

    # Find best matching job
    matched_job = find_best_matching_job(target_job, job_skills_mapping)

    required_skills = []
    if matched_job:
        required_skills = job_skills_mapping[matched_job]

    # Compare skills
    resume_skills_lower = set(s.lower() for s in resume_skills)
    required_skills_lower = set(s.lower() for s in required_skills)

    matched = sorted(resume_skills_lower & required_skills_lower)
    missing = sorted(required_skills_lower - resume_skills_lower)

    # Calculate gap percentage
    if required_skills_lower:
        gap_percentage = round(
            (len(missing) / len(required_skills_lower)) * 100, 1
        )
    else:
        gap_percentage = 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "required_skills": sorted(required_skills),
        "matched": matched,
        "missing": missing,
        "gap_percentage": gap_percentage,
        "target_job_matched": matched_job,
    }
