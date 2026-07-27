"""Centralized dataset-loading helpers."""

from pathlib import Path
from typing import Iterable
import argparse

import joblib
import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV and normalize accidental byte-order marks in headers."""
    frame = pd.read_csv(path, **kwargs)
    frame.columns = [column.lstrip("\ufeffï»¿") for column in frame.columns]
    return frame


def load_job_sources(paths: Iterable[str | Path]) -> dict[str, pd.DataFrame]:
    """Load job sources keyed by their input filename."""
    return {Path(path).stem: load_csv(path) for path in paths}


def build_merged_job_corpus(
    source_paths: Iterable[str | Path],
    interim_path: str | Path,
    processed_path: str | Path,
    linkedin_limit: int = 50_000,
) -> pd.DataFrame:
    """Normalize, merge, clean, and persist the job corpus.

    The full LinkedIn CSV remains under ``data/raw``. A deterministic subset is
    used for the laptop-friendly live demo, then merged with all Naukri rows.
    """
    from src.models.recommender import load_job_dataframe

    frames = []
    for path in source_paths:
        columns = pd.read_csv(path, nrows=0).columns
        max_rows = (
            linkedin_limit
            if {"title", "description"}.issubset(columns)
            else None
        )
        frames.append(load_job_dataframe(str(path), max_rows=max_rows))

    interim = pd.concat(frames, ignore_index=True, sort=False)
    interim_path = Path(interim_path)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    interim.to_csv(interim_path, index=False)

    processed = (
        interim.dropna(subset=["positions", "clean_job_text"])
        .drop_duplicates(
            subset=["positions", "company", "location"], keep="first"
        )
        .reset_index(drop=True)
    )
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    return processed


def build_project_data_artifacts() -> None:
    """Build the exact interim, processed, and model artifacts for SmartHire."""
    from src.config import (
        JOB_DATA_PATH,
        JOB_RECOMMENDER_PATH,
        LINKEDIN_DATA_PATH,
        ROOT_DIR,
    )
    from src.models.recommender import build_job_index

    sources = [JOB_DATA_PATH]
    if LINKEDIN_DATA_PATH.exists():
        sources.append(LINKEDIN_DATA_PATH)

    processed = build_merged_job_corpus(
        sources,
        ROOT_DIR / "data" / "interim" / "merged_job_corpus.csv",
        ROOT_DIR / "data" / "processed" / "job_corpus.csv",
    )
    vectorizer, vectors = build_job_index(processed, "clean_job_text")
    joblib.dump(
        (processed, vectorizer, vectors),
        JOB_RECOMMENDER_PATH,
        compress=3,
    )
    print(
        f"Built {len(processed):,} processed jobs and saved "
        f"{JOB_RECOMMENDER_PATH}"
    )


def export_saved_processed_corpus() -> None:
    """Export the validated saved recommendation corpus to data/processed."""
    from src.config import JOB_RECOMMENDER_PATH, ROOT_DIR

    processed, _, _ = joblib.load(JOB_RECOMMENDER_PATH)
    destination = ROOT_DIR / "data" / "processed" / "job_corpus.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination, index=False)
    print(f"Exported {len(processed):,} rows to {destination}")


def build_deployment_job_index(sample_size: int = 8_000) -> None:
    """Build a GitHub-friendly compact index while preserving both sources."""
    from src.config import (
        DEPLOY_JOB_RECOMMENDER_PATH,
        JOB_RECOMMENDER_PATH,
    )
    from src.models.recommender import build_job_index

    jobs, _, _ = joblib.load(JOB_RECOMMENDER_PATH)
    per_source = max(1, sample_size // jobs["source"].nunique())
    sampled = pd.concat(
        [
            group.sample(min(per_source, len(group)), random_state=42)
            for _, group in jobs.groupby("source")
        ],
        ignore_index=True,
    )
    vectorizer, vectors = build_job_index(
        sampled, "clean_job_text", max_features=3_000
    )
    joblib.dump(
        (sampled, vectorizer, vectors),
        DEPLOY_JOB_RECOMMENDER_PATH,
        compress=5,
    )
    print(
        f"Saved {len(sampled):,} deployment jobs to "
        f"{DEPLOY_JOB_RECOMMENDER_PATH}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-processed",
        action="store_true",
        help="Export the saved index without rebuilding raw/interim data.",
    )
    parser.add_argument(
        "--build-deploy-index",
        action="store_true",
        help="Build a compact recommendation index suitable for GitHub.",
    )
    args = parser.parse_args()
    if args.build_deploy_index:
        build_deployment_job_index()
    elif args.export_processed:
        export_saved_processed_corpus()
    else:
        build_project_data_artifacts()
