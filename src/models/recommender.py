"""Job recommendation system.

Uses TF-IDF + cosine similarity to recommend jobs based on resume text.
Extracted from notebook 03 logic.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.preprocess import clean_text


def build_job_index(
    df: pd.DataFrame,
    text_column: str = "job_text",
    max_features: int = 5000,
):
    """Build TF-IDF vectors for all jobs in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a text column for each job posting.
    text_column : str
        Column name containing the combined job text.
    max_features : int
        Maximum TF-IDF features. Defaults to 5000.

    Returns
    -------
    tuple of (TfidfVectorizer, sparse matrix)
        Fitted vectorizer and the job vectors.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    job_vectors = tfidf.fit_transform(df[text_column])
    return tfidf, job_vectors


def recommend_jobs(
    resume_text: str,
    df: pd.DataFrame,
    tfidf: TfidfVectorizer,
    job_vectors,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend top-N jobs based on cosine similarity with resume text.

    Parameters
    ----------
    resume_text : str
        The resume text to match against.
    df : pd.DataFrame
        The full job dataset.
    tfidf : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    job_vectors : sparse matrix
        Pre-computed TF-IDF vectors for all jobs.
    top_n : int
        Number of recommendations. Defaults to 5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: positions, skills, related_skils_in_job, Similarity Score
    """
    cleaned = clean_text(resume_text)
    resume_vector = tfidf.transform([cleaned])
    similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()

    result_df = df.copy()
    result_df["Similarity Score"] = similarity_scores

    # Simplify positions for display
    result_df["positions"] = result_df["positions"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x)
    )

    result_df = (
        result_df.sort_values("Similarity Score", ascending=False)
        .drop_duplicates(subset=["positions"])
    )

    output_columns = [
        column
        for column in [
            "positions",
            "source",
            "company",
            "location",
            "experience",
            "skills",
            "related_skils_in_job",
            "Similarity Score",
        ]
        if column in result_df.columns
    ]
    return result_df[output_columns].head(top_n)


def load_job_dataframe(data_path: str, max_rows=None) -> pd.DataFrame:
    """Load and normalize one supported job dataset."""
    available_columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    if {"jobtitle", "jobdescription"}.issubset(available_columns):
        use_columns = [
            column
            for column in [
                "jobtitle",
                "jobdescription",
                "joblocation_address",
                "experience",
                "company",
                "skills",
            ]
            if column in available_columns
        ]
    elif {"title", "description"}.issubset(available_columns):
        use_columns = [
            column
            for column in [
                "title",
                "description",
                "company_name",
                "location",
                "formatted_experience_level",
                "skills_desc",
            ]
            if column in available_columns
        ]
    else:
        use_columns = None

    # Loading only searchable/display fields keeps the 500+ MB LinkedIn CSV
    # practical for a local Streamlit app.
    df = pd.read_csv(data_path, usecols=use_columns, nrows=max_rows)
    df.columns = [col.lstrip("\ufeffï»¿") for col in df.columns]

    # Normalize the public Naukri sample into a common job-corpus schema.
    if {"jobtitle", "jobdescription"}.issubset(df.columns):
        df = df.rename(
            columns={
                "jobtitle": "positions",
                "jobdescription": "description",
                "joblocation_address": "location",
            }
        )
        df["source"] = "Naukri"
        df["related_skils_in_job"] = df.get("skills", "")
        df["job_text"] = (
            df[
                [
                    "positions",
                    "company",
                    "location",
                    "experience",
                    "skills",
                    "description",
                ]
            ]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        df["clean_job_text"] = df["job_text"].apply(clean_text)
        return df

    # Normalize the LinkedIn 2023–2024 postings dataset.
    if {"title", "description"}.issubset(df.columns):
        df = df.rename(
            columns={
                "title": "positions",
                "company_name": "company",
                "formatted_experience_level": "experience",
                "skills_desc": "skills",
            }
        )
        for column in ("company", "location", "experience", "skills"):
            if column not in df.columns:
                df[column] = ""
        df["source"] = "LinkedIn"
        df["related_skils_in_job"] = df["skills"]
        df["job_text"] = (
            df[
                [
                    "positions",
                    "company",
                    "location",
                    "experience",
                    "skills",
                    "description",
                ]
            ]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        df["clean_job_text"] = df["job_text"].apply(clean_text)
        return df

    text_columns = [
        "career_objective",
        "skills",
        "major_field_of_studies",
        "related_skils_in_job",
        "positions",
        "responsibilities",
    ]

    df["job_text"] = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    df["clean_job_text"] = df["job_text"].apply(clean_text)

    return df


def load_recommender(data_path: str):
    """Load one dataset and build its recommendation index."""
    df = load_job_dataframe(data_path)
    tfidf, job_vectors = build_job_index(df, "clean_job_text")
    return df, tfidf, job_vectors


def load_combined_recommender(data_paths):
    """Load and merge normalized job datasets into one searchable corpus."""
    frames = []
    for path in data_paths:
        header = pd.read_csv(path, nrows=0).columns
        # Keep the complete source CSV for analysis while bounding the live
        # demo index so its initial build remains responsive on a laptop.
        max_rows = 50_000 if {"title", "description"}.issubset(header) else None
        frame = load_job_dataframe(str(path), max_rows=max_rows)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.dropna(subset=["positions", "clean_job_text"])
    combined = combined.drop_duplicates(
        subset=["positions", "company", "location"], keep="first"
    ).reset_index(drop=True)
    tfidf, job_vectors = build_job_index(combined, "clean_job_text")
    return combined, tfidf, job_vectors
