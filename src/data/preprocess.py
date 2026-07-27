"""Text preprocessing utilities.

Extracts logic from notebooks 02 and 04 for cleaning resume text,
building combined text fields, and loading the dataset.
"""

import re
import ast
import pandas as pd
from typing import List, Optional


# Columns used to build the combined resume text
TEXT_COLUMNS = [
    "career_objective",
    "skills",
    "major_field_of_studies",
    "related_skils_in_job",
    "positions",
    "responsibilities",
]


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, non-alpha characters, and extra whitespace.

    This is the same cleaning function used across all notebooks.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_resume_text(row: pd.Series, columns: Optional[List[str]] = None) -> str:
    """Combine specified text columns into a single string for a row.

    Parameters
    ----------
    row : pd.Series
        A single row of the DataFrame.
    columns : list of str, optional
        Columns to combine. Defaults to TEXT_COLUMNS.

    Returns
    -------
    str
        Concatenated text from all specified columns.
    """
    cols = columns or TEXT_COLUMNS
    parts = []
    for col in cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                parts.append(str(val))
    return " ".join(parts)


def load_and_preprocess(csv_path: str, text_col_name: str = "resume_text") -> pd.DataFrame:
    """Load the resume dataset and add cleaned text columns.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.
    text_col_name : str
        Name for the combined text column. Defaults to "resume_text".

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'resume_text' and 'clean_resume' columns.
    """
    df = pd.read_csv(csv_path)

    # Fix BOM in column name if present
    df.columns = [col.lstrip("\ufeffï»¿") for col in df.columns]

    # Build combined text
    df[text_col_name] = df.apply(
        lambda row: build_resume_text(row, TEXT_COLUMNS), axis=1
    )

    # Clean text
    clean_col = (
        "clean_resume"
        if text_col_name == "resume_text"
        else f"clean_{text_col_name.removesuffix('_text')}"
    )
    df[clean_col] = df[text_col_name].apply(clean_text)

    return df


def prepare_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse positions column from string lists and explode to one row per position.

    Returns a DataFrame with one row per position, dropping N/A values.
    """
    df = df.copy()

    # Convert string representation of lists to actual lists
    df["positions"] = df["positions"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # One row per position
    df = df.explode("positions")

    # Remove empty values
    df = df.dropna(subset=["positions"])

    return df


def get_top_positions(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Filter to top N most frequent positions, excluding 'N/A'."""
    top_roles = df["positions"].value_counts().head(top_n).index
    df_top = df[df["positions"].isin(top_roles)].copy()
    df_top = df_top[df_top["positions"] != "N/A"].copy()
    return df_top
