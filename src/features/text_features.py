"""TF-IDF text-feature utilities.

Handles fitting, transforming, saving, and loading TF-IDF vectorizers.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 5000,
    stop_words: str = "english",
) -> TfidfVectorizer:
    """Create and return a new TfidfVectorizer with specified parameters.

    Parameters
    ----------
    max_features : int
        Maximum number of features (vocabulary size). Defaults to 5000.
    stop_words : str
        Stop words language. Defaults to "english".

    Returns
    -------
    TfidfVectorizer
        Unfitted vectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
    )


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Save a fitted TfidfVectorizer to disk.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer to save.
    path : str
        File path for the saved model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load a TfidfVectorizer from disk.

    Parameters
    ----------
    path : str
        Path to the saved vectorizer file.

    Returns
    -------
    TfidfVectorizer
        Loaded vectorizer instance.
    """
    return joblib.load(path)
