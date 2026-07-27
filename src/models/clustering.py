"""KMeans clustering for resume topic discovery.

Extracted from notebook 04 logic. Clusters resumes into topic groups
and provides cluster-level summaries.
"""

import os
import joblib
import pandas as pd
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def train_kmeans(
    X_tfidf,
    n_clusters: int = 10,
    random_state: int = 42,
    n_init: int = 10,
) -> KMeans:
    """Train a KMeans clustering model.

    Parameters
    ----------
    X_tfidf : sparse matrix
        TF-IDF vectorized features.
    n_clusters : int
        Number of clusters. Defaults to 10.
    random_state : int
        Random seed. Defaults to 42.
    n_init : int
        Number of initializations. Defaults to 10.

    Returns
    -------
    KMeans
        Trained KMeans model.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    kmeans.fit(X_tfidf)
    return kmeans


def save_kmeans(model: KMeans, path: str) -> None:
    """Save KMeans model to disk.

    Parameters
    ----------
    model : KMeans
        Trained KMeans model.
    path : str
        File path for the saved model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_kmeans(path: str) -> KMeans:
    """Load KMeans model from disk.

    Parameters
    ----------
    path : str
        Path to the saved KMeans model.

    Returns
    -------
    KMeans
        Loaded KMeans model.
    """
    return joblib.load(path)


def get_cluster_assignments(df: pd.DataFrame, kmeans: KMeans, X_tfidf) -> pd.DataFrame:
    """Add cluster labels to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add cluster labels to.
    kmeans : KMeans
        Trained KMeans model.
    X_tfidf : sparse matrix
        TF-IDF vectors corresponding to the DataFrame rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'cluster' column.
    """
    df = df.copy()
    df["cluster"] = kmeans.labels_
    return df


def get_top_positions_per_cluster(
    df: pd.DataFrame, top_n: int = 5
) -> Dict[int, List[str]]:
    """Get the top-N most common positions per cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'positions' and 'cluster' columns.
    top_n : int
        Number of top positions per cluster. Defaults to 5.

    Returns
    -------
    dict
        Mapping of cluster_id -> list of top position names.
    """
    result = {}
    for cluster in sorted(df["cluster"].unique()):
        positions = (
            df[df["cluster"] == cluster]["positions"]
            .value_counts()
            .head(top_n)
            .index.tolist()
        )
        result[cluster] = positions
    return result


def get_top_features_per_cluster(
    tfidf: TfidfVectorizer, kmeans: KMeans, top_n: int = 10
) -> Dict[int, List[str]]:
    """Get the top-N TF-IDF features (words) per cluster centroid.

    Parameters
    ----------
    tfidf : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    kmeans : KMeans
        Trained KMeans model.
    top_n : int
        Number of top features per cluster. Defaults to 10.

    Returns
    -------
    dict
        Mapping of cluster_id -> list of top feature names.
    """
    feature_names = tfidf.get_feature_names_out()
    result = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[-top_n:][::-1]
        result[i] = [feature_names[j] for j in top_indices]
    return result
