"""Resume category classifier.

Handles training, saving, loading, and predicting job roles
using TF-IDF + Logistic Regression (from notebook 02).
"""

import os
import numpy as np
import joblib
from typing import List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def train_classifier(
    X_train_tfidf,
    y_train,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train_tfidf : sparse matrix
        TF-IDF vectorized training features.
    y_train : array-like
        Encoded training labels.
    max_iter : int
        Maximum iterations for the solver. Defaults to 1000.

    Returns
    -------
    LogisticRegression
        Trained classifier.
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_tfidf, y_train)
    return model


def save_model(model, path: str) -> None:
    """Save a trained model to disk using joblib.

    Parameters
    ----------
    model : estimator
        Trained scikit-learn model or transformer.
    path : str
        File path for the saved model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path: str) -> LogisticRegression:
    """Load a trained classifier from disk.

    Parameters
    ----------
    path : str
        Path to the saved classifier file.

    Returns
    -------
    LogisticRegression
        Loaded classifier.
    """
    return joblib.load(path)


def load_label_encoder(path: str) -> LabelEncoder:
    """Load the label encoder from disk.

    Parameters
    ----------
    path : str
        Path to the saved LabelEncoder file.

    Returns
    -------
    LabelEncoder
        Loaded label encoder.
    """
    return joblib.load(path)


def predict_roles(
    resume_text: str,
    classifier,
    vectorizer,
    label_encoder,
    clean_fn,
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """Predict top-N job roles for a given resume text.

    Parameters
    ----------
    resume_text : str
        Raw resume text input.
    classifier : LogisticRegression
        Trained classifier.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    label_encoder : LabelEncoder
        Fitted label encoder for role names.
    clean_fn : callable
        Text cleaning function (e.g., clean_text from preprocess).
    top_n : int
        Number of top roles to return. Defaults to 5.

    Returns
    -------
    list of dict
        List of {"role": str, "probability": float} sorted by probability descending.
    """
    cleaned = clean_fn(resume_text)
    vector = vectorizer.transform([cleaned])
    probabilities = classifier.predict_proba(vector)[0]
    top_indices = np.argsort(probabilities)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        role = label_encoder.inverse_transform([idx])[0]
        results.append({
            "role": role,
            "probability": round(float(probabilities[idx]), 4),
        })
    return results
