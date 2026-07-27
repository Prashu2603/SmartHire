"""Reproducible evaluation metrics and report figures for SmartHire."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_classifier(y_true, y_pred) -> dict:
    """Return core multi-class classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
    }


def evaluate_clustering(vectors, labels) -> dict:
    """Return silhouette score for a fitted clustering result."""
    return {"silhouette_score": silhouette_score(vectors, labels)}


def precision_at_k(relevant, recommended, k: int = 5) -> float:
    """Calculate Precision@K from relevant and ranked recommended item IDs."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    return sum(item in relevant_set for item in top_k) / len(top_k)


def _save_confusion_matrix(matrix, labels, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, cmap="Purples")
    axis.set(
        title="Resume classifier confusion matrix",
        xlabel="Predicted category",
        ylabel="True category",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def evaluate_saved_classifier(root: Path, figures: Path) -> dict:
    """Reconstruct the documented test split and evaluate saved artifacts."""
    from src.data.preprocess import load_and_preprocess, prepare_positions

    frame = load_and_preprocess(root / "data" / "raw" / "resume_data.csv")
    frame = prepare_positions(frame)
    encoder = joblib.load(root / "models" / "label_encoder.pkl")
    frame = frame[frame["positions"].isin(encoder.classes_)].copy()
    labels = encoder.transform(frame["positions"])
    _, test_indices = train_test_split(
        np.arange(len(frame)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    vectorizer = joblib.load(root / "models" / "tfidf_vectorizer.pkl")
    classifier = joblib.load(root / "models" / "classifier.pkl")
    test_vectors = vectorizer.transform(
        frame.iloc[test_indices]["clean_resume"].fillna("")
    )
    y_true = labels[test_indices]
    y_pred = classifier.predict(test_vectors)
    metrics = evaluate_classifier(y_true, y_pred)
    _save_confusion_matrix(
        metrics["confusion_matrix"],
        encoder.classes_,
        figures / "confusion_matrix.png",
    )
    return metrics


def evaluate_clusters(root: Path, figures: Path) -> dict:
    """Evaluate K-Means across k and save elbow/silhouette and SVD plots."""
    from src.data.preprocess import load_and_preprocess

    frame = load_and_preprocess(root / "data" / "raw" / "resume_data.csv")
    texts = frame["clean_resume"].fillna("")
    if len(texts) > 3000:
        texts = texts.sample(3000, random_state=42)
    vectorizer = TfidfVectorizer(
        max_features=1500, stop_words="english", min_df=2
    )
    vectors = vectorizer.fit_transform(texts)

    ks = list(range(2, 11))
    inertias, silhouettes = [], []
    models = {}
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(vectors)
        models[k] = model
        inertias.append(float(model.inertia_))
        silhouettes.append(
            float(
                silhouette_score(
                    vectors,
                    labels,
                    sample_size=min(1000, vectors.shape[0]),
                    random_state=42,
                )
            )
        )

    best_k = ks[int(np.argmax(silhouettes))]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ks, inertias, marker="o", color="#6C3CE9")
    axes[0].set(title="Elbow curve", xlabel="Number of clusters (k)", ylabel="Inertia")
    axes[1].plot(ks, silhouettes, marker="o", color="#078C6A")
    axes[1].axvline(best_k, linestyle="--", color="#E76F24")
    axes[1].set(
        title="Silhouette scores",
        xlabel="Number of clusters (k)",
        ylabel="Silhouette score",
    )
    figure.tight_layout()
    figure.savefig(figures / "clustering_metrics.png", dpi=180)
    plt.close(figure)

    labels = models[best_k].labels_
    projection = TruncatedSVD(n_components=2, random_state=42).fit_transform(vectors)
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.65,
    )
    axis.set(title=f"Job-family clusters (k={best_k})", xlabel="SVD 1", ylabel="SVD 2")
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.tight_layout()
    figure.savefig(figures / "cluster_projection.png", dpi=180)
    plt.close(figure)
    return {
        "best_k": best_k,
        "silhouette_score": max(silhouettes),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def evaluate_recommender(root: Path, sample_size: int = 50, k: int = 5) -> dict:
    """Measure title-level Precision@K using repeated job titles as relevance."""
    jobs, _, vectors = joblib.load(root / "models" / "job_recommender.pkl")
    titles = jobs["positions"].fillna("").str.strip().str.lower()
    counts = titles.value_counts()
    candidates = jobs.index[titles.isin(counts[counts >= 2].index)].to_numpy()
    rng = np.random.default_rng(42)
    chosen = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)
    scores = []
    for index in chosen:
        similarities = cosine_similarity(vectors[index], vectors).ravel()
        ranked = np.argsort(similarities)[::-1]
        ranked = ranked[ranked != index][:k]
        relevant = set(jobs.index[titles == titles.iloc[index]]) - {index}
        scores.append(precision_at_k(relevant, ranked, k))
    return {
        "precision_at_5": float(np.mean(scores)) if scores else 0.0,
        "queries": len(scores),
        "relevance": "same normalized job title",
    }


def train_and_evaluate_fit_predictor(root: Path, figures: Path) -> dict:
    """Train the optional fit classifier and save ROC evaluation."""
    from src.models.fit_predictor import (
        engineer_fit_features,
        predict_fit_probability,
        save_fit_predictor,
        train_fit_predictor,
    )

    frame = pd.read_csv(root / "data" / "raw" / "resume_data.csv")
    features = engineer_fit_features(frame)
    scores = pd.to_numeric(frame["matched_score"], errors="coerce").fillna(0)
    labels = (scores >= 0.70).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = train_fit_predictor(x_train, y_train)
    probabilities = predict_fit_probability(model, x_test)
    predictions = (probabilities >= 0.5).astype(int)
    save_fit_predictor(model, root / "models" / "fit_predictor.pkl")
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, color="#6C3CE9", label=f"ROC-AUC = {roc_auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777")
    axis.set(
        title="Fit predictor ROC curve",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "fit_predictor_roc.png", dpi=180)
    plt.close(figure)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc,
        "threshold": 0.70,
        "test_samples": len(y_test),
    }


def generate_evaluation_artifacts(root: Path | None = None) -> dict:
    """Generate every required metric, figure, and machine-readable summary."""
    root = root or Path(__file__).resolve().parents[1]
    figures = root / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    classifier = evaluate_saved_classifier(root, figures)
    summary = {
        "classifier": {
            key: float(value)
            for key, value in classifier.items()
            if key not in {"confusion_matrix", "classification_report"}
        },
        "clustering": evaluate_clusters(root, figures),
        "recommender": evaluate_recommender(root),
        "fit_predictor": train_and_evaluate_fit_predictor(root, figures),
    }
    (root / "reports" / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "SmartHire - Evaluation Report",
        "=" * 31,
        "",
        json.dumps(summary, indent=2),
        "",
        "Classifier per-class report",
        "-" * 28,
        classifier["classification_report"],
    ]
    (root / "reports" / "evaluation_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_evaluation_artifacts(), indent=2))
