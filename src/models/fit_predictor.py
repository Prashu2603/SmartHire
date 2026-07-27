"""Optional supervised shortlisting/fit predictor."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FIT_FEATURE_COLUMNS = [
    "skill_overlap",
    "skill_jaccard",
    "resume_skill_count",
    "required_skill_count",
    "resume_text_length",
    "requirement_text_length",
]


def _tokens(value) -> set[str]:
    """Convert list-like job skill text into normalized word tokens."""
    import re

    return set(re.findall(r"[a-z][a-z0-9+#.-]{1,}", str(value).lower()))


def engineer_fit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable resume/job compatibility features."""
    rows = []
    for _, row in frame.iterrows():
        resume_text = " ".join(
            str(row.get(column, ""))
            for column in ("skills", "career_objective", "responsibilities")
        )
        requirement_text = " ".join(
            str(row.get(column, ""))
            for column in (
                "skills_required",
                "related_skils_in_job",
                "responsibilities.1",
                "experiencere_requirement",
            )
        )
        resume_tokens = _tokens(resume_text)
        required_tokens = _tokens(requirement_text)
        overlap = resume_tokens & required_tokens
        union = resume_tokens | required_tokens
        rows.append(
            {
                "skill_overlap": (
                    len(overlap) / len(required_tokens) if required_tokens else 0.0
                ),
                "skill_jaccard": len(overlap) / len(union) if union else 0.0,
                "resume_skill_count": len(resume_tokens),
                "required_skill_count": len(required_tokens),
                "resume_text_length": len(resume_text),
                "requirement_text_length": len(requirement_text),
            }
        )
    return pd.DataFrame(rows, columns=FIT_FEATURE_COLUMNS)


def train_fit_predictor(features, labels, random_state: int = 42):
    """Train a binary logistic-regression fit model."""
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return model.fit(features, labels)


def predict_fit_probability(model, features) -> np.ndarray:
    """Return positive-class shortlisting probabilities."""
    return model.predict_proba(features)[:, 1]


def save_fit_predictor(model, path) -> None:
    """Persist a trained fit predictor."""
    joblib.dump(model, path)


def load_fit_predictor(path):
    """Load a persisted fit predictor."""
    return joblib.load(path)
