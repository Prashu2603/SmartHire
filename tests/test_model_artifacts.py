"""Validate that similarly named model artifacts have distinct purposes."""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def test_label_encoder_artifact():
    encoder = joblib.load("models/label_encoder.pkl")
    assert isinstance(encoder, LabelEncoder)
    assert len(encoder.classes_) == 19


def test_fit_predictor_artifact():
    predictor = joblib.load("models/fit_predictor.pkl")
    assert isinstance(predictor, Pipeline)
    assert hasattr(predictor, "predict_proba")
