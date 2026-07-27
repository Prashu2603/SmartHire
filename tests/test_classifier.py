"""Tests for the classifier module."""

import sys
import os
import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.classifier import load_classifier, load_label_encoder


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class TestClassifier:
    """Tests for the classifier model loading and prediction."""

    @pytest.fixture
    def models(self):
        """Load models once for all tests."""
        classifier_path = os.path.join(MODELS_DIR, "classifier.pkl")
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")

        classifier = load_classifier(classifier_path)
        vectorizer = joblib.load(vectorizer_path)
        encoder = load_label_encoder(encoder_path)

        return classifier, vectorizer, encoder

    def test_classifier_loads(self, models):
        """Should load the Logistic Regression classifier."""
        classifier = models[0]
        assert classifier is not None
        assert hasattr(classifier, "predict")

    def test_vectorizer_loads(self, models):
        """Should load the TF-IDF vectorizer."""
        vectorizer = models[1]
        assert vectorizer is not None
        assert hasattr(vectorizer, "transform")

    def test_encoder_loads(self, models):
        """Should load the label encoder."""
        encoder = models[2]
        assert encoder is not None
        assert hasattr(encoder, "inverse_transform")

    def test_predict_returns_valid_labels(self, models):
        """Should return valid class labels for a sample text."""
        classifier, vectorizer, encoder = models

        sample = "python machine learning data analysis sql tensorflow"
        cleaned = sample.lower()
        vector = vectorizer.transform([cleaned])
        pred = classifier.predict(vector)[0]

        # Pred should be a valid label (not out of range)
        assert 0 <= pred < len(encoder.classes_)

    def test_predict_top_roles_format(self, models):
        """Should produce predictions with correct format."""
        from src.data.preprocess import clean_text
        from src.models.classifier import predict_roles

        classifier, vectorizer, encoder = models

        sample = "python machine learning data analysis sql tensorflow"
        results = predict_roles(sample, classifier, vectorizer, encoder, clean_text, top_n=3)

        assert len(results) == 3
        for r in results:
            assert "role" in r
            assert "probability" in r
            assert 0.0 <= r["probability"] <= 1.0


def test_model_files_exist():
    """All required model files should exist in the models directory."""
    required = [
        "classifier.pkl",
        "tfidf_vectorizer.pkl",
        "label_encoder.pkl",
        "fit_predictor.pkl",
    ]
    for f in required:
        path = os.path.join(MODELS_DIR, f)
        assert os.path.exists(path), f"Missing model: {f}"
        assert os.path.getsize(path) > 0, f"Empty model file: {f}"
