"""Project paths and reproducible model settings."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESUME_DATA_PATH = DATA_DIR / "raw" / "resume_data.csv"
JOB_DATA_PATH = DATA_DIR / "raw" / "naukri_com-job_sample.csv"
LINKEDIN_DATA_PATH = DATA_DIR / "raw" / "postings.csv"
MODELS_DIR = ROOT_DIR / "models"
JOB_RECOMMENDER_PATH = MODELS_DIR / "job_recommender.pkl"
DEPLOY_JOB_RECOMMENDER_PATH = MODELS_DIR / "job_recommender_deploy.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FIT_PREDICTOR_PATH = MODELS_DIR / "fit_predictor.pkl"
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000
