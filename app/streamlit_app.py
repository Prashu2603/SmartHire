"""SmartHire Streamlit portal."""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = ROOT / "app" / "assets" / "smarthire_logo.svg"
sys.path.insert(0, str(ROOT))

from src.config import (
    DEPLOY_JOB_RECOMMENDER_PATH,
    JOB_DATA_PATH,
    JOB_RECOMMENDER_PATH,
    LABEL_ENCODER_PATH,
    LINKEDIN_DATA_PATH,
    MODELS_DIR,
    RESUME_DATA_PATH,
)
from src.data.preprocess import clean_text
from src.features.match_features import calculate_fit_score
from src.models.classifier import load_classifier, load_label_encoder, predict_roles
from src.models.recommender import load_combined_recommender, recommend_jobs
from src.models.skill_gap import analyze_skill_gap
from src.parsing.resume_parser import parse_resume_pdf

st.set_page_config(
    page_title="SmartHire",
    page_icon=":material/work:",
    layout="wide",
)
st.logo(str(LOGO_PATH), size="large")


@st.cache_resource(show_spinner="Loading classification models…")
def load_models():
    return (
        load_classifier(str(MODELS_DIR / "classifier.pkl")),
        joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl"),
        load_label_encoder(str(LABEL_ENCODER_PATH)),
    )


@st.cache_resource(show_spinner="Indexing the job corpus…")
def load_job_index():
    if JOB_RECOMMENDER_PATH.exists():
        return joblib.load(JOB_RECOMMENDER_PATH)
    if DEPLOY_JOB_RECOMMENDER_PATH.exists():
        return joblib.load(DEPLOY_JOB_RECOMMENDER_PATH)
    paths = [JOB_DATA_PATH]
    if LINKEDIN_DATA_PATH.exists():
        paths.append(LINKEDIN_DATA_PATH)
    return load_combined_recommender(paths)


def resume_text(text: str, uploaded_file) -> str:
    """Use pasted text when supplied; otherwise extract an uploaded PDF."""
    if text and text.strip():
        return text.strip()
    if uploaded_file is not None:
        return parse_resume_pdf(uploaded_file).strip()
    return ""


with st.sidebar:
    st.title(":material/work: SmartHire")
    st.caption("Your AI-powered career companion")
    st.markdown(":violet-badge[CLASSICAL ML] :green-badge[RESPONSIBLE AI]")
    st.markdown(
        """
        ### Everything in one place

        :material/check_circle: **Resume classification**  
        :material/search: **Smart job matching**  
        :material/analytics: **Fit and skill-gap scores**  
        :material/picture_as_pdf: **Instant PDF parsing**
        """
    )
    with st.container(border=True):
        st.caption("POWERED BY")
        st.write("TF-IDF · Logistic regression")
        st.write("Cosine similarity · K-Means")

with st.container(border=True):
    hero_copy, hero_visual = st.columns([3, 1], vertical_alignment="center")
    with hero_copy:
        st.markdown(":violet-badge[SMART CAREER PLATFORM]")
        st.title("Turn your resume into your next opportunity")
        st.markdown(
            "Discover the right career path, search **Naukri + LinkedIn job "
            "listings**, and get a personalized roadmap to improve your profile."
        )
        st.markdown(
            ":blue-badge[Upload resume]  →  :violet-badge[Analyze profile]  →  "
            ":green-badge[Find best matches]"
        )
    with hero_visual:
        st.image(str(LOGO_PATH), width="stretch")

with st.container(horizontal=True):
    st.metric("Resume records", "9,544", "Rich training data", border=True)
    st.metric("Job sources", "2", "Naukri + LinkedIn", border=True)
    st.metric("Model accuracy", "86.46%", "Tested performance", border=True)
    st.metric("Career categories", "19", "Multiple domains", border=True)

st.subheader("Your career journey, simplified")
journey = st.columns(3, gap="medium")
with journey[0].container(border=True, height="stretch"):
    st.markdown(":blue-badge[STEP 1]")
    st.subheader(":material/upload_file: Share your story")
    st.write("Paste your resume or upload a PDF. Your data stays inside this app.")
with journey[1].container(border=True, height="stretch"):
    st.markdown(":violet-badge[STEP 2]")
    st.subheader(":material/neurology: Let ML connect the dots")
    st.write("Classical ML identifies your profile, strengths, and best-fit roles.")
with journey[2].container(border=True, height="stretch"):
    st.markdown(":green-badge[STEP 3]")
    st.subheader(":material/rocket_launch: Take the next step")
    st.write("Explore ranked jobs and a focused skill roadmap built for your goal.")

predict_tab, jobs_tab, gap_tab, about_tab = st.tabs(
    [
        ":material/person_search: Profile classification",
        ":material/work: Job matches",
        ":material/trending_up: Skill gap",
        ":material/info: Project details",
    ]
)

with predict_tab:
    st.header("Discover your strongest career category")
    st.caption("Our trained classifier studies your skills and experience in seconds.")
    with st.form("classification_form", border=True):
        text = st.text_area(
            "Resume text",
            height=220,
            placeholder="Paste skills, education, and experience here…",
        )
        pdf = st.file_uploader("Or upload a PDF resume", type=["pdf"], key="class_pdf")
        top_n = st.slider("Predictions to show", 3, 10, 5)
        submitted = st.form_submit_button(
            "Classify resume", type="primary", icon=":material/auto_awesome:"
        )

    if submitted:
        try:
            content = resume_text(text, pdf)
            if not content:
                st.warning("Add resume text or upload a readable PDF.")
            else:
                classifier, vectorizer, encoder = load_models()
                results = predict_roles(
                    content, classifier, vectorizer, encoder, clean_text, top_n
                )
                result_df = pd.DataFrame(results)
                result_df["probability"] *= 100
                best = result_df.iloc[0]
                st.success(
                    f"Best profile: **{best['role']}** ({best['probability']:.1f}% confidence)",
                    icon=":material/check_circle:",
                )
                st.bar_chart(result_df, x="role", y="probability", horizontal=True)
                st.dataframe(
                    result_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "role": st.column_config.TextColumn("Job category"),
                        "probability": st.column_config.ProgressColumn(
                            "Confidence", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                )
        except Exception as exc:
            st.error(f"Could not classify this resume: {exc}", icon=":material/error:")

with jobs_tab:
    st.header("Find jobs built around your strengths")
    st.caption("Every listing is ranked using resume similarity and skill coverage.")
    with st.form("recommendation_form", border=True):
        text = st.text_area(
            "Resume text",
            height=220,
            key="jobs_text",
            placeholder="Paste your resume or key experience…",
        )
        pdf = st.file_uploader("Or upload a PDF resume", type=["pdf"], key="jobs_pdf")
        top_n = st.slider("Jobs to show", 3, 20, 10)
        submitted = st.form_submit_button(
            "Find job matches", type="primary", icon=":material/search:"
        )

    if submitted:
        try:
            content = resume_text(text, pdf)
            if not content:
                st.warning("Add resume text or upload a readable PDF.")
            else:
                jobs, tfidf, vectors = load_job_index()
                matches = recommend_jobs(content, jobs, tfidf, vectors, top_n)
                fits = []
                for _, row in matches.iterrows():
                    job_text = " ".join(
                        str(row.get(column, ""))
                        for column in ("positions", "skills", "related_skils_in_job")
                    )
                    fits.append(
                        calculate_fit_score(
                            content, job_text, row["Similarity Score"]
                        )
                    )
                matches = matches.copy()
                matches["Fit score"] = [item["fit_score"] for item in fits]
                matches["Skill coverage"] = [
                    item["skill_coverage"] for item in fits
                ]
                matches["Matched skills"] = [
                    ", ".join(item["matched_skills"]) or "—" for item in fits
                ]
                matches = matches.drop(
                    columns=["related_skils_in_job", "Similarity Score"],
                    errors="ignore",
                ).rename(
                    columns={
                        "positions": "Job title",
                        "company": "Company",
                        "location": "Location",
                        "experience": "Experience",
                        "skills": "Required skills",
                    }
                )
                st.caption(
                    "Fit score combines 70% TF-IDF similarity and 30% detected-skill coverage."
                )
                st.dataframe(
                    matches,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Job title": st.column_config.TextColumn(pinned=True),
                        "source": st.column_config.TextColumn("Source"),
                        "Fit score": st.column_config.ProgressColumn(
                            format="%.1f%%", min_value=0, max_value=100
                        ),
                        "Skill coverage": st.column_config.ProgressColumn(
                            format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                )
        except Exception as exc:
            st.error(f"Could not build recommendations: {exc}", icon=":material/error:")

with gap_tab:
    st.header("Build your personalized growth roadmap")
    st.caption("Compare your current profile with a target role and close the gap.")
    with st.form("gap_form", border=True):
        text = st.text_area(
            "Resume text",
            height=200,
            key="gap_text",
            placeholder="Paste your skills and experience…",
        )
        target = st.text_input(
            "Target role", placeholder="Example: Data Scientist"
        )
        pdf = st.file_uploader("Or upload a PDF resume", type=["pdf"], key="gap_pdf")
        submitted = st.form_submit_button(
            "Analyze skill gap", type="primary", icon=":material/analytics:"
        )

    if submitted:
        try:
            content = resume_text(text, pdf)
            if not content or not target.strip():
                st.warning("Add a resume and enter a target role.")
            else:
                result = analyze_skill_gap(content, target, str(RESUME_DATA_PATH))
                if not result["target_job_matched"]:
                    st.warning("No related target role was found in the dataset.")
                else:
                    coverage = 100 - result["gap_percentage"]
                    with st.container(horizontal=True):
                        st.metric("Skill coverage", f"{coverage:.1f}%", border=True)
                        st.metric(
                            "Matched skills", len(result["matched"]), border=True
                        )
                        st.metric(
                            "Missing skills", len(result["missing"]), border=True
                        )
                    st.caption(
                        f"Closest dataset role: {result['target_job_matched'].title()}"
                    )
                    left, right = st.columns(2)
                    with left.container(border=True):
                        st.subheader("Skills already present")
                        st.write(", ".join(result["matched"]) or "None detected")
                    with right.container(border=True):
                        st.subheader("Skills to improve")
                        st.write(", ".join(result["missing"]) or "No gap detected")
        except Exception as exc:
            st.error(f"Could not analyze this role: {exc}", icon=":material/error:")

with about_tab:
    st.header("Transparent machine learning, meaningful guidance")
    st.markdown(
        ":violet-badge[NO LLM] :blue-badge[NO LIVE SCRAPING] "
        ":green-badge[INTERPRETABLE SCORES]"
    )
    st.write(
        "SmartHire is an educational classical-machine-learning project. It does "
        "not scrape live job sites and does not use an LLM."
    )
    st.markdown(
        """
        | Component | Method | Evaluation |
        |---|---|---|
        | Resume classifier | TF-IDF + logistic regression | Accuracy 86.46%; macro F1 83.67% |
        | Job recommender | TF-IDF + cosine similarity | Precision@5 27.6% |
        | Fit scoring | Similarity + skill coverage | Interpretable 0–100 score |
        | Fit predictor | Logistic regression | ROC-AUC 69.3% |
        | Role discovery | K-Means, 10 clusters | Silhouette score 0.0564 |
        | Skill gap | Curated skill extraction | Matched and missing skill coverage |
        """
    )
    st.warning(
        "Scores are guidance, not hiring decisions. Keyword-based matching can miss "
        "synonyms and PDF extraction quality depends on the source file.",
        icon=":material/info:",
    )
