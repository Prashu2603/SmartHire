import pandas as pd

from src.models.recommender import load_recommender, recommend_jobs


def test_linkedin_schema_is_normalized(tmp_path):
    path = tmp_path / "postings.csv"
    pd.DataFrame(
        {
            "title": ["Data Scientist", "Accountant"],
            "description": ["Python machine learning SQL", "Ledger audit"],
            "company_name": ["Example AI", "Example Finance"],
            "location": ["Hyderabad", "Mumbai"],
            "formatted_experience_level": ["Entry level", "Associate"],
        }
    ).to_csv(path, index=False)

    jobs, tfidf, vectors = load_recommender(str(path))
    result = recommend_jobs("Python ML SQL", jobs, tfidf, vectors, top_n=1)

    assert jobs.loc[0, "source"] == "LinkedIn"
    assert result.iloc[0]["positions"] == "Data Scientist"
