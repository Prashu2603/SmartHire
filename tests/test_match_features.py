from src.features.match_features import calculate_fit_score


def test_fit_score_rewards_skill_overlap():
    high = calculate_fit_score("Python SQL", "Python SQL analytics", 0.5)
    low = calculate_fit_score("Java", "Python SQL analytics", 0.5)
    assert high["fit_score"] > low["fit_score"]
    assert high["skill_coverage"] == 100.0


def test_fit_score_is_bounded():
    result = calculate_fit_score("Python", "Python", 2.0)
    assert 0 <= result["fit_score"] <= 100
