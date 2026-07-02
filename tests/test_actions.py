"""Action projection: the grade is a vector, not a scalar."""
import polars as pl

from engine import actions


def test_all_actions_score_without_error(graded):
    for act in actions.ACTIONS:
        d = actions.recommend(graded, act, limit=10)
        assert d["action"] == act
        assert len(d["targets"]) <= 10
        assert "rationale" in d and "drivers" in d


def test_grade_is_a_vector_not_a_scalar(graded):
    """The core taxonomy claim: an outlet's action-fit differs across actions.
    If every action ranked outlets identically, a single scalar would suffice."""
    prem = actions.score(graded, "premium_launch").select("outletid", "target_score")
    reac = actions.score(graded, "reactivation").select(
        "outletid", pl.col("target_score").alias("r"))
    j = prem.join(reac, on="outletid").drop_nulls()
    # Spearman between the two action rankings should be far from 1.0
    rho = j.select(pl.corr(pl.col("target_score").rank(), pl.col("r").rank())).item()
    assert rho is None or abs(rho) < 0.9, f"actions too collinear (rho={rho})"


def test_scheme_targets_high_headroom(graded):
    d = actions.recommend(graded, "volume_scheme", limit=30)
    tiers = [t["tier"] for t in d["targets"]]
    # scheme should skew to lower tiers (headroom), not T1
    assert tiers.count("T1") < len(tiers) / 2


def test_retention_targets_high_ri(graded):
    d = actions.recommend(graded, "retention", limit=30)
    ris = [t["RI"] for t in d["targets"] if t["RI"] is not None]
    assert sum(ris) / len(ris) > 0.6  # retention picks realised outlets


def test_format_filter(graded):
    d = actions.recommend(graded, "premium_launch", fmt="kirana", limit=20)
    assert all(t["format"] == "kirana" for t in d["targets"])


def test_unknown_action_raises(graded):
    import pytest
    with pytest.raises(ValueError):
        actions.recommend(graded, "not_an_action")
