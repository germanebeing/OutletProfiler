"""Mission-led weighting layer — the agent-usable surface."""
import polars as pl

from engine import mission


def test_weights_from_mission_reads_intent():
    p = mission.weights_from_mission("launch a premium sku in Delhi", regions=["Delhi"], formats=["kirana"])
    assert p["archetype"] == "premium_launch"
    assert abs(sum(p["weights"].values()) - 1.0) < 1e-6
    assert p["weights"]["value"] == max(p["weights"].values())   # premium weights value
    assert p["filters"]["regions"] == ["Delhi"]


def test_grade_is_a_vector_under_weights(graded):
    """Same outlets, different weights -> different tiers. If tiers were fixed the
    two weightings would produce identical tier columns."""
    a = mission.apply_weights(graded, {"range": 1, "cadence": 0, "recency": 0, "value": 0})
    b = mission.apply_weights(graded, {"range": 0, "cadence": 0, "recency": 1, "value": 0})
    ta = a.filter(pl.col("has_data"))["tier_w"].to_list()
    tb = b.filter(pl.col("has_data"))["tier_w"].to_list()
    assert ta != tb


def test_guard_detects_the_leakier_lever(graded):
    """The guard must react to weights: cadence carries more size than the clean
    range lever, so weighting all-cadence should score higher than all-range."""
    def g(w):
        return mission.guard_on_weights(mission.apply_weights(graded, w), w)["size_correlation"]
    range_only = g({"range": 1, "cadence": 0, "recency": 0, "value": 0})
    cadence_only = g({"range": 0, "cadence": 1, "recency": 0, "value": 0})
    assert cadence_only > range_only


def test_promote_gives_a_path_up(graded):
    # a T3/T4 outlet under balanced weights should get a promotion task or projection
    w = mission.ARCHETYPES["balanced"]["weights"]
    gw = mission.apply_weights(graded, w)
    low = gw.filter(pl.col("has_data") & pl.col("mature") & pl.col("tier_w").is_in(["T3", "T4"]))
    o = low.head(1).to_dicts()[0]
    p = mission.promote(o, graded, w)
    assert p["projected_tier"] in {"T1", "T2", "T3", "T4"}
    assert len(p["tasks"]) >= 1


def test_classify_mission_applies_filters(graded):
    r = mission.classify_mission(graded, "volume scheme in BIHAR kirana", limit=5)
    assert r["archetype"] == "volume_scheme"
    for t in r["targets"]:
        assert t["regionname"] == "BIHAR" and t["format"] == "kirana"
    assert "run_id" in r and r["guard"]["size_correlation"] is not None
