"""Engine mechanics: typing, maturity gate, realisation, cold-start."""
import polars as pl

from engine import segment
from conftest import write_synth


# ─── typing ───

def test_clean_format():
    f = segment._clean_format
    assert f("Kirana Store/General Store.") == "kirana"
    assert f("General /Kirana shop") == "kirana"
    assert f("Chemist & Druggist") == "chemist"
    assert f("Pan Plus") == "pan_kiosk"
    assert f("D-Mart Supermarket") == "supermarket"
    assert f("Bakery & Confectionery") == "horeca"
    assert f("Wholesale Stockist") == "wholesale"
    assert f(None) == "unknown"
    assert f("") == "unknown"
    assert f("SMT OUTLET") == "other"


def test_clean_channel():
    c = segment._clean_channel
    assert c("General Trade") == "GT"
    assert c("GT") == "GT"
    assert c("MT") == "MT"
    assert c("Modern Trade") == "MT"
    assert c("HORECA") == "HoReCa"
    assert c(None) == "unknown"
    assert c("Fusion") == "GT"  # company-specific route -> GT base


# ─── maturity gate ───

def test_maturity_gate_flags_thin(tmp_path):
    rows = [
        # one thin outlet (1 bill, 1 week) and one mature (5 bills, 5 weeks)
        {"has_data": True, "bills": 1, "order_weeks": 1, "distinct_skus": 1,
         "line_value": 100.0, "total_value": 100.0, "last_bill": "2026-06-01 10:00:00.000"},
        {"has_data": True, "bills": 5, "order_weeks": 5, "distinct_skus": 10,
         "line_value": 5000.0, "total_value": 5000.0, "last_bill": "2026-06-20 10:00:00.000"},
    ]
    g = segment.run_engine(write_synth(tmp_path, rows)).graded
    thin = g.filter(pl.col("bills") == 1)
    mature = g.filter(pl.col("bills") == 5)
    assert thin["mature"][0] is False
    assert thin["confidence"][0] == "low"
    assert mature["mature"][0] is True


def test_thin_outlet_not_in_recommendations(tmp_path):
    from engine import actions
    rows = [{"has_data": True, "bills": 1, "order_weeks": 1, "distinct_skus": 1,
             "line_value": 100.0, "total_value": 100.0, "last_bill": "2026-06-01 10:00:00.000"}
            for _ in range(3)]
    rows += [{"has_data": True, "bills": 6, "order_weeks": 6, "distinct_skus": 12,
              "line_value": 6000.0, "total_value": 6000.0, "last_bill": "2026-06-25 10:00:00.000"}
             for _ in range(3)]
    g = segment.run_engine(write_synth(tmp_path, rows)).graded
    rec = actions.recommend(g, "premium_launch", limit=50)
    assert rec["excluded_thin"] == 3
    assert all(t["outletid"] not in g.filter(pl.col("bills") == 1)["outletid"].to_list()
               for t in rec["targets"])


# ─── realisation / robustness ───

def test_no_crash_on_zero_bills_has_data(tmp_path):
    # defensive: a has_data row with 0 bills must not divide-by-zero
    rows = [{"has_data": True, "bills": 0, "order_weeks": 0, "distinct_skus": 0,
             "line_value": 0.0, "total_value": 0.0, "last_bill": None}]
    g = segment.run_engine(write_synth(tmp_path, rows)).graded
    assert g["range_intensity"][0] is None  # ok=False -> null, no crash


def test_recency_parses(tmp_path):
    rows = [{"has_data": True, "bills": 4, "order_weeks": 4, "distinct_skus": 8,
             "line_value": 4000.0, "total_value": 4000.0, "last_bill": "2026-06-21 12:00:00.000"}]
    g = segment.run_engine(write_synth(tmp_path, rows)).graded
    # TODAY is 2026-07-01 -> ~10 days
    assert g["recency"][0] is not None
    assert 5 <= g["recency"][0] <= 15


def test_realisation_capped_at_one(graded):
    for c in ["real_range_intensity", "real_cadence", "real_recency"]:
        col = graded[c].drop_nulls()
        if col.len():
            assert col.max() <= 1.0 + 1e-9


def test_ri_between_zero_and_one(graded):
    ri = graded.filter(pl.col("has_data"))["RI"].drop_nulls()
    assert ri.min() >= 0.0 and ri.max() <= 1.0


# ─── cold start ───

def test_nodata_never_gets_realisation_tier(graded):
    nod = graded.filter(~pl.col("has_data"))
    assert nod.height > 0
    assert set(nod["tier"].unique().to_list()) == {"provisional"}


def test_return_rate_bounded_and_null_without_coverage(tmp_path):
    """return_rate is [0,1] where returns exist, and NULL (not 0) for companies
    with no returns coverage — so 'no returns data' != 'zero returns'."""
    rows = [
        {"has_data": True, "bills": 5, "order_weeks": 5, "distinct_skus": 10,
         "line_value": 1000.0, "total_value": 1000.0, "returns_value": 200.0, "return_bills": 3,
         "last_bill": "2026-06-20 00:00:00.000"},
        {"has_data": True, "bills": 5, "order_weeks": 5, "distinct_skus": 10,
         "line_value": 1000.0, "total_value": 1000.0, "returns_value": 0.0, "return_bills": 0,
         "last_bill": "2026-06-20 00:00:00.000"},
    ]
    g = segment.run_engine(write_synth(tmp_path, rows)).graded
    rr = g["return_rate"].drop_nulls()
    assert rr.min() >= 0.0 and rr.max() <= 1.0
    # 200 / (1000+200) = 0.1667
    assert abs(g.filter(pl.col("returns_value") == 200.0)["return_rate"][0] - 0.1667) < 0.01

    # a synth frame with no returns column at all -> return_rate all null
    plain = [{"has_data": True, "bills": 5, "order_weeks": 5, "distinct_skus": 10,
              "line_value": 1000.0, "total_value": 1000.0, "last_bill": "2026-06-20 00:00:00.000"}]
    g2 = segment.run_engine(write_synth(tmp_path, plain)).graded
    assert g2["return_rate"].null_count() == g2.height


def test_cold_start_matches_baseline(result):
    from engine import actions
    cs = actions.cold_start(result.graded, result.baseline)
    assert cs["n_nodata"] > 0
    assert cs["match_rate_pct"] >= 80.0
    # potential is a band, not a tier
    assert set(cs["potential_distribution"]) <= {"high", "medium", "low", "unknown"}


def test_cold_start_cross_company_is_honest(result):
    """Genuine cross-company support must EXCLUDE an outlet's own company — the
    total match rate overstated cross-company generalisation before this fix."""
    from engine import actions
    cs = actions.cold_start(result.graded, result.baseline)
    assert "cross_company_match_rate_pct" in cs and "n_own_company_only" in cs
    # cross-company support can only be <= total match, never inflated past it
    assert cs["cross_company_match_rate_pct"] <= cs["match_rate_pct"]
    assert cs["n_cross_company_support"] + cs["n_own_company_only"] == cs["n_matched_to_baseline"]
