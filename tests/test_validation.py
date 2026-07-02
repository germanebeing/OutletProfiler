"""The hypothesis tests — these are the whole point of the lab.

If any of these fail on real data, the segmentation approach does NOT hold and
must not be wired into the agent contract.
"""


def test_guard_passes(result):
    v = result.validation
    assert v["guard_pass"] is True, v
    assert abs(v["decorrelation_RI_vs_logsize"]) <= 0.5


def test_every_scored_lever_is_size_neutral(result):
    for lever, corr in result.validation["per_lever_vs_logsize"].items():
        assert corr != corr or abs(corr) <= 0.5, f"{lever} leaks size: {corr}"


def test_raw_counts_would_have_leaked(result):
    """Proves the trap is real: the extensive counts we replaced DO correlate
    with size past threshold — so the intensity swap was necessary, not cosmetic."""
    raw = result.validation["raw_count_levers_vs_logsize (the trap)"]
    assert raw["raw_bills"] > 0.5
    assert raw["raw_distinct_skus"] > 0.5


def test_basket_value_correctly_held_out(result):
    # basket_value leaks size hard; it must not be a scored lever.
    assert result.validation["diagnostic_levers_vs_logsize"]["basket_value"] > 0.5


def test_identical_sales_divergence(result):
    # near-equal-throughput outlets must frequently land in different tiers.
    assert result.validation["identical_sales_divergence_pct"] >= 25.0


def test_tier_spread_is_real(result):
    td = result.validation["tier_distribution"]
    assert set(td) >= {"T1", "T2", "T3", "T4"}
    total = sum(td.values())
    # no single tier dominates (the failure mode where everyone is "T1")
    assert max(td.values()) / total < 0.6


def test_multi_company_multi_region(result):
    import polars as pl
    g = result.graded
    assert g["company_name"].n_unique() >= 3
    regions = g.group_by("company_name").agg(pl.col("regionname").n_unique().alias("r"))
    assert regions["r"].min() >= 3, "every company must have >=3 regions"


def test_returns_folded_in_and_size_neutral(result):
    """If returns data is present (Colgate), it must be surfaced and be a
    size-neutral quality signal — not a back-door size proxy."""
    rb = result.validation.get("returns")
    if rb is None:
        import pytest
        pytest.skip("no returns coverage in this dataset")
    assert rb["n_outlets"] > 0
    assert rb["pct_outlets_with_any_return"] > 0
    # honest: return_rate is weakly size-correlated but must stay under the guard
    assert abs(rb["return_rate_vs_logsize"]) <= 0.5


def test_guard_holds_with_all_companies(result):
    # the point of adding Colgate (different warehouse + data model): the guard
    # must still pass on the pooled 5-company set.
    assert result.validation["guard_pass"] is True
    assert result.graded["company_name"].n_unique() >= 4


def test_per_company_decorrelation_is_surfaced(result):
    """The pooled decorrelation must not hide a per-company leak — every company
    gets its own number and breaches are flagged (honesty over headline)."""
    v = result.validation
    pc = v["per_company_decorrelation"]
    assert len(pc) >= 4
    # guard_pass_every_company must agree with the per-company numbers
    breaches = [c for c, r in pc.items() if abs(r) > 0.5]
    assert (len(breaches) == 0) == v["guard_pass_every_company"]
    assert set(v["companies_over_guard"]) == set(breaches)
