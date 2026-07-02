"""Action projection + cold-start — the "so what" layer on top of the grade.

The grade is a VECTOR, never a scalar A. An outlet is never just "good"; it is
"good FOR X". Each action reads the stored primitives and projects an
action-specific target score, so the same outlet can be a top premium-launch
target and a poor scheme target. This is what the launch recommender calls.

Cold-start: no-history outlets can't have a realisation, but their structural
peer-tuple has a cross-company baseline — so we can still say what comparable
outlets achieve and hence the outlet's POTENTIAL band (never a realisation tier).
"""
from __future__ import annotations

import polars as pl

# action -> (label, one-line rationale, which primitives drive targeting)
ACTIONS = {
    "premium_launch": (
        "Premium / new-SKU launch",
        "Proven executors in premium-capacity peers — they can sell breadth and reorder reliably.",
        ["basket_value (premium-capacity proxy)", "RI (execution)", "range_intensity"],
    ),
    "volume_scheme": (
        "Volume trade scheme",
        "High headroom in high-potential peers — a scheme converts unrealised frontier into volume.",
        ["gap = 1-RI", "peer potential", "cadence"],
    ),
    "assortment_expansion": (
        "Assortment / must-stock expansion",
        "Narrow baskets versus their peer frontier — the clearest range-gap to close.",
        ["range_intensity vs peer frontier"],
    ),
    "retention": (
        "Retention / protect",
        "Realised, consistent outlets worth defending against competitive switching.",
        ["RI", "cadence"],
    ),
    "reactivation": (
        "Reactivation",
        "Were active, now lapsing — recency is stretched but the outlet is real.",
        ["recency", "prior activity"],
    ),
}


def _pctile(col: str) -> pl.Expr:
    # min-max normalised rank in [0,1]; robust to outliers via rank not raw value.
    # Guard the degenerate single-value case (max==min) so we never emit NaN.
    r = pl.col(col).rank()
    span = r.max() - r.min()
    return pl.when(span > 0).then((r - r.min()) / span).otherwise(pl.lit(0.5)).fill_null(0.0)


def _peer_premium_capacity(graded: pl.DataFrame) -> pl.DataFrame:
    # premium capacity is a PEER property (structural-ish): the peer's typical
    # basket value, ranked across peers. Uses basket_value knowingly — size is
    # allowed to inform ACTION targeting, it is only banned from the SEGMENT axis.
    cap = (graded.filter(pl.col("has_data"))
           .group_by("peer").agg(pl.col("basket_value").median().alias("peer_basket")))
    cap = cap.with_columns(premium_capacity=_pctile("peer_basket"))
    return cap.select(["peer", "peer_basket", "premium_capacity"])


def score(graded: pl.DataFrame, action: str) -> pl.DataFrame:
    """Return has_data outlets ranked by target score for `action`, with reasons."""
    g = graded.filter(pl.col("has_data"))
    cap = _peer_premium_capacity(graded)
    g = g.join(cap, on="peer", how="left")

    if action == "premium_launch":
        # returns are a launch risk: a high return-rate outlet is a poor place to
        # push a new premium SKU. Penalise it where returns data exists (Colgate).
        rr = pl.col("return_rate").fill_null(0.0) if "return_rate" in g.columns else pl.lit(0.0)
        g = g.with_columns(
            target_score=(0.55 * pl.col("premium_capacity").fill_null(0.0)
                          + 0.45 * pl.col("RI").fill_null(0.0)
                          - 0.30 * rr).clip(0.0, 1.0),
        ).with_columns(reason=pl.format(
            "premium-capacity peer {} · execution RI {}{}",
            (pl.col("premium_capacity") * 100).round(0).cast(pl.Int64),
            pl.col("RI").round(2),
            pl.when(rr > 0.05).then(pl.format(" · ⚠ returns {}%", (rr * 100).round(0).cast(pl.Int64)))
            .otherwise(pl.lit(""))))
    elif action == "volume_scheme":
        g = g.with_columns(
            target_score=(0.6 * pl.col("gap").fill_null(0.0)
                          + 0.4 * _pctile("basket_value")),
        ).with_columns(reason=pl.format(
            "headroom {} · basket ₹{}",
            (pl.col("gap") * 100).round(0).cast(pl.Int64),
            pl.col("basket_value").round(0).cast(pl.Int64)))
    elif action == "assortment_expansion":
        g = g.with_columns(
            target_score=(1 - pl.col("real_range_intensity").fill_null(1.0)),
        ).with_columns(reason=pl.format(
            "range realisation {} of peer frontier",
            pl.col("real_range_intensity").round(2)))
    elif action == "retention":
        g = g.with_columns(
            target_score=(0.7 * pl.col("RI").fill_null(0.0)
                          + 0.3 * pl.col("real_cadence").fill_null(0.0)),
        ).with_columns(reason=pl.format("RI {} · cadence {}",
            pl.col("RI").round(2), pl.col("real_cadence").round(2)))
    elif action == "reactivation":
        g = g.with_columns(
            target_score=_pctile("recency"),
        ).with_columns(reason=pl.format("{} days since last order",
            pl.col("recency").round(0).cast(pl.Int64)))
    else:
        raise ValueError(f"unknown action {action}")

    return g.sort("target_score", descending=True)


def recommend(graded: pl.DataFrame, action: str, company: str | None = None,
              region: str | None = None, tiers: list[str] | None = None,
              fmt: str | None = None, limit: int = 50, include_thin: bool = False) -> dict:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action}")
    ranked = score(graded, action)
    excluded_thin = 0
    if not include_thin and "mature" in ranked.columns:
        # never prescribe against a thin, low-confidence grade
        excluded_thin = int(ranked.filter(~pl.col("mature")).height)
        ranked = ranked.filter(pl.col("mature"))
    if company:
        ranked = ranked.filter(pl.col("company_name") == company)
    if region:
        ranked = ranked.filter(pl.col("regionname") == region)
    if fmt:
        ranked = ranked.filter(pl.col("format") == fmt)
    if tiers:
        ranked = ranked.filter(pl.col("tier").is_in(tiers))
    top = ranked.head(limit)
    cols = ["outletid", "company_name", "regionname", "peer", "format", "tier", "RI",
            "range_intensity", "cadence", "recency", "basket_value", "return_rate",
            "target_score", "reason"]
    label, rationale, drivers = ACTIONS[action]
    return {
        "action": action, "label": label, "rationale": rationale, "drivers": drivers,
        "n_candidates": ranked.height, "excluded_thin": excluded_thin,
        "targets": top.select([c for c in cols if c in top.columns]).to_dicts(),
    }


def cold_start(graded: pl.DataFrame, baseline: pl.DataFrame) -> dict:
    """Grade no-data outlets off the cross-company peer-tuple baseline.

    Output is a POTENTIAL band (from what comparable outlets achieve), never a
    realisation tier — the outlet has no transactions to realise anything yet.
    """
    nod = graded.filter(~pl.col("has_data")).with_columns(
        tuple_key=(pl.col("channel") + "·" + pl.col("format")))
    joined = nod.join(baseline, on="tuple_key", how="left")
    # GENUINE cross-company support: a no-data outlet's baseline must draw on a
    # company OTHER than its own, else the "cross-company" transfer is circular
    # (just other same-company outlets). Excluding the own company is the honest test.
    def other_cos(row):
        ids = row["company_ids"]
        if ids is None:
            return 0
        return len([c for c in ids if c != row["company_id"]])
    joined = joined.with_columns(
        n_other_companies=pl.struct(["company_ids", "company_id"])
        .map_elements(other_cos, return_dtype=pl.Int64))
    joined = joined.with_columns(
        potential=pl.when(pl.col("front_range_intensity").is_null()).then(pl.lit("unknown"))
        .when(pl.col("front_range_intensity") >= pl.col("front_range_intensity").quantile(0.66))
            .then(pl.lit("high"))
        .when(pl.col("front_range_intensity") >= pl.col("front_range_intensity").quantile(0.33))
            .then(pl.lit("medium"))
        .otherwise(pl.lit("low")),
        matched=pl.col("companies").is_not_null(),
        cross_company=pl.col("n_other_companies") >= 1,
    )
    dist = joined.group_by("potential").len().sort("len", descending=True)
    matched = int(joined.filter(pl.col("matched")).height)
    cross = int(joined.filter(pl.col("cross_company")).height)
    cols = ["outletid", "company_name", "regionname", "tuple_key", "potential",
            "companies", "n_other_companies", "front_range_intensity", "front_cadence", "med_basket_value"]
    return {
        "n_nodata": joined.height,
        "n_matched_to_baseline": matched,
        "match_rate_pct": round(100 * matched / joined.height, 1) if joined.height else 0.0,
        # the honest headline: how many rest on a genuinely DIFFERENT company
        "n_cross_company_support": cross,
        "cross_company_match_rate_pct": round(100 * cross / joined.height, 1) if joined.height else 0.0,
        "n_own_company_only": matched - cross,
        "potential_distribution": dict(zip(*dist.to_dict(as_series=False).values())),
        "sample": joined.select([c for c in cols if c in joined.columns]).head(50).to_dicts(),
    }
