"""Mission-led classification — the layer that makes the grade agent-usable.

The insight we locked: the SEGMENT (channel·format·region) is a stored, context-
free property of an outlet; the TIER is never stored — it is the outlet's
per-lever realisation *weighted by the business mission*. Same outlet is T1 for a
premium launch and T3 for a volume scheme.

This module:
  weights_from_mission(text)  -> proposes lever weights + WHY, from a plain ask
  apply_weights(graded, w)    -> re-derives RI + tier under those weights (cheap,
                                 from the stored realisations — no re-pull)
  guard_on_weights(graded, w) -> checks whether THOSE weights turned the grade
                                 back into a size ranking (live safety check)
  promote(outlet, graded, w)  -> gap decomposition -> the OPE/Frontier task that
                                 moves an outlet up a tier, with a projected tier
  classify_mission(...)       -> assembles the full, persistable graded-run record
"""
from __future__ import annotations

import datetime as _dt
import hashlib

import polars as pl

# scored levers -> the stored realisation column that carries them
LEVERS = {"range": "real_range_intensity", "cadence": "real_cadence",
          "recency": "real_recency", "value": "real_value"}
RAW = {"range": "range_intensity", "cadence": "cadence", "recency": "recency", "value": "basket_value"}

# each play carries default weights, a per-lever rationale, and who to target
ARCHETYPES = {
    "premium_launch": {
        "label": "Premium / new-SKU launch",
        "weights": {"range": 0.30, "cadence": 0.10, "recency": 0.20, "value": 0.40},
        "why": {"value": "premium SKUs need outlets whose shoppers already pay up (basket value)",
                "range": "breadth shows they can carry an added premium line",
                "recency": "must be currently active to stock a launch",
                "cadence": "raw frequency matters less than value for a premium play"},
        "target_tiers": ["T1", "T2"], "target_note": "launch to proven, premium-capable outlets first"},
    "volume_scheme": {
        "label": "Volume trade scheme",
        "weights": {"range": 0.25, "cadence": 0.45, "recency": 0.20, "value": 0.10},
        "why": {"cadence": "volume rewards consistent, frequent reordering",
                "range": "enough breadth to absorb the extra volume",
                "recency": "recently active so the scheme lands",
                "value": "kept low — we want throughput, not premium"},
        "target_tiers": ["T3", "T4"], "target_note": "schemes convert unrealised headroom into volume"},
    "frequency": {
        "label": "Order-frequency lift",
        "weights": {"range": 0.15, "cadence": 0.60, "recency": 0.20, "value": 0.05},
        "why": {"cadence": "order frequency IS ordering rhythm vs the peer frontier — the lever being lifted",
                "recency": "must be active enough to reorder more often",
                "range": "secondary for a frequency play",
                "value": "held low — this is about rhythm, not premium"},
        "target_tiers": ["T3", "T4"], "target_note": "lift outlets whose ordering rhythm lags their peer frontier"},
    "distribution": {
        "label": "Assortment / distribution expansion",
        "weights": {"range": 0.55, "cadence": 0.15, "recency": 0.15, "value": 0.15},
        "why": {"range": "distribution IS basket breadth versus the peer frontier",
                "cadence": "some ordering rhythm to sustain new lines",
                "recency": "active enough to add SKUs", "value": "minor here"},
        "target_tiers": ["T3", "T4"], "target_note": "widen the range where breadth lags the peer frontier"},
    "retention": {
        "label": "Retention / protect",
        "weights": {"range": 0.15, "cadence": 0.35, "recency": 0.40, "value": 0.10},
        "why": {"recency": "protect outlets that are still ordering recently",
                "cadence": "consistency is the signal of a defensible account",
                "range": "breadth is secondary when defending", "value": "minor here"},
        "target_tiers": ["T1", "T2"], "target_note": "defend your realised, consistent base"},
    "reactivation": {
        "label": "Reactivation",
        "weights": {"range": 0.15, "cadence": 0.25, "recency": 0.50, "value": 0.10},
        "why": {"recency": "recency is the whole story — find the ones going quiet",
                "cadence": "were they ever consistent?", "range": "minor", "value": "minor"},
        "target_tiers": ["T3", "T4"], "target_note": "win back outlets whose ordering is lapsing"},
    "balanced": {
        "label": "Overall segmentation",
        "weights": {"range": 0.34, "cadence": 0.33, "recency": 0.33, "value": 0.0},
        "why": {"range": "basket breadth", "cadence": "ordering consistency",
                "recency": "freshness", "value": "held out to keep the grade off size"},
        "target_tiers": ["T1", "T2", "T3", "T4"], "target_note": "portfolio view with promotion tasks per outlet"},
}

_KEYWORDS = [
    ("premium_launch", ["premium", "launch", "new sku", "new product", "upgrade", "high-end", "trade up"]),
    ("frequency", ["order frequency", "ordering frequency", "frequency", "order more often", "order more frequently",
                   "reorder", "re-order", "reordering", "ordering rhythm", "order regularly", "how often", "cadence"]),
    ("volume_scheme", ["volume", "scheme", "trade offer", "push volume", "grow volume", "discount", "promo"]),
    ("distribution", ["distribution", "assortment", "must-sell", "must stock", "width", "range expansion", "stock more", "coverage"]),
    ("retention", ["retain", "retention", "protect", "defend", "loyal", "churn risk"]),
    ("reactivation", ["reactivat", "win back", "lapsing", "lapsed", "dormant", "inactive", "gone quiet"]),
    ("balanced", ["segment", "grade all", "improve segmentation", "overall", "classify my outlets"]),
]

TIER_BANDS = [(0.85, "T1"), (0.65, "T2"), (0.45, "T3")]


def weights_from_mission(text: str, regions: list[str] | None = None,
                         formats: list[str] | None = None) -> dict:
    t = (text or "").lower()
    arch = "balanced"
    for name, kws in _KEYWORDS:
        if any(k in t for k in kws):
            arch = name
            break
    a = ARCHETYPES[arch]
    # detect geo / format filters mentioned in the ask
    reg = [r for r in (regions or []) if r and r.lower() in t]
    fmt = [f for f in (formats or []) if f and f.lower() in t]
    return {
        "archetype": arch, "label": a["label"], "weights": dict(a["weights"]),
        "why": dict(a["why"]), "target_tiers": list(a["target_tiers"]),
        "target_note": a["target_note"],
        "filters": {"regions": reg, "formats": fmt},
        "interpretation": _interpret(arch, reg, fmt),
    }


def _interpret(arch: str, reg: list[str], fmt: list[str]) -> str:
    a = ARCHETYPES[arch]
    top = sorted(a["weights"].items(), key=lambda x: -x[1])[:2]
    lead = " and ".join(k for k, _ in top)
    scope = ""
    if reg:
        scope += f" in {', '.join(reg)}"
    if fmt:
        scope += f" ({', '.join(fmt)} only)"
    return f"Read as a {a['label'].lower()}{scope} — weighting {lead} highest."


# ranking of the "outlets to act on" per play: headroom = worst/most-to-gain
# first, best = strongest first, lapsing = least-recent first.
_RANK_BY_ARCH = {"reactivation": "lapsing", "premium_launch": "best", "retention": "best"}


def _interpret_llm(lp: dict) -> str:
    top = sorted(lp["weights"].items(), key=lambda x: -x[1])[:2]
    lead = " and ".join(k for k, w in top if w > 0) or "all levers"
    scope = ""
    if lp["regions"]:
        scope += f" in {', '.join(lp['regions'])}"
    if lp["formats"]:
        scope += f" ({', '.join(lp['formats'])} only)"
    return f"Read as {lp['label'].lower()}{scope} — weighting {lead} highest."


def plan_from_mission(text: str, regions=None, formats=None, company=None) -> dict:
    """LLM lens first (handles arbitrary phrasing); deterministic keyword fallback.
    Tags reasoning_mode so the CPG-OS contract records which path graded the run."""
    try:
        from engine import llm_parse
    except Exception:
        try:
            import llm_parse
        except Exception:
            llm_parse = None
    lp = None
    if llm_parse is not None:
        try:
            if llm_parse.available():
                lp = llm_parse.plan(text, company=company, regions=regions, formats=formats)
        except Exception:
            lp = None
    if lp:
        return {"archetype": "custom", "label": lp["label"], "weights": lp["weights"],
                "why": {}, "target_tiers": lp["target_tiers"],
                "target_note": lp["rationale"] or "planned from your ask",
                "ranking": lp["ranking"],
                "filters": {"regions": lp["regions"], "formats": lp["formats"]},
                "interpretation": _interpret_llm(lp), "reasoning_mode": "reasoning"}
    p = weights_from_mission(text, regions, formats)
    p["ranking"] = _RANK_BY_ARCH.get(p["archetype"], "headroom")
    p["reasoning_mode"] = "deterministic"
    return p


def _ri_weighted(weights: dict) -> pl.Expr:
    # weighted mean over the levers that are present (renormalise over non-null)
    num, den = pl.lit(0.0), pl.lit(0.0)
    for k, col in LEVERS.items():
        w = float(weights.get(k, 0.0))
        if w <= 0:
            continue
        present = pl.col(col).is_not_null()
        num = num + pl.when(present).then(w * pl.col(col)).otherwise(0.0)
        den = den + pl.when(present).then(w).otherwise(0.0)
    return pl.when(den > 0).then(num / den).otherwise(None)


def apply_weights(graded: pl.DataFrame, weights: dict) -> pl.DataFrame:
    ri = _ri_weighted(weights)
    tier = (pl.when(ri >= 0.85).then(pl.lit("T1"))
            .when(ri >= 0.65).then(pl.lit("T2"))
            .when(ri >= 0.45).then(pl.lit("T3")).otherwise(pl.lit("T4")))
    return graded.with_columns(
        RI_w=ri,
        tier_w=pl.when(pl.col("has_data")).then(tier).otherwise(pl.lit("provisional")),
        gap_w=pl.when(ri.is_not_null()).then(1 - ri).otherwise(None))


def _spearman(a: pl.Series, b: pl.Series) -> float:
    m = a.is_not_null() & b.is_not_null()
    a, b = a.filter(m), b.filter(m)
    if a.len() < 20:
        return float("nan")
    return float(pl.DataFrame({"a": a.rank(), "b": b.rank()}).select(pl.corr("a", "b")).item() or 0.0)


def guard_on_weights(graded_w: pl.DataFrame, weights: dict) -> dict:
    d = graded_w.filter(pl.col("mature") & pl.col("RI_w").is_not_null() & (pl.col("line_value") > 0))
    rho = round(abs(_spearman(d.select("RI_w").to_series(),
                              d.select(pl.col("line_value").log1p()).to_series())), 3)
    ok = rho <= 0.5
    if ok and rho < 0.35:
        msg = "These weights keep the grade independent of size — safe."
    elif ok:
        msg = f"Borderline: at these weights the grade is {rho} correlated with size. Usable, watch it."
    else:
        lead = max(weights, key=lambda k: weights.get(k, 0))
        msg = (f"Warning: these weights push the grade to {rho} correlation with size — it is now "
               f"partly a size ranking again. Ease off the leakier levers (you've up-weighted {lead}).")
    return {"size_correlation": rho, "safe": ok, "message": msg,
            "value_weight": round(float(weights.get("value", 0)), 2)}


def promote(outlet: dict, graded: pl.DataFrame, weights: dict) -> dict:
    """Gap decomposition -> the task that moves this outlet up a tier (OPE/Frontier)."""
    peer = graded.filter((pl.col("peer") == outlet["peer"]) &
                         (pl.col("company_name") == outlet["company_name"]) & pl.col("has_data"))
    fr = {
        "range": peer.select(pl.col("range_intensity").quantile(0.8)).item() or 0,
        "cadence": peer.select(pl.col("cadence").quantile(0.8)).item() or 0,
        "recency": peer.select(pl.col("recency").quantile(0.2)).item() or 0,
    }
    reals = {"range": outlet.get("real_range_intensity"), "cadence": outlet.get("real_cadence"),
             "recency": outlet.get("real_recency")}
    # binding lever = the scored lever furthest from frontier, weighted by the mission
    ranked = sorted([(k, (1 - (reals[k] or 1)) * float(weights.get(k, 0) or 0.01)) for k in reals],
                    key=lambda x: -x[1])
    tips = {
        "range": f"widen the basket — carries {round(outlet.get('range_intensity') or 0,1)} SKUs/bill vs a peer frontier of {round(fr['range'],1)}; add the missing must-stock SKUs",
        "cadence": f"order more regularly — active {int((outlet.get('cadence') or 0)*13)} of ~13 weeks vs {int(fr['cadence']*13)} at the frontier",
        "recency": f"re-engage — last order was {int(outlet.get('recency') or 0)} days ago",
    }
    tasks = [{"lever": k, "task": tips[k], "realisation": round(reals[k] or 0, 2)}
             for k, _ in ranked if (reals[k] or 1) < 0.98][:2]
    # projected tier if the binding lever is lifted to frontier
    proj = dict(outlet)
    if ranked:
        proj[LEVERS[ranked[0][0]]] = 1.0
    proj_ri = apply_weights(pl.DataFrame([proj]), weights).select("RI_w").item()
    proj_tier = ("T1" if proj_ri >= 0.85 else "T2" if proj_ri >= 0.65 else "T3" if proj_ri >= 0.45 else "T4")
    return {"tasks": tasks, "current_tier": outlet.get("tier_w") or outlet.get("tier"),
            "projected_tier": proj_tier, "projected_RI": round(proj_ri or 0, 2)}


def _run_id(text: str) -> str:
    stamp = _dt.datetime(2026, 7, 2).isoformat()
    return "run_" + hashlib.sha1((text + stamp).encode()).hexdigest()[:8]


def classify_mission(graded: pl.DataFrame, text: str, company: str | None = None,
                     weight_override: dict | None = None, adjust_reasons: dict | None = None,
                     limit: int = 40, region_filter: list[str] | None = None) -> dict:
    regions = [r for r in graded.select(pl.col("regionname").unique()).to_series().to_list() if r]
    formats = [f for f in graded.select(pl.col("format").unique()).to_series().to_list() if f]
    plan = plan_from_mission(text, regions, formats, company=company)
    weights = weight_override or plan["weights"]
    total = sum(weights.values()) or 1.0
    weights = {k: round(v / total, 3) for k, v in weights.items()}
    # explicit region selection (from the UI / agent payload) overrides any
    # region parsed from the mission text; empty/None = all regions.
    regions_sel = [r for r in (region_filter or []) if r] or plan["filters"]["regions"]
    plan["filters"]["regions"] = regions_sel

    gw = apply_weights(graded, weights)
    guard = guard_on_weights(gw, weights)

    g = gw.filter(pl.col("has_data") & pl.col("mature"))
    if company:
        g = g.filter(pl.col("company_name") == company)
    if regions_sel:
        g = g.filter(pl.col("regionname").is_in(regions_sel))
    for f in plan["filters"]["formats"]:
        g = g.filter(pl.col("format") == f)
    targets = g.filter(pl.col("tier_w").is_in(plan["target_tiers"]))

    # rank the "outlets to act on":
    #  - reactivation: the lapsing (lowest recency realisation) first
    #  - proven plays (premium launch, retention): the best/proven first — you
    #    launch to or defend your strongest outlets
    #  - everyone else (volume, frequency, distribution, balanced): the outlets
    #    with the most UNREALISED headroom first (lowest weighted RI), since the
    #    list is "how to move each one up a tier" — an already-maxed T1 has none.
    def _rank(df: pl.DataFrame) -> pl.DataFrame:
        rk = plan.get("ranking", "headroom")
        if rk == "lapsing":
            return df.sort("real_recency", nulls_last=True)
        if rk == "best":
            return df.sort("RI_w", descending=True, nulls_last=True)
        return df.sort("RI_w", descending=False, nulls_last=True)  # headroom (default)

    targets = _rank(targets)
    # stratify the shown list across the actionable tiers so EVERY tier that has
    # outlets is represented (and therefore filterable) — otherwise a single-key
    # sort + cap collapses the whole page to one tier.
    present = [t for t in ["T1", "T2", "T3", "T4"] if t in plan["target_tiers"]]
    per = max(1, -(-limit // max(1, len(present))))
    parts = [p for p in (targets.filter(pl.col("tier_w") == t).head(per) for t in present) if p.height]
    shown = _rank(pl.concat(parts)) if parts else targets.head(0)
    tier_candidates = dict(zip(*targets.group_by("tier_w").len().to_dict(as_series=False).values()))

    cols = ["outletid", "company_name", "regionname", "city", "format", "channel",
            "peer", "tier_w", "RI_w", "confidence", "range_intensity", "cadence",
            "recency", "basket_value", "return_rate", "line_value"]
    rows = shown.head(limit).select([c for c in cols if c in shown.columns]).to_dicts()
    # tier split is scoped to the same universe as the targets (company + any
    # region/format the ask named) so a single-company run reports that
    # company's split, not the whole portfolio.
    scope = gw.filter(pl.col("has_data"))
    if company:
        scope = scope.filter(pl.col("company_name") == company)
    if regions_sel:
        scope = scope.filter(pl.col("regionname").is_in(regions_sel))
    for f in plan["filters"]["formats"]:
        scope = scope.filter(pl.col("format") == f)
    dist = scope.group_by("tier_w").len().to_dict(as_series=False)

    return {
        "run_id": _run_id(text + str(weights)), "mission": text, "company": company,
        "archetype": plan["archetype"], "label": plan["label"],
        "reasoning_mode": plan.get("reasoning_mode", "deterministic"),
        "ranking": plan.get("ranking", "headroom"),
        "interpretation": plan["interpretation"], "weights": weights, "why": plan["why"],
        "adjust_reasons": adjust_reasons or {}, "guard": guard,
        "target_tiers": plan["target_tiers"], "target_note": plan["target_note"],
        "n_candidates": int(targets.height), "n_shown": len(rows),
        "tier_distribution": dict(zip(dist.get("tier_w", []), dist.get("len", []))),
        "tier_candidates": {k: int(v) for k, v in tier_candidates.items()},
        "targets": rows, "computed_at": _dt.datetime(2026, 7, 2).isoformat(),
    }
