"""Action handlers — the command seam that calls the segmentation engine and
wraps its output into CPG-OS contract objects (Observation / Diagnosis /
Opportunity), each tagged with reasoning_mode. This is where the Profiler
becomes the first agent to actually EMIT typed contracts (the reference
classifier only persists dict records).

Each handler returns:
    {"outputs": [<contract .model_dump(mode='json')>...],
     "counters": {...}, "produces": [...], "reasoning_modes": [...]}
"""
from __future__ import annotations

from typing import Any, Callable

import polars as pl

from engine import contracts, mission
from engine.contracts import Diagnosis, EntityRefs, Evidence, Observation, Opportunity

from . import identity

_CONF = {"high": 0.9, "low": 0.6, "none": 0.3}
MONTHS_IN_WINDOW = 3  # the pull window (see engine ingestion)
# plays that capture unrealised headroom (target the low-realisation tiers) size
# the opportunity as throughput × gap; plays that place a line at already-proven
# outlets (target T1/T2, gap≈0) size it as an incremental uplift instead.
_GAP_PLAYS = {"volume_scheme", "frequency", "distribution", "reactivation", "balanced"}
_LAUNCH_UPLIFT = 0.15  # assumed incremental share for launch/retention at proven outlets


def _conf(row: dict) -> float:
    return _CONF.get(row.get("confidence"), 0.7)


def _annual_throughput(row: dict) -> float:
    return ((row.get("line_value") or 0.0) / MONTHS_IN_WINDOW) * 12.0


def _annual_opportunity(row: dict, archetype: str) -> float:
    """₹/yr opportunity, play-aware. Gap-closing plays: throughput × unrealised
    fraction (1-RI). Launch/retention plays: throughput × incremental uplift
    (proven outlets have ~no gap, so the value is the new/defended line)."""
    thr = _annual_throughput(row)
    ri = row.get("RI_w")
    if archetype in _GAP_PLAYS:
        basis = max(0.0, 1.0 - ri) if ri is not None else 0.0
    else:
        basis = _LAUNCH_UPLIFT
    return round(thr * basis, 0)


def _obs(run_id: str, tenant_id: str, signal_id: str | None, row: dict,
         guard: dict) -> Observation:
    return Observation(
        agent_id=identity.AGENT_ID, agent_version=identity.AGENT_VERSION,
        run_id=run_id, signal_id=signal_id,
        entity_refs=EntityRefs(tenant_id=tenant_id, outlet_id=str(row["outletid"]),
                               region=row.get("regionname")),
        confidence=_conf(row), reasoning_mode="deterministic",
        kind="outlet_opportunity_grade",
        value={
            "tier": row.get("tier_w"), "RI": row.get("RI_w"), "peer": row.get("peer"),
            "format": row.get("format"), "channel": row.get("channel"),
            "range_intensity": row.get("range_intensity"), "cadence": row.get("cadence"),
            "recency": row.get("recency"), "basket_value": row.get("basket_value"),
            "return_rate": row.get("return_rate"), "company": row.get("company_name"),
        },
        evidence=[
            Evidence(kind="realisation_levers",
                     value={"RI_w": row.get("RI_w"), "gap": None if row.get("RI_w") is None
                            else round(1 - row["RI_w"], 3)}),
            Evidence(kind="decorrelation_guard",
                     value={"size_correlation": guard.get("size_correlation"),
                            "safe": guard.get("safe")},
                     weight=1.0 if guard.get("safe") else 0.4),
        ],
    )


def _opp(run_id: str, tenant_id: str, signal_id: str | None, row: dict,
         label: str, horizon_days: int, safe: bool, archetype: str) -> Opportunity:
    inr = _annual_opportunity(row, archetype)
    ri = row.get("RI_w") or 0.0
    gap_play = archetype in _GAP_PLAYS
    basis_txt = ("unrealised headroom to close" if gap_play
                 else "incremental from placing the line at this proven outlet")
    lvl = "high" if safe else "low"
    return Opportunity(
        agent_id=identity.AGENT_ID, agent_version=identity.AGENT_VERSION,
        run_id=run_id, signal_id=signal_id,
        entity_refs=EntityRefs(tenant_id=tenant_id, outlet_id=str(row["outletid"]),
                               region=row.get("regionname")),
        confidence=_conf(row), reasoning_mode="deterministic",
        summary=(f"{label}: outlet #{row['outletid']} ({row.get('tier_w')}, "
                 f"{row.get('format')}) realises {round(ri, 2)} of its peer frontier — "
                 f"~₹{int(inr):,}/yr {basis_txt}."),
        inr_value=inr, horizon_days=horizon_days, confidence_level=lvl,
    )


# ─── grade_outlets ───────────────────────────────────────────────────────

def grade_outlets(get_store: Callable[[], Any], run_id: str, tenant_id: str,
                  signal_id: str | None, payload: dict) -> dict:
    store = get_store()
    company = payload.get("company")
    if company in ("", "all", "__all", "*"):
        company = None
    text = (payload.get("mission") or payload.get("text") or "").strip()
    weights = payload.get("weights")
    limit = max(1, min(int(payload.get("limit", 40) or 40), 500))  # bound cost per run
    # region/geography selection: an explicit list, a single region, or all (None)
    regions = [r for r in (payload.get("regions") or []) if r]
    if not regions and payload.get("region"):
        regions = [payload["region"]]

    res = mission.classify_mission(store.graded, text, company=company,
                                   weight_override=weights, limit=limit,
                                   region_filter=regions or None)
    guard = res["guard"]
    horizon = 90
    targets = res.get("targets", [])
    label = res.get("label", "Opportunity grade")
    archetype = res.get("archetype", "balanced")

    outputs: list[dict] = []
    for row in targets:
        outputs.append(_obs(run_id, tenant_id, signal_id, row, guard).model_dump(mode="json"))
        outputs.append(_opp(run_id, tenant_id, signal_id, row, label, horizon,
                            bool(guard.get("safe")), archetype).model_dump(mode="json"))

    total_inr = round(sum(_annual_opportunity(r, archetype) for r in targets), 0)
    # mission-text interpretation (text -> weights) is a reasoning step; the
    # grades themselves are deterministic.
    interpreted = bool(text) and not weights
    reasoning_modes = (["reasoning", "deterministic"] if interpreted else ["deterministic"])
    scope_txt = (" in " + ", ".join(regions)) if regions else ""
    summary = (f"{label}: graded {res.get('n_candidates')} actionable outlets for "
               f"{company or 'all companies'}{scope_txt}; emitted {len(targets)} observations + "
               f"{len(targets)} opportunities (~₹{int(total_inr):,}/yr total headroom).")
    counters = {
        "n_observations": len(targets), "n_opportunities": len(targets),
        "n_candidates": res.get("n_candidates"), "tier_distribution": res.get("tier_distribution"),
        "total_inr_opportunity": total_inr, "weights": res.get("weights"),
        "guard": guard, "interpretation": res.get("interpretation"),
        "archetype": res.get("archetype"), "label": label,
        "target_tiers": res.get("target_tiers"), "company": company or "all companies",
        "regions": regions or "all",
    }
    return {"outputs": outputs, "counters": counters, "summary": summary, "verdict": None,
            "produces": ["observation", "opportunity"], "reasoning_modes": reasoning_modes}


# ─── validate_opportunity_hypothesis ─────────────────────────────────────

_DORMANT = ("dormant", "lapsed", "lapsing", "inactive", "gone quiet", "gone dormant", "stopped ordering", "sleeping")
_WEAK = ("underperform", "under-perform", "under performing", "underperforming", "weak", "struggling", "poor", "low potential", "declining")
_STRONG = ("top", "best", "strong", "high potential", "premium", "proven", "high performing", "outperform", "star")


def _scope(g: pl.DataFrame, payload: dict) -> tuple[pl.DataFrame, dict]:
    company = payload.get("company")
    if company in ("", "all", "__all", "*", None):
        company = None
    scope = payload.get("scope") or {}
    sub = g.filter(pl.col("has_data"))
    applied = {}
    if company:
        sub = sub.filter(pl.col("company_name") == company); applied["company"] = company
    if scope.get("region"):
        sub = sub.filter(pl.col("regionname") == scope["region"]); applied["region"] = scope["region"]
    if scope.get("format"):
        sub = sub.filter(pl.col("format") == scope["format"]); applied["format"] = scope["format"]
    if scope.get("tier"):
        sub = sub.filter(pl.col("tier") == scope["tier"]); applied["tier"] = scope["tier"]
    return sub, applied


def validate_opportunity_hypothesis(get_store: Callable[[], Any], run_id: str,
                                    tenant_id: str, signal_id: str | None,
                                    payload: dict) -> dict:
    store = get_store()
    g = store.graded
    hyp = (payload.get("hypothesis") or payload.get("text") or "").strip()
    t = hyp.lower()
    sub, applied = _scope(g, payload)
    n = sub.height

    # decorrelation guard on this company (evidence the grade isn't a size proxy)
    guard_corr = None
    try:
        pcd = store.validation.get("per_company_decorrelation", {})
        guard_corr = pcd.get(applied.get("company")) if applied.get("company") else \
            store.validation.get("decorrelation_RI_vs_logsize")
    except Exception:
        pass

    verdict = "inconclusive"
    reasoning_mode: str = "reasoning"
    metric_name = "n/a"
    share = None
    root_causes: list[str] = []

    if n < 10:
        verdict, reasoning_mode = "inconclusive", "deterministic"
        root_causes = [f"only {n} outlets in scope — too few to settle the claim"]
        summary = f"Inconclusive: {n} outlets match the scope {applied or 'all'}; need ≥10."
    else:
        def _share(expr) -> float:
            return round(sub.select(expr.mean()).item() or 0.0, 3)

        if any(k in t for k in _DORMANT):
            metric_name = "stale_share (recency realisation < 0.5)"
            share = _share((pl.col("real_recency") < 0.5).cast(pl.Float64))
            reasoning_mode = "deterministic"
            verdict = "confirm" if share >= 0.5 else "refute" if share <= 0.2 else "inconclusive"
            root_causes = [f"{int(share*100)}% of the {n} in-scope outlets are stale "
                           f"(recency realisation < 0.5)"]
        elif any(k in t for k in _WEAK):
            metric_name = "underperforming_share (tier in T3/T4)"
            share = _share(pl.col("tier").is_in(["T3", "T4"]).cast(pl.Float64))
            reasoning_mode = "deterministic"
            verdict = "confirm" if share >= 0.6 else "refute" if share <= 0.3 else "inconclusive"
            root_causes = [f"{int(share*100)}% of the {n} in-scope outlets sit in T3/T4 "
                           f"(below their peer frontier)"]
        elif any(k in t for k in _STRONG):
            metric_name = "strong_share (tier in T1/T2)"
            share = _share(pl.col("tier").is_in(["T1", "T2"]).cast(pl.Float64))
            reasoning_mode = "deterministic"
            verdict = "confirm" if share >= 0.6 else "refute" if share <= 0.3 else "inconclusive"
            root_causes = [f"{int(share*100)}% of the {n} in-scope outlets are T1/T2 "
                           f"(at/near their peer frontier)"]
        else:
            verdict, reasoning_mode = "inconclusive", "reasoning"
            root_causes = ["the hypothesis did not map to a measurable claim "
                           "(dormancy / under- or over-performance); rephrase or scope it"]
        share_txt = f" ({metric_name} = {share})" if share is not None else ""
        summary = (f"{verdict.upper()}: over {n} in-scope outlets{share_txt}. "
                   f"Re-graded against their peer frontiers.")

    # supporting observations (capped) — the freshly re-graded in-scope outlets
    cols = ["outletid", "company_name", "regionname", "format", "channel", "peer",
            "tier", "RI", "range_intensity", "cadence", "recency", "basket_value",
            "return_rate", "confidence"]
    sample = sub.sort("RI", descending=True, nulls_last=True).head(25) \
        .select([c for c in cols if c in sub.columns]).to_dicts()
    obs_outputs = []
    for row in sample:
        row = dict(row)
        row["tier_w"] = row.get("tier"); row["RI_w"] = row.get("RI")
        obs_outputs.append(
            _obs(run_id, tenant_id, signal_id, row,
                 {"size_correlation": guard_corr, "safe": (guard_corr or 1) <= 0.5}).model_dump(mode="json"))

    diag = Diagnosis(
        agent_id=identity.AGENT_ID, agent_version=identity.AGENT_VERSION,
        run_id=run_id, signal_id=signal_id,
        entity_refs=EntityRefs(tenant_id=tenant_id, region=(payload.get("scope") or {}).get("region")),
        confidence=0.85 if verdict != "inconclusive" else 0.5,
        reasoning_mode=reasoning_mode, verdict=verdict,
        summary=summary, root_causes=root_causes,
        contributing_output_ids=[str(r["outletid"]) for r in sample],
        evidence=[
            Evidence(kind="distribution_compare",
                     value={"scope": applied, "n_in_scope": n, "metric": metric_name, "value": share}),
            Evidence(kind="decorrelation_guard",
                     value={"company_size_correlation": guard_corr,
                            "note": "grade is decorrelated from size; not a size proxy"},
                     weight=1.0 if (guard_corr or 1) <= 0.5 else 0.4),
        ],
    )
    outputs = [diag.model_dump(mode="json")] + obs_outputs
    counters = {"verdict": verdict, "reasoning_mode": reasoning_mode, "n_in_scope": n,
                "metric": metric_name, "value": share, "scope": applied,
                "hypothesis": hyp, "n_observations": len(obs_outputs)}
    return {"outputs": outputs, "counters": counters, "summary": summary, "verdict": verdict,
            "produces": ["diagnosis", "observation"], "reasoning_modes": [reasoning_mode]}


# ─── analyze_outcome (M5 stub) ───────────────────────────────────────────

def analyze_outcome(get_store: Callable[[], Any], run_id: str, tenant_id: str,
                    signal_id: str | None, payload: dict) -> dict:
    diag = Diagnosis(
        agent_id=identity.AGENT_ID, agent_version=identity.AGENT_VERSION,
        run_id=run_id, signal_id=signal_id,
        entity_refs=EntityRefs(tenant_id=tenant_id),
        confidence=0.3, reasoning_mode="deterministic", verdict="inconclusive",
        summary="Outcome measurement is not yet implemented (CPG-OS M5 'measure' milestone).",
        root_causes=["no post-action outcome data is collected yet"],
    )
    return {"outputs": [diag.model_dump(mode="json")], "counters": {"stub": True},
            "summary": diag.summary, "verdict": "inconclusive",
            "produces": ["diagnosis"], "reasoning_modes": ["deterministic"]}


HANDLERS = {
    "grade_outlets": grade_outlets,
    "validate_opportunity_hypothesis": validate_opportunity_hypothesis,
    "analyze_outcome": analyze_outcome,
}
