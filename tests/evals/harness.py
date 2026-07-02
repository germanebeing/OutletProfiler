"""Scored behaviour-eval harness for the Outlet Profiler.

Runs the labeled fixtures in cases.py against the real engine + agent handlers
and scores three behaviours with pass gates:

  mission             — weights_from_mission picks the right archetype (absolute)
  reasoning_mode      — validate tags deterministic vs reasoning correctly (absolute)
  verdict_absolute    — forced verdicts (no-keyword / empty-scope) hold (absolute)
  verdict_differential— the agent's verdict matches an INDEPENDENT re-computation
                        of the same metric+threshold over the graded frame

Run standalone:  PYTHONPATH=.:tests python -m tests.evals.harness
Gated in CI:     tests/test_evals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent import handlers  # noqa: E402
from engine import mission  # noqa: E402
from tests.evals.cases import GATES, MISSION_CASES, VERDICT_CASES  # noqa: E402

# independent reference of the documented verdict rule (a second implementation,
# NOT importing the handler internals — so the differential check is meaningful)
_DORMANT = ("dormant", "lapsed", "lapsing", "inactive", "gone quiet", "gone dormant", "stopped ordering", "sleeping")
_WEAK = ("underperform", "under-perform", "under performing", "underperforming", "weak", "struggling", "poor", "low potential", "declining")
_STRONG = ("top", "best", "strong", "high potential", "premium", "proven", "high performing", "outperform", "star")


def _band(share: float, hi: float = 0.6, lo: float = 0.3) -> str:
    return "confirm" if share >= hi else "refute" if share <= lo else "inconclusive"


def _reference(graded: pl.DataFrame, company: str, hypothesis: str, scope: dict) -> tuple[str, str]:
    sub = graded.filter(pl.col("has_data"))
    if company:
        sub = sub.filter(pl.col("company_name") == company)
    if scope.get("region"):
        sub = sub.filter(pl.col("regionname") == scope["region"])
    if scope.get("format"):
        sub = sub.filter(pl.col("format") == scope["format"])
    if scope.get("tier"):
        sub = sub.filter(pl.col("tier") == scope["tier"])
    n = sub.height
    t = hypothesis.lower()
    if n < 10:
        return "inconclusive", "deterministic"

    def share(expr) -> float:
        return round(sub.select(expr.mean()).item() or 0.0, 3)

    if any(k in t for k in _DORMANT):
        return _band(share((pl.col("real_recency") < 0.5).cast(pl.Float64)), 0.5, 0.2), "deterministic"
    if any(k in t for k in _WEAK):
        return _band(share(pl.col("tier").is_in(["T3", "T4"]).cast(pl.Float64))), "deterministic"
    if any(k in t for k in _STRONG):
        return _band(share(pl.col("tier").is_in(["T1", "T2"]).cast(pl.Float64))), "deterministic"
    return "inconclusive", "reasoning"


class _FakeStore:
    def __init__(self, result):
        self.graded = result.graded
        self.validation = result.validation


def run_evals(result) -> dict:
    graded = result.graded
    fake = _FakeStore(result)
    # guard: a labeled company must actually be in the graded frame, else the
    # case would pass vacuously (n=0 -> inconclusive). Catches fixture drift.
    present = set(graded.select(pl.col("company_name").unique()).to_series().to_list())
    missing = [c["company"] for c in VERDICT_CASES if c["company"] not in present]
    assert not missing, f"eval fixtures reference companies absent from the graded frame: {missing}"
    report: dict = {"mission": [], "verdict": [], "counts": {}, "gate_pass": True}

    # 1. mission -> archetype
    m_ok = 0
    for text, expect in MISSION_CASES:
        got = mission.weights_from_mission(text, regions=[], formats=[])["archetype"]
        ok = got == expect
        m_ok += ok
        report["mission"].append({"text": text, "expect": expect, "got": got, "ok": ok})

    # 2/3. verdict + reasoning_mode
    mode_ok = v_abs_ok = v_abs_n = v_diff_ok = 0
    for c in VERDICT_CASES:
        out = handlers.validate_opportunity_hypothesis(
            lambda: fake, "eval", "default", None,
            {"company": c["company"], "hypothesis": c["hypothesis"], "scope": c.get("scope", {})})
        diag = next(o for o in out["outputs"] if o["type"] == "diagnosis")
        got_v, got_m = diag["verdict"], diag["reasoning_mode"]
        ref_v, ref_m = _reference(graded, c["company"], c["hypothesis"], c.get("scope", {}))
        mode_ok += (got_m == c["mode"])
        diff_ok = (got_v == ref_v)
        v_diff_ok += diff_ok
        row = {"company": c["company"], "hyp": c["hypothesis"], "scope": c.get("scope", {}),
               "verdict": got_v, "ref_verdict": ref_v, "mode": got_m, "expect_mode": c["mode"],
               "mode_ok": got_m == c["mode"], "diff_ok": diff_ok,
               "n_in_scope": out["counters"].get("n_in_scope"), "metric": out["counters"].get("value")}
        if "verdict" in c:
            v_abs_n += 1
            row["expect_verdict"] = c["verdict"]
            row["abs_ok"] = got_v == c["verdict"]
            v_abs_ok += row["abs_ok"]
        report["verdict"].append(row)

    nm, nv = len(MISSION_CASES), len(VERDICT_CASES)
    scores = {
        "mission": m_ok / nm,
        "reasoning_mode": mode_ok / nv,
        "verdict_absolute": (v_abs_ok / v_abs_n) if v_abs_n else 1.0,
        "verdict_differential": v_diff_ok / nv,
    }
    report["counts"] = {"mission": f"{m_ok}/{nm}", "reasoning_mode": f"{mode_ok}/{nv}",
                        "verdict_absolute": f"{v_abs_ok}/{v_abs_n}", "verdict_differential": f"{v_diff_ok}/{nv}"}
    report["scores"] = scores
    report["gates"] = GATES
    report["gate_pass"] = all(scores[k] >= GATES[k] for k in GATES)
    return report


def _print(report: dict) -> None:
    print("\n=== Outlet Profiler — behaviour evals ===")
    for r in report["mission"]:
        if not r["ok"]:
            print(f"  MISSION FAIL: {r['text']!r} -> {r['got']} (expect {r['expect']})")
    for r in report["verdict"]:
        flag = "" if (r["mode_ok"] and r["diff_ok"] and r.get("abs_ok", True)) else "  <-- FAIL"
        print(f"  {r['company']:14} | {r['hyp'][:40]:40} | n={r['n_in_scope']} "
              f"metric={r['metric']} verdict={r['verdict']} mode={r['mode']}{flag}")
    for k, v in report["counts"].items():
        print(f"  {k:22} {v}  (gate {report['gates'][k]:.0%})  score {report['scores'][k]:.0%}")
    print(f"  GATE: {'PASS' if report['gate_pass'] else 'FAIL'}\n")


if __name__ == "__main__":
    from engine import segment
    data = next((p for p in (ROOT / "data" / "outlets_geo2.parquet", ROOT / "data" / "outlets_geo.parquet",
                             ROOT / "data" / "outlets_all.parquet", ROOT / "data" / "outlets.parquet")
                 if p.exists()), None)
    if not data:
        print("no data parquet present — run the pulls first"); sys.exit(1)
    rep = run_evals(segment.run_engine(str(data)))
    _print(rep)
    sys.exit(0 if rep["gate_pass"] else 1)
