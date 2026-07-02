"""Optional LLM parse layer: free-text mission -> a structured grading LENS.

Turns an open-ended ask ("Delhi outlets ordering less than last quarter but with
wide baskets") into {weights, target_tiers, ranking, regions, formats}. The
DETERMINISTIC engine then grades and guards under that lens, so the numbers stay
auditable — the model only chooses the lens, never the grades. Falls back to the
keyword rules (mission.weights_from_mission) when there is no key / no network.

Auth: the API key comes ONLY from the environment (ANTHROPIC_API_KEY), or a
gitignored `secrets.local` if the operator chooses to drop one there. It is never
written to a tracked file and never committed.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVERS = ["range", "cadence", "recency", "value"]
_MODEL = os.environ.get("PROFILER_LLM_MODEL", "claude-haiku-4-5-20251001")
_KEY_CACHE: str | None = None


def _api_key() -> str | None:
    global _KEY_CACHE
    if _KEY_CACHE is not None:
        return _KEY_CACHE or None
    k = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not k:
        f = ROOT / "secrets.local"          # gitignored; operator-provided, optional
        if f.exists():
            for ln in f.read_text().splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "=" in ln:
                    name, val = ln.split("=", 1)
                    if name.strip() in ("ANTHROPIC_API_KEY", "PROFILER_LLM_KEY"):
                        k = val.strip().strip('"').strip("'")
                        break
                elif ln.startswith("sk-"):
                    k = ln
                    break
    _KEY_CACHE = k
    return k or None


def available() -> bool:
    import importlib.util as u
    return bool(_api_key()) and bool(u.find_spec("httpx"))


_SYS = (
    "You translate a field-sales manager's plain-English ask into a grading LENS "
    "for a retail-outlet opportunity model. You do NOT grade outlets — you only "
    "choose how to weight four size-neutral levers and which outlets to surface.\n\n"
    "Levers (weights 0..1, should sum to ~1.0):\n"
    "- range: basket breadth (distinct SKUs per bill) vs peers\n"
    "- cadence: ordering frequency / rhythm (active weeks) vs peers\n"
    "- recency: freshness (how recently they ordered) vs peers\n"
    "- value: basket value — KEEP LOW (usually 0) unless the ask is explicitly "
    "about premium / high-value, because value re-encodes outlet SIZE and the "
    "model deliberately holds it out to stay size-neutral.\n\n"
    "target_tiers: which of T1(best)..T4(worst) to act on. Improve / grow / fix / "
    "win-more asks -> ['T3','T4']. Launch-to-proven or protect/defend asks -> "
    "['T1','T2']. Broad overview -> all four.\n"
    "ranking: 'headroom' (worst / most to gain first — default for improve/grow/"
    "fix), 'best' (strongest first — for launch/protect), or 'lapsing' (least "
    "recent first — for win-back/reactivation).\n"
    "regions / formats: include ONLY ones the ask explicitly names, drawn from the "
    "provided lists; otherwise leave empty."
)


def plan(text: str, company: str | None = None,
         regions: list[str] | None = None, formats: list[str] | None = None,
         timeout: float = 12.0) -> dict | None:
    """Return a normalized lens dict, or None (parse failed / unavailable)."""
    key = _api_key()
    if not key or not (text or "").strip():
        return None
    import httpx
    tool = {
        "name": "grading_lens",
        "description": "Return the grading lens for the manager's ask.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {"type": "object",
                            "properties": {k: {"type": "number"} for k in LEVERS},
                            "required": LEVERS},
                "target_tiers": {"type": "array",
                                 "items": {"type": "string", "enum": ["T1", "T2", "T3", "T4"]}},
                "ranking": {"type": "string", "enum": ["headroom", "best", "lapsing"]},
                "regions": {"type": "array", "items": {"type": "string"}},
                "formats": {"type": "array", "items": {"type": "string"}},
                "label": {"type": "string", "description": "3-5 word name for this play"},
                "rationale": {"type": "string", "description": "one line: why these weights"},
            },
            "required": ["weights", "target_tiers", "ranking", "label", "rationale"],
        },
    }
    user = (f"Ask: {text}\nCompany: {company or 'all'}\n"
            f"Available regions: {', '.join((regions or []))[:900]}\n"
            f"Available formats: {', '.join((formats or []))[:400]}")
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": _MODEL, "max_tokens": 700, "temperature": 0, "system": _SYS,
                  "tools": [tool], "tool_choice": {"type": "tool", "name": "grading_lens"},
                  "messages": [{"role": "user", "content": user}]},
            timeout=timeout)
        r.raise_for_status()
        tu = next((b for b in r.json().get("content", []) if b.get("type") == "tool_use"), None)
        return _normalize(tu["input"], regions, formats) if tu else None
    except Exception:
        return None


def _normalize(d: dict, regions: list[str] | None, formats: list[str] | None) -> dict:
    w = {k: max(0.0, float((d.get("weights") or {}).get(k, 0) or 0)) for k in LEVERS}
    s = sum(w.values()) or 1.0
    w = {k: round(v / s, 3) for k, v in w.items()}
    tiers = [t for t in ["T1", "T2", "T3", "T4"] if t in (d.get("target_tiers") or [])] \
        or ["T1", "T2", "T3", "T4"]
    ranking = d.get("ranking") if d.get("ranking") in ("headroom", "best", "lapsing") else "headroom"
    reg = [r for r in (d.get("regions") or []) if r in set(regions or [])]
    fmt = [f for f in (d.get("formats") or []) if f in set(formats or [])]
    return {"weights": w, "target_tiers": tiers, "ranking": ranking, "regions": reg,
            "formats": fmt, "label": (d.get("label") or "Custom lens")[:44],
            "rationale": (d.get("rationale") or "")[:220]}
