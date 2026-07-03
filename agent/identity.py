"""Agent identity + config. Self-describing so the manifest/A2A/MCP never drift
from what the app actually serves. Open auth by default (validation product),
with a dev-token boundary that can be tightened for production."""
from __future__ import annotations

import os

AGENT_ID = "outlet-profiler"
AGENT_VERSION = "0.1.0"
DISPLAY_NAME = "Outlet Profiler"
OWNER_SQUAD = "fieldassist-categorisation"
TIER = "S"

DESCRIPTION = (
    "Profiles every outlet by opportunity — how well it performs versus "
    "structurally-similar peers on size-neutral intensity levers — and assigns "
    "an opportunity tier (T1–T4) plus a grade-as-vector per business play. A "
    "CPG-OS Case-C depth agent: it OWNS the outlet opportunity-tier ground truth "
    "other products defer to, validates supervisor hypotheses about that truth "
    "data-first, and never orchestrates or sets cross-product priority. Every "
    "response is tagged deterministic (a rule over the data) or reasoning (an "
    "interpretation call)."
)

# what this agent emits / ingests
PRODUCES = ["observation", "diagnosis", "opportunity"]
ACCEPTS = ["diagnosis", "opportunity"]
REASONING_MODES = ["deterministic", "reasoning"]

# public URLs (env-overridable so the manifest matches the deployment).
# Render injects RENDER_EXTERNAL_URL at runtime, so the manifest self-describes
# with the real public URL without any manual config.
PUBLIC_API_URL = (os.environ.get("PROFILER_API_URL")
                  or os.environ.get("RENDER_EXTERNAL_URL")
                  or "http://localhost:8100")
# the operator UI is served on the same host as the API (FastAPI serves `/`), so
# default it to the API URL rather than localhost — keeps the deployed manifest right.
PUBLIC_UI_URL = os.environ.get("PROFILER_UI_URL") or PUBLIC_API_URL
PUBLIC_MCP_URL = PUBLIC_API_URL.rstrip("/") + "/mcp"
PUBLIC_A2A_URL = PUBLIC_API_URL.rstrip("/") + "/a2a"
CLI_NAME = "profiler"

DEFAULT_TENANT = os.environ.get("PROFILER_TENANT", "default")
# production bearer token(s) come from the environment (PROFILER_AUTH_TOKENS,
# comma-separated, or PROFILER_AUTH_TOKEN); when set they REPLACE the dev token so
# 'dev-token' does not work in prod. Falls back to 'dev-token' for local dev only.
_env_tokens = [t.strip() for t in os.environ.get(
    "PROFILER_AUTH_TOKENS", os.environ.get("PROFILER_AUTH_TOKEN", "")).split(",") if t.strip()]
DEV_AUTH_TOKENS = {t: DEFAULT_TENANT for t in _env_tokens} or {"dev-token": DEFAULT_TENANT}
REQUIRE_AUTH = os.environ.get("PROFILER_REQUIRE_AUTH", "0") == "1"

STATUS = "beta"

# in-process worker pool size — effective concurrency == this. Capacity below is
# set honestly to what a small pool on a single container actually sustains
# (not an aspirational number); load_tested_to_rps stays null until measured.
WORKER_POOL = int(os.environ.get("PROFILER_WORKERS", "4"))
CAPACITY = {
    "sustained_rps": 8,
    "peak_rps": 15,
    "max_concurrent_runs": WORKER_POOL,
    "p95_latency_ms": 4000,
    "rate_limit_per_tenant_rps": 5,
    "load_tested_at": None,
    "load_tested_to_rps": None,  # stays null until a real load test runs
}
SLO = {"availability": 0.995}

# the actions the agent exposes (also the ActionRequest.action allow-list)
ACTION_NAMES = ["grade_outlets", "validate_opportunity_hypothesis", "analyze_outcome"]
