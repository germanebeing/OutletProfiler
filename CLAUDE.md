# Outlet Profiler — repo rules

CPG-OS Case-C **depth agent** (`agent_id: outlet-profiler`). Owns outlet
opportunity-tier truth; grades against structural peer + intensity frontier,
validates supervisor hypotheses data-first. Engineering standard: `AGENTS.md`
(distilled below). Behaviour: the CPG-OS Case-C model.

## Architecture (keep the seams)
- **One core, thin adapters.** Business logic lives only in `engine/` (domain)
  and `agent/handlers.py` (application). `agent/api.py` (REST/A2A/MCP), the CLI,
  and `web/` are thin — no logic per surface; the same handler sits behind all.
- `engine/` is storage-agnostic — it returns data; the agent layer persists +
  serves and wraps output into CPG-OS contracts (`engine/contracts.py`).
- Run state is durable (SQLite, `agent/store.py`). No process memory as the
  source of truth for runs.

## Non-negotiables
- `Idempotency-Key` required on run submission (re-runs are no-ops).
- Deterministic-by-default; interpretation is the only `reasoning` path; every
  emitted contract carries `reasoning_mode`.
- Read-only: the agent emits Observation/Diagnosis/Opportunity, never mutates a
  master-of-record. `tenant_id` scopes every run.
- `load_tested_to_rps` stays `null` until a real load test measures it.
- Ask before adding a dependency; secrets via env only, never committed
  (warehouse creds live in `CH_PW`, hostnames configurable).

## Before claiming done
```bash
PYTHONPATH=.:tests python -m pytest tests -q          # unit/contract/agent + eval gate
PYTHONPATH=.:tests python -m tests.evals.harness      # behaviour evals (gated)
PYTHONPATH=. python -m compileall -q engine agent api
```

## Layout
`engine/` grade + mission + contracts · `agent/` manifest/api/worker/store/handlers ·
`api/app.py` host · `web/` UI · `tests/` + `tests/evals/` · `Dockerfile`/`render.yaml`.
Docs: `README.md`, `OUTLET_PROFILER_INTEGRATION.md` (supervisor contract), `HANDOFF_DELTA.md`.
