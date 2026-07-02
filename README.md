# Outlet Profiler

> **Regrade every outlet by opportunity, not size.**

A **CPG-OS Case-C depth agent** (`agent_id: outlet-profiler`, Operations) that owns the outlet **opportunity-tier** truth. It grades each outlet against its *structural-peer + intensity frontier*, sizes the unrealised headroom in ₹, and validates a supervisor's hypotheses about that truth — **data-first**. Every response is tagged **`deterministic`** (a rule over the data) or **`reasoning`** (an interpretation call). It's also a usable operator product (a Nuxt-free single-file UI) with an **Agent** mode.

- **Who it's for:** CPG KAM / sales leaders, and the CPG-OS supervisor that routes work to it.
- **What it replaces:** the size-biased A/B/C/D grade. An outlet is never "an A" — it's "an A *for a premium launch*" and maybe "a C *for a scheme*."
- **Contract:** `contract_version: "1"`, wire-compatible with the reference CPG-OS agents. Produces `observation · diagnosis · opportunity`; accepts `diagnosis · opportunity`.

---

## Why it works (validated on real data)

The one risk is rebuilding the size grade under a new name. The engine guards against it and the guard runs live. On the pulled data — **6 companies × 3 regions**, 12,274 outlets (10,674 selling), 70 peer groups, at the validated `channel·format·region` segmentation:

| Check | Result | Meaning |
|---|---|---|
| Decorrelation `RI ↔ log(size)` | **0.245** (< 0.50) | the grade is **not** a size proxy |
| The trap avoided (raw counts) | bills **0.756** · SKUs **0.609** · basket **0.857** | raw extensive counts *do* leak — the size-neutral intensity swap was necessary |
| Scored levers vs size | range **0.07** · cadence **0.29** · recency **0.12** | all size-neutral |
| Identical-sales divergence | **51.9%** (> 25%) | near-equal-sales outlets land in different tiers |
| Per-company (the honest cut) | Everest ~**0.54**, GIL Live ~**0.62** breach | surfaced, not hidden — the real open gap (see §Open) |

Two warehouses, two data models (Trino `bse` + ClickHouse/`f2k`) slot into the same peer-frontier grade. Full write-up: [`HANDOFF_DELTA.md`](HANDOFF_DELTA.md).

---

## Call it (the agent)

Discover → submit (async) → poll **or** receive a webhook.

```bash
# 1. discover
curl $BASE/.well-known/agent.json

# 2. submit a hypothesis (Bearer + Idempotency-Key required; signal nests in triggered_by)
curl -X POST $BASE/v1/runs \
  -H "Authorization: Bearer dev-token" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{"action":"validate_opportunity_hypothesis",
       "tenant_id":"cocacola-india",
       "triggered_by":{"signal_id":"sig_123"},
       "callback_url":"https://cpgos/hooks/agent",   // optional: pushed on completion
       "agent_specific_payload":{"company":"Anchor",
         "hypothesis":"my T1 outlets in Delhi have gone dormant","scope":{"region":"Delhi"}}}'
#  -> {"run_id":"run_…","status":"queued"}

# 3a. poll                        3b. or receive a POST to callback_url:
curl $BASE/v1/runs/run_… \            #   {event:"run.completed", status, run_id,
  -H "Authorization: Bearer dev-token"  #    signal_id, outcome{…}, outputs_url}
```

- **Actions:** `grade_outlets` (→ Observations + ₹-sized Opportunities), `validate_opportunity_hypothesis` (→ confirm/refute/inconclusive Diagnosis), `analyze_outcome` (M5 stub). Add `regions:[…]` to scope a run; omit = all.
- **Surfaces:** REST (`/v1/runs`, `…/outputs`, `…/cancel`, list with filters + cursor), **A2A** JSON-RPC (`POST /a2a`: `message/send`, `tasks/get`), **MCP** (`POST /mcp` + `GET /mcp/tools`; the six standard tools incl. `agent.simulate`), **CLI** (`profiler_cli.py`), **UI**.
- **Full field-level integration brief for the supervisor:** [`OUTLET_PROFILER_INTEGRATION.md`](OUTLET_PROFILER_INTEGRATION.md).

---

## Run locally in 5 minutes

```bash
pip install -r requirements-serve.txt          # or use the repo venv
PYTHONPATH=. python -m uvicorn api.app:app --port 8100
# open http://127.0.0.1:8100   (Product | Agent toggle top-right)

python profiler_cli.py --api http://localhost:8100 smoke   # pings every surface
PYTHONPATH=.:tests python -m pytest tests -q               # 49 tests
PYTHONPATH=.:tests python -m tests.evals.harness           # behaviour evals (gated 100%)
```

The base graded dataset ships in `data/` — no warehouse needed to run, grade, or validate. (Onboarding a *new* company hits Trino/ClickHouse and needs warehouse access.)

---

## Deploy

Single stateful container — the grader holds the graded frame in memory; runs are persisted in a durable SQLite store, drained by an in-process worker pool. Runs as-is on Render / Fly / Railway / Azure Container Apps.

```bash
docker build -t outlet-profiler . && docker run -p 8100:8100 outlet-profiler
```

**Render:** push the repo, then New → Blueprint (reads [`render.yaml`](render.yaml)). It injects `RENDER_EXTERNAL_URL`, so the manifest self-describes with the public URL — no config. Set `PROFILER_REQUIRE_AUTH=1` + real tokens past the pilot.

---

## Contract & reasoning-mode

Outputs are CPG-OS contract objects on a shared envelope (`engine/contracts.py`):

- **Observation** — `outlet_opportunity_grade`: tier + realisation index + levers + peer.
- **Opportunity** — `inr_value` of the unrealised headroom, `horizon_days`, `confidence_level`.
- **Diagnosis** — a hypothesis `verdict` (confirm/refute/inconclusive) + `root_causes`.
- Every output carries **`reasoning_mode`**: `deterministic` (the tier, the guard, peer-frontier realisation) or `reasoning` (parsing a plain-English mission/hypothesis). `signal_id` is propagated from `triggered_by` onto every output. Deterministic-by-default; interpretation is the only reasoning path.

---

## Repo layout

```
engine/segment.py    type → peer → intensity frontier → RI → tier → guard → baseline
engine/mission.py    plain-English → weights → weighted tier → guard-on-weights → promote
engine/actions.py    per-play action projection (grade-as-vector) + cold-start
engine/contracts.py  CPG-OS Observation/Diagnosis/Opportunity/Plug + reasoning_mode
agent/               manifest · A2A card · /v1/runs (idempotency) · MCP · worker · run store
api/app.py           FastAPI: grades once, serves the product + mounts the agent surfaces
web/index.html       single-file UI (Product + Agent modes)
tests/ · tests/evals behaviour evals (mission · reasoning_mode · verdict) + contract/engine tests
Dockerfile · render.yaml · agent.manifest.json · profiler_cli.py
```

---

## Open (in priority order)

1. **Outcome proof** — A/B or holdout: does acting on the grade grow sales? (Blocked on post-action field data.)
2. **Per-company lever leak** — Everest (~0.54) and GIL Live (~0.62) breach the guard; tighten the leaky lever (cadence) so it holds for every client, not just pooled.
3. **Production hardening** — the run store is durable SQLite; remaining: a mounted volume (so it survives redeploy), JWKS auth, and a split/scaled worker (the seams exist; it's a config change, not a rewrite).

Behaviour is governed by the CPG-OS Case-C model; engineering by `AGENTS.md`. This agent is validation-first: it graduated into the contract only because the grade held.
