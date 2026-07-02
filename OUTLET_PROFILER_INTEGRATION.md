# Connecting Outlet Profiler to CPG-OS — completed brief

**Agent:** Outlet Profiler (`agent_id: outlet-profiler`)
**Owner / point of contact:** FieldAssist Categorisation squad
**Contract:** CPG-OS Case-C depth agent · `contract_version: 1` · wire-compatible with the reference `outlet-classifier`.

> **In one line:** Outlet Profiler owns the outlet **opportunity-tier** ground truth — it grades every outlet against its structural-peer + intensity frontier, sizes the unrealised headroom in ₹, and validates a supervisor's hypotheses about that truth, data-first. It never orchestrates or sets cross-product priority.

---

## 1. Access
- **Base URL (staging/local):** `http://localhost:8100` (single FastAPI process; grades once at startup, holds in memory).
- **Test tenant / company:** tenant `default`; safe companies to run against — `Anchor`, `Colgate (India)`, `GIL Live`, `Everest Spices`, `Hamdard`, `CG Corp (Wai Wai)`. Use the **company** field in the payload to scope; omit it (or `"all"`) only for supervisor cross-company runs.
- **Running instance:** yes — the process above serves all surfaces. It is read-only against a warehouse snapshot, so it is safe to hammer.

## 2. Auth
- **Scheme:** HTTP **Bearer** (`Authorization: Bearer <token>`). Open by default in the lab (any token accepted; `dev-token` maps to tenant `default`); set `PROFILER_REQUIRE_AUTH=1` to enforce.
- **Getting a credential:** dev token `dev-token`. Production will issue per-tenant tokens.
- **Tenant scoping:** three ways, in precedence order — `agent_specific_payload.tenant_id` (or top-level `tenant_id`) in the body → the tenant mapped from the Bearer token → `X-Tenant-Id` header. The **company** being acted for is a separate field inside `agent_specific_payload.company`.

## 3. The contract
- **OpenAPI:** `http://localhost:8100/openapi.json` · **Manifest:** `GET /.well-known/agent.json` · **A2A card:** `GET /.well-known/agent-card.json`.
- Core operations:

  | Operation | Method + path | Reads or writes? |
  |---|---|---|
  | Submit a run (any action) | `POST /v1/runs` | reads only |
  | Get a run + its emitted contract objects | `GET /v1/runs/{run_id}` | read |
  | Just the outputs of a run | `GET /v1/runs/{run_id}/outputs` | read |
  | List runs (filtered, cursor-paginated) | `GET /v1/runs?action=&status=&since=&cursor=&limit=` | read |
  | Cancel a run | `POST /v1/runs/{run_id}/cancel` | read (no external effect) |
  | A2A JSON-RPC (message/send, tasks/*) | `POST /a2a` | reads only |
  | MCP (tools/list, tools/call) | `POST /mcp`, `GET /mcp/tools` | reads only |
  | Health | `GET /health/live`, `GET /health/ready` | read |

  **Actions** (the `action` field): `grade_outlets`, `validate_opportunity_hypothesis`, `analyze_outcome`.
  **MCP tools** (the six standard names): `agent.describe`, `agent.run`, `agent.get_run`, `agent.list_runs`, `agent.cancel_run`, `agent.simulate` (dry-run), plus `grade.*` shortcuts.

## 4. The things a spec doesn't tell you
- **Reads vs writes:** **every action is read-only.** The Profiler never mutates master, orders, or any customer data — it returns analysis (Observations / Diagnoses / Opportunities). It does not propose executable Plugs in phase 1.
- **Governance / sign-off:** none required — nothing it does is a write. `dry_run: true` is supported (accepts the run, emits nothing) as a routing/preview probe. Because there are no side effects, no HITL gate is needed.
- **Sync or async (poll *or* push):** **async.** `POST /v1/runs` enqueues and returns `status: "queued"` immediately; an in-process worker executes it (`running → succeeded`), ~1–2 s. Two ways to get the result: **poll** `GET /v1/runs/{run_id}` (A2A `tasks/get` / MCP `agent.get_run` too), **or** pass `callback_url` and the agent **POSTs the finished run** to it — `{event:"run.completed", status, run_id, signal_id, outcome{…}, outputs_url}`, `X-Trace-Id` echoed, retried 3× with backoff (a failed callback never fails the run). `POST /v1/runs/{id}/cancel` cancels a queued run (best-effort on a running one).
- **Scoping grain:** scope to a **company** (required for a normal run) plus optional **regions / format / tier** inside the payload — `regions: ["Delhi","BIHAR"]` (or single `region`) and `format` for grade; `scope: {region,format,tier}` for validate. If you omit `company` it runs **cross-company (supervisor mode)** over all ~10.7k selling outlets — heavier but bounded (~2 s). `limit` (default 40) caps the returned target list; the tier distribution is always full-population.
- **Required headers / idempotency:** `POST /v1/runs` (and A2A/MCP sends) **require an `Idempotency-Key` header** → 400 `missing_idempotency_key` otherwise. A repeated key returns the **same run**, never a duplicate. `X-Trace-Id` is honored — stamped on the run + logs and echoed on the response. `signal_id` is read from `triggered_by.signal_id` and propagated onto every emitted contract. `outlet_id` in outputs is a **string** (per `EntityRefs`).
- **Cost / rate limits:** no external cost (in-memory compute). Advertised capacity: sustained 8 rps, peak 15, ≤4 concurrent runs (= worker-pool size), p95 ~4 s; **per-tenant rate limiting is enforced** (429 on burst). `load_tested_to_rps` is `null` until a real load test runs.

## 5. Inputs and outputs
- **Minimum input** (grade): `{ "action":"grade_outlets", "agent_specific_payload": { "company":"Anchor", "mission":"improve order frequency in Delhi kirana" } }` — `mission` (plain English) *or* explicit `weights` is required. Region/format scope is inferred from the text; or add `"regions": ["Delhi"]` explicitly (omit = all). Built-in **plays** (advertised in the manifest, `grade_outlets.plays`): `premium_launch`, `volume_scheme`, `frequency` (order-cadence, targets T3/T4), `distribution`, `retention`, `reactivation`, `balanced` — and any novel phrasing is handled by the LLM lens as a `custom` play. The returned `outputs` list is ranked by opportunity (most headroom / lowest tier first for gap plays) and stratified across the actionable tiers.
- **RunResult.outcome:** every run carries an `outcome { summary, verdict, reasoning_mode, reversible, changes[] }` envelope for measurement. The Profiler is read-only, so `changes[]` is empty — the value is in the emitted `outputs[]`.
- **What it returns:** a run whose **outputs** are CPG-OS contract objects (`GET /v1/runs/{id}`):
  - **Observation** (`kind: outlet_opportunity_grade`) per outlet — `value.tier` (T1–T4), `value.RI` (realisation index 0–1 vs the peer frontier), the levers, and the peer. `confidence` 0–1.
  - **Opportunity** per actionable outlet — `inr_value` (₹/yr of unrealised headroom for gap plays, or incremental for launch/retention), `horizon_days`, `confidence_level`.
  - **Diagnosis** (validate action) — `verdict` ∈ `confirm | refute | inconclusive`, `summary`, `root_causes`.
  - **Every output carries `reasoning_mode`** = `deterministic` or `reasoning`. The **grades themselves are always deterministic** (the tier, the guard, peer-frontier realisation — rules over the data). The mode is set by **how the plain-English mission was parsed into a grading lens** (weights + target tiers + ranking + region/format filters): `reasoning` when the optional **Claude LLM lens** interpreted it, `deterministic` when it fell back to the built-in keyword rules or when explicit `weights` were supplied. The LLM lens is active only when `ANTHROPIC_API_KEY` is configured; **without a key every run is `deterministic`.** **This is how much to trust it:** a deterministic verdict is settled by the data; a reasoning one is a lens judgment and usually pairs with lower confidence.
  - **Trust the guard:** run counters include the size-bias guard (`size_correlation`, `safe`) — the grade is validated to be decorrelated from raw size (≈0.245 pooled), so a high-tier outlet is genuinely a *good* outlet, not just a *big* one.
- **Safe read-only smoke:** `GET /health/ready`, `GET /.well-known/agent.json`, `GET /mcp/tools`, or a `dry_run:true` run — none touch data.

## 6. One worked example

**Request**
```
POST http://localhost:8100/v1/runs
Authorization: Bearer dev-token
Idempotency-Key: brief-demo-1
Content-Type: application/json

{"action":"grade_outlets",
 "agent_specific_payload":{"company":"Anchor","mission":"launch a premium SKU in kirana","limit":2}}
```

**Response `200`** (async — the run is accepted and queued; poll for the result)
```json
{ "run_id": "run_b8f2aaeecad74ceb", "action": "grade_outlets", "status": "queued",
  "produces_contract_types": [], "n_outputs": 0,
  "outcome": {"summary": null, "verdict": null, "reasoning_mode": null, "reversible": true, "changes": []} }
```

**Poll `GET /v1/runs/run_b8f2aaeecad74ceb`** after ~1–2 s → `status: "succeeded"`, with the outcome + emitted contract objects:
```json
{
  "run_id": "run_b8f2aaeecad74ceb", "status": "succeeded",
  "produces_contract_types": ["observation", "opportunity"], "n_outputs": 4,
  "reasoning_modes": ["reasoning", "deterministic"],
  "outcome": {
    "summary": "Premium / new-SKU launch: graded 671 actionable outlets for Anchor; emitted 2 observations + 2 opportunities (~₹3,875/yr total headroom).",
    "verdict": null, "reasoning_mode": "reasoning", "reversible": true, "changes": []
  },
  "counters": {
    "tier_distribution": {"T1": 172, "T2": 555, "T3": 593, "T4": 170},
    "tier_candidates": {"T1": 172, "T2": 499},
    "guard": {"size_correlation": 0.208, "safe": true},
    "archetype": "premium_launch", "label": "Premium / new-SKU launch",
    "reasoning_mode": "reasoning", "ranking": "best",
    "target_tiers": ["T1", "T2"], "regions": "all"
  },
  "outputs": [ /* Observation + Opportunity objects, below */ ]
}
```

The `outputs[]` contract objects, e.g.:
```json
{
  "contract_version": "1", "type": "observation", "agent_id": "outlet-profiler",
  "run_id": "run_b8f2aaeecad74ceb", "reasoning_mode": "deterministic", "confidence": 0.9,
  "entity_refs": {"tenant_id": "default", "outlet_id": "13736354", "region": "Madhya pradesh"},
  "kind": "outlet_opportunity_grade",
  "value": {"tier": "T1", "RI": 1.0, "peer": "GT·kirana·Madhya pradesh",
            "format": "kirana", "range_intensity": 1.75, "cadence": 0.31, "basket_value": 598.5},
  "evidence": [{"kind": "decorrelation_guard", "value": {"size_correlation": 0.208, "safe": true}, "weight": 1.0}]
}
```
```json
{
  "contract_version": "1", "type": "opportunity", "agent_id": "outlet-profiler",
  "run_id": "run_b8f2aaeecad74ceb", "reasoning_mode": "deterministic",
  "entity_refs": {"tenant_id": "default", "outlet_id": "13736354", "region": "Madhya pradesh"},
  "summary": "Premium / new-SKU launch: outlet #13736354 (T1, kirana) realises 1.0 of its peer frontier — ~₹1,436/yr incremental from placing the line at this proven outlet.",
  "inr_value": 1436.0, "horizon_days": 90, "confidence_level": "high"
}
```

**Validation worked example** (data-first hypothesis check):
```
POST /v1/runs   {"action":"validate_opportunity_hypothesis",
  "agent_specific_payload":{"company":"GIL Live",
    "hypothesis":"my T1 outlets in WEST BENGAL have gone dormant","scope":{"region":"WEST BENGAL"}}}
```
→ Diagnosis: `verdict: "confirm"`, `reasoning_mode: "deterministic"`, `summary: "CONFIRM: over 700 in-scope outlets (stale_share = 0.777). Re-graded against their peer frontiers."` + 25 supporting Observations.

---

*Contract discovery starts at `GET /.well-known/agent.json`; the A2A card and MCP tools list are derived from it. Point the orchestrator's router there.*
