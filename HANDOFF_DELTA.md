# Outlet Grader — delta vs the original Case-C handoff

**For:** the agent that authored `CASE_C_OUTLET_CLASSIFIER_HANDOFF.md`.
**Purpose:** what was actually built, and every place we deviated from that handoff (with the reasoning), so you can review the delta.
**Live product:** `http://localhost:8100` (FastAPI + a single-file UI). **Repo:** `/Users/prithvi/outlet-segmentation-lab`.
**Last updated:** 2026-07-02 — after a product-readiness UI pass. Every number and claim below was re-verified against the code and the live server on that date (§5 numbers are at the validated default segmentation `channel·format·region`; see §8).

> **⚠️ Superseded in part:** since this doc was written, the CPG-OS **agent layer was built** — the product is now the **Outlet Profiler** (`agent_id: outlet-profiler`) with manifest, `/v1/runs` (idempotency + async worker + callback), A2A, MCP (6 tools), typed contract emission (Observation/Diagnosis/Opportunity + `reasoning_mode`), an eval suite, and a deployable container. So §5/§6 rows marking **MCP/A2A/contract-emission "Deferred"** and any "**33 tests**" count are **out of date** — see [`OUTLET_PROFILER_INTEGRATION.md`](OUTLET_PROFILER_INTEGRATION.md) and [`README.md`](README.md) for current state (59+ tests + evals). Still genuinely deferred: durable Azure/Helm/JWKS, outcome/A-B proof, per-company lever leak.

---

## 1. The one-paragraph delta

The original handoff described an **image/site-stage classifier** to be built as a CPG-OS **Case-C depth agent** (supervisor↔agent contract, MCP/A2A surfaces, durable Azure state). We deliberately **did not build that agent yet.** Instead, on the user's instruction, we built a **standalone validation product** that proves the harder, more valuable idea first: an **outlet *opportunity* grader** — regrade every outlet by *how well it is doing versus structurally-similar peers*, not by size — validated on **real multi-company warehouse data** before any agent wiring. As of this session it is also a **usable single-operator tool**: one active company at a time, a real classification run-flow, and the grade surfaced as a per-play vector. Agent integration is still explicitly deferred to after validation.

---

## 2. Scope changes (what we changed and why)

| # | Change | Why (from the working sessions) |
|---|--------|---------------------------------|
| A | **Reframed the product**: image/site-stage classification → **outlet opportunity grading** (structural peer + size-neutral intensity frontier → tier). | The valuable question is "where is my headroom," not "what stage/type is this photo." The old A/B/C/D grade is just size; we replace it. |
| B | **Validation-first, agent-later.** Built a lab (API+UI+tests) that answers "does the grade work?" instead of the Case-C agent. | User: *"validate whether it works… before we add it to agent contract."* Agent/supervisor flow comes after validation. |
| C | **The grade is a mission-weighted *vector*, never a stored scalar.** Segment (peer) is stored/looked-up; tier is *computed per mission*. **Now shipped concretely:** the outlet lightbox shows an A–D + percentile grade for each play, and every tier states the weighting it was computed under (see §4). | Long discussion: a tier depends on which parameters the business weights, so it cannot be a context-free stored value. Same outlet is T1 for a premium launch, T3 for a scheme. |
| D | **Size-bias guard is a first-class product feature** (per-company + on the chosen weights), not just a test. | The whole risk is rebuilding the size grade. The guard runs live and warns when a user's weighting turns the grade back into size. |
| E | **Images: tested, not assumed.** Off-the-shelf CLIP failed; **SigLIP works** (accurate typing) but did **not** improve the guard. | We proved image typing is not the bottleneck — ~85% of GT outlets are one format (kirana), so shop-type barely differentiates. |
| F | **Multi-model ingestion** (Trino "bse" model + Trino/CH "f2k"/NonFA model) with **live company onboarding**. | Real clients diverge (GIL Live keys on `f2klocations.id`, Colgate on NonFA invoices). Onboarding auto-detects the model. |

---

## 3. The product, end to end

What has actually been built is a complete, runnable system: **warehouse → per-outlet features → structural peers → size-neutral grade + guard → mission-weighted tier → action targeting → API → operator UI**. This section describes the whole thing as it stands. (Every number/constant below was read out of the code on 2026-07-02.)

### 3.1 What it does, in one line
For a chosen company, it regrades every outlet by **how well it performs versus structurally-similar peers** (not by its raw size), lets an operator weight that grade to a **specific business play** in plain English, checks the weighting has not just recreated the size grade, and returns a ranked, reasoned list of outlets to act on with **how to move each one up a tier**.

### 3.2 Data foundation (ingestion)
Read-only pulls from **two FieldAssist warehouses** over a fixed window — sales since **2026-04-01** (~3 months, ≥1 reorder cycle):
- **Trino** (the FA Trino warehouse, REST — host from env) for the 4 base companies via `pull_data.py`.
- **ClickHouse** (a tenant warehouse, `NonFAInvoiceDetail`) for the returns-carrying company via `pull_colgate.py`, which folds in **returns** (`InvoiceType` 1/2 = credit-note/return).
- **On-demand onboarding** (`pull_company.py`, run in an isolated subprocess so a heavy/failed pull cannot crash the API) for any other client, e.g. GIL Live.
- **Two client data models, auto-detected by test-join** (`detect_model`): `bse` — `secondarysales.outletid = famasters.buyersellerentity.entityid`, region = `regionname`; `f2k` — `= masterdb.f2klocations.id`, geo unit = `state`, channel codes mapped via `CHANNEL_ENUM`. Real clients diverge, so ingestion adapts rather than assuming one schema.
- **Single-catalog rule (the key ingestion lesson):** cross-catalog joins (line-items JOIN master) blew Trino's **12 GB** limit and hung, so every warehouse query is pure `transactiondb` **or** pure master, and all cross-model joins are done in **Polars**.
- **Per-outlet features** pulled/derived: `bills`, `order_weeks` (distinct weeks with an order), `last_bill`/`first_bill`, `distinct_skus`, `line_value` (`qty × billedptr` from line items — the real throughput, because the header `invoiceamount` is 0 for ~54% of bills), `returns_value`/`return_bills`.
- **Sampling:** ≤ **700 selling outlets/region** (ranked by bills); no-sales ("cold-start") outlets are **kept, never dropped**, sampled ~250–300/company.
- **Enrichment:** `enrich_geo.py` adds lat/lon/pincode/market; `affluence_enrich.py` derives `affluence_tier` from scraped GeoNames population bins — **found weak, reported not scored**.
- **Active dataset** the engine grades: `data/_active.parquet` — **12,274 rows × 31 columns**, 6 companies, **10,674 selling / 1,600 no-data**. (Note: the raw columns are `shoptypename`/`channelname`/`affluence_tier`; the engine derives clean `format`/`channel` from them — the source master data is dirty, e.g. 60 distinct shop-type strings.)

### 3.3 The grading engine (`engine/segment.py`, `run_engine()` — a 9-stage Polars pipeline)
1. **Read + cast** the parquet; synthesise `affluence_tier='na'` if absent.
2. **Returns** folded in as a size-neutral **quality** ratio (share of gross value returned) — surfaced, and used as an action penalty, never a scored frontier lever.
3. **Typing + controllable levers + maturity.** Dirty `shoptypename` → canonical **format** (kirana / chemist / pan_kiosk / supermarket / horeca / wholesale / cosmetics / other / unknown), with a **channel fallback** when the free-text is absent (e.g. Colgate populates channel, not shop-type); `channelname` → **GT / MT / HoReCa**. The four levers are all **size-neutral ratios**:

   | Lever | Definition | Direction |
   |-------|-----------|-----------|
   | range_intensity | `distinct_skus / bills` (basket breadth) | higher better |
   | cadence | `min(1, order_weeks / 13)` (share of the 13-wk window ordered) | higher better |
   | recency | days since last bill (anchor `TODAY = 2026-07-01`) | lower better |
   | basket_value | `line_value / bills` — **DIAGNOSTIC, held out of the grade** (size-leaky premium-capacity proxy) | — |

   **Maturity gate:** `mature = has_data & bills ≥ 2 & order_weeks ≥ 2` (a 1-bill outlet trivially maxes an intensity).
4. **Density band** (within-company) — available as an optional peer axis.
5. **Peer assignment.** `peer = channel · format · region` (the validated `DEFAULT_AXES`), **coarsened to `channel · format`** when a peer cell has `< 20` selling outlets. Axes are runtime-configurable (`AXIS_COL` also offers affluence, density).
6. **Frontier → realisation → tier.** Per-peer **p80 frontier** from **mature** outlets (≥ 10 mature, else fall back to all selling). Each lever becomes a **realisation ratio** capped at 1.0 (`min(1, value/frontier)`, or `min(1, frontier/value)` for recency). `RI_intensity` = mean of the three controllable realisations; **`RI = 0.8·RI_intensity + 0.2·real_value`** (`VALUE_WEIGHT = 0.20`, kept small and separable). **Tier:** RI ≥ 0.85 → T1, ≥ 0.65 → T2, ≥ 0.45 → T3, else T4 (`provisional` when no data); `gap = 1 − RI`. **Confidence:** none (no data) / low (thin, or unknown format) / high.
7. **Cold-start baseline.** A cross-company table keyed on **`channel · format`** only (region/density dropped — names differ per company), recording *which* companies contribute so an outlet's own company can be excluded.
8. **Validation / size-bias guard.** Over mature, selling outlets it computes: `decorrelation_RI_vs_logsize` (the headline `|Spearman(RI, log1p(line_value))|`), per-lever vs size, the **raw-count "trap"** (bills, SKUs), **identical-sales divergence** (near-equal-throughput outlets landing in different tiers), **per-company decorrelation** + `companies_over_guard`, the returns block, and a **value-weight curve**. `guard_pass` requires the pooled and every-lever correlation ≤ `GUARD_MAX = 0.5`; `guard_pass_every_company` requires no company over guard.
9. **Select** the graded columns into the in-memory frame.

### 3.4 From grade to action (`engine/actions.py`, `engine/mission.py`)
- **The grade is a vector, not a scalar.** `actions.py` projects an action-specific `target_score` per outlet across **5 plays** — the same outlet is a top *premium-launch* target and a poor *scheme* target:

  | Play | Score (verified formula) |
  |------|--------------------------|
  | Premium / new-SKU launch | `0.55·peer premium-capacity + 0.45·RI − 0.30·return_rate`, clipped |
  | Volume trade scheme | `0.6·gap + 0.4·pctile(basket_value)` |
  | Assortment / must-stock | `1 − range realisation vs frontier` |
  | Retention / protect | `0.7·RI + 0.3·cadence` |
  | Reactivation | `pctile(recency)` (days-since-last) |

  Each target carries a plain reason string; only premium applies a returns penalty; thin (non-mature) outlets are excluded from recommendations.
- **The tier is computed per mission** (`mission.py`). Plain-English text → 1 of **6 archetypes** (premium_launch, volume_scheme, distribution, retention, reactivation, balanced), each with default lever weights (+ a per-lever "why") and target tiers → weighted `RI_w`/`tier_w` → **guard-on-weights** (re-checks size for *the chosen weighting* and names the leakier lever if it breaches) → target outlets in the play's tiers → **promote-to-next-tier** (decomposes the peer-frontier gap into the top ≤ 2 binding levers and projects the tier if the top one is lifted). The **segment (peer) is the stored, context-free attribute; the tier is always recomputed** for the mission — cheaply, from cached realisations, no data re-pull.
- **Cold-start** outlets (no sales yet) get a **potential band** (high / medium / low) from the cross-company baseline — never a realisation tier — with an honest "how many rest on a genuinely *different* company" metric (97.8% here).

### 3.5 Surfaces
- **JSON API** (`api/app.py`): a single-process, read-only FastAPI host that grades once at startup and holds the frame in memory. `GET /api/summary`, `/api/companies`, `/api/peers`, `/api/outlets`, `/api/outlet/{id}`, `/api/recommend`, `/api/segment`, `/api/coldstart`, `/api/run/{id}`, `/api/runs`; `POST /api/resegment`, `/api/company/add`, `/api/mission`, `/api/promote`. `resegment` and `company/add` rebuild the in-memory store and re-grade everything.
- **UI** (`web/index.html`, single file, Floodlight-styled): **9 sections** — Classify, a dedicated **run view**, a transactional **Runs** log, Recommender, Segments, Peer groups, Cold-start, Validation, Outlet lookup — under the **single-active-company model** (§4).
- **Tests** (`tests/`): **33 passing** across 4 files (`test_validation.py` 10, `test_engine.py` 12, `test_actions.py` 6, `test_mission.py` 5); `compileall` clean.

### 3.6 The operator journey (what a user actually does)
1. **Pick the company** in the top bar (a KAM works one at a time; "All companies · supervisor" is the only scope that spans all).
2. **Describe the play** in plain English ("launch a premium SKU", "push a scheme in kirana").
3. **Run** → land on a **dedicated run page**: interpretation, the size-bias guard verdict, the weights the tier was computed from (adjustable), and the tier split — with a **pop-up summary** of what the weighting did.
4. **Review** the filterable list of outlets to act on; **open one** to see its **grade-vector** (A–D per play) and the concrete steps to move it up a tier.
5. **Recalibrate** — drag the weights, note *why* (captured for the audit trail), re-grade in place and see the T1–T4 deltas; or start a new run.
6. Every run is written to the **Runs log** with its weights, guard result and reasons — **the record that goes back to the supervisor**. That log is the intended payload for the eventual agent/supervisor integration.

---

## 4. What changed this session — UI + correctness delta

This is the headline for your review: a product-readiness pass turned the dashboard into a usable single-operator tool and fixed two correctness bugs. All verified live.

| Change | Files | Why |
|--------|-------|-----|
| **Global single-active-company model + supervisor scope** | `web/index.html` | A KAM works one company at a time. A header company selector drives every section; per-page "All companies" dropdowns were removed. Only the special **"All companies · supervisor"** scope spans all companies at once — and a *Company* column then appears in the tables. Matches how an operator vs a supervisor actually work. |
| **Fixed region filters (were silently broken)** | `web/index.html`, `GET /api/companies` | Region dropdowns read `regions` off `/api/summary`, where it is a **count**, not a list — so `.map()` produced nothing and the filters were empty. Now sourced from `/api/companies` real region lists, scoped to the active company. |
| **Dedicated run view + post-run summary overlay + filterable classified list** | `web/index.html` | A classification is a transaction: it opens its own page (interpretation, guard, adjustable weights, tier split), pops an overlay showing *what the weighting did* (tier split, guard, and T1–T4 deltas on a re-grade), then drops into the filterable outlet list (region/tier/format). Recalibrate / reset / new-run in place. |
| **Transactional Runs log** | `web/index.html`, `GET /api/runs` | Runs is now a dense table — *When · Mission · Company · Play · Weights (R/C/F/V) · Guard · Targets* — framed as the record that goes back to the supervisor. `/api/runs` now surfaces per-run `weights`, `guard_safe`, `target_tiers`, `company`, and `edited`; weight-change reasons are captured with the run. |
| **Grade-as-vector restored + weights shown wherever a tier is** | `web/index.html` | Scope-change C made concrete: each outlet shows **A–D + percentile** for Premium launch / Volume scheme / Distribution / Retention / Reactivation (from the `_build_vectors` action-fit frame), and every tier states the weighting it was computed under (mission weights, or "default balanced weighting" in the lookup). |
| **Segments before → after** | `web/index.html` | Re-segment now shows peers and grade↔size **moving** (e.g. 70→119 peers, 0.245→0.226) so it is evidently recomputing, not cached, plus links straight to the re-graded outlets and new peer groups. |
| **Bugfix — mission tier split was portfolio-wide** | `engine/mission.py` (classify_mission) | A single-company run reported the whole-portfolio tier split (~10,674). The split is now scoped to the run's company (+ any region/format the ask named); Anchor now reads its own ~1,582. |
| **Bugfix — `/api/peers` fragmented by density** | `api/app.py` (peers) | Peers were grouped by `density`, which is **not** a peer axis, splitting one peer into duplicate-looking rows (two `GT·kirana·Delhi`). Now grouped by `["company_name","peer"]` only — the peer string already encodes the active axes. |

---

## 5. Key validated findings (real data, 6 companies)

Numbers below are at the **validated default segmentation `channel·format·region`** (the API is left in this state). They move with the axes — see §8.

- **Guard passes pooled: RI ↔ log(size) = 0.245** (< 0.50). The raw counts we rejected leak hard: bills **0.756**, distinct-SKUs **0.609**, basket_value **0.857**. The scored (size-neutral) levers stay low: range **0.068**, cadence **0.287**, recency **0.115**. Identical-sales divergence **51.9%** (near-equal-sales outlets land in different tiers).
- **Per-company is the honest cut:** most companies clear 0.50, but **Everest Spices (~0.54) and GIL Live (~0.62) breach** — surfaced, not hidden. `guard_pass=true` is the pooled pass; `guard_pass_every_company=false`. This residual per-company lever leak is the real open problem, not a typing problem.
- **Value:** peer-normalised value can be weighted in cheaply (stays under guard — decorrelation-by-value-weight stays ~0.23–0.25 across weights 0.0–0.5). **Returns** fold in as a weakly-size-correlated quality signal — Colgate only (2,100 outlets): mean return rate **4.4%**, size-corr **0.252** (weak), 37% of outlets with any return; reported, not scored.
- **Scale:** 12,274 outlets, **10,674 selling**, 1,600 no-data, 6 companies, 70 peers at the default. Cold-start: **97.8%** of no-data outlets have genuine cross-company support.
- **Adversarially verified** across sessions (a multi-agent workflow re-derived every number here on 2026-07-02).

---

## 6. Delta against specific original-handoff items

| Original handoff item | Status now |
|-----------------------|------------|
| Case-C supervisor↔agent contract, conflict handling | **Deferred** — validation-first. Design intact for later. |
| MCP / A2A / manifest surfaces | **Deferred.** The engine is behind a JSON API instead. |
| Durable Azure SQL/Blob/Queue state | **Not built** — lab uses parquet + a `data/runs/` JSON store (the "agent DB" stand-in). |
| Image classification as the core | **Reframed** — image is one *typing input*, tested (SigLIP), not the core. |
| Emit CPG-OS contract types (Observation/Diagnosis/Plug/Opportunity) | **Not yet** — mission runs return targets + promote tasks; mapping to contract types is the agent-integration step. |
| Site/outlet *stage/type* taxonomy | **Replaced** by the opportunity tier (T1–T4) + segment. |

---

## 7. Open / deferred (in priority order)

1. **Outcome proof** — A/B or holdout: does acting on the grade grow sales? (Turns "clever" into "buy it." Biggest gap.)
2. **Per-company lever leak** — fix the residual size-lean for breaching clients (Everest Spices ~0.54, GIL Live ~0.62 at the default segmentation).
3. **Agent integration** — wire as the Case-C depth agent, emit the contract types, supervisor loop. (User's explicit "later.") The Runs log + weight-change reasons are the intended payload back to the supervisor.
4. **Productionise** — durable store, auth, scheduled refresh, scale/load test. Onboarding SKU-range query is ~2 min for large clients (optimise or async-queue).
5. Cold-start outlets are browsable but not yet in *mission* targeting.

*(Two defects found this session — the portfolio-wide mission split and the density-fragmented peers — are now fixed, see §4; they are not open items.)*

---

## 8. Known sharp edges (read before you rely on a number)

- **`/api/summary` peers + decorrelation are stateful** to the active segmentation. At `channel·format` → 10 peers / 0.311; `channel·format·region` → 70 / 0.245 (default); `channel·format·region·density` → 119 / 0.226. Always state which axes a number belongs to. The server is left at the validated default.
- **`guard_pass` is the pooled pass, not per-company.** `guard_pass_every_company=false` (Everest, GIL Live breach). Don't report the guard as universally passing.
- **Run timestamps are pinned.** `run_id` and `computed_at` are frozen to a fixed date in `engine/mission.py` for determinism — not wall-clock; the Runs log shows the same date for every run.
- **`promote()` never surfaces a basket-value task.** Its gap decomposition covers range/cadence/recency only, even for premium (value-weighted) missions.
- **Single mutable in-memory `STORE`, no lock, no auth** (by design for a single-process validation product). `/api/resegment` and `/api/company/add` rebuild it globally; a concurrent request mid-rebuild can observe stale state.
- **Region-name casing is inconsistent in source data** (`Uttrakhand` vs `UTTARAKHAND`) — reproduced as-is, not normalised.
- `@app.on_event("startup")` is a deprecated FastAPI pattern (move to lifespan when productionising).

---

## 9. Run it
```
PYTHONPATH=. python -m uvicorn api.app:app --port 8100   # http://localhost:8100
PYTHONPATH=.:tests python -m pytest tests -q               # 33 tests
```
