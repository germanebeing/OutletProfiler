# Outlet Profiler — Go-Live Runbook

How to take the agent + product fully live, mirrored to **GitHub** and **Azure DevOps**,
so both the **CPG-OS supervisor** (agent API) and a **human operator** (product UI) can use it.

---

## 0. What's live already

| Piece | Status |
|---|---|
| Code on **GitHub** | ✅ `origin` → `github.com/germanebeing/OutletProfiler` (public, incremental commits) |
| Code on **Azure DevOps** | ⏳ remote `azure` configured — needs one authenticated push (§2) |
| Deployed service | ⏳ not yet — one deploy step (§3) gives it a public URL |

Runs locally with zero setup (`data/` ships a graded dataset). Everything below is to make it reachable over the internet.

---

## 1. Two audiences, one service

**The supervisor (machine)** talks to the agent surfaces:
- `GET /.well-known/agent.json` — manifest (actions, plays, schemas, reasoning modes)
- `GET /.well-known/agent-card.json` — A2A card
- `POST /v1/runs` — submit a run (`Idempotency-Key` + `Bearer` required); poll `GET /v1/runs/{id}`
- `/mcp` — 9 MCP tools (`agent.run`, `grade.grade_outlets`, …)
- Actions: **`grade_outlets`** (opportunity grade + ₹ headroom), **`validate_opportunity_hypothesis`** (confirm/refute/inconclusive), `analyze_outcome` (forward-looking stub, §7)

**A human operator** uses the product UI at `/`:
- **Ask** (plain-English play → graded outlets to act on), **Runs**, **Recommender**, **Segments**, **Peer groups**, **Cold-start**, **Validation**, **Outlet lookup**
- **+ Add** company (onboard live from Trino — text / storefront-photo / both segmentation)
- **Agent** mode — the "how to call this agent" panel (API/MCP/CLI/A2A/manifest)

Same container serves both. Nothing is UI-only.

---

## 2. Push to Azure DevOps (mirror of GitHub)

The remote is already added:
```bash
git remote -v          # azure  →  https://dev.azure.com/flick2know/FA%20-%20Ai/_git/OutletProfiler
```

**One-time:**
1. In Azure DevOps → project **FA - Ai** → **Repos** → create a repo named **OutletProfiler** (empty, no README).
2. Create a **Personal Access Token**: User settings → *Personal access tokens* → **New** → scope **Code = Read & Write** → copy it.
3. Push (git will prompt for username = anything, password = the PAT; macOS keychain caches it):
   ```bash
   git push azure main
   ```
   Or bake the PAT into the remote once (kept out of the repo):
   ```bash
   git remote set-url azure "https://<PAT>@dev.azure.com/flick2know/FA%20-%20Ai/_git/OutletProfiler"
   git push azure main
   ```

**Push to both from now on** (one command):
```bash
git remote set-url --add --push origin https://github.com/germanebeing/OutletProfiler.git
git remote set-url --add --push origin "https://dev.azure.com/flick2know/FA%20-%20Ai/_git/OutletProfiler"
# now `git push origin main` writes to GitHub AND Azure
```

---

## 3. Deploy the live service

The `Dockerfile` serves everything on `$PORT` (`uvicorn api.app:app`). Pick one host.

### Option A — Render (fastest; `render.yaml` is ready)
1. render.com → **New → Blueprint** → connect the GitHub repo → it reads `render.yaml` (web service + persistent disk).
2. Set env vars (§4) in the service **Environment** tab.
3. Deploy → you get `https://outlet-profiler.onrender.com`. Auto-deploys on every push to `main`.

### Option B — Azure (same org as your repo)
Using **Azure Web App for Containers** (or Container Apps):
```bash
# build + push image to Azure Container Registry
az acr create -g <rg> -n faoutletprofiler --sku Basic
az acr build -r faoutletprofiler -t outlet-profiler:latest .

# create the web app from the image
az appservice plan create -g <rg> -n op-plan --is-linux --sku B1
az webapp create -g <rg> -p op-plan -n outlet-profiler \
  --deployment-container-image-name faoutletprofiler.azurecr.io/outlet-profiler:latest
az webapp config appsettings set -g <rg> -n outlet-profiler --settings \
  WEBSITES_PORT=8100 PROFILER_API_URL=https://outlet-profiler.azurewebsites.net \
  PROFILER_REQUIRE_AUTH=1 PROFILER_AUTH_TOKENS=<real-token> \
  ANTHROPIC_API_KEY=<key> PROFILER_TRINO_HOSTS=trino.fieldassist.io
```
Or wire an **Azure DevOps Pipeline** (you already have the org): a `azure-pipelines.yml` that runs `az acr build` + `az webapp` on each push to `main` gives you CI/CD from the Azure repo.

> **Persistence:** the agent's run store is SQLite (`data/agent.db`) + `data/`. Render's persistent disk keeps it across deploys. On Azure, mount **Azure Files** to `/app/data` (or accept ephemeral run history — grades are recomputed, only run/idempotency history resets).

---

## 4. Environment / config reference

Set these on the deployed service (never in the repo):

| Var | Purpose | Needed for |
|---|---|---|
| `PROFILER_API_URL` | Public base URL stamped into the manifest surfaces | **required** (supervisor discovery) |
| `PROFILER_REQUIRE_AUTH` | `1` to enforce Bearer on `/v1/runs` | **required for prod** |
| `PROFILER_AUTH_TOKENS` | Comma-separated bearer token(s); replaces `dev-token` | **required for prod** |
| `ANTHROPIC_API_KEY` | Enables the Claude LLM lens (else deterministic keyword parsing) | natural-language missions |
| `PROFILER_TRINO_HOSTS` | Warehouse hosts for the onboarding dropdown | onboarding new companies |
| `PROFILER_IMAGE_URL_TEMPLATE` | Override outlet-photo URL (default `static.fieldassist.io/outletimages/{imageid}`) | image segmentation |
| `PROFILER_WORKERS` | Async worker-pool size (default 4 = max concurrency) | throughput |
| `PROFILER_UI_URL` | Public UI URL for the manifest | cosmetic |
| `PROFILER_LLM_MODEL` | Override the lens model (default `claude-haiku-4-5-20251001`) | optional |

Without a warehouse/key it still runs fully on the shipped dataset (grades, validate, UI) — those vars unlock onboarding, live language, and prod auth.

---

## 5. Register with the supervisor + verify the handshake

Give the supervisor the manifest URL, then confirm end-to-end:
```bash
BASE=https://<your-host>
curl -s $BASE/.well-known/agent.json | jq '.agent_id, .actions[].name'   # outlet-profiler + 3 actions
curl -s $BASE/health/ready

# a real graded run (Bearer + Idempotency-Key required)
curl -s -X POST $BASE/v1/runs -H "Authorization: Bearer <token>" \
  -H "Idempotency-Key: live-check-1" -H "Content-Type: application/json" \
  -d '{"action":"grade_outlets","agent_specific_payload":{"company":"Anchor","mission":"improve order frequency in Delhi"}}'
# → poll GET $BASE/v1/runs/{run_id} for outputs (Observation + Opportunity) + counters
```
The supervisor now routes work by `action`; each run returns typed CPG-OS contract objects tagged `reasoning_mode`, plus counters (`ranking`, `tier_candidates`, `guard`, ₹ opportunity).

---

## 6. Go-live checklist

- [ ] `git push azure main` succeeds (§2); optional dual-push remote set
- [ ] Service deployed, `GET /health/ready` = 200 (§3)
- [ ] `PROFILER_API_URL`, `PROFILER_REQUIRE_AUTH=1`, `PROFILER_AUTH_TOKENS` set (§4)
- [ ] `ANTHROPIC_API_KEY` set → a free-text mission returns `reasoning_mode: "reasoning"`
- [ ] `PROFILER_TRINO_HOSTS` set → onboarding dropdown populated
- [ ] Supervisor pointed at `/.well-known/agent.json`; a `grade_outlets` run round-trips
- [ ] Operator can open `/`, run **Ask**, and onboard a company

---

## 7. The one honest boundary

Two of the three agent actions are **fully functional**: `grade_outlets` and `validate_opportunity_hypothesis`. The third, **`analyze_outcome`**, is an intentional stub that returns an `inconclusive` Diagnosis — it measures the effect of an *already-acted-on* opportunity, which needs post-intervention sales data that only exists after the supervisor acts on a recommendation. It's advertised as a stub in the manifest and docs; it is not a broken feature, it is a phase boundary waiting on that data feed. Everything a supervisor or operator needs to grade, size, validate, onboard, and act is live today.
