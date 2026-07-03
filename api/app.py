"""Outlet Segmentation Lab — validation API + UI host.

Loads the parquet once, runs the engine, holds the graded frame in memory, and
serves it to the dashboard. This is a validation product: read-only, single
process, no auth. It exists to answer one question with real data — does the
structural-peer + intensity-frontier grade hold up, and are its action
projections sensible?
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import polars as pl
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import actions, mission, segment  # noqa: E402

# Writable state (onboarded companies, product runs, cached photos, active frame)
# lives under PROFILER_DATA_DIR — point this at a mounted disk in production so it
# survives redeploys; defaults to the repo's data/ for local dev.
DATA_DIR = Path(os.environ.get("PROFILER_DATA_DIR", str(ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS = DATA_DIR / "runs"
RUNS.mkdir(exist_ok=True)
ONBOARD = DATA_DIR / "onboarded"
ONBOARD.mkdir(exist_ok=True)
ACTIVE = DATA_DIR / "_active.parquet"
IMG_DIR = DATA_DIR / "imgs"
IMG_TYPE_CAP = int(os.environ.get("PROFILER_IMAGE_TYPE_CAP", "2000"))  # bound typing work

# seed the shipped onboarded companies (baked into the image at data/onboarded)
# into the persistent dir on first boot, so the demo companies are present but
# freshly-onboarded ones persist alongside them.
_SEED = ROOT / "data" / "onboarded"
if _SEED.exists() and _SEED.resolve() != ONBOARD.resolve():
    import shutil
    for _p in _SEED.glob("*.parquet"):
        _dst = ONBOARD / _p.name
        if not _dst.exists():
            shutil.copy2(_p, _dst)

DATA = next((p for p in (ROOT / "data" / "outlets_geo2.parquet", ROOT / "data" / "outlets_geo.parquet",
                         ROOT / "data" / "outlets_all.parquet",
                         ROOT / "data" / "outlets.parquet") if p.exists()),
            ROOT / "data" / "outlets.parquet")
WEB = ROOT / "web"


def _trino_hosts() -> list[str]:
    """Warehouse hosts for the onboarding dropdown. Config, not code — so the
    public repo ships none. Source: env PROFILER_TRINO_HOSTS (comma-separated),
    else a gitignored local file `hosts.local` (one host per line)."""
    env = [h.strip() for h in os.environ.get("PROFILER_TRINO_HOSTS", "").split(",") if h.strip()]
    if env:
        return env
    local = ROOT / "hosts.local"
    if local.exists():
        return [ln.strip() for ln in local.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    return []


app = FastAPI(title="Outlet Profiler", version="1.0")


SEG_MODES = DATA_DIR / "seg_modes.json"        # {company_name: text|image|both}
IMG_OVERRIDES = DATA_DIR / "img_overrides.parquet"  # outletid -> image format (base cos)


def _seg_mode_overrides() -> dict:
    try:
        return json.loads(SEG_MODES.read_text()) if SEG_MODES.exists() else {}
    except Exception:
        return {}


class Store:
    """In-memory hold of the engine output. Loads the base dataset plus any
    onboarded companies, then grades under the current segment (peer) axes."""
    def __init__(self, peer_axes: list[str] | None = None) -> None:
        base = pl.read_parquet(DATA)
        extra = [pl.read_parquet(p) for p in sorted(ONBOARD.glob("*.parquet"))]
        combined = pl.concat([base] + extra, how="diagonal_relaxed") if extra else base
        # persistent segmentation-mode choice per company (set at onboard or on the
        # Segments page). Controls how hard peers are clubbed; see segment._MODE_CAPS.
        if "seg_mode" not in combined.columns:
            combined = combined.with_columns(seg_mode=pl.lit(None, dtype=pl.Utf8))
        combined = combined.with_columns(pl.col("seg_mode").fill_null("text"))
        ov = _seg_mode_overrides()
        if ov:
            expr = pl.col("seg_mode")
            for co, m in ov.items():
                expr = pl.when(pl.col("company_name") == co).then(pl.lit(m)).otherwise(expr)
            combined = combined.with_columns(seg_mode=expr)
        # image-typed formats for base companies (Segments "segment by image")
        if IMG_OVERRIDES.exists():
            imo = pl.read_parquet(IMG_OVERRIDES).with_columns(pl.col("outletid").cast(pl.Int64))
            combined = (combined.with_columns(pl.col("outletid").cast(pl.Int64))
                        .join(imo, on="outletid", how="left")
                        .with_columns(shoptypename=pl.coalesce(["_imgfmt", "shoptypename"])).drop("_imgfmt"))
        combined.write_parquet(ACTIVE)
        self.axes = peer_axes or list(segment.DEFAULT_AXES)
        res = segment.run_engine(str(ACTIVE), peer_axes=self.axes)
        self.graded = res.graded
        self.peers = res.peers
        self.baseline = res.baseline
        self.validation = res.validation
        self.coldstart = actions.cold_start(self.graded, self.baseline)
        self.vectors = self._build_vectors()

    def _build_vectors(self) -> pl.DataFrame:
        """Per-outlet action-fit vector: percentile of each action's target
        score. This is the grade-as-vector — one outlet, many action fits."""
        base = self.graded.filter(pl.col("has_data")).select("outletid").unique()
        for act in actions.ACTIONS:
            s = actions.score(self.graded, act).select(
                "outletid",
                (pl.col("target_score").rank() / pl.len()).round(3).alias(f"fit_{act}"))
            base = base.join(s, on="outletid", how="left")
        return base


STORE: Store | None = None


@app.on_event("startup")
def _startup() -> None:
    global STORE
    STORE = Store()


def store() -> Store:
    if STORE is None:
        raise HTTPException(503, "store not ready")
    return STORE


# ─── data endpoints ──────────────────────────────────────────────────────

@app.get("/api/summary")
def summary() -> dict:
    s = store()
    g = s.graded
    # peer counts are over SELLING outlets — a peer that only holds no-data
    # outlets is not a real frontier group and would inflate the count past the cap.
    comps = (g.group_by("company_name").agg(
        outlets=pl.len(),
        selling=pl.col("has_data").sum(),
        regions=pl.col("regionname").n_unique(),
        peers=pl.col("peer").filter(pl.col("has_data")).n_unique(),
    ).sort("company_name").to_dicts())
    return {
        "validation": s.validation,
        "totals": {
            "outlets": g.height,
            "selling": int(g.filter(pl.col("has_data")).height),
            "nodata": int(g.filter(~pl.col("has_data")).height),
            "companies": g.select(pl.col("company_name").n_unique()).item(),
            "peers": g.filter(pl.col("has_data")).select(pl.col("peer").n_unique()).item(),
        },
        "companies": comps,
        "coldstart": {k: s.coldstart[k] for k in
                      ("n_nodata", "n_matched_to_baseline", "match_rate_pct",
                       "n_cross_company_support", "cross_company_match_rate_pct",
                       "n_own_company_only", "potential_distribution")},
        "actions": {k: {"label": v[0], "rationale": v[1], "drivers": v[2]}
                    for k, v in actions.ACTIONS.items()},
        "formats": sorted(g.filter(pl.col("has_data")).select(pl.col("format").unique())
                          .to_series().to_list()),
        "peer_axes": s.axes, "axis_options": list(segment.AXIS_COL.keys()),
        "seg_modes": _company_seg_modes(),
        # warehouse hosts for the onboarding dropdown — config, not code (empty
        # in the public repo; set PROFILER_TRINO_HOSTS or a gitignored hosts.local).
        "trino_hosts": _trino_hosts(),
    }


def _company_seg_modes() -> dict:
    """Effective segmentation mode per company (from the active dataset)."""
    if not ACTIVE.exists():
        return {}
    try:
        d = pl.read_parquet(ACTIVE, columns=["company_name", "seg_mode"])
        return {r["company_name"]: (r["seg_mode"] or "text")
                for r in d.group_by("company_name").agg(pl.col("seg_mode").first()).to_dicts()}
    except Exception:
        return {}


@app.get("/api/companies")
def companies() -> dict:
    s = store()
    out = []
    for row in s.graded.group_by("company_name").agg(
            regions=pl.col("regionname").unique()).sort("company_name").iter_rows(named=True):
        cg = s.graded.filter((pl.col("company_name") == row["company_name"]) & pl.col("has_data"))
        tiers = dict(zip(*cg.group_by("tier").len().to_dict(as_series=False).values()))
        out.append({
            "company": row["company_name"],
            "regions": sorted([r for r in row["regions"] if r]),
            "selling": cg.height,
            "tier_distribution": tiers,
        })
    return {"companies": out}


@app.get("/api/peers")
def peers(company: str | None = None) -> dict:
    s = store()
    g = s.graded.filter(pl.col("has_data"))
    if company:
        g = g.filter(pl.col("company_name") == company)
    # group by the peer key only — the peer string already encodes the active
    # axes, so grouping by density (not an axis) would fragment one peer into
    # duplicate-looking rows.
    rows = (g.group_by(["company_name", "peer"]).agg(
        n=pl.len(),
        avg_RI=pl.col("RI").mean().round(3),
        t1=(pl.col("tier") == "T1").sum(),
        t4=(pl.col("tier") == "T4").sum(),
        front_range=pl.col("range_intensity").quantile(0.8).round(2),
        front_cadence=pl.col("cadence").quantile(0.8).round(2),
        basket=pl.col("basket_value").median().round(0),
    ).filter(pl.col("n") >= 5).sort("n", descending=True).to_dicts())
    return {"peers": rows, "n": len(rows)}


@app.get("/api/outlets")
def outlets(company: str | None = None, region: str | None = None,
            peer: str | None = None, tier: str | None = None,
            has_data: bool | None = None,
            limit: int = Query(100, le=1000), offset: int = 0) -> dict:
    s = store()
    g = s.graded
    if company: g = g.filter(pl.col("company_name") == company)
    if region: g = g.filter(pl.col("regionname") == region)
    if peer: g = g.filter(pl.col("peer") == peer)
    if tier: g = g.filter(pl.col("tier") == tier)
    if has_data is not None: g = g.filter(pl.col("has_data") == has_data)
    total = g.height
    cols = ["outletid", "company_name", "regionname", "city", "format", "channel",
            "density", "peer", "tier", "confidence", "mature", "RI", "gap", "range_intensity",
            "cadence", "recency", "basket_value", "return_rate", "return_bills",
            "bills", "line_value", "segmentationname", "has_data"]
    page = g.sort(["has_data", "RI"], descending=[True, True], nulls_last=True).slice(offset, limit)
    return {"total": total, "offset": offset, "limit": limit,
            "outlets": page.select([c for c in cols if c in page.columns]).to_dicts()}


@app.get("/api/outlet/{outletid}")
def outlet(outletid: int) -> dict:
    s = store()
    row = s.graded.filter(pl.col("outletid") == outletid)
    if row.height == 0:
        raise HTTPException(404, "outlet not found")
    o = row.to_dicts()[0]
    vec = s.vectors.filter(pl.col("outletid") == outletid).to_dicts()
    o["action_fit"] = vec[0] if vec else {}
    # peer frontier context
    pr = s.graded.filter((pl.col("peer") == o["peer"]) &
                         (pl.col("company_name") == o["company_name"]) & pl.col("has_data"))
    o["peer_context"] = {
        "peer_n": pr.height,
        "front_range_intensity": round(pr.select(pl.col("range_intensity").quantile(0.8)).item() or 0, 2),
        "front_cadence": round(pr.select(pl.col("cadence").quantile(0.8)).item() or 0, 2),
        "median_basket": round(pr.select(pl.col("basket_value").median()).item() or 0, 0),
    }
    return o


@app.get("/api/recommend")
def recommend(action: str, company: str | None = None, region: str | None = None,
              tiers: str | None = None, fmt: str | None = None,
              limit: int = Query(50, le=500)) -> dict:
    s = store()
    tlist = [t.strip() for t in tiers.split(",")] if tiers else None
    try:
        return actions.recommend(s.graded, action, company=company, region=region,
                                 tiers=tlist, fmt=fmt, limit=limit)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/segment")
def segment_get() -> dict:
    return {"axes": store().axes, "options": list(segment.AXIS_COL.keys()),
            "n_peers": store().graded.select(pl.col("peer").n_unique()).item()}


@app.post("/api/resegment")
def resegment(body: dict = Body(...)) -> dict:
    global STORE
    axes = [a for a in body.get("axes", []) if a in segment.AXIS_COL] or list(segment.DEFAULT_AXES)
    STORE = Store(peer_axes=axes)
    v = STORE.validation
    return {"axes": axes, "n_peers": STORE.graded.select(pl.col("peer").n_unique()).item(),
            "decorrelation": v["decorrelation_RI_vs_logsize"],
            "guard_pass": v["guard_pass"], "companies_over_guard": v["companies_over_guard"]}


@app.post("/api/company/regions")
def company_regions(body: dict = Body(...)) -> dict:
    """Discover every region a company sells in, so the operator can choose a
    subset (or all) before onboarding. Isolated subprocess like the pull."""
    import os
    import subprocess
    import sys
    host = (body.get("host") or os.environ.get("PROFILER_TRINO_HOST") or "").strip()
    cid = int(body["company_id"])
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    try:
        proc = subprocess.run([sys.executable, str(ROOT / "pull_company.py"), "--regions", host, str(cid)],
                              capture_output=True, text=True, timeout=120, cwd=str(ROOT), env=env)
    except subprocess.TimeoutExpired:
        raise HTTPException(400, "region discovery timed out")
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "unknown error").strip().splitlines()[-1:] or ["failed"]
        raise HTTPException(400, f"region discovery failed: {msg[0][:180]}")
    info = json.loads((proc.stdout.strip().splitlines()[-1] or "OK {}")[3:] or "{}")
    return {"regions": info.get("regions", []), "window": info.get("window"),
            "all_time_sales": info.get("all_time_sales"), "latest_sale": info.get("latest_sale")}


# ─── onboarding: sync path + async job (with image-typing progress) ──────────

JOBS: dict[str, dict] = {}  # job_id -> live progress record


def _pull_cmd(host: str, cid: int, name: str, regions: list[str], with_images: bool = False) -> list[str]:
    cmd = [sys.executable, str(ROOT / "pull_company.py"), host, str(cid), name,
           str(ONBOARD / f"{host.replace('.', '_')}_{cid}.parquet"),
           ",".join(regions)]  # empty string = auto top-3
    if with_images:
        cmd.append("--images")
    return cmd


def _run_onboard_job(jid: str, host: str, cid: int, name: str, regions: list[str], mode: str) -> None:
    """Background onboarding with progress: pull → (classify photos) → grade.
    mode ∈ {image, both} here (text goes through the sync path)."""
    j = JOBS[jid]
    out = ONBOARD / f"{host.replace('.', '_')}_{cid}.parquet"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    typed_ids: list[int] = []
    try:
        j.update(phase="pulling", message="Looking at your outlets — pulling them from the warehouse…")
        proc = subprocess.run(_pull_cmd(host, cid, name, regions, with_images=True), capture_output=True,
                              text=True, timeout=1200, cwd=str(ROOT), env=env)  # no 200s ceiling here
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError((proc.stderr or proc.stdout or "pull failed").strip().splitlines()[-1][:180])
        info = json.loads((proc.stdout.strip().splitlines()[-1] or "OK {}")[3:] or "{}")

        df = pl.read_parquet(out).with_columns(pl.col("outletid").cast(pl.Int64))
        j.update(phase="typing", message="Looking at storefront photos…")
        import image_typing
        if not image_typing.available():
            j.update(message="Image model unavailable on this host — using text typing.")
        elif "image_id" not in df.columns:
            j.update(message="No storefront photos found for this client — using text typing.")
        else:
            # type the selling outlets that have a storefront photo, bounded so a
            # huge client can't run forever; the rest fall back to text.
            have = df.filter(pl.col("has_data") & pl.col("image_id").is_not_null())
            items = [(int(r["outletid"]), r["image_id"]) for r in
                     have.select("outletid", "image_id").head(IMG_TYPE_CAP).to_dicts()]
            start = time.time()

            def fcb(done: int, total: int) -> None:
                j.update(phase="typing", processed=done, total=total, pct=round(100 * done / max(total, 1)),
                         message=f"Fetching storefront photos {done}/{total}…")

            def cb(done: int, total: int) -> None:
                el = time.time() - start
                rate = done / el if el > 0 else 0
                j.update(processed=done, total=total, pct=round(100 * done / max(total, 1)),
                         eta_s=int((total - done) / rate) if rate > 0 else None,
                         message=f"Reading photos {done}/{total}…")

            typed, seen = image_typing.type_outlets(items, progress_cb=cb, fetch_cb=fcb)
            typed_ids = list(typed.keys())
            if typed:  # inject the image format via shoptypename so the engine picks it up
                m = pl.DataFrame({"outletid": list(typed.keys()), "_imgfmt": list(typed.values())})
                df = df.join(m, on="outletid", how="left")
                df = df.with_columns(shoptypename=pl.coalesce(["_imgfmt", "shoptypename"])).drop("_imgfmt")
            j.update(images_typed=len(typed), images_total=seen,
                     message=f"Classified {len(typed)} of {seen} photos")

        # no usable photos → don't leave the company uncapped with text data;
        # fall back to text segmentation so the ≤6 cap still applies.
        eff_mode = mode if typed_ids else "text"
        df = df.with_columns(seg_mode=pl.lit(eff_mode))
        df.write_parquet(out)
        mode = eff_mode

        j.update(phase="grading", message="Re-grading all outlets…")
        global STORE
        STORE = Store(peer_axes=STORE.axes if STORE else None)
        j.update(status="succeeded", phase="done", message="Onboarded.",
                 result=_onboard_result(name, mode, info, len(typed_ids)))
    except subprocess.TimeoutExpired:
        j.update(status="failed", phase="done", error="pull timed out", message="Pull timed out.")
    except Exception as e:  # noqa: BLE001
        j.update(status="failed", phase="done", error=str(e)[:200], message="Failed: " + str(e)[:120])


def _sample_image_outlets(company: str, n: int = 24) -> list[int]:
    """Outletids for the company that have a cached photo — review thumbnail grid.
    Prefers typed/selling outlets; survives across runs (reads the image cache)."""
    s = STORE
    if s is None:
        return []
    g = s.graded.filter((pl.col("company_name") == company) & pl.col("has_data"))
    ids = [int(o) for o in g.select("outletid").to_series().to_list()]
    return [i for i in ids if (IMG_DIR / f"{i}.jpg").exists()][:n]


def _onboard_result(name: str, mode: str, info: dict, images_typed: int) -> dict:
    s = store()
    cg = s.graded.filter((pl.col("company_name") == name) & pl.col("has_data"))
    tiers = dict(zip(*cg.group_by("tier").len().to_dict(as_series=False).values())) if cg.height else {}
    regions = sorted([r for r in cg.select(pl.col("regionname").unique()).to_series().to_list() if r])
    return {"company": name, "seg_mode": mode, "regions": info.get("regions", regions),
            "selling_outlets": info.get("selling", cg.height),
            "images_typed": images_typed, "sample_outlets": _sample_image_outlets(name),
            "peers": cg.select(pl.col("peer").n_unique()).item() if cg.height else 0,
            "tier_distribution": tiers,
            "companies_now": s.graded.select(pl.col("company_name").n_unique()).item()}


@app.get("/api/company/job/{job_id}")
def company_job(job_id: str) -> dict:
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


def _seg_mode(body: dict) -> str:
    """How to segment the onboarded company: text (attributes only), image (the
    photo classification is the segmentation), or both."""
    m = (body.get("mode") or ("image" if body.get("use_images") else "text")).strip().lower()
    return m if m in ("text", "image", "both") else "text"


@app.post("/api/company/add")
def company_add(body: dict = Body(...)) -> dict:
    host = (body.get("host") or os.environ.get("PROFILER_TRINO_HOST") or "").strip()
    cid = int(body["company_id"])
    name = (body.get("name") or f"Company {cid}").strip()
    regions = [str(r) for r in (body.get("regions") or []) if r]
    mode = _seg_mode(body)
    out = ONBOARD / f"{host.replace('.', '_')}_{cid}.parquet"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    if mode in ("image", "both"):  # async job with photo-classification progress
        jid = "job_" + uuid.uuid4().hex[:12]
        JOBS[jid] = {"job_id": jid, "status": "running", "phase": "queued", "total": 0,
                     "processed": 0, "pct": 0, "eta_s": None, "message": "Queued…",
                     "company": name, "mode": mode, "result": None, "error": None}
        threading.Thread(target=_run_onboard_job, args=(jid, host, cid, name, regions, mode),
                         daemon=True).start()
        return {"async": True, "job_id": jid}

    try:  # synchronous path (text attributes) — isolated subprocess
        proc = subprocess.run(_pull_cmd(host, cid, name, regions), capture_output=True,
                              text=True, timeout=200, cwd=str(ROOT), env=env)
    except subprocess.TimeoutExpired:
        raise HTTPException(400, "onboarding timed out — try image-based (async) or fewer regions")
    if proc.returncode != 0 or not out.exists():
        msg = (proc.stderr or proc.stdout or "unknown error").strip().splitlines()[-1:] or ["failed"]
        raise HTTPException(400, f"onboarding failed: {msg[0][:180]}")
    info = json.loads((proc.stdout.strip().splitlines()[-1] or "OK {}")[3:] or "{}")
    pl.read_parquet(out).with_columns(seg_mode=pl.lit("text")).write_parquet(out)
    global STORE
    STORE = Store(peer_axes=STORE.axes if STORE else None)
    return {"ok": True, **_onboard_result(name, "text", info, 0)}


@app.get("/api/outlet/{outletid}/image")
def outlet_image(outletid: int):
    """Serve the cached storefront thumbnail for an outlet (image onboarding)."""
    p = IMG_DIR / f"{outletid}.jpg"
    if not p.exists():
        raise HTTPException(404, "no image")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/api/company/images")
def company_images(company: str, limit: int = Query(24, le=120)) -> dict:
    """Outletids for a company that have a cached storefront photo — thumbnail
    grid on the onboarding-review and Segments pages."""
    s = store()
    g = s.graded.filter((pl.col("company_name") == company) & pl.col("has_data"))
    ids = [int(o) for o in g.select("outletid").to_series().to_list()]
    have = [i for i in ids if (IMG_DIR / f"{i}.jpg").exists()]
    return {"company": company, "n": len(have), "outlets": have[:limit]}


def _company_source_parquet(company: str) -> Path | None:
    """The onboarded parquet that holds this company (None if it is a base co)."""
    for p in sorted(ONBOARD.glob("*.parquet")):
        try:
            names = pl.read_parquet(p, columns=["company_name"])["company_name"].unique().to_list()
            if company in names:
                return p
        except Exception:
            pass
    return None


def _type_company_images(company: str, progress=None) -> int:
    """Classify whatever storefront photos are available for a company and persist
    the formats: into its onboarded parquet (onboarded co) or an outletid→format
    override (base co, typed from the local image cache). Returns count typed."""
    import image_typing
    if not image_typing.available():
        return 0
    s = store()
    g = s.graded.filter((pl.col("company_name") == company) & pl.col("has_data"))
    ids = [int(o) for o in g.select("outletid").to_series().to_list()]
    src = _company_source_parquet(company)
    ref_by = {}
    if src is not None:
        d = pl.read_parquet(src)
        if "image_id" in d.columns:
            ref_by = {int(r["outletid"]): r["image_id"]
                      for r in d.select("outletid", "image_id").to_dicts() if r["image_id"]}
    items = [(i, ref_by.get(i)) for i in ids
             if (i in ref_by) or (IMG_DIR / f"{i}.jpg").exists()][:IMG_TYPE_CAP]
    if not items:
        return 0
    typed, _seen = image_typing.type_outlets(items, progress_cb=progress, fetch_cb=progress)
    if not typed:
        return 0
    m = pl.DataFrame({"outletid": list(typed.keys()), "_imgfmt": list(typed.values())})
    if src is not None:  # onboarded: bake into its parquet's shoptypename
        d = pl.read_parquet(src).with_columns(pl.col("outletid").cast(pl.Int64))
        d = d.join(m, on="outletid", how="left")
        d = d.with_columns(shoptypename=pl.coalesce(["_imgfmt", "shoptypename"])).drop("_imgfmt")
        d.write_parquet(src)
    else:  # base company: merge into the outletid→format override
        if IMG_OVERRIDES.exists():
            prev = pl.read_parquet(IMG_OVERRIDES)
            m = pl.concat([prev.filter(~pl.col("outletid").is_in(list(typed.keys()))), m],
                          how="diagonal_relaxed")
        m.write_parquet(IMG_OVERRIDES)
    return len(typed)


def _run_segmode_job(jid: str, company: str, mode: str) -> None:
    j = JOBS[jid]
    try:
        ov = _seg_mode_overrides(); ov[company] = mode
        SEG_MODES.write_text(json.dumps(ov))
        typed = 0
        if mode in ("image", "both"):
            j.update(phase="typing", message="Looking at storefront photos…")

            def cb(done, total):
                j.update(processed=done, total=total, pct=round(100 * done / max(total, 1)),
                         message=f"Reading photos {done}/{total}…")
            typed = _type_company_images(company, progress=cb)
            if typed == 0:  # no usable photos → keep text so the ≤6 cap still applies
                mode = "text"
                ov = _seg_mode_overrides(); ov[company] = "text"; SEG_MODES.write_text(json.dumps(ov))
                j.update(message="No storefront photos on file — kept text segmentation.")
        j.update(phase="grading", message="Re-segmenting…")
        global STORE
        STORE = Store(peer_axes=STORE.axes if STORE else None)
        j.update(status="succeeded", phase="done", message="Re-segmented.",
                 result=_onboard_result(company, mode, {}, typed))
    except Exception as e:  # noqa: BLE001
        j.update(status="failed", phase="done", error=str(e)[:200], message="Failed: " + str(e)[:120])


@app.post("/api/company/segmode")
def company_segmode(body: dict = Body(...)) -> dict:
    company = (body.get("company") or "").strip()
    mode = _seg_mode(body)
    if not company:
        raise HTTPException(400, "company required")
    if mode == "text":  # fast — no typing, just re-cap + re-grade
        ov = _seg_mode_overrides(); ov[company] = mode
        SEG_MODES.write_text(json.dumps(ov))
        global STORE
        STORE = Store(peer_axes=STORE.axes if STORE else None)
        return {"async": False, **_onboard_result(company, mode, {}, 0)}
    jid = "job_" + uuid.uuid4().hex[:12]
    JOBS[jid] = {"job_id": jid, "status": "running", "phase": "queued", "total": 0,
                 "processed": 0, "pct": 0, "eta_s": None, "message": "Queued…",
                 "company": company, "mode": mode, "result": None, "error": None}
    threading.Thread(target=_run_segmode_job, args=(jid, company, mode), daemon=True).start()
    return {"async": True, "job_id": jid}


@app.post("/api/mission")
def mission_run(body: dict = Body(...)) -> dict:
    s = store()
    res = mission.classify_mission(
        s.graded, body.get("text", ""), company=body.get("company"),
        weight_override=body.get("weights"), adjust_reasons=body.get("adjust_reasons"),
        limit=int(body.get("limit", 40)), region_filter=body.get("regions"))
    (RUNS / f"{res['run_id']}.json").write_text(json.dumps(res, default=str))
    return res


@app.post("/api/promote")
def promote(body: dict = Body(...)) -> dict:
    s = store()
    row = s.graded.filter(pl.col("outletid") == int(body["outletid"]))
    if row.height == 0:
        raise HTTPException(404, "outlet not found")
    w = body.get("weights") or mission.ARCHETYPES["balanced"]["weights"]
    o = row.to_dicts()[0]
    return mission.promote(o, s.graded, w)


@app.get("/api/run/{run_id}")
def run_get(run_id: str) -> dict:
    p = RUNS / f"{run_id}.json"
    if not p.exists():
        raise HTTPException(404, "run not found")
    return json.loads(p.read_text())


@app.get("/api/runs")
def runs() -> dict:
    out = []
    for p in sorted(RUNS.glob("run_*.json")):
        try:
            d = json.loads(p.read_text())
            rec = {k: d[k] for k in ("run_id", "mission", "label", "archetype", "company",
                                     "n_candidates", "computed_at") if k in d}
            rec["weights"] = d.get("weights")
            rec["target_tiers"] = d.get("target_tiers")
            rec["guard_safe"] = (d.get("guard") or {}).get("safe")
            rec["edited"] = bool(d.get("adjust_reasons"))
            out.append(rec)
        except Exception:
            pass
    return {"runs": out[-30:][::-1]}


@app.get("/api/agent-runs")
def agent_runs(limit: int = 30) -> dict:
    """The supervisor's runs (agent /v1/runs), so the operator can see contract
    calls in the UI — they live in the separate agent SQLite store, not data/runs."""
    try:
        from agent import store as agent_store
        rows = agent_store.list_runs(limit=limit)  # all tenants, newest first
    except Exception as e:  # noqa: BLE001
        return {"runs": [], "error": str(e)[:150]}
    out = []
    for r in rows:
        c = r.get("counters") or {}
        out.append({
            "run_id": r.get("id"), "action": r.get("action"), "status": r.get("status"),
            "created_at": r.get("created_at"), "tenant": r.get("tenant_id"),
            "company": c.get("company"), "label": c.get("label"),
            "reasoning_mode": c.get("reasoning_mode"),
            "n_outputs": len(r.get("outputs") or []),
            "summary": (r.get("outcome") or {}).get("summary") or r.get("summary"),
        })
    return {"runs": out}


@app.get("/api/coldstart")
def coldstart(company: str | None = None) -> dict:
    s = store()
    cs = dict(s.coldstart)
    if company:
        cs["sample"] = [r for r in cs["sample"] if r.get("company_name") == company]
    return cs


# ─── UI host ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": STORE is not None})


@app.get("/api/health/vision")
def health_vision(warm: bool = False) -> dict:
    """Is the storefront-photo (SigLIP) stack ready? `available` = torch+open_clip
    installed; `loaded` = the model is in memory (lazy — set on the first photo
    job). Call with ?warm=1 to force-load it now and time it."""
    import image_typing
    avail = image_typing.available()
    out = {"available": avail, "loaded": image_typing.loaded() if avail else False,
           "cached_images": len(list(IMG_DIR.glob("*.jpg"))) if IMG_DIR.exists() else 0}
    if warm and avail and not out["loaded"]:
        t = time.time()
        try:
            image_typing._model()          # downloads (first time) + loads into memory
            out.update(loaded=True, warmed=True, load_seconds=round(time.time() - t, 1))
        except Exception as e:  # noqa: BLE001
            out.update(warmed=False, error=str(e)[:200])
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


# ─── CPG-OS Case-C agent surfaces (Outlet Profiler) ──────────────────────
# Manifest, A2A card, /v1/runs (idempotency), MCP, health — mounted over the
# same in-memory grader. The engine stays storage-agnostic; the agent package
# wraps its output into CPG-OS contract objects.
from agent import mount_agent  # noqa: E402

mount_agent(app, store)


if (WEB / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")
