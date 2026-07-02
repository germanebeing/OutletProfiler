"""Durable run store — SQLite (WAL), ACID and concurrency-safe across the worker
pool. Exposes the run lifecycle + idempotency the agent surfaces need:

  create_run / get_run / list_runs         (lifecycle, tenant-scoped)
  get_run_id_for_key / bind_key            (idempotency: (tenant,action,key)->run_id)
  claim_next / request_cancel / update     (queue + transitions)

A run is stored as a JSON document with indexed columns (tenant_id, action,
status, created_at) for filtered, cursor-paginated listing. `claim_next` uses a
`BEGIN IMMEDIATE` transaction so concurrent workers never double-claim.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__import__("os").environ.get("PROFILER_DB", str(ROOT / "data" / "agent.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    return c


def _init() -> None:
    with closing(_conn()) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, tenant_id TEXT, action TEXT, status TEXT,
            created_at TEXT, updated_at TEXT, data TEXT)""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_runs_list ON runs(tenant_id, created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status, created_at)")
        c.execute("""CREATE TABLE IF NOT EXISTS idempotency (
            tenant_id TEXT, action TEXT, key TEXT, run_id TEXT,
            PRIMARY KEY (tenant_id, action, key))""")


_init()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:16]


def _save(run: dict) -> None:
    with closing(_conn()) as c:
        c.execute("""INSERT INTO runs (id, tenant_id, action, status, created_at, updated_at, data)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at,
                data=excluded.data""",
            (run["id"], run["tenant_id"], run["action"], run["status"],
             run["created_at"], run["updated_at"], json.dumps(run, default=str)))


# ─── idempotency ─────────────────────────────────────────────────────────

def get_run_id_for_key(tenant_id: str, action: str, key: str) -> str | None:
    with closing(_conn()) as c:
        row = c.execute("SELECT run_id FROM idempotency WHERE tenant_id=? AND action=? AND key=?",
                        (tenant_id, action, key)).fetchone()
    return row["run_id"] if row else None


def bind_key(tenant_id: str, action: str, key: str, run_id: str) -> None:
    with closing(_conn()) as c:  # first writer wins
        c.execute("INSERT OR IGNORE INTO idempotency (tenant_id, action, key, run_id) VALUES (?,?,?,?)",
                  (tenant_id, action, key, run_id))


# ─── lifecycle ───────────────────────────────────────────────────────────

def create_run(*, tenant_id: str, action: str, name: str, agent_id: str,
               agent_version: str, input_payload: dict, idempotency_key: str | None,
               signal_id: str | None, triggered_by: dict | None,
               context: dict | None, dry_run: bool, trace_id: str | None = None,
               callback_url: str | None = None) -> dict:
    run_id = new_run_id()
    run = {
        "id": run_id, "run_id": run_id, "tenant_id": tenant_id, "action": action,
        "name": name, "status": "queued", "created_at": _now(), "updated_at": _now(),
        "idempotency_key": idempotency_key, "signal_id": signal_id, "trace_id": trace_id,
        "callback_url": callback_url,
        "agent_id": agent_id, "agent_version": agent_version,
        "input": input_payload, "triggered_by": triggered_by or {},
        "context": context or {}, "dry_run": bool(dry_run),
        "produces_contract_types": [], "outputs": [], "counters": {},
        "reasoning_modes": [], "summary": None, "verdict": None,
        "cancel_requested": False, "error": None,
        "timeline": [{"at": _now(), "event": "submitted", "message": "Run accepted"}],
    }
    _save(run)
    return run


def get_run(run_id: str, tenant_id: str | None = None) -> dict | None:
    with closing(_conn()) as c:
        row = c.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    run = json.loads(row["data"])
    if tenant_id and run.get("tenant_id") != tenant_id:
        return None
    return run


def list_runs(tenant_id: str | None = None, action: str | None = None,
              status: str | None = None, since: str | None = None,
              cursor: str | None = None, limit: int = 100) -> list[dict]:
    q = "SELECT data FROM runs WHERE 1=1"
    args: list = []
    if tenant_id and tenant_id != "*":
        q += " AND tenant_id=?"; args.append(tenant_id)
    if action:
        q += " AND action=?"; args.append(action)
    if status:
        q += " AND status=?"; args.append(status)
    if since:
        q += " AND created_at>=?"; args.append(since)
    if cursor:
        q += " AND created_at<?"; args.append(cursor)
    q += " ORDER BY created_at DESC LIMIT ?"; args.append(int(limit))
    with closing(_conn()) as c:
        rows = c.execute(q, args).fetchall()
    return [json.loads(r["data"]) for r in rows]


def claim_next(worker_id: str) -> dict | None:
    """Claim the oldest queued run and flip it to running, atomically — a
    BEGIN IMMEDIATE transaction so concurrent workers can't double-claim.
    A queued run flagged cancel_requested is marked cancelled and skipped."""
    with closing(_conn()) as c:
        while True:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT data FROM runs WHERE status='queued' "
                            "ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                c.execute("COMMIT")
                return None
            run = json.loads(row["data"])
            if run.get("cancel_requested"):
                run["status"] = "cancelled"; run["updated_at"] = _now()
                run.setdefault("timeline", []).append({"at": _now(), "event": "cancelled", "message": "Cancelled before start"})
            else:
                run["status"] = "running"; run["locked_by"] = worker_id; run["updated_at"] = _now()
                run.setdefault("timeline", []).append({"at": _now(), "event": "started", "message": f"Claimed by {worker_id}"})
            c.execute("UPDATE runs SET status=?, updated_at=?, data=? WHERE id=?",
                      (run["status"], run["updated_at"], json.dumps(run, default=str), run["id"]))
            c.execute("COMMIT")
            if run["status"] == "running":
                return run
            # else it was cancelled — keep scanning for the next queued run


def requeue_running() -> int:
    """Reclaim runs left in 'running' by a crashed/killed process — a fresh
    process has no legitimately in-flight run, so any 'running' row is stale and
    is put back to 'queued' for the worker to pick up. Called on worker start."""
    with closing(_conn()) as c:
        rows = c.execute("SELECT data FROM runs WHERE status='running'").fetchall()
        n = 0
        for r in rows:
            run = json.loads(r["data"])
            run["status"] = "queued"; run["updated_at"] = _now()
            run.setdefault("timeline", []).append(
                {"at": _now(), "event": "requeued", "message": "Reclaimed stale running run on restart"})
            c.execute("UPDATE runs SET status='queued', updated_at=?, data=? WHERE id=?",
                      (run["updated_at"], json.dumps(run, default=str), run["id"]))
            n += 1
    return n


def request_cancel(run_id: str) -> dict | None:
    run = get_run(run_id)
    if not run:
        return None
    if run["status"] == "queued":
        event(run, "cancelled", "Cancelled by caller")
        return update(run, status="cancelled")
    if run["status"] == "running":  # best-effort; the worker checks the flag
        run["cancel_requested"] = True
        event(run, "cancel_requested", "Cancel requested mid-run")
        _save(run)
    return run


def event(run: dict, ev: str, message: str = "") -> None:
    run.setdefault("timeline", []).append({"at": _now(), "event": ev, "message": message})


def update(run: dict, *, status: str | None = None, outputs: list[dict] | None = None,
           counters: dict | None = None, error: dict | None = None,
           produces: list[str] | None = None, reasoning_modes: list[str] | None = None,
           summary: str | None = None, verdict: str | None = None) -> dict:
    if status:
        run["status"] = status
    if outputs is not None:
        run["outputs"] = outputs
    if counters is not None:
        run["counters"] = counters
    if error is not None:
        run["error"] = error
    if produces is not None:
        run["produces_contract_types"] = produces
    if reasoning_modes is not None:
        run["reasoning_modes"] = reasoning_modes
    if summary is not None:
        run["summary"] = summary
    if verdict is not None:
        run["verdict"] = verdict
    run["updated_at"] = _now()
    _save(run)
    return run


# ─── views ───────────────────────────────────────────────────────────────

def _outcome(run: dict) -> dict:
    modes = run.get("reasoning_modes") or []
    mode = "reasoning" if "reasoning" in modes else ("deterministic" if modes else None)
    # changes[] = master-of-record mutations. The Profiler is read-only, so it
    # emits none — value flows via the emitted Observation/Opportunity/Diagnosis.
    return {"summary": run.get("summary"), "verdict": run.get("verdict"),
            "reasoning_mode": mode, "reversible": True, "changes": []}


def summary(run: dict) -> dict:
    return {
        "id": run["id"], "run_id": run["run_id"], "tenant_id": run["tenant_id"],
        "action": run["action"], "name": run.get("name"), "status": run["status"],
        "created_at": run["created_at"], "updated_at": run.get("updated_at"),
        "produces_contract_types": run.get("produces_contract_types", []),
        "n_outputs": len(run.get("outputs", [])), "counters": run.get("counters", {}),
        "reasoning_modes": run.get("reasoning_modes", []), "outcome": _outcome(run),
        "signal_id": run.get("signal_id"), "trace_id": run.get("trace_id"),
        "dry_run": run.get("dry_run", False), "error": run.get("error"),
    }


def detail(run: dict) -> dict:
    d = summary(run)
    d.update({"agent_id": run.get("agent_id"), "agent_version": run.get("agent_version"),
              "input": run.get("input"), "outputs": run.get("outputs", []),
              "timeline": run.get("timeline", [])})
    return d
