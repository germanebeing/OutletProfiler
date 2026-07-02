"""In-process async worker — the validation-lab analog of the reference agent's
run_one / worker_loop. POST /v1/runs only enqueues (status queued); this worker
claims queued runs, executes the handler, and transitions them to
succeeded/failed. Callers poll GET /v1/runs/{id}. Production would split this
into a KEDA-scaled worker per the AGENTS.md tier model; here it is a daemon
thread over the same durable SQLite run store."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from . import handlers, identity, log, store

_started = False
_STOP = threading.Event()  # set on shutdown so workers drain and exit cleanly
_CB_BACKOFF = (1.0, 3.0)  # retry waits after attempts 1 and 2 (3 attempts total)


def stop() -> None:
    """Signal the pool to stop claiming (in-flight jobs finish first)."""
    _STOP.set()


def _fire_callback(run: dict) -> None:
    """POST the finished run to the caller's callback_url (webhook push), so a
    supervisor that doesn't poll still learns the outcome. Best-effort: retries
    with backoff + timeout, and a callback failure NEVER fails the run."""
    url = run.get("callback_url")
    if not url or run.get("dry_run"):
        return
    import time
    import httpx
    payload = {"event": "run.completed", **store.summary(run),
               "outputs_url": f"{identity.PUBLIC_API_URL.rstrip('/')}/v1/runs/{run['id']}/outputs"}
    headers = {"Content-Type": "application/json"}
    if run.get("trace_id"):
        headers["X-Trace-Id"] = run["trace_id"]
    for attempt in range(3):
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=10.0)
            if 200 <= r.status_code < 300:
                log.log_event("callback.sent", trace_id=run.get("trace_id"),
                              tenant_id=run["tenant_id"], run_id=run["id"], action=run["action"],
                              url=url, status=r.status_code)
                return
            raise RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            if attempt < 2:
                time.sleep(_CB_BACKOFF[attempt])
            else:
                log.log_event("callback.failed", trace_id=run.get("trace_id"),
                              tenant_id=run["tenant_id"], run_id=run["id"], action=run["action"],
                              url=url, error=str(e))


def execute_run(run: dict, get_store: Callable[[], Any]) -> dict:
    """Run one claimed job to a terminal state. Honors a cancel requested
    before execution; the grade itself is a short synchronous pass."""
    action = run["action"]
    tid, rid = run["tenant_id"], run["id"]
    if run.get("cancel_requested"):
        store.event(run, "cancelled", "Cancelled before execution")
        return store.update(run, status="cancelled")
    log.log_event("run.started", trace_id=run.get("trace_id"), tenant_id=tid, run_id=rid, action=action)
    try:
        res = handlers.HANDLERS[action](get_store, rid, tid, run.get("signal_id"),
                                        run.get("input") or {})
        store.event(run, "succeeded", f"Emitted {len(res['outputs'])} contract objects")
        run = store.update(run, status="succeeded", outputs=res["outputs"], counters=res["counters"],
                           produces=res["produces"], reasoning_modes=res["reasoning_modes"],
                           summary=res.get("summary"), verdict=res.get("verdict"))
        log.log_event("run.succeeded", trace_id=run.get("trace_id"), tenant_id=tid, run_id=rid,
                      action=action, n_outputs=len(res["outputs"]))
        _fire_callback(run)
        return run
    except Exception as e:  # noqa: BLE001
        store.event(run, "failed", str(e))
        log.log_event("run.failed", trace_id=run.get("trace_id"), tenant_id=tid, run_id=rid,
                      action=action, error=str(e))
        run = store.update(run, status="failed", error={"code": "handler_error", "message": str(e)})
        _fire_callback(run)
        return run


def _store_ready(get_store: Callable[[], Any]) -> bool:
    try:
        st = get_store()
        return st is not None and getattr(st, "graded", None) is not None
    except Exception:
        return False


def run_pending_once(get_store: Callable[[], Any]) -> bool:
    """Claim + execute at most one queued run. Returns False if none pending.
    Exposed so tests and the CLI can drive the queue deterministically."""
    if not _store_ready(get_store):
        return False
    run = store.claim_next("worker-once")
    if not run:
        return False
    execute_run(run, get_store)
    return True


def worker_loop(get_store: Callable[[], Any], poll: float = 0.25) -> None:
    while not _STOP.is_set():
        try:
            if not (_store_ready(get_store) and _drain_one(get_store)):
                time.sleep(poll)
        except Exception:  # noqa: BLE001 — a worker must never die on one bad job
            time.sleep(poll)


def _drain_one(get_store: Callable[[], Any]) -> bool:
    run = store.claim_next("worker-loop")
    if not run:
        return False
    execute_run(run, get_store)
    return True


def start_worker(get_store: Callable[[], Any]) -> None:
    """Start a small pool of worker threads. claim_next is lock-guarded so
    concurrent workers never double-claim; effective concurrency == pool size
    (== capacity.max_concurrent_runs)."""
    global _started
    if _started:
        return
    _started = True
    _STOP.clear()
    store.requeue_running()  # reclaim runs left 'running' by a prior crash
    for i in range(max(1, identity.WORKER_POOL)):
        threading.Thread(target=worker_loop, args=(get_store,), daemon=True,
                         name=f"profiler-worker-{i}").start()
