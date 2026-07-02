"""Mounts the CPG-OS Case-C agent surfaces onto the existing FastAPI app:

  GET  /.well-known/agent.json          manifest
  GET  /.well-known/agent-card.json     A2A card
  POST /v1/runs                         submit (Idempotency-Key + Bearer required)
  GET  /v1/runs                         list
  GET  /v1/runs/{id}                    detail (+ emitted contract outputs)
  GET  /v1/runs/{id}/outputs            just the outputs
  POST /v1/runs/{id}/cancel             cancel
  POST /a2a                             A2A JSON-RPC (message/send, tasks/*)
  POST /mcp  GET /mcp/tools  POST /mcp/tools/{name}   MCP surface
  GET  /health/live  /health/ready  /healthz  /readyz

Execution is asynchronous: POST /v1/runs enqueues (status queued) and an
in-process worker pool (agent/worker.py) drains it to succeeded/failed over the
durable SQLite run store (agent/store.py). Callers poll GET /v1/runs/{id} or set
callback_url for a push. Per-tenant rate limiting is enforced in _submit, so it
applies uniformly across REST / MCP / A2A.
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Header, HTTPException, Request, Response

from . import identity, log, store, worker
from .manifest import build_agent_card, build_manifest, build_tools

_GET_STORE: Callable[[], Any] | None = None
router = APIRouter()

# ─── per-tenant rate limit (advertised in the manifest capacity) ────────────
import time as _time
from collections import defaultdict, deque

_RL: dict = defaultdict(deque)
_RL_RPS = identity.CAPACITY.get("rate_limit_per_tenant_rps", 5)


def _rate_check(tenant: str) -> None:
    now = _time.monotonic()
    q = _RL[tenant]
    while q and now - q[0] > 1.0:
        q.popleft()
    if len(q) >= _RL_RPS:
        raise HTTPException(429, {"code": "rate_limited",
                                  "message": f"> {_RL_RPS} runs/s for tenant {tenant}"})
    q.append(now)


# ─── auth + tenant ────────────────────────────────────────────────────────

def _tenant(authorization: str | None, x_tenant: str | None, body_tenant: str | None) -> str:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if identity.REQUIRE_AUTH and (not token or token not in identity.DEV_AUTH_TOKENS):
        raise HTTPException(401, {"code": "unauthorized", "message": "Bearer token required"})
    if body_tenant:
        return body_tenant
    if token and token in identity.DEV_AUTH_TOKENS:
        return identity.DEV_AUTH_TOKENS[token]
    if x_tenant:
        return x_tenant
    return identity.DEFAULT_TENANT


# ─── run submission (enqueue; the worker executes) ──────────────────────────

def _sig(body: dict) -> str | None:
    """signal_id from the canonical envelope: triggered_by.signal_id (or .signal),
    falling back to a top-level signal_id."""
    tb = body.get("triggered_by") or {}
    return body.get("signal_id") or tb.get("signal_id") or tb.get("signal")


def _submit(*, tenant_id: str, action: str, payload: dict, idempotency_key: str | None,
            signal_id: str | None, triggered_by: dict | None, context: dict | None,
            dry_run: bool, trace_id: str | None = None, callback_url: str | None = None) -> dict:
    _rate_check(tenant_id)  # enforced on every surface (REST / MCP / A2A) via this seam
    if action not in identity.ACTION_NAMES:
        raise HTTPException(400, {"code": "unknown_action", "message": action,
                                  "allowed": identity.ACTION_NAMES})
    # idempotency: an already-seen key returns the existing run unchanged
    if idempotency_key:
        existing = store.get_run_id_for_key(tenant_id, action, idempotency_key)
        if existing:
            run = store.get_run(existing, tenant_id)
            if run is None:
                raise HTTPException(404, {"code": "run_not_found",
                                          "message": f"idempotent run {existing} missing"})
            return run
    name = payload.get("mission") or payload.get("hypothesis") or action
    run = store.create_run(
        tenant_id=tenant_id, action=action, name=str(name)[:200],
        agent_id=identity.AGENT_ID, agent_version=identity.AGENT_VERSION,
        input_payload=payload, idempotency_key=idempotency_key, signal_id=signal_id,
        triggered_by=triggered_by, context=context, dry_run=dry_run, trace_id=trace_id,
        callback_url=callback_url)
    if idempotency_key:
        store.bind_key(tenant_id, action, idempotency_key, run["id"])
    log.log_event("run.queued", trace_id=trace_id, tenant_id=tenant_id, run_id=run["id"],
                  action=action, signal_id=signal_id)
    if dry_run:  # simulate: accept, emit nothing
        store.event(run, "succeeded", "Dry run accepted; no grading performed")
        return store.update(run, status="succeeded", summary="Dry run accepted.")
    return run  # left queued; the worker thread will pick it up


# ─── manifest + card ────────────────────────────────────────────────────────

@router.get("/.well-known/agent.json")
def manifest() -> dict:
    return build_manifest()


@router.get("/.well-known/agent-card.json")
def agent_card() -> dict:
    return build_agent_card()


# ─── /v1/runs ──────────────────────────────────────────────────────────────

@router.post("/v1/runs")
async def create_run(request: Request, response: Response,
                     authorization: str | None = Header(default=None),
                     x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
                     x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
                     idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    body = await request.json()
    if not idempotency_key and not body.get("idempotency_key"):
        raise HTTPException(400, {"code": "missing_idempotency_key",
                                  "message": "Idempotency-Key header is required for run submission"})
    key = idempotency_key or body.get("idempotency_key")
    tenant = _tenant(authorization, x_tenant_id, body.get("tenant_id"))
    if x_trace_id:
        response.headers["X-Trace-Id"] = x_trace_id  # propagate
    run = _submit(
        tenant_id=tenant, action=body.get("action", ""),
        payload=body.get("agent_specific_payload") or body.get("payload") or {},
        idempotency_key=key, signal_id=_sig(body),
        triggered_by=body.get("triggered_by"), context=body.get("context"),
        dry_run=bool(body.get("dry_run", False)), trace_id=x_trace_id,
        callback_url=body.get("callback_url"))
    return store.summary(run)


@router.get("/v1/runs")
def list_runs(authorization: str | None = Header(default=None),
              x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
              action: str | None = None, status: str | None = None, since: str | None = None,
              cursor: str | None = None, limit: int = 100, all_tenants: bool = False) -> dict:
    tenant = None if all_tenants else _tenant(authorization, x_tenant_id, None)
    rows = store.list_runs(tenant, action=action, status=status, since=since,
                           cursor=cursor, limit=limit)
    next_cursor = rows[-1]["created_at"] if len(rows) == limit and rows else None
    return {"runs": [store.summary(r) for r in rows], "next_cursor": next_cursor}


@router.get("/v1/runs/{run_id}")
def get_run(run_id: str,
            authorization: str | None = Header(default=None),
            x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id")) -> dict:
    tenant = _tenant(authorization, x_tenant_id, None)
    run = store.get_run(run_id, tenant)  # tenant-scoped — no cross-tenant fallback
    if not run:
        raise HTTPException(404, {"code": "run_not_found", "message": run_id})
    return store.detail(run)


@router.get("/v1/runs/{run_id}/outputs")
def get_outputs(run_id: str,
                authorization: str | None = Header(default=None),
                x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id")) -> dict:
    tenant = _tenant(authorization, x_tenant_id, None)
    run = store.get_run(run_id, tenant)
    if not run:
        raise HTTPException(404, {"code": "run_not_found", "message": run_id})
    return {"run_id": run_id, "status": run["status"],
            "produces_contract_types": run.get("produces_contract_types", []),
            "outputs": run.get("outputs", [])}


@router.post("/v1/runs/{run_id}/cancel")
def cancel_run(run_id: str,
               authorization: str | None = Header(default=None),
               x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id")) -> dict:
    tenant = _tenant(authorization, x_tenant_id, None)
    if store.get_run(run_id, tenant) is None:  # enforce tenant ownership first
        raise HTTPException(404, {"code": "run_not_found", "message": run_id})
    return store.summary(store.request_cancel(run_id))


# ─── A2A JSON-RPC ────────────────────────────────────────────────────────────

_TASK_STATE = {"queued": "submitted", "running": "working", "succeeded": "completed",
               "failed": "failed", "cancelled": "canceled"}


def _task(run: dict) -> dict:
    return {"id": run["id"], "status": {"state": _TASK_STATE.get(run["status"], "unknown")},
            "artifacts": [{"parts": [{"type": "data", "data": o}]} for o in run.get("outputs", [])],
            "metadata": {"agent_id": run["agent_id"], "action": run["action"],
                         "counters": run.get("counters", {})}}


@router.post("/a2a")
async def a2a_rpc(request: Request,
                  authorization: str | None = Header(default=None),
                  x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
                  idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    msg = await request.json()
    method, params, mid = msg.get("method"), msg.get("params") or {}, msg.get("id")
    try:
        if method in ("message/send", "tasks/send"):
            req = params.get("action_request") or params
            key = idempotency_key or params.get("idempotency_key")
            if not key:
                raise HTTPException(400, {"code": "missing_idempotency_key"})
            tenant = _tenant(authorization, x_tenant_id, req.get("tenant_id"))
            run = _submit(tenant_id=tenant, action=req.get("action", ""),
                          payload=req.get("agent_specific_payload") or req.get("payload") or {},
                          idempotency_key=key, signal_id=_sig(req),
                          triggered_by=req.get("triggered_by"), context=req.get("context"),
                          dry_run=bool(req.get("dry_run", False)),
                          callback_url=req.get("callback_url"))
            result = _task(run)
        elif method in ("tasks/get", "tasks/status"):
            tenant = _tenant(authorization, x_tenant_id, None)
            run = store.get_run(params.get("id") or params.get("task_id") or "", tenant)
            if not run:
                raise HTTPException(404, {"code": "task_not_found"})
            result = _task(run)
        elif method == "tasks/cancel":
            tenant = _tenant(authorization, x_tenant_id, None)
            rid = params.get("id") or ""
            if store.get_run(rid, tenant) is None:  # enforce tenant ownership
                raise HTTPException(404, {"code": "task_not_found"})
            run = store.request_cancel(rid)  # signals the worker; no silent overwrite
            result = _task(run)
        else:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"method_not_found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    except HTTPException as e:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": e.status_code, "message": e.detail}}


# ─── MCP ──────────────────────────────────────────────────────────────────

def _invoke_tool(name: str, args: dict, tenant: str, key: str | None) -> Any:
    if name == "agent.describe":
        return build_manifest()
    if name in ("agent.run", "agent.simulate", "grade.grade_outlets", "grade.validate_hypothesis"):
        action = ({"grade.grade_outlets": "grade_outlets",
                   "grade.validate_hypothesis": "validate_opportunity_hypothesis"}
                  .get(name) or args.get("action"))
        payload = (args.get("agent_specific_payload")
                   if name in ("agent.run", "agent.simulate") else args) or {}
        k = args.get("idempotency_key") or key
        if not k:
            raise HTTPException(400, {"code": "missing_idempotency_key"})
        run = _submit(tenant_id=args.get("tenant_id") or tenant, action=action, payload=payload,
                      idempotency_key=k, signal_id=_sig(args),
                      triggered_by=args.get("triggered_by"), context=None,
                      dry_run=(name == "agent.simulate"), callback_url=args.get("callback_url"))
        return store.detail(run)
    if name == "agent.get_run":
        run = store.get_run(args["run_id"], tenant)  # tenant-scoped
        if not run:
            raise HTTPException(404, {"code": "run_not_found"})
        return store.detail(run)
    if name == "grade.get_run_outputs":
        run = store.get_run(args["run_id"], tenant)  # tenant-scoped
        if not run:
            raise HTTPException(404, {"code": "run_not_found"})
        return {"outputs": run.get("outputs", [])}
    if name == "agent.list_runs":
        return {"runs": [store.summary(r) for r in
                         store.list_runs(tenant, action=args.get("action"), status=args.get("status"),
                                         cursor=args.get("cursor"), limit=args.get("limit", 50))]}
    if name == "agent.cancel_run":
        if store.get_run(args["run_id"], tenant) is None:
            raise HTTPException(404, {"code": "run_not_found"})
        return store.summary(store.request_cancel(args["run_id"]))
    raise HTTPException(404, {"code": "tool_not_found", "message": name})


@router.get("/mcp/tools")
def mcp_tools() -> dict:
    return {"tools": build_tools()}


@router.post("/mcp/tools/{tool_name}")
async def mcp_tool_call(tool_name: str, request: Request,
                        authorization: str | None = Header(default=None),
                        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
                        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    args = await request.json()
    tenant = _tenant(authorization, x_tenant_id, args.get("tenant_id"))
    return _invoke_tool(tool_name, args, tenant, idempotency_key)


@router.post("/mcp")
async def mcp_jsonrpc(request: Request,
                      authorization: str | None = Header(default=None),
                      x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
                      idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    m = await request.json()
    method, params, mid = m.get("method"), m.get("params") or {}, m.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": params.get("protocolVersion") or "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": identity.AGENT_ID, "version": identity.AGENT_VERSION}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": build_tools()}}
    if method == "tools/call":
        tenant = _tenant(authorization, x_tenant_id, None)
        try:
            out = _invoke_tool(params.get("name", ""), params.get("arguments") or {}, tenant, idempotency_key)
        except HTTPException as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": e.status_code, "message": e.detail}}
        import json as _json
        return {"jsonrpc": "2.0", "id": mid,
                "result": {"content": [{"type": "text", "text": _json.dumps(out, default=str)}]}}
    if method in ("notifications/initialized", "ping"):
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method_not_found: {method}"}}


# ─── health ─────────────────────────────────────────────────────────────────

@router.get("/health/live")
def live() -> dict:
    return {"status": "live"}


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def ready() -> dict:
    ok = _GET_STORE is not None and getattr(_GET_STORE(), "graded", None) is not None
    if not ok:
        raise HTTPException(503, {"status": "not_ready"})
    return {"status": "ready", "agent_id": identity.AGENT_ID}


@router.get("/readyz")
def readyz() -> dict:
    return ready()


def mount_agent(app, get_store: Callable[[], Any]) -> None:
    """Wire the agent surfaces onto an existing FastAPI app and start the
    in-process run worker (queue → running → succeeded)."""
    global _GET_STORE
    _GET_STORE = get_store
    app.include_router(router)
    worker.start_worker(get_store)

    @app.on_event("shutdown")
    def _drain_workers() -> None:  # graceful shutdown: stop claiming, let in-flight finish
        worker.stop()
