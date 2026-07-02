#!/usr/bin/env python3
"""profiler — CLI for the Outlet Profiler agent. Purely API-backed (urllib), no
business logic. Mirrors the reference agent's `stage` CLI.

  profiler describe
  profiler run --action grade_outlets --company "Anchor" --mission "launch a premium SKU"
  profiler run --action validate_opportunity_hypothesis --company "GIL Live" \
               --hypothesis "my T1 outlets in WEST BENGAL have gone dormant"
  profiler get <run_id>
  profiler list
  profiler cancel <run_id>
  profiler smoke
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import uuid

BASE = os.environ.get("PROFILER_API_BASE", "http://localhost:8100")
TOKEN = os.environ.get("PROFILER_API_TOKEN", "dev-token")


def _call(method: str, path: str, body: dict | None = None, key: str | None = None) -> dict:
    url = BASE.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Idempotency-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return {"error": e.code, "detail": e.read().decode()[:400]}


def main(argv: list[str] | None = None) -> None:
    global BASE, TOKEN
    p = argparse.ArgumentParser(prog="profiler")
    p.add_argument("--api", default=BASE)
    p.add_argument("--token", default=TOKEN)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("describe")
    r = sub.add_parser("run")
    r.add_argument("--action", default="grade_outlets")
    r.add_argument("--company")
    r.add_argument("--mission")
    r.add_argument("--hypothesis")
    r.add_argument("--region")
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--idempotency-key", dest="key")
    g = sub.add_parser("get"); g.add_argument("run_id")
    sub.add_parser("list")
    c = sub.add_parser("cancel"); c.add_argument("run_id")
    sub.add_parser("smoke")

    a = p.parse_args(argv)
    BASE, TOKEN = a.api, a.token

    if a.cmd == "describe":
        print(json.dumps(_call("GET", "/.well-known/agent.json"), indent=2)[:2000])
    elif a.cmd == "run":
        payload: dict = {}
        if a.company:
            payload["company"] = a.company
        if a.mission:
            payload["mission"] = a.mission
        if a.hypothesis:
            payload["hypothesis"] = a.hypothesis
        if a.region:
            payload.setdefault("scope", {})["region"] = a.region
        payload["limit"] = a.limit
        body = {"action": a.action, "agent_specific_payload": payload}
        res = _call("POST", "/v1/runs", body, key=a.key or uuid.uuid4().hex)
        print(json.dumps(res, indent=2))
    elif a.cmd == "get":
        print(json.dumps(_call("GET", f"/v1/runs/{a.run_id}"), indent=2))
    elif a.cmd == "list":
        print(json.dumps(_call("GET", "/v1/runs"), indent=2))
    elif a.cmd == "cancel":
        print(json.dumps(_call("POST", f"/v1/runs/{a.run_id}/cancel"), indent=2))
    elif a.cmd == "smoke":
        for path in ["/health/live", "/health/ready", "/.well-known/agent.json",
                     "/.well-known/agent-card.json", "/mcp/tools", "/v1/runs"]:
            res = _call("GET", path)
            ok = "error" not in res
            print(f"  {'OK ' if ok else 'ERR'} GET {path}")


if __name__ == "__main__":
    main()
