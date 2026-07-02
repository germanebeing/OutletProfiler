"""Minimal read-only ClickHouse HTTP client for the Colgate (colpal) tenant.

Credentials come from the environment — never hard-coded here:
  CH_HOST (warehouse host, from env), CH_PORT (8123),
  CH_USER (admin), CH_PW (required), CH_DB (unify).
Read-only usage only (SELECT / SHOW / DESCRIBE).
"""
from __future__ import annotations

import os

import httpx

HOST = os.environ.get("CH_HOST", "")
PORT = os.environ.get("CH_PORT", "8123")
USER = os.environ.get("CH_USER", "admin")
PW = os.environ.get("CH_PW", "")
DB = os.environ.get("CH_DB", "unify")
BASE = f"http://{HOST}:{PORT}/"


def run(sql: str, fmt: str = "JSONCompact", db: str | None = None, timeout: int = 180):
    q = sql.strip().rstrip(";") + (f" FORMAT {fmt}" if fmt else "")
    r = httpx.post(BASE, params={"database": db or DB}, content=q.encode(),
                   headers={"X-ClickHouse-User": USER, "X-ClickHouse-Key": PW}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    j = r.json()
    return j.get("meta"), j.get("data")
