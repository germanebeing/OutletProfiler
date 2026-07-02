"""Structured JSON logging — one line per event with the required fields
(TraceId, TenantId, RunId, Action). Stdlib only (no new dependency); satisfies
the 'structured JSON logs on every line' standard for the validation lab."""
from __future__ import annotations

import json
import logging
import sys

_logger = logging.getLogger("outlet_profiler")
if not _logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)


def log_event(event: str, *, trace_id: str | None = None, tenant_id: str | None = None,
              run_id: str | None = None, action: str | None = None, **fields) -> None:
    rec = {"event": event, "TraceId": trace_id, "TenantId": tenant_id,
           "RunId": run_id, "Action": action}
    rec.update(fields)
    _logger.info(json.dumps({k: v for k, v in rec.items() if v is not None}, default=str))
