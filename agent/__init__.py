"""Outlet Profiler — CPG-OS Case-C depth agent layer over the segmentation engine.

Adds the headless-agent contract surfaces (manifest, A2A card, /v1/runs with
idempotency, MCP, health, CLI) on top of the in-memory grader, mirroring the
reference `outlet-classifier` agent. The engine stays storage-agnostic; this
package is the adapter that wraps engine output into CPG-OS contract objects.
"""
from .api import mount_agent

__all__ = ["mount_agent"]
