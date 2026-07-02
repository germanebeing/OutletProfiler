"""Builders for the agent's machine-readable contract: the manifest
(GET /.well-known/agent.json), the A2A card (GET /.well-known/agent-card.json),
and the MCP tools list. Generated from `identity` + `engine.contracts` so the
advertised schema never drifts from what the app emits."""
from __future__ import annotations

from typing import Any

from engine.contracts import components_schemas, output_envelope_schema

from . import identity as I

# ─── per-action input schemas ────────────────────────────────────────────

_GRADE_INPUT = {
    "type": "object",
    "properties": {
        "company": {"type": ["string", "null"], "description": "Active company (tenant scope). Omit / 'all' = supervisor cross-company."},
        "mission": {"type": "string", "description": "Plain-English play, e.g. 'launch a premium SKU in kirana'. Interpreted into lever weights (reasoning)."},
        "weights": {"type": "object", "description": "Explicit lever weights {range,cadence,recency,value}; overrides mission interpretation (deterministic)."},
        "regions": {"type": "array", "items": {"type": "string"}, "description": "Regions/geographies to scope the run to; omit or empty = all regions of the company."},
        "region": {"type": ["string", "null"], "description": "Single-region shorthand for `regions`."},
        "format": {"type": ["string", "null"]},
        "limit": {"type": "integer", "default": 40, "minimum": 1, "maximum": 500},
    },
    "anyOf": [{"required": ["mission"]}, {"required": ["weights"]}],
}

_VALIDATE_INPUT = {
    "type": "object",
    "required": ["hypothesis"],
    "properties": {
        "company": {"type": ["string", "null"]},
        "hypothesis": {"type": "string", "description": "Supervisor assertion to test data-first, e.g. 'my T1 outlets in Delhi have gone dormant'."},
        "scope": {"type": "object", "properties": {
            "region": {"type": "string"}, "format": {"type": "string"}, "tier": {"type": "string"}}},
    },
}

_OUTCOME_INPUT = {
    "type": "object",
    "properties": {"opportunity_id": {"type": "string"}, "run_id": {"type": "string"}},
}


def _actions() -> list[dict[str, Any]]:
    return [
        {
            "name": "grade_outlets",
            "description": "Grade outlets by opportunity for a business play (plain-English mission or explicit lever weights). Emits one Observation per outlet (tier + realisation + grade-vector) and one ₹-sized Opportunity per actionable outlet. Read-only.",
            "async": True, "reversible": True,
            "reasoning_mode": "reasoning",
            "side_effects": ["grade.run_created", "grade.tiers_assigned", "grade.opportunities_emitted"],
            "input_schema": _GRADE_INPUT,
            "output_schema": output_envelope_schema(["observation", "opportunity"]),
            "produces": ["observation", "opportunity"],
        },
        {
            "name": "validate_opportunity_hypothesis",
            "description": "Validate a supervisor hypothesis about outlet opportunity/tiers, data-first: re-grade the in-scope outlets against their peer frontiers, compare to the assertion, and return a confirm | refute | inconclusive Diagnosis tagged with reasoning_mode, plus the supporting Observations. Read-only.",
            "async": True, "reversible": True,
            "reasoning_mode": "reasoning",
            "side_effects": ["grade.run_created", "grade.observations_emitted", "grade.diagnosis_emitted"],
            "input_schema": _VALIDATE_INPUT,
            "output_schema": output_envelope_schema(["diagnosis", "observation"]),
            "produces": ["diagnosis", "observation"],
        },
        {
            "name": "analyze_outcome",
            "description": "Measure the effect of an acted-on opportunity and write it into the learning loop. Phase-1 stub (M5) — returns an inconclusive Diagnosis until post-action data is collected.",
            "async": True, "reversible": True,
            "reasoning_mode": "deterministic",
            "side_effects": ["outcome.measured"],
            "input_schema": _OUTCOME_INPUT,
            "output_schema": output_envelope_schema(["diagnosis"]),
            "produces": ["diagnosis"],
        },
    ]


def build_manifest() -> dict[str, Any]:
    return {
        "agent_id": I.AGENT_ID,
        "agent_version": I.AGENT_VERSION,
        "display_name": I.DISPLAY_NAME,
        "description": I.DESCRIPTION,
        "owner_squad": I.OWNER_SQUAD,
        "tier": I.TIER,
        "status": I.STATUS,
        "capacity": I.CAPACITY,
        "slo": I.SLO,
        "reasoning_modes": I.REASONING_MODES,
        "produces_contract_types": I.PRODUCES,
        "accepts_contract_types": I.ACCEPTS,
        "actions": _actions(),
        "surfaces": {
            "api": I.PUBLIC_API_URL, "mcp": I.PUBLIC_MCP_URL, "ui": I.PUBLIC_UI_URL,
            "cli": I.CLI_NAME, "a2a": I.PUBLIC_A2A_URL,
        },
        "components": {"schemas": components_schemas()},
    }


def build_agent_card() -> dict[str, Any]:
    return {
        "name": I.DISPLAY_NAME,
        "agent_id": I.AGENT_ID,
        "description": I.DESCRIPTION,
        "url": I.PUBLIC_A2A_URL,
        "version": I.AGENT_VERSION,
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
        "security": [{"bearer": []}],
        "produces_contract_types": I.PRODUCES,
        "accepts_contract_types": I.ACCEPTS,
        "reasoning_modes": I.REASONING_MODES,
        "skills": [
            {"id": "grade_outlets", "name": "Grade outlets by opportunity",
             "description": "Opportunity tier + grade-vector + ₹-sized headroom per outlet."},
            {"id": "validate_opportunity_hypothesis", "name": "Validate a hypothesis",
             "description": "Confirm/refute/inconclusive verdict on a supervisor assertion, data-first."},
            {"id": "analyze_outcome", "name": "Measure an outcome",
             "description": "Post-action impact into the learning loop (stub)."},
        ],
    }


# ─── MCP tools ────────────────────────────────────────────────────────────

def build_tools() -> list[dict[str, Any]]:
    _id = {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}
    return [
        {"name": "agent.describe", "description": "Return this agent's manifest.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "agent.run", "description": "Submit a run. args: {action, agent_specific_payload, idempotency_key?, tenant_id?, signal_id?}. Idempotency-Key required.",
         "input_schema": {"type": "object", "required": ["action", "agent_specific_payload"],
                          "properties": {"action": {"type": "string", "enum": I.ACTION_NAMES},
                                         "agent_specific_payload": {"type": "object"},
                                         "idempotency_key": {"type": "string"},
                                         "tenant_id": {"type": "string"},
                                         "signal_id": {"type": "string"}}}},
        {"name": "agent.get_run", "description": "Fetch a run (status + emitted contract outputs).", "input_schema": _id},
        {"name": "agent.list_runs", "description": "List recent runs for the tenant (filters + cursor pagination).",
         "input_schema": {"type": "object", "properties": {
             "action": {"type": "string"}, "status": {"type": "string"},
             "cursor": {"type": "string"}, "limit": {"type": "integer"}}}},
        {"name": "agent.cancel_run", "description": "Cancel a run.", "input_schema": _id},
        {"name": "agent.simulate", "description": "Dry-run an action: validate the request and accept it without grading (returns a succeeded run with no outputs). args: {action, agent_specific_payload}.",
         "input_schema": {"type": "object", "required": ["action", "agent_specific_payload"],
                          "properties": {"action": {"type": "string", "enum": I.ACTION_NAMES},
                                         "agent_specific_payload": {"type": "object"}}}},
        {"name": "grade.grade_outlets", "description": "Shortcut for action grade_outlets.",
         "input_schema": _GRADE_INPUT},
        {"name": "grade.validate_hypothesis", "description": "Shortcut for action validate_opportunity_hypothesis.",
         "input_schema": _VALIDATE_INPUT},
        {"name": "grade.get_run_outputs", "description": "Return just the contract outputs of a run.", "input_schema": _id},
    ]
