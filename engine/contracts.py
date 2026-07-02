"""CPG-OS Case-C output contracts — the shared, storage-agnostic types every
depth agent emits so the supervisor can correlate outputs across agents.

Mirrored VERBATIM from the reference agent's `engine/contracts/types.py`
(agent id `outlet-classifier`, running on :8080) so the Outlet Profiler is
wire-compatible: same `contract_version="1"`, same `_Envelope`, same four
types (Observation / Diagnosis / Plug / Opportunity), same `EntityRefs` /
`Evidence` sub-shapes, same `components_schemas()` / `output_envelope_schema()`
manifest helpers.

Two additive, backward-compatible fields the Profiler carries (they appear in
this agent's own advertised component schemas, so a supervisor reading our
manifest sees them):
  - `reasoning_mode` on the envelope — whether an output is a deterministic
    rule over the data or a reasoning/interpretation call. Every Profiler
    output is tagged.
  - `verdict` on Diagnosis — confirm | refute | inconclusive, for the
    hypothesis-validation action.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ContractType = Literal["observation", "diagnosis", "plug", "opportunity"]
ReasoningMode = Literal["deterministic", "reasoning"]


class Evidence(BaseModel):
    """One piece of evidence supporting a contract object. Free-form by value so
    each engine carries its domain payload; the supervisor propagates it for
    human review and stores it for traceability."""

    model_config = ConfigDict(extra="allow")

    kind: str = Field(
        ...,
        description="What this evidence is. Examples: peer_frontier, "
        "decorrelation_guard, realisation_levers, distribution_compare, "
        "statistical_test.",
    )
    value: Any = Field(..., description="The evidence payload.")
    weight: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Optional 0..1 weight of this evidence in the overall confidence.",
    )


class EntityRefs(BaseModel):
    """References to business entities this output is about. The supervisor uses
    these to correlate outputs across agents. `tenant_id` is required for
    isolation; everything else is optional."""

    model_config = ConfigDict(extra="allow")

    tenant_id: str
    outlet_id: str | None = None
    site_id: str | None = None
    sku_id: str | None = None
    distributor_id: str | None = None
    rep_id: str | None = None
    region: str | None = None


class _Envelope(BaseModel):
    """Common fields on every contract object — the supervisor reads only this
    envelope plus the discriminator `type` to correlate."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1"] = "1"
    type: ContractType
    agent_id: str = Field(..., description="Which agent produced this.")
    agent_version: str
    run_id: str = Field(..., description="The run this output came from.")
    signal_id: str | None = Field(
        default=None,
        description="If this run was triggered by a supervisor signal, the "
        "signal_id is propagated here for traceability.",
    )
    entity_refs: EntityRefs
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_mode: ReasoningMode | None = Field(
        default=None,
        description="Basis of this output: 'deterministic' (a rule/formula over "
        "the warehouse data — the tier, the guard, peer-frontier realisation) or "
        "'reasoning' (an interpretation/judgment, e.g. parsing a plain-English "
        "mission or hypothesis into lever weights).",
    )
    evidence: list[Evidence] = Field(default_factory=list)
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Observation(_Envelope):
    """A perceived fact about an entity. The Profiler emits one per graded
    outlet: 'outlet Y realises 0.62 of its peer frontier → tier T2', carrying
    the levers and peer as value/evidence."""

    type: Literal["observation"] = "observation"
    kind: str = Field(
        ...,
        description="Domain label: outlet_opportunity_grade | grade_vector | …",
    )
    value: dict[str, Any] = Field(
        ...,
        description="The perception payload. Shape varies by `kind`.",
    )


class Diagnosis(_Envelope):
    """A causal claim / verdict built from Observations. The Profiler emits one
    per hypothesis validation, tagged with a verdict and reasoning_mode."""

    type: Literal["diagnosis"] = "diagnosis"
    summary: str = Field(..., description="One-line plain-English claim/verdict.")
    verdict: Literal["confirm", "refute", "inconclusive"] | None = Field(
        default=None,
        description="For hypothesis validation: does the data confirm/refute the "
        "supervisor's assertion, or is it inconclusive.",
    )
    root_causes: list[str] = Field(..., description="Ordered causes, most likely first.")
    contributing_output_ids: list[str] = Field(
        default_factory=list,
        description="run_id/outlet ids of the Observations this was built from.",
    )


class Plug(_Envelope):
    """A proposed action that, if executed, addresses a Diagnosis or captures an
    Opportunity. Named + parameterised but NOT executed (HITL + MCP)."""

    type: Literal["plug"] = "plug"
    action: str = Field(..., description="Short verb-phrase action name.")
    description: str = Field(..., description="Plain-English what + why.")
    parameters: dict[str, Any] = Field(..., description="Payload for the executor / MCP tool.")
    reversible: bool
    blast_radius: dict[str, Any] = Field(
        default_factory=dict,
        description="Scope of impact if executed — outlet_count, inr_value, etc.",
    )


class Opportunity(_Envelope):
    """A sized commercial chance, ₹-denominated. The Profiler sizes an outlet's
    unrealised headroom against its peer frontier."""

    type: Literal["opportunity"] = "opportunity"
    summary: str
    inr_value: float = Field(..., description="₹ value of the opportunity.")
    horizon_days: int | None = Field(
        default=None,
        description="Time horizon over which the inr_value is expected, if known.",
    )
    confidence_level: Literal["low", "medium", "high"] = "medium"


# ─── manifest helpers ────────────────────────────────────────────────────

_RUN_STATUS = ["queued", "running", "succeeded", "failed", "cancelled"]


def output_envelope_schema(produces: list[str]) -> dict[str, Any]:
    """The per-action `output_schema` block for the manifest: an envelope of
    run_id/status/produces_contract_types + an outputs[] stream of $refs into
    #/components/schemas."""
    refs = [{"$ref": f"#/components/schemas/{ct.capitalize()}"} for ct in produces]
    items_schema = refs[0] if len(refs) == 1 else {"oneOf": refs}
    return {
        "type": "object",
        "required": ["run_id", "status", "produces_contract_types"],
        "properties": {
            "run_id": {"type": "string"},
            "status": {"type": "string", "enum": _RUN_STATUS},
            "produces_contract_types": {
                "type": "array",
                "items": {"type": "string", "enum": list(produces)},
            },
            "outputs": {"type": "array", "items": items_schema},
        },
    }


def components_schemas() -> dict[str, Any]:
    """The four full JSON schemas keyed by capitalized type name — spliced into
    the manifest's `components.schemas` so callers can validate our outputs."""
    return {
        "Observation": Observation.model_json_schema(),
        "Diagnosis": Diagnosis.model_json_schema(),
        "Plug": Plug.model_json_schema(),
        "Opportunity": Opportunity.model_json_schema(),
    }
