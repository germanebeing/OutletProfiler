"""Agent layer — CPG-OS Case-C contract surfaces over the grader engine."""
import uuid

import pytest

from agent import api, handlers, identity, store, worker
from agent.manifest import build_agent_card, build_manifest, build_tools
from engine.contracts import Diagnosis, EntityRefs, Observation, Opportunity

_STD_MCP_TOOLS = {"agent.describe", "agent.run", "agent.get_run", "agent.list_runs",
                  "agent.cancel_run", "agent.simulate"}


class _FakeStore:
    def __init__(self, result):
        self.graded = result.graded
        self.validation = result.validation


# ─── contract types ────────────────────────────────────────────────────────

def test_contract_version_and_reasoning_mode():
    o = Observation(agent_id="outlet-profiler", agent_version="0.1.0", run_id="r",
                    entity_refs=EntityRefs(tenant_id="t", outlet_id="1"), confidence=0.9,
                    reasoning_mode="deterministic", kind="outlet_opportunity_grade", value={"tier": "T2"})
    d = o.model_dump()
    assert d["contract_version"] == "1"
    assert d["type"] == "observation"
    assert d["reasoning_mode"] == "deterministic"


def test_diagnosis_carries_verdict_and_reasoning_mode():
    d = Diagnosis(agent_id="outlet-profiler", agent_version="0.1.0", run_id="r",
                  entity_refs=EntityRefs(tenant_id="t"), confidence=0.8,
                  reasoning_mode="reasoning", verdict="refute", summary="x", root_causes=["y"])
    dd = d.model_dump()
    assert dd["verdict"] == "refute"
    assert dd["reasoning_mode"] == "reasoning"


# ─── manifest / card ────────────────────────────────────────────────────────

def test_manifest_matches_case_c_shape():
    m = build_manifest()
    assert m["agent_id"] == "outlet-profiler"
    assert m["produces_contract_types"] == ["observation", "diagnosis", "opportunity"]
    assert m["accepts_contract_types"] == ["diagnosis", "opportunity"]
    assert m["reasoning_modes"] == ["deterministic", "reasoning"]
    # load_tested_to_rps stays null until a real load test runs
    assert m["capacity"]["load_tested_to_rps"] is None
    assert set(m["surfaces"]) == {"api", "mcp", "ui", "cli", "a2a"}
    assert set(m["components"]["schemas"]) == {"Observation", "Diagnosis", "Plug", "Opportunity"}
    for a in m["actions"]:
        # per-action reasoning is dual-path (LLM lens vs deterministic rules), so
        # the action advertises the set of modes it can produce, not a fixed one.
        assert a["reasoning_modes"] and all(
            rm in ("deterministic", "reasoning") for rm in a["reasoning_modes"])
        assert a["produces"] and "output_schema" in a
    # the new order-frequency play is discoverable on grade_outlets
    ga = next(a for a in m["actions"] if a["name"] == "grade_outlets")
    assert "frequency" in {p["name"] for p in ga["plays"]}
    assert {a["name"] for a in m["actions"]} == set(identity.ACTION_NAMES)


def test_agent_card_shape():
    c = build_agent_card()
    assert c["agent_id"] == "outlet-profiler"
    assert c["securitySchemes"]["bearer"]["scheme"] == "bearer"
    assert c["produces_contract_types"] == identity.PRODUCES
    assert {s["id"] for s in c["skills"]} == set(identity.ACTION_NAMES)


def test_handlers_cover_manifest_actions():
    assert set(handlers.HANDLERS) == set(identity.ACTION_NAMES)


def test_six_standard_mcp_tools_present():
    # AGENTS.md hard requirement: the six exact tool names, incl. agent.simulate
    assert _STD_MCP_TOOLS <= {t["name"] for t in build_tools()}


def test_signal_id_read_from_triggered_by():
    # canonical envelope nests signal under triggered_by
    assert api._sig({"triggered_by": {"signal_id": "sig_9"}}) == "sig_9"
    assert api._sig({"triggered_by": {"signal": "sig_alt"}}) == "sig_alt"
    assert api._sig({"signal_id": "sig_top"}) == "sig_top"
    assert api._sig({}) is None


# ─── idempotency store ──────────────────────────────────────────────────────

def test_store_is_tenant_scoped():
    run = store.create_run(tenant_id="tenantA", action="grade_outlets", name="t",
                           agent_id="outlet-profiler", agent_version="0.1.0", input_payload={},
                           idempotency_key=None, signal_id=None, triggered_by=None,
                           context=None, dry_run=True)
    rid = run["id"]
    assert store.get_run(rid, "tenantA") is not None       # owner reads it
    assert store.get_run(rid, "intruder-co") is None        # cross-tenant blocked (IDOR)
    assert store.get_run(rid) is not None                    # unscoped internal lookup still works


def test_rate_limit_raises_after_burst():
    tenant = "rl-" + uuid.uuid4().hex[:8]
    for _ in range(api._RL_RPS):
        api._rate_check(tenant)                              # first N ok
    with pytest.raises(Exception) as e:
        api._rate_check(tenant)                              # N+1 -> 429
    assert getattr(e.value, "status_code", None) == 429


def test_requeue_reclaims_stale_running():
    run = store.create_run(tenant_id="default", action="grade_outlets", name="t",
                           agent_id="outlet-profiler", agent_version="0.1.0", input_payload={},
                           idempotency_key=None, signal_id=None, triggered_by=None,
                           context=None, dry_run=True)
    store.update(run, status="running")                     # simulate a crash mid-run
    store.requeue_running()
    assert store.get_run(run["id"])["status"] == "queued"   # reclaimed


def test_idempotency_first_writer_wins():
    key = "test-" + uuid.uuid4().hex
    assert store.get_run_id_for_key("t", "grade_outlets", key) is None
    store.bind_key("t", "grade_outlets", key, "run_A")
    store.bind_key("t", "grade_outlets", key, "run_B")  # ignored
    assert store.get_run_id_for_key("t", "grade_outlets", key) == "run_A"


# ─── handlers emit typed contracts ──────────────────────────────────────────

def test_grade_outlets_emits_observations_and_opportunities(result):
    fake = _FakeStore(result)
    company = result.graded.filter(result.graded["has_data"])["company_name"][0]
    out = handlers.grade_outlets(lambda: fake, "run_test", "default", None,
                                 {"company": company, "mission": "launch a premium SKU", "limit": 5})
    assert out["produces"] == ["observation", "opportunity"]
    types = {o["type"] for o in out["outputs"]}
    assert "observation" in types and "opportunity" in types
    for o in out["outputs"]:
        assert o["agent_id"] == "outlet-profiler"
        assert o["contract_version"] == "1"
        assert o["reasoning_mode"] in ("deterministic", "reasoning")
    opp = next(o for o in out["outputs"] if o["type"] == "opportunity")
    assert "inr_value" in opp and opp["horizon_days"]


def test_grade_outlets_reasoning_path_propagates(result, monkeypatch):
    """When the LLM lens fires, the run + Opportunity carry reasoning_mode
    'reasoning' and the ₹-sizing uses the ranking-driven headroom basis — the
    exact contract the supervisor consumes. Mocks the lens so it runs keyless."""
    from engine import llm_parse
    monkeypatch.setattr(llm_parse, "available", lambda: True)
    monkeypatch.setattr(llm_parse, "plan", lambda *a, **k: {
        "weights": {"range": 0.15, "cadence": 0.6, "recency": 0.2, "value": 0.05},
        "target_tiers": ["T3", "T4"], "ranking": "headroom", "regions": [], "formats": [],
        "label": "Order-frequency lift", "rationale": "cadence is the binding lever"})
    fake = _FakeStore(result)
    company = result.graded.filter(result.graded["has_data"])["company_name"][0]
    out = handlers.grade_outlets(lambda: fake, "run_r", "default", None,
                                 {"company": company, "mission": "lift the ones ordering rarely", "limit": 5})
    assert out["reasoning_modes"] == ["reasoning", "deterministic"]
    c = out["counters"]
    assert c["reasoning_mode"] == "reasoning" and c["ranking"] == "headroom"
    assert c["tier_candidates"] is not None
    assert "headroom" in out["summary"]  # gap-play ₹ basis, not launch-uplift
    opps = [o for o in out["outputs"] if o["type"] == "opportunity"]
    assert opps and all(o["reasoning_mode"] == "reasoning" for o in opps)


def test_grade_outlets_deterministic_when_lens_off(result, monkeypatch):
    """No LLM key -> keyword fallback -> the run is tagged deterministic, not
    falsely 'reasoning' just because the mission was free text (the audit bug)."""
    from engine import llm_parse
    monkeypatch.setattr(llm_parse, "available", lambda: False)
    fake = _FakeStore(result)
    company = result.graded.filter(result.graded["has_data"])["company_name"][0]
    out = handlers.grade_outlets(lambda: fake, "run_d", "default", None,
                                 {"company": company, "mission": "improve order frequency", "limit": 5})
    assert out["reasoning_modes"] == ["deterministic"]
    assert out["counters"]["reasoning_mode"] == "deterministic"
    assert all(o["reasoning_mode"] == "deterministic"
               for o in out["outputs"] if o["type"] == "opportunity")


def test_validate_hypothesis_returns_verdict_diagnosis(result):
    fake = _FakeStore(result)
    company = result.graded.filter(result.graded["has_data"])["company_name"][0]
    out = handlers.validate_opportunity_hypothesis(
        lambda: fake, "run_v", "default", None,
        {"company": company, "hypothesis": "my outlets have gone dormant"})
    assert "diagnosis" in out["produces"]
    diag = next(o for o in out["outputs"] if o["type"] == "diagnosis")
    assert diag["verdict"] in ("confirm", "refute", "inconclusive")
    assert diag["reasoning_mode"] in ("deterministic", "reasoning")


def test_analyze_outcome_is_stub(result):
    out = handlers.analyze_outcome(lambda: _FakeStore(result), "run_o", "default", None, {})
    diag = out["outputs"][0]
    assert diag["type"] == "diagnosis" and diag["verdict"] == "inconclusive"


def test_region_filter_scopes_the_grade(result):
    fake = _FakeStore(result)
    g = result.graded.filter(result.graded["has_data"])
    company = g["company_name"][0]
    region = g.filter(g["company_name"] == company)["regionname"][0]
    out = handlers.grade_outlets(lambda: fake, "run_r", "default", None,
                                 {"company": company, "mission": "grade", "regions": [region], "limit": 20})
    regs = {o["entity_refs"]["region"] for o in out["outputs"]}
    assert regs <= {region}  # every emitted output is in the requested region
    assert out["counters"]["regions"] == [region]


def _grade_run(company, callback_url=None):
    return store.create_run(
        tenant_id="default", action="grade_outlets", name="t",
        agent_id="outlet-profiler", agent_version="0.1.0",
        input_payload={"company": company, "mission": "launch a premium SKU", "limit": 2},
        idempotency_key=None, signal_id=None, triggered_by=None, context=None,
        dry_run=False, callback_url=callback_url)


def test_callback_fired_on_completion(result, monkeypatch):
    import httpx
    calls = []

    class _Resp:
        status_code = 200

    monkeypatch.setattr(httpx, "post", lambda url, **kw: calls.append((url, kw.get("json"))) or _Resp())
    company = result.graded.filter(result.graded["has_data"])["company_name"][0]
    done = worker.execute_run(_grade_run(company, "https://cb.example/hook"), lambda: _FakeStore(result))
    assert done["status"] == "succeeded"
    assert calls and calls[0][0] == "https://cb.example/hook"
    body = calls[0][1]
    assert body["status"] == "succeeded" and body["run_id"] == done["id"] and "outputs_url" in body


def test_callback_failure_never_fails_the_run(result, monkeypatch):
    import httpx

    def boom(url, **kw):
        raise RuntimeError("unreachable")

    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setattr(worker, "_CB_BACKOFF", (0.0, 0.0))  # keep the retry loop instant
    company = result.graded.filter(result.graded["has_data"])["company_name"][0]
    done = worker.execute_run(_grade_run(company, "https://cb.example/hook"), lambda: _FakeStore(result))
    assert done["status"] == "succeeded"  # callback blew up 3x, run still succeeded


def test_async_worker_runs_a_queued_run_and_builds_outcome(result):
    fake = _FakeStore(result)
    company = result.graded.filter(result.graded["has_data"])["company_name"][0]
    run = store.create_run(tenant_id="default", action="grade_outlets", name="t",
                           agent_id="outlet-profiler", agent_version="0.1.0",
                           input_payload={"company": company, "mission": "launch a premium SKU", "limit": 3},
                           idempotency_key=None, signal_id="sig_test", triggered_by={"signal_id": "sig_test"},
                           context=None, dry_run=False)
    assert run["status"] == "queued"
    done = worker.execute_run(run, lambda: fake)
    assert done["status"] == "succeeded"
    s = store.summary(done)
    # RunResult.outcome envelope is present and populated
    assert s["outcome"]["summary"] and s["outcome"]["reasoning_mode"] in ("deterministic", "reasoning")
    assert s["outcome"]["reversible"] is True and s["outcome"]["changes"] == []
    # signal_id propagated onto emitted contracts
    assert all(o.get("signal_id") == "sig_test" for o in done["outputs"])
