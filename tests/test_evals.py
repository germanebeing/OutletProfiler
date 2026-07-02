"""CI gate for the behaviour-eval suite (tests/evals/). Fails the build if the
agent's mission interpretation, reasoning_mode tagging, or hypothesis verdicts
regress below their pass gates."""
from tests.evals.cases import GATES
from tests.evals.harness import run_evals


def test_behaviour_evals_pass_gates(result):
    rep = run_evals(result)
    scores = rep["scores"]
    for k, gate in GATES.items():
        assert scores[k] >= gate, (
            f"eval '{k}' scored {scores[k]:.0%} < gate {gate:.0%} — counts {rep['counts'][k]}")
    assert rep["gate_pass"]
