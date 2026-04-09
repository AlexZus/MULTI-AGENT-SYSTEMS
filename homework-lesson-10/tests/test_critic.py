"""Critic agent component tests.

Tests:
1. test_critique_approve — GEval: high-quality findings get APPROVE with specific strengths
2. test_critique_revise  — GEval: poor findings get REVISE with actionable revision_requests
"""

import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.critic import critique


# ---------------------------------------------------------------------------
# Shared metric factory
# ---------------------------------------------------------------------------

def _critique_quality_metric(judge, threshold: float = 0.7):
    return GEval(
        name="Critique Quality",
        evaluation_steps=[
            "Check that the verdict is exactly 'APPROVE' or 'REVISE'",
            "If verdict is APPROVE: strengths must be a non-empty list with specific points; "
            "gaps and revision_requests should be empty or list only minor items",
            "If verdict is REVISE: revision_requests must be non-empty; "
            "each revision_request must be actionable (researcher can act on it); "
            "vague requests like 'improve quality' without specifics should lower the score",
            "gaps must list concrete missing topics or issues, not vague statements",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_critique_approve(judge, sample_findings):
    """GEval: high-quality findings should get APPROVE with substantive strengths."""
    result = critique.invoke({"findings": sample_findings})

    # Parse and do a quick structural check before the GEval
    try:
        parsed = json.loads(result)
        assert "verdict" in parsed, f"Missing 'verdict' in critique output: {result}"
    except json.JSONDecodeError:
        pytest.fail(f"Critic output is not valid JSON:\n{result}")

    metric = _critique_quality_metric(judge, threshold=0.7)
    tc = LLMTestCase(
        input=sample_findings,
        actual_output=result,
    )
    assert_test(tc, [metric])


def test_critique_revise(judge, sample_findings_poor):
    """GEval: poor findings should get REVISE with specific actionable requests."""
    result = critique.invoke({"findings": sample_findings_poor})

    try:
        parsed = json.loads(result)
        assert "verdict" in parsed, f"Missing 'verdict' in critique output: {result}"
    except json.JSONDecodeError:
        pytest.fail(f"Critic output is not valid JSON:\n{result}")

    # For poor input the verdict should be REVISE
    parsed = json.loads(result)
    assert parsed.get("verdict") == "REVISE", (
        f"Expected verdict=REVISE for poor findings, got: {parsed.get('verdict')}\n"
        f"Full critique:\n{result}"
    )

    metric = _critique_quality_metric(judge, threshold=0.7)
    tc = LLMTestCase(
        input=sample_findings_poor,
        actual_output=result,
    )
    assert_test(tc, [metric])
