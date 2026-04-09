"""Research agent component tests.

Tests:
1. test_research_grounded    — GEval: all claims in output can be traced to cited sources
2. test_research_completeness — GEval: output covers all queries from the plan
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.research import research


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_research_grounded(judge, sample_plan_json):
    """GEval: research findings contain source citations for factual claims."""
    findings = research.invoke({"request": sample_plan_json})

    groundedness = GEval(
        name="Groundedness",
        evaluation_steps=[
            "Extract every factual claim from actual_output",
            "For each factual claim, check whether it is accompanied by a citation "
            "(e.g. a filename, URL, page reference, or 'Source:' marker)",
            "Claims presented without any citation count as ungrounded",
            "Score = number of cited claims / total claims; "
            "if there are no claims at all, score 0",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.7,
    )

    tc = LLMTestCase(
        input=sample_plan_json,
        actual_output=findings,
    )
    assert_test(tc, [groundedness])


def test_research_completeness(judge, sample_plan_json):
    """GEval: research findings address all queries specified in the plan."""
    import json
    plan = json.loads(sample_plan_json)
    queries_summary = "\n".join(f"- {q}" for q in plan["search_queries"])

    findings = research.invoke({"request": sample_plan_json})

    completeness = GEval(
        name="Research Completeness",
        evaluation_steps=[
            f"The research plan required findings on these topics:\n{queries_summary}",
            "For each topic, check whether actual_output contains a relevant section or paragraph",
            "Score = number of topics addressed / total topics in the plan",
            "A topic is 'addressed' if there is at least one substantive sentence about it",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.6,
    )

    tc = LLMTestCase(
        input=f"Research plan queries:\n{queries_summary}",
        actual_output=findings,
    )
    assert_test(tc, [completeness])
