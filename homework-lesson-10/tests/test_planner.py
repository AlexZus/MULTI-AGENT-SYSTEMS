"""Planner agent component tests.

Tests:
1. test_plan_quality        — GEval: plan is specific, sources set, format described
2. test_plan_required_fields — structural: all fields present and non-empty
"""

import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import plan


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _call_plan(request: str) -> tuple[str, dict | None]:
    """Call the planner and return (raw_output, parsed_dict_or_None)."""
    raw = plan.invoke({"request": request})
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    return raw, parsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plan_quality(judge):
    """GEval: plan search queries are specific and sources/format are set."""
    request = "What architecture patterns exist for building AI-powered Telegram bots?"
    raw, _ = _call_plan(request)

    plan_quality = GEval(
        name="Plan Quality",
        evaluation_steps=[
            "Check that search_queries contains at least 3 specific queries (not single generic words like 'telegram' or 'bot')",
            "Check that sources_to_check includes at least one of 'knowledge_base' or 'web'",
            "Check that output_format is non-empty and describes a structured deliverable",
            "Check that goal clearly restates the user's research objective",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.7,
    )

    tc = LLMTestCase(input=request, actual_output=raw)
    assert_test(tc, [plan_quality])


def test_plan_required_fields(judge):
    """Structural: plan JSON has all required fields with non-empty values."""
    request = "How does webhook handling work in Telegram bots compared to long polling?"
    raw, parsed = _call_plan(request)

    assert parsed is not None, f"Planner output is not valid JSON.\nRaw output:\n{raw}"

    required = ["goal", "search_queries", "sources_to_check", "output_format"]
    for field in required:
        assert field in parsed, f"Missing field '{field}' in plan: {parsed}"
        assert parsed[field], f"Field '{field}' is empty in plan: {parsed}"

    assert isinstance(parsed["search_queries"], list), "search_queries must be a list"
    assert len(parsed["search_queries"]) >= 2, (
        f"Expected at least 2 search queries, got {len(parsed['search_queries'])}: "
        f"{parsed['search_queries']}"
    )

    assert isinstance(parsed["sources_to_check"], list), "sources_to_check must be a list"
    valid_sources = {"knowledge_base", "web"}
    for src in parsed["sources_to_check"]:
        assert src in valid_sources, (
            f"Unknown source '{src}'. Expected one of {valid_sources}"
        )
