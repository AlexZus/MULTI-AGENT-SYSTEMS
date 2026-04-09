"""Tool correctness tests.

Verifies that each agent calls the right tools given its input.
Uses DeepEval ToolCorrectnessMetric.

Tests:
1. test_planner_uses_search_tools   — planner calls knowledge_search and/or web_search
2. test_researcher_uses_plan_sources — researcher calls tools from both plan sources
3. test_supervisor_calls_save_report — supervisor calls save_report after critic APPROVE
"""

import json
import sys
import os
from unittest.mock import patch

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware, _tool_budget
from config import Settings, get_model, SUPERVISOR_SYSTEM_PROMPT
from tests.conftest import SAMPLE_PLAN_JSON, SAMPLE_FINDINGS, SAMPLE_CRITIQUE_APPROVE

_settings = Settings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_calls(messages: list) -> list[ToolCall]:
    """Convert LangGraph message history to DeepEval ToolCall objects."""
    tool_outputs: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.tool_call_id] = str(msg.content)

    calls = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in (msg.tool_calls or []):
            output = tool_outputs.get(tc.get("id", ""), None)
            calls.append(ToolCall(
                name=tc["name"],
                input_parameters=tc.get("args", {}),
                output=output,
            ))
    return calls


def _run_planner_agent(request: str) -> list[ToolCall]:
    from agents.planner import _planner_agent
    token = _tool_budget.set({"remaining": _settings.max_iterations})
    try:
        result = _planner_agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={
                "recursion_limit": _settings.max_iterations * 2 + 2,
                "configurable": {"thread_id": "test:planner_tools"},
            },
        )
    finally:
        _tool_budget.reset(token)
    return _extract_tool_calls(result.get("messages", []))


def _run_research_agent(plan_json: str) -> list[ToolCall]:
    from agents.research import _research_agent
    token = _tool_budget.set({"remaining": _settings.max_iterations})
    try:
        result = _research_agent.invoke(
            {"messages": [{"role": "user", "content": plan_json}]},
            config={
                "recursion_limit": _settings.max_iterations * 2 + 2,
                "configurable": {"thread_id": "test:research_tools"},
            },
        )
    finally:
        _tool_budget.reset(token)
    return _extract_tool_calls(result.get("messages", []))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_planner_uses_search_tools(judge):
    """Planner must call at least one search tool (web_search or knowledge_search)."""
    request = "What are the main architecture patterns for AI-powered Telegram bots?"
    tools_called = _run_planner_agent(request)

    assert tools_called, "Planner made no tool calls at all"

    metric = ToolCorrectnessMetric(
        threshold=0.5,
        model=judge,
        should_exact_match=False,
        should_consider_ordering=False,
    )

    tc = LLMTestCase(
        input=request,
        actual_output=f"Planner called {len(tools_called)} tools: "
                      f"{[t.name for t in tools_called]}",
        tools_called=tools_called,
        expected_tools=[
            ToolCall(name="knowledge_search"),
            ToolCall(name="web_search"),
        ],
    )
    assert_test(tc, [metric])


def test_researcher_uses_plan_sources(judge):
    """Researcher must use both knowledge_search and web_search when plan requires both."""
    tools_called = _run_research_agent(SAMPLE_PLAN_JSON)

    assert tools_called, "Researcher made no tool calls at all"

    metric = ToolCorrectnessMetric(
        threshold=0.5,
        model=judge,
        should_exact_match=False,
        should_consider_ordering=False,
    )

    tc = LLMTestCase(
        input=SAMPLE_PLAN_JSON,
        actual_output=f"Researcher called {len(tools_called)} tools: "
                      f"{[t.name for t in tools_called]}",
        tools_called=tools_called,
        expected_tools=[
            ToolCall(name="knowledge_search"),
            ToolCall(name="web_search"),
        ],
    )
    assert_test(tc, [metric])


def test_supervisor_calls_save_report(judge):
    """Supervisor must call save_report after receiving an APPROVE critique.

    plan, research, and critique are mocked to return pre-built fixtures so the
    test is fast and deterministic. The supervisor model still runs and decides
    which tool to call next.
    """
    from langchain.agents import create_agent
    from agents.planner import plan
    from agents.research import research
    from agents.critic import critique
    from tools import save_report

    # Build a supervisor without HumanInTheLoop so save_report actually executes
    test_supervisor = create_agent(
        get_model(),
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=[
            BudgetMiddleware(),
            InvalidToolCallRetryMiddleware(
                max_retries=_settings.subagent_output_retry_number_on_validation_fail
            ),
        ],
        checkpointer=InMemorySaver(),
    )

    request = "Summarize AI-powered Telegram bot architectures"

    # Mock sub-agents to return pre-built responses instantly.
    # StructuredTool stores the wrapped function in the `func` field; patching
    # it intercepts _run without touching Pydantic's frozen attribute machinery.
    with (
        patch.object(plan, "func", return_value=SAMPLE_PLAN_JSON),
        patch.object(research, "func", return_value=SAMPLE_FINDINGS),
        patch.object(critique, "func", return_value=SAMPLE_CRITIQUE_APPROVE),
    ):
        token = _tool_budget.set({"remaining": _settings.max_iterations})
        try:
            result = test_supervisor.invoke(
                {"messages": [{"role": "user", "content": request}]},
                config={
                    "recursion_limit": _settings.max_iterations * 2 + 2,
                    "configurable": {"thread_id": "test:supervisor_tools"},
                },
            )
        finally:
            _tool_budget.reset(token)

    messages = result.get("messages", [])
    tools_called = _extract_tool_calls(messages)
    tool_names = [t.name for t in tools_called]

    assert "save_report" in tool_names, (
        f"Expected save_report to be called, but tools called were: {tool_names}"
    )

    metric = ToolCorrectnessMetric(
        threshold=0.8,
        model=judge,
        should_exact_match=False,
        should_consider_ordering=False,
    )

    tc = LLMTestCase(
        input=request,
        actual_output=f"Supervisor pipeline called tools: {tool_names}",
        tools_called=tools_called,
        expected_tools=[
            ToolCall(name="plan"),
            ToolCall(name="research"),
            ToolCall(name="critique"),
            ToolCall(name="save_report"),
        ],
    )
    assert_test(tc, [metric])
