"""End-to-end evaluation on the golden dataset.

Tests:
1. test_e2e_happy_path      — full Supervisor pipeline on happy_path examples;
                               evaluated with AnswerRelevancy + Correctness + CitationPresence
2. test_failure_graceful    — planner-only on failure_case examples;
                               evaluated with GracefulHandling GEval
"""

import json
import os
import glob
import sys

import pytest
from deepeval import assert_test, evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner import plan
from config import Settings

_settings = Settings()


# ---------------------------------------------------------------------------
# Custom business metric: Citation Presence
# ---------------------------------------------------------------------------

def citation_presence_metric(judge):
    """Domain rule: every approved research report must contain inline citations."""
    return GEval(
        name="Citation Presence",
        evaluation_steps=[
            "Check if the actual_output contains at least one URL (starting with http:// or https://) "
            "or a citation marker such as 'Source:', '[Source:', a filename like 'file.txt', "
            "or a page reference like 'page N'",
            "Score 1.0 if citations appear throughout the text (every major section has at least one)",
            "Score 0.5 if at least one citation is present but coverage is sparse",
            "Score 0.0 if no citations are present at all",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.5,
    )


def graceful_handling_metric(judge):
    """Planner should handle out-of-domain / nonsense inputs without hallucinating."""
    return GEval(
        name="Graceful Handling",
        evaluation_steps=[
            "Check if the response is a valid JSON research plan or an error/fallback message",
            "Score 1.0 if the plan acknowledges the input is vague, out-of-domain, or problematic "
            "(e.g., limited scope, notes impossibility of the task)",
            "Score 0.7 if the plan attempts a reasonable interpretation without hallucinating "
            "confident answers about nonsense or future events",
            "Score 0.3 if the plan produces something plausible-sounding but unrelated to the input",
            "Score 0.0 if the plan hallucinates specific facts about the nonsense input",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.4,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_latest_output(output_dir: str, before_paths: set[str]) -> str | None:
    """Return text content of the newest .md file written to output_dir since we started."""
    pattern = os.path.join(output_dir, "*.md")
    current = set(glob.glob(pattern))
    new_files = current - before_paths
    if not new_files:
        return None
    latest = max(new_files, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        return f.read()


def _run_full_pipeline(request: str) -> str:
    """Run the full Supervisor pipeline and return the saved report text.

    Uses a supervisor without HumanInTheLoopMiddleware so save_report executes
    automatically during testing (no interactive approval required).
    """
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from agents.planner import plan
    from agents.research import research
    from agents.critic import critique
    from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware
    from config import SUPERVISOR_SYSTEM_PROMPT, get_model
    from tools import save_report

    supervisor = create_agent(
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

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        _settings.output_dir,
    )
    os.makedirs(output_dir, exist_ok=True)
    existing = set(glob.glob(os.path.join(output_dir, "*.md")))

    from agents.middleware import _tool_budget
    token = _tool_budget.set({"remaining": _settings.max_iterations})
    try:
        result = supervisor.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={
                "recursion_limit": _settings.max_iterations * 2 + 2,
                "configurable": {"thread_id": f"test:e2e:{hash(request)}"},
            },
        )
    finally:
        _tool_budget.reset(token)

    # Try to get saved report first
    report = _get_latest_output(output_dir, existing)
    if report:
        return report

    # Fall back: find the most substantial AI message in the conversation.
    # The research findings / final report are typically the longest AI text,
    # whereas the final supervisor summary may be short or a JSON snippet.
    from langchain_core.messages import AIMessage, ToolMessage

    messages = result.get("messages", [])

    # Collect all ToolMessage contents (which include research findings, critique, etc.)
    # and AI messages; return the longest substantive one.
    candidates = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = str(m.content)
            if len(content) > 200:
                candidates.append(content)
        elif isinstance(m, AIMessage):
            text = m.text if isinstance(getattr(m, "text", None), str) else (
                m.text() if callable(getattr(m, "text", None)) else str(m.content)
            )
            if len(text) > 200:
                candidates.append(text)

    if candidates:
        return max(candidates, key=len)
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "example",
    [ex for ex in json.load(open(
        os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    )) if ex["category"] == "happy_path"],
    ids=lambda ex: ex["input"][:50],
)
def test_e2e_happy_path(judge, example):
    """Full pipeline: Supervisor → Planner → Researcher → Critic → SaveReport.

    Metrics: AnswerRelevancy + Correctness + CitationPresence.
    Baseline thresholds — expect some failures until system is tuned.
    """
    report = _run_full_pipeline(example["input"])

    assert report, f"Pipeline produced no output for: {example['input']}"

    metrics = [
        GEval(
            name="Answer Relevancy",
            evaluation_steps=[
                "Check whether actual_output directly addresses the user's input question",
                "Score 1.0 if the output is fully on-topic and answers the question",
                "Score 0.5 if the output is partially relevant but drifts or adds irrelevant content",
                "Score 0.0 if the output is unrelated to the question",
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.7,
        ),
        GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether the facts in actual_output contradict expected_output",
                "Penalize omission of critical details mentioned in expected_output",
                "Different wording of the same concept is acceptable",
                "Score higher if actual_output is more detailed than expected_output "
                "as long as it's accurate",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            threshold=0.6,
        ),
        citation_presence_metric(judge),
    ]

    tc = LLMTestCase(
        input=example["input"],
        actual_output=report,
        expected_output=example["expected_output"],
    )
    assert_test(tc, metrics)


@pytest.mark.parametrize(
    "example",
    [ex for ex in json.load(open(
        os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    )) if ex["category"] == "failure_case"],
    ids=lambda ex: ex["input"][:40],
)
def test_failure_graceful_handling(judge, example):
    """Planner-only test: out-of-domain inputs should be handled gracefully.

    Threshold 0.4 — this is a baseline; we just want no confident hallucinations.
    """
    raw = plan.invoke({"request": example["input"]})

    metric = graceful_handling_metric(judge)
    tc = LLMTestCase(
        input=example["input"],
        actual_output=raw,
    )
    assert_test(tc, [metric])
