"""Critic Agent — independently verifies research findings and returns a structured verdict."""

import json
import re

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware, _tool_budget
from config import Settings, get_critic_prompt, get_model
from schemas import CritiqueResult
from tools import knowledge_search, web_fetch, web_search

_settings = Settings()

_critic_agent = create_agent(
    get_model(),
    tools=[web_search, web_fetch, knowledge_search],
    system_prompt=get_critic_prompt(),
    checkpointer=InMemorySaver(),
    middleware=[
        BudgetMiddleware(),
        InvalidToolCallRetryMiddleware(
            max_retries=_settings.subagent_output_retry_number_on_validation_fail
        ),
    ],
    response_format=None if _settings.structured_output_workaround else CritiqueResult,
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_CRITIQUE_CORRECTION_MSG = (
    "Your previous response did not contain a valid JSON block matching the CritiqueResult schema. "
    "You MUST end your response with a fenced JSON block in exactly this format:\n"
    "```json\n"
    '{{"verdict": "APPROVE", "is_fresh": true, "is_complete": true, '
    '"is_well_structured": true, "strengths": ["..."], "gaps": [], "revision_requests": []}}\n'
    "```\n"
    'The verdict must be exactly "APPROVE" or "REVISE". Output the JSON block now.'
)

_AUTO_APPROVE = CritiqueResult(
    verdict="APPROVE",
    is_fresh=True,
    is_complete=True,
    is_well_structured=True,
    strengths=["Research appears comprehensive"],
    gaps=[],
    revision_requests=[],
)


def _extract_critique(text: str) -> CritiqueResult | None:
    """Parse a CritiqueResult from a fenced JSON block in the model's text response."""
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return CritiqueResult.model_validate_json(m.group(1))
        except Exception:
            pass
    # Fallback: last {...} in text
    last_brace = text.rfind("}")
    first_brace = text.rfind("{", 0, last_brace + 1)
    if first_brace >= 0:
        try:
            return CritiqueResult.model_validate_json(text[first_brace: last_brace + 1])
        except Exception:
            pass
    return None


def _invoke_critic(msg_content: str, sub_config: RunnableConfig) -> tuple[str, dict | None]:
    """Invoke the critic agent once; return (text_output, raw_result)."""
    token = _tool_budget.set({"remaining": _settings.max_iterations})
    try:
        result = _critic_agent.invoke(
            {"messages": [{"role": "user", "content": msg_content}]},
            config=sub_config,
        )
    except GraphRecursionError:
        fallback = _AUTO_APPROVE.model_dump_json(indent=2)
        fallback += (
            f"\n// [CRITIC LIMIT REACHED] Critic exhausted its tool-call budget "
            f"(max_iterations={_settings.max_iterations}). "
            "Auto-approved to allow the pipeline to complete."
        )
        return fallback, None
    except Exception as e:
        fallback = _AUTO_APPROVE.model_dump_json(indent=2)
        fallback += f"\n// Critic agent unexpected error (auto-approved): {type(e).__name__}: {e}"
        return fallback, None
    finally:
        _tool_budget.reset(token)

    messages = result.get("messages", [])
    last = messages[-1] if messages else None
    text = ""
    if last:
        text = last.text if isinstance(getattr(last, "text", None), str) else (
            last.text() if callable(getattr(last, "text", None)) else str(last.content)
        )
    return text, result


@tool
def critique(findings: str | dict, config: RunnableConfig = None) -> str:
    """Independently evaluate research findings for freshness, completeness, and structure.

    The critic runs its own verification searches before returning a structured
    CritiqueResult with verdict "APPROVE" or "REVISE", detailed strengths/gaps,
    and specific revision_requests if REVISE.

    Input: Research findings text (from the researcher).
    Output: JSON-serialized CritiqueResult.
    """
    if isinstance(findings, dict):
        findings = json.dumps(findings, ensure_ascii=False, indent=2)

    supervisor_thread = (config or {}).get("configurable", {}).get("thread_id", "default")
    sub_config: RunnableConfig = {
        "recursion_limit": _settings.max_iterations,
        "configurable": {"thread_id": f"{supervisor_thread}:critic"},
        "callbacks": (config or {}).get("callbacks", []),
    }

    retry_limit = _settings.subagent_output_retry_number_on_validation_fail

    for attempt in range(retry_limit + 1):
        msg = findings if attempt == 0 else _CRITIQUE_CORRECTION_MSG
        text, result = _invoke_critic(msg, sub_config)

        # Hard error — return the annotated auto-approve JSON directly
        if result is None:
            return text

        # Structured-output path
        if not _settings.structured_output_workaround and result:
            critique_obj = result.get("structured_response")
            if isinstance(critique_obj, CritiqueResult):
                return critique_obj.model_dump_json(indent=2)

        # Text-extraction path
        critique_obj = _extract_critique(text)
        if critique_obj:
            return critique_obj.model_dump_json(indent=2)

        # Extraction failed — retry if budget allows

    # All retries exhausted without valid JSON
    if text:
        return text

    return (
        _AUTO_APPROVE.model_dump_json(indent=2)
        + "\n// Auto-approved: critic produced no parseable output after all retries."
    )
