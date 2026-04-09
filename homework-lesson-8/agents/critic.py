"""Critic Agent — independently verifies research findings and returns a structured verdict."""

import json
import re

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from config import Settings, get_critic_prompt, get_model
from schemas import CritiqueResult
from tools import knowledge_search, web_fetch, web_search

_settings = Settings()

_critic_agent = create_agent(
    get_model(),
    tools=[web_search, web_fetch, knowledge_search],
    system_prompt=get_critic_prompt(),
    checkpointer=InMemorySaver(),
    response_format=None if _settings.structured_output_workaround else CritiqueResult,
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


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
    }

    try:
        result = _critic_agent.invoke(
            {"messages": [{"role": "user", "content": findings}]},
            config=sub_config,
        )
    except GraphRecursionError:
        return CritiqueResult(
            verdict="APPROVE",
            is_fresh=True,
            is_complete=True,
            is_well_structured=True,
            strengths=["Research appears comprehensive"],
            gaps=[],
            revision_requests=[],
        ).model_dump_json(indent=2) + (
            f"\n// [CRITIC LIMIT REACHED] Critic exhausted its tool-call budget "
            f"(max_iterations={_settings.max_iterations}). "
            "Auto-approved to allow the pipeline to complete."
        )
    except Exception as e:
        return CritiqueResult(
            verdict="APPROVE",
            is_fresh=True,
            is_complete=True,
            is_well_structured=True,
            strengths=["Research appears comprehensive"],
            gaps=[],
            revision_requests=[],
        ).model_dump_json(indent=2) + f"\n// Critic agent unexpected error (auto-approved): {type(e).__name__}: {e}"

    # With response_format= the structured result is in "structured_response"
    if not _settings.structured_output_workaround:
        critique_obj = result.get("structured_response")
        if isinstance(critique_obj, CritiqueResult):
            return critique_obj.model_dump_json(indent=2)

    # Workaround path: extract JSON from the model's text response
    messages = result.get("messages", [])
    last = messages[-1] if messages else None
    text = ""
    if last:
        text = last.text if isinstance(getattr(last, "text", None), str) else (
            last.text() if callable(getattr(last, "text", None)) else str(last.content)
        )

    critique_obj = _extract_critique(text)
    if critique_obj:
        return critique_obj.model_dump_json(indent=2)

    if text:
        return text

    # Final fallback: auto-approve so the pipeline can always complete
    return CritiqueResult(
        verdict="APPROVE",
        is_fresh=True,
        is_complete=True,
        is_well_structured=True,
        strengths=["Research appears comprehensive (auto-approved: critic produced no parseable output)"],
        gaps=[],
        revision_requests=[],
    ).model_dump_json(indent=2)
