"""Planner Agent — decomposes a research request into a structured ResearchPlan."""

import json
import re

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware, _tool_budget
from config import Settings, get_model, get_planner_prompt
from schemas import ResearchPlan
from tools import knowledge_search, web_fetch, web_search

_settings = Settings()

_planner_agent = create_agent(
    get_model(),
    tools=[web_search, web_fetch, knowledge_search],
    system_prompt=get_planner_prompt(),
    checkpointer=InMemorySaver(),
    middleware=[
        BudgetMiddleware(),
        InvalidToolCallRetryMiddleware(
            max_retries=_settings.subagent_output_retry_number_on_validation_fail
        ),
    ],
    response_format=None if _settings.structured_output_workaround else ResearchPlan,
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_PLAN_CORRECTION_MSG = (
    "Your previous response did not contain a valid JSON block matching the ResearchPlan schema. "
    "You MUST end your response with a fenced JSON block in exactly this format:\n"
    "```json\n"
    '{{"goal": "...", "search_queries": ["q1", "q2"], '
    '"sources_to_check": ["knowledge_base", "web"], "output_format": "..."}}\n'
    "```\n"
    "Output the JSON block now."
)


def _extract_plan(text: str) -> ResearchPlan | None:
    """Parse a ResearchPlan from a fenced JSON block in the model's text response."""
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return ResearchPlan.model_validate_json(m.group(1))
        except Exception:
            pass
    # Fallback: last {...} in the text
    last_brace = text.rfind("}")
    first_brace = text.rfind("{", 0, last_brace + 1)
    if first_brace >= 0:
        try:
            return ResearchPlan.model_validate_json(text[first_brace: last_brace + 1])
        except Exception:
            pass
    return None


def _invoke_planner(msg_content: str, sub_config: RunnableConfig) -> tuple[str, dict | None]:
    """Invoke the planner agent once; return (text_output, raw_result)."""
    token = _tool_budget.set({"remaining": _settings.max_iterations})
    try:
        result = _planner_agent.invoke(
            {"messages": [{"role": "user", "content": msg_content}]},
            config=sub_config,
        )
    except GraphRecursionError:
        return (
            json.dumps({
                "goal": msg_content[:120],
                "search_queries": [msg_content[:80]],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": "comprehensive Markdown report with sections and citations",
                "_error": (
                    f"[PLAN LIMIT REACHED] Planner exhausted its tool-call budget "
                    f"(max_iterations={_settings.max_iterations}). "
                    "A minimal fallback plan has been generated. Proceed with research()."
                ),
            }, indent=2),
            None,
        )
    except Exception as e:
        return (
            json.dumps({
                "goal": msg_content[:120],
                "search_queries": [msg_content[:80]],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": "comprehensive Markdown report with sections and citations",
                "_error": f"Planner unexpected error: {type(e).__name__}: {e}",
            }, indent=2),
            None,
        )
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
def plan(request: str, config: RunnableConfig = None) -> str:
    """Decompose a research request into a structured ResearchPlan.

    The planner does preliminary domain reconnaissance (knowledge base +
    web search) before producing a plan with: goal, search_queries,
    sources_to_check, and output_format.

    Input: The user's original research request (natural language).
    Output: JSON-serialized ResearchPlan.
    """
    supervisor_thread = (config or {}).get("configurable", {}).get("thread_id", "default")
    sub_config: RunnableConfig = {
        "recursion_limit": _settings.max_iterations,
        "configurable": {"thread_id": f"{supervisor_thread}:planner"},
    }

    retry_limit = _settings.subagent_output_retry_number_on_validation_fail

    for attempt in range(retry_limit + 1):
        msg = request if attempt == 0 else _PLAN_CORRECTION_MSG
        text, result = _invoke_planner(msg, sub_config)

        # Hard error from invoke (GraphRecursionError / Exception) — text is already
        # a fallback JSON string; return it directly.
        if result is None:
            return text

        # Structured-output path
        if not _settings.structured_output_workaround and result:
            plan_obj = result.get("structured_response")
            if isinstance(plan_obj, ResearchPlan):
                return plan_obj.model_dump_json(indent=2)

        # Text-extraction path
        plan_obj = _extract_plan(text)
        if plan_obj:
            return plan_obj.model_dump_json(indent=2)

        # Extraction failed — retry if budget allows
        # (next iteration sends _PLAN_CORRECTION_MSG on the same thread)

    return text or "Planner produced no output."
