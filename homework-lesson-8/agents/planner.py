"""Planner Agent — decomposes a research request into a structured ResearchPlan."""

import json
import re

from langchain.agents import create_agent
from langchain.tools import tool

from config import Settings, get_model, get_planner_prompt
from schemas import ResearchPlan
from tools import knowledge_search, web_fetch, web_search

_settings = Settings()

_planner_agent = create_agent(
    get_model(),
    tools=[web_search, web_fetch, knowledge_search],
    system_prompt=get_planner_prompt(),
    response_format=None if _settings.structured_output_workaround else ResearchPlan,
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


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


@tool
def plan(request: str) -> str:
    """Decompose a research request into a structured ResearchPlan.

    The planner does preliminary domain reconnaissance (knowledge base +
    web search) before producing a plan with: goal, search_queries,
    sources_to_check, and output_format.

    Input: The user's original research request (natural language).
    Output: JSON-serialized ResearchPlan.
    """
    try:
        result = _planner_agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        )
    except Exception as e:
        return json.dumps({
            "goal": request,
            "search_queries": [request],
            "sources_to_check": ["knowledge_base", "web"],
            "output_format": "comprehensive Markdown report with sections and citations",
            "_error": f"Planner failed: {e}",
        }, indent=2)

    # With response_format= the structured result is in "structured_response"
    if not _settings.structured_output_workaround:
        plan_obj = result.get("structured_response")
        if isinstance(plan_obj, ResearchPlan):
            return plan_obj.model_dump_json(indent=2)

    # Workaround path: extract JSON from the model's text response
    messages = result.get("messages", [])
    last = messages[-1] if messages else None
    text = ""
    if last:
        text = last.text if isinstance(getattr(last, "text", None), str) else (
            last.text() if callable(getattr(last, "text", None)) else str(last.content)
        )

    plan_obj = _extract_plan(text)
    if plan_obj:
        return plan_obj.model_dump_json(indent=2)

    return text or "Planner produced no output."
