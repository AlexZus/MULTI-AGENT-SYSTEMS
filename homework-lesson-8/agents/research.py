"""Research Agent — executes a research plan using web search, read_url, and knowledge base."""

import json

from langchain.agents import create_agent
from langchain.tools import tool

from config import RESEARCHER_SYSTEM_PROMPT, get_model
from tools import knowledge_search, web_fetch, web_search

_research_agent = create_agent(
    get_model(),
    tools=[web_search, web_fetch, knowledge_search],
    system_prompt=RESEARCHER_SYSTEM_PROMPT,
)


@tool
def research(request: str | dict) -> str:
    """Execute a research plan and return comprehensive findings.

    Pass the full ResearchPlan JSON (from the planner) along with any
    additional revision guidance from the critic. The researcher will
    search the knowledge base and web, then return structured findings.

    Input: ResearchPlan JSON + optional revision guidance.
    Output: Detailed research findings in Markdown format.
    """
    # Normalize: LLM may pass a dict (ResearchPlan) instead of a string
    if isinstance(request, dict):
        request = json.dumps(request, ensure_ascii=False, indent=2)

    try:
        result = _research_agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        )
    except Exception as e:
        return f"Research agent error: {e}"

    messages = result.get("messages", [])
    if not messages:
        return "Researcher produced no output."
    last = messages[-1]
    if hasattr(last, "text") and isinstance(last.text, str):
        return last.text
    if callable(getattr(last, "text", None)):
        return last.text()
    return str(last.content)
