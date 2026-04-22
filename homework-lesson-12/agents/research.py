"""Research Agent — executes a research plan using web search, read_url, and knowledge base."""

import json

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware, _tool_budget
from config import Settings, get_model, get_researcher_prompt
from tools import knowledge_search, web_fetch, web_search

_settings = Settings()

_research_agent = create_agent(
    get_model(),
    tools=[web_search, web_fetch, knowledge_search],
    system_prompt=get_researcher_prompt(),
    checkpointer=InMemorySaver(),
    middleware=[
        BudgetMiddleware(),
        InvalidToolCallRetryMiddleware(
            max_retries=_settings.subagent_output_retry_number_on_validation_fail
        ),
    ],
)


@tool
def research(request: str | dict, config: RunnableConfig = None) -> str:
    """Execute a research plan and return comprehensive findings.

    Pass the full ResearchPlan JSON (from the planner) along with any
    additional revision guidance from the critic. The researcher will
    search the knowledge base and web, then return structured findings.

    Input: ResearchPlan JSON + optional revision guidance.
    Output: Detailed research findings in Markdown format.
    """
    if isinstance(request, dict):
        request = json.dumps(request, ensure_ascii=False, indent=2)

    supervisor_thread = (config or {}).get("configurable", {}).get("thread_id", "default")
    sub_config: RunnableConfig = {
        "recursion_limit": _settings.max_iterations,
        "configurable": {"thread_id": f"{supervisor_thread}:researcher"},
        "callbacks": (config or {}).get("callbacks", []),
    }

    token = _tool_budget.set({"remaining": _settings.max_iterations})
    try:
        result = _research_agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config=sub_config,
        )
    except GraphRecursionError:
        return (
            "[RESEARCH LIMIT REACHED] The researcher exhausted its tool-call budget "
            f"(max_iterations={_settings.max_iterations}) without producing a final answer. "
            "The topic may be too broad or the model called redundant tools. "
            "Recommended next step: call critique() on whatever partial findings exist, "
            "or ask the user to narrow the research scope. "
            "Do NOT call research() again with the same plan."
        )
    except Exception as e:
        return f"Research agent unexpected error: {type(e).__name__}: {e}"
    finally:
        _tool_budget.reset(token)

    messages = result.get("messages", [])
    if not messages:
        return "Researcher produced no output."
    last = messages[-1]
    if hasattr(last, "text") and isinstance(last.text, str):
        return last.text
    if callable(getattr(last, "text", None)):
        return last.text()
    return str(last.content)
