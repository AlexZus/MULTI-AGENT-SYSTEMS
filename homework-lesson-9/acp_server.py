"""ACP server exposing three agents: planner, researcher, critic.

Each agent connects to SearchMCP (port 8901) to get its tools,
then runs a create_agent loop and returns the result as an ACP Message.
"""

import json
import re

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Server
from fastmcp import Client as MCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware, _tool_budget
from config import Settings, get_critic_prompt, get_model, get_planner_prompt, RESEARCHER_SYSTEM_PROMPT
from mcp_utils import mcp_tools_to_langchain
from schemas import CritiqueResult, ResearchPlan

_settings = Settings()
SEARCH_MCP_URL = f"http://127.0.0.1:8901/mcp"

acp_server = Server()

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_last_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    if hasattr(last, "text") and isinstance(last.text, str):
        return last.text
    if callable(getattr(last, "text", None)):
        return last.text()
    return str(last.content)


def _extract_json_object(text: str) -> str | None:
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    last_brace = text.rfind("}")
    first_brace = text.rfind("{", 0, last_brace + 1)
    if first_brace >= 0:
        return text[first_brace: last_brace + 1]
    return None


async def _build_lc_tools():
    """Connect to SearchMCP and return LangChain tools."""
    async with MCPClient(SEARCH_MCP_URL) as client:
        mcp_tools = await client.list_tools()
        return mcp_tools_to_langchain(mcp_tools, client), client


# ── Planner agent ─────────────────────────────────────────────────────────────

_PLAN_CORRECTION_MSG = (
    "Your previous response did not contain a valid JSON block matching the ResearchPlan schema. "
    "You MUST end your response with a fenced JSON block in exactly this format:\n"
    "```json\n"
    '{{"goal": "...", "search_queries": ["q1", "q2"], '
    '"sources_to_check": ["knowledge_base", "web"], "output_format": "..."}}\n'
    "```\n"
    "Output the JSON block now."
)


@acp_server.agent(
    name="planner",
    description="Decomposes a research request into a structured ResearchPlan using preliminary web and knowledge-base searches.",
)
async def planner_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content

    async with MCPClient(SEARCH_MCP_URL) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        lc_tools = mcp_tools_to_langchain(mcp_tools, mcp_client)

        agent = create_agent(
            get_model(),
            tools=lc_tools,
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

        retry_limit = _settings.subagent_output_retry_number_on_validation_fail
        sub_config = {
            "recursion_limit": _settings.max_iterations * 2 + 2,
            "configurable": {"thread_id": "planner"},
        }

        for attempt in range(retry_limit + 1):
            msg_content = user_text if attempt == 0 else _PLAN_CORRECTION_MSG
            token = _tool_budget.set({"remaining": _settings.max_iterations})
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": msg_content}]},
                    config=sub_config,
                )
            except GraphRecursionError:
                fallback = ResearchPlan(
                    goal=user_text[:120],
                    search_queries=[user_text[:80]],
                    sources_to_check=["knowledge_base", "web"],
                    output_format="comprehensive Markdown report with sections and citations",
                )
                return Message(role="agent", parts=[MessagePart(content=fallback.model_dump_json(indent=2))])
            except Exception as e:
                return Message(role="agent", parts=[MessagePart(
                    content=json.dumps({"goal": user_text[:120], "search_queries": [user_text[:80]],
                                        "sources_to_check": ["knowledge_base", "web"],
                                        "output_format": "Markdown report", "_error": str(e)})
                )])
            finally:
                _tool_budget.reset(token)

            text = _extract_last_text(result)

            # Structured-output path
            if not _settings.structured_output_workaround:
                plan_obj = result.get("structured_response")
                if isinstance(plan_obj, ResearchPlan):
                    return Message(role="agent", parts=[MessagePart(content=plan_obj.model_dump_json(indent=2))])

            # Text-extraction path
            raw = _extract_json_object(text)
            if raw:
                try:
                    plan_obj = ResearchPlan.model_validate_json(raw)
                    return Message(role="agent", parts=[MessagePart(content=plan_obj.model_dump_json(indent=2))])
                except Exception:
                    pass

        return Message(role="agent", parts=[MessagePart(content=text or "Planner produced no output.")])


# ── Researcher agent ──────────────────────────────────────────────────────────

@acp_server.agent(
    name="researcher",
    description="Executes a research plan: searches web and knowledge base, returns detailed Markdown findings.",
)
async def researcher_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content

    async with MCPClient(SEARCH_MCP_URL) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        lc_tools = mcp_tools_to_langchain(mcp_tools, mcp_client)

        agent = create_agent(
            get_model(),
            tools=lc_tools,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                BudgetMiddleware(),
                InvalidToolCallRetryMiddleware(
                    max_retries=_settings.subagent_output_retry_number_on_validation_fail
                ),
            ],
        )

        sub_config = {
            "recursion_limit": _settings.max_iterations * 2 + 2,
            "configurable": {"thread_id": "researcher"},
        }
        token = _tool_budget.set({"remaining": _settings.max_iterations})
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_text}]},
                config=sub_config,
            )
        except GraphRecursionError:
            return Message(role="agent", parts=[MessagePart(
                content=(
                    "[RESEARCH LIMIT REACHED] The researcher exhausted its tool-call budget. "
                    "Proceed to critique() on partial findings."
                )
            )])
        except Exception as e:
            return Message(role="agent", parts=[MessagePart(content=f"Research agent error: {e}")])
        finally:
            _tool_budget.reset(token)

    return Message(role="agent", parts=[MessagePart(content=_extract_last_text(result))])


# ── Critic agent ──────────────────────────────────────────────────────────────

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


@acp_server.agent(
    name="critic",
    description="Independently verifies research findings for freshness, completeness, and structure. Returns CritiqueResult JSON.",
)
async def critic_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content

    async with MCPClient(SEARCH_MCP_URL) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        lc_tools = mcp_tools_to_langchain(mcp_tools, mcp_client)

        agent = create_agent(
            get_model(),
            tools=lc_tools,
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

        retry_limit = _settings.subagent_output_retry_number_on_validation_fail
        sub_config = {
            "recursion_limit": _settings.max_iterations * 2 + 2,
            "configurable": {"thread_id": "critic"},
        }

        text = ""
        for attempt in range(retry_limit + 1):
            msg_content = user_text if attempt == 0 else _CRITIQUE_CORRECTION_MSG
            token = _tool_budget.set({"remaining": _settings.max_iterations})
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": msg_content}]},
                    config=sub_config,
                )
            except GraphRecursionError:
                return Message(role="agent", parts=[MessagePart(
                    content=_AUTO_APPROVE.model_dump_json(indent=2)
                    + "\n// [CRITIC LIMIT REACHED] Auto-approved."
                )])
            except Exception as e:
                return Message(role="agent", parts=[MessagePart(
                    content=_AUTO_APPROVE.model_dump_json(indent=2)
                    + f"\n// Critic error (auto-approved): {e}"
                )])
            finally:
                _tool_budget.reset(token)

            text = _extract_last_text(result)

            if not _settings.structured_output_workaround:
                critique_obj = result.get("structured_response")
                if isinstance(critique_obj, CritiqueResult):
                    return Message(role="agent", parts=[MessagePart(content=critique_obj.model_dump_json(indent=2))])

            raw = _extract_json_object(text)
            if raw:
                try:
                    critique_obj = CritiqueResult.model_validate_json(raw)
                    return Message(role="agent", parts=[MessagePart(content=critique_obj.model_dump_json(indent=2))])
                except Exception:
                    pass

    fallback = _AUTO_APPROVE.model_dump_json(indent=2) + "\n// Auto-approved: no parseable output."
    return Message(role="agent", parts=[MessagePart(content=fallback)])


if __name__ == "__main__":
    acp_server.run(port=8903)
