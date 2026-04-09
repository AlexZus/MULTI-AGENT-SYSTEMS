"""Supervisor Agent — orchestrates Plan → Research → Critique → Save pipeline.

Sub-agents are called via ACP (acp_sdk.client.Client).
save_report is called via ReportMCP (fastmcp.Client).
"""

import asyncio

from acp_sdk.client import Client as ACPClient
from acp_sdk.models import Message, MessagePart
from fastmcp import Client as MCPClient
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware
from config import (
    ACP_PORT,
    REPORT_MCP_PORT,
    Settings,
    SUPERVISOR_SYSTEM_PROMPT,
    get_model,
)

_settings = Settings()
_ACP_BASE = f"http://127.0.0.1:{ACP_PORT}"
_REPORT_MCP_URL = f"http://127.0.0.1:{REPORT_MCP_PORT}/mcp"


# ── ACP delegation helpers ────────────────────────────────────────────────────

def _acp_call(agent_name: str, content: str) -> str:
    """Synchronous wrapper: call an ACP agent and return its text output."""
    async def _run():
        async with ACPClient(
            base_url=_ACP_BASE,
            headers={"Content-Type": "application/json"},
        ) as client:
            run = await client.run_sync(
                agent=agent_name,
                input=[Message(role="user", parts=[MessagePart(content=content)])],
            )
            return run.output[-1].parts[0].content

    return asyncio.run(_run())


# ── Supervisor tools ──────────────────────────────────────────────────────────

@tool
def delegate_to_planner(request: str, config: RunnableConfig = None) -> str:
    """Delegate research planning to the Planner ACP agent.

    The planner does preliminary domain reconnaissance (knowledge base +
    web search) before producing a plan with: goal, search_queries,
    sources_to_check, and output_format.

    Input: The user's original research request (natural language).
    Output: JSON-serialized ResearchPlan.
    """
    return _acp_call("planner", request)


@tool
def delegate_to_researcher(request: str, config: RunnableConfig = None) -> str:
    """Delegate research execution to the Researcher ACP agent.

    Pass the full ResearchPlan JSON (from the planner) along with any
    additional revision guidance from the critic. The researcher will
    search the knowledge base and web, then return structured findings.

    Input: ResearchPlan JSON + optional revision guidance.
    Output: Detailed research findings in Markdown format.
    """
    return _acp_call("researcher", request)


@tool
def delegate_to_critic(findings: str, config: RunnableConfig = None) -> str:
    """Delegate critique to the Critic ACP agent.

    The critic independently evaluates research findings for freshness,
    completeness, and structure. Returns a CritiqueResult JSON with
    verdict "APPROVE" or "REVISE".

    Input: Research findings text (from the researcher).
    Output: JSON-serialized CritiqueResult.
    """
    return _acp_call("critic", findings)


@tool
def save_report(filename: str, content: str) -> str:
    """Save a Markdown research report to a file via ReportMCP.

    Call this as the final step when the research has been approved by the Critic.
    The file will be saved in the output/ directory.
    """
    async def _run():
        async with MCPClient(_REPORT_MCP_URL) as client:
            result = await client.call_tool("save_report", {"filename": filename, "content": content})
            return result.data or result.content[0].text

    return asyncio.run(_run())


# ── Supervisor factory ────────────────────────────────────────────────────────

def create_supervisor():
    """Create a new Supervisor agent with HITL on save_report and an in-memory checkpointer."""
    return create_agent(
        get_model(),
        tools=[delegate_to_planner, delegate_to_researcher, delegate_to_critic, save_report],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=[
            BudgetMiddleware(),
            InvalidToolCallRetryMiddleware(
                max_retries=_settings.subagent_output_retry_number_on_validation_fail
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={"save_report": True},
                description_prefix="Research report pending approval",
            ),
        ],
        checkpointer=InMemorySaver(),
    )
