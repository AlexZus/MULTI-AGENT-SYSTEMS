"""Supervisor Agent — orchestrates Plan → Research → Critique → Save pipeline."""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from agents.critic import critique
from agents.planner import plan
from agents.research import research
from config import SUPERVISOR_SYSTEM_PROMPT, get_model
from tools import save_report


def create_supervisor():
    """Create a new Supervisor agent with HITL on save_report and an in-memory checkpointer."""
    return create_agent(
        get_model(),
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"save_report": True},
                description_prefix="Research report pending approval",
            ),
        ],
        checkpointer=InMemorySaver(),
    )
