"""Supervisor Agent — orchestrates Plan → Research → Critique → Save pipeline."""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from agents.critic import critique
from agents.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware
from agents.planner import plan
from agents.research import research
from config import Settings, get_supervisor_prompt, get_model
from tools import save_report

_settings = Settings()


def create_supervisor():
    """Create a new Supervisor agent with HITL on save_report and an in-memory checkpointer."""
    return create_agent(
        get_model(),
        tools=[plan, research, critique, save_report],
        system_prompt=get_supervisor_prompt(),
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
