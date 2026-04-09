"""Custom middleware for sub-agents.

BudgetMiddleware     — wraps every tool result with XML budget info so the LLM
                       knows how many tool calls remain before it must finalize.
InvalidToolCallRetryMiddleware — retries model calls where the LLM produced tool
                       calls with invalid JSON arguments, injecting a corrective
                       user message each time.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Shared ContextVar — set by each @tool wrapper before calling sub-agent.invoke
# ---------------------------------------------------------------------------
_tool_budget: ContextVar[dict[str, int] | None] = ContextVar("_tool_budget", default=None)


# ---------------------------------------------------------------------------
# BudgetMiddleware
# ---------------------------------------------------------------------------

class BudgetMiddleware(AgentMiddleware):
    """Append XML budget tags to every tool result.

    Reads remaining call budget from the ``_tool_budget`` ContextVar (a mutable
    dict with a ``"remaining"`` key).  Must be initialised by the @tool caller
    before each agent.invoke() call::

        token = _tool_budget.set({"remaining": budget})
        try:
            agent.invoke(...)
        finally:
            _tool_budget.reset(token)
    """

    tools: list = []

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        result = handler(request)

        budget = _tool_budget.get()
        if budget is None or not isinstance(result, ToolMessage):
            return result

        budget["remaining"] -= 1
        remaining = budget["remaining"]

        if remaining <= 1:
            limit_msg = "You spend all tool call budget. You must provide the final answer."
        else:
            limit_msg = (
                f"You can call tools another {remaining-1} times before producing the final answer."
            )

        wrapped_content = (
            f"<tool_call_output>\n{result.content}\n</tool_call_output>\n"
            f"<tool_call_limits_info>\n    {limit_msg}\n</tool_call_limits_info>"
        )
        return result.model_copy(update={"content": wrapped_content})


# ---------------------------------------------------------------------------
# InvalidToolCallRetryMiddleware
# ---------------------------------------------------------------------------

class InvalidToolCallRetryMiddleware(AgentMiddleware):
    """Retry model calls that return tool calls with invalid JSON arguments.

    When the LLM produces a tool call whose ``args`` field is not valid JSON,
    LangChain stores it in ``AIMessage.invalid_tool_calls``.  This middleware
    injects a corrective HumanMessage and retries the model up to
    ``max_retries`` times.
    """

    tools: list = []

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(self.max_retries + 1):
            response = handler(request)

            if not response.result:
                return response

            ai_msg = response.result[0]
            if not isinstance(ai_msg, AIMessage):
                return response

            invalid = ai_msg.invalid_tool_calls or []
            if not invalid or attempt >= self.max_retries:
                return response

            # Build human-readable error description
            error_lines = []
            for tc in invalid:
                name = tc.get("name") or "unknown"
                args = tc.get("args") or ""
                err = tc.get("error") or "invalid JSON"
                error_lines.append(f"  - tool '{name}': {err}\n    raw args: {args!r}")

            correction = HumanMessage(
                content=(
                    "Your previous tool call could not be executed because the arguments "
                    "contained invalid JSON syntax. Please fix the JSON and retry.\n\n"
                    "Error details:\n" + "\n".join(error_lines)
                )
            )
            # Extend the in-flight message list: bad AI message + corrective user message
            request = request.override(messages=request.messages + [ai_msg, correction])

        return response  # return last response even if still invalid
