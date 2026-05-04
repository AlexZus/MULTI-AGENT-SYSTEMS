"""Middleware components for the agentflow ReAct loop.

BudgetMiddleware             — wraps every tool result with <tool_call_limits_info> XML
InvalidToolCallRetryMiddleware — catches malformed JSON args, injects correction, retries
"""

from __future__ import annotations

import json


class BudgetMiddleware:
    """Appends XML budget info to every tool result so the LLM knows remaining calls.

    Usage::

        budget = BudgetMiddleware(max_tool_calls=30)
        budget.reset()
        result_text = budget.wrap_tool_result(raw_result)
    """

    def __init__(self, max_tool_calls: int) -> None:
        self.max_tool_calls = max_tool_calls
        self._remaining = max_tool_calls

    def reset(self) -> None:
        self._remaining = self.max_tool_calls

    @property
    def remaining(self) -> int:
        return self._remaining

    def wrap_tool_result(self, result: str) -> str:
        """Decrement counter and wrap result with budget XML."""
        self._remaining -= 1
        remaining = self._remaining

        if remaining <= 0:
            limit_msg = "You have spent all tool call budget. You MUST provide the final answer now."
        elif remaining == 1:
            limit_msg = "You have 1 tool call remaining before you MUST provide the final answer."
        else:
            limit_msg = (
                f"You can call tools {remaining} more times before providing the final answer."
            )

        return (
            f"<tool_call_output>\n{result}\n</tool_call_output>\n"
            f"<tool_call_limits_info>\n    {limit_msg}\n</tool_call_limits_info>"
        )


class InvalidToolCallRetryMiddleware:
    """Detects tool calls with invalid JSON arguments and injects a correction message.

    The AgentRunner checks this middleware after each LLM response.  If invalid
    JSON args are detected, this middleware returns a corrective user message to
    inject into the conversation before retrying the model.

    Usage::

        retry_mw = InvalidToolCallRetryMiddleware(max_retries=3)
        correction = retry_mw.check_and_build_correction(tool_calls)
        if correction:
            messages.append({"role": "user", "content": correction})
            # retry LLM call
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._retry_count = 0

    def reset(self) -> None:
        self._retry_count = 0

    @property
    def retries_left(self) -> int:
        return self.max_retries - self._retry_count

    def check_and_build_correction(self, tool_calls: list[dict]) -> str | None:
        """Inspect raw tool_call dicts from the LLM response.

        Returns a correction message string if invalid JSON args are found and
        retries remain, otherwise None (caller should proceed normally or stop).
        Increments the retry counter when a correction is returned.
        """
        if self._retry_count >= self.max_retries:
            return None

        error_lines: list[str] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "unknown")
            args_str = fn.get("arguments", "")
            if args_str:
                try:
                    json.loads(args_str)
                except json.JSONDecodeError as exc:
                    error_lines.append(
                        f"  - tool '{name}': invalid JSON — {exc}\n    raw args: {args_str!r}"
                    )

        if not error_lines:
            return None

        self._retry_count += 1
        return (
            "Your previous tool call could not be executed because the arguments "
            "contained invalid JSON syntax. Please fix the JSON and retry.\n\n"
            "Error details:\n" + "\n".join(error_lines)
        )
