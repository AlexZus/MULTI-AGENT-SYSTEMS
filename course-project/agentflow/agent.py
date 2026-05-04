"""AgentRunner — async ReAct loop using the OpenAI-compatible SDK.

Supports:
- Tool execution with configurable max iterations
- _try_fix_tool_call: recover tool calls emitted as plain JSON in content
- BudgetMiddleware: budget XML wrapping
- InvalidToolCallRetryMiddleware: retry on bad JSON args
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from openai import AsyncOpenAI

from agentflow.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware


# ---------------------------------------------------------------------------
# Tool-call JSON detection helpers
# ---------------------------------------------------------------------------

def _build_schema_lookup(tools: list[dict]) -> dict[frozenset, str]:
    """Map frozenset(all_property_names) → tool_name for content-field detection.

    Using *all* property names (required + optional) avoids collisions when two
    tools share the same required params but differ in optional ones (e.g.
    web_search with max_results vs knowledge_search with top_k both require
    only "query" but have different optional params).
    """
    lookup: dict[frozenset, str] = {}
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        # Prefer all-properties key; fall back to required-only if no properties defined
        all_props = frozenset(params.get("properties", {}).keys())
        key = all_props if all_props else frozenset(params.get("required") or [])
        if key:
            lookup[key] = fn["name"]
    return lookup


def _try_fix_tool_call(message: dict, schema_lookup: dict[frozenset, str]) -> dict:
    """If the model put tool-call JSON in the content field, convert it to tool_calls.

    This handles local LLMs that emit tool calls as raw JSON text instead of
    using the structured tool_calls field (HW4 pattern).
    """
    if message.get("tool_calls"):
        return message

    content = (message.get("content") or "").strip()
    if not content.startswith("{"):
        return message

    try:
        args = json.loads(content)
    except json.JSONDecodeError:
        return message

    if not isinstance(args, dict):
        return message

    tool_name = schema_lookup.get(frozenset(args.keys()))
    if not tool_name:
        return message

    fake_tc = {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args),
        },
    }
    return {**message, "tool_calls": [fake_tc], "content": ""}


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    content: str
    messages: list[dict]
    tool_calls_made: int = 0
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_call_records: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

ToolCallbackType = Callable[[str, dict, str], Awaitable[None]] | None


class AgentRunner:
    """Async ReAct agent loop.

    Parameters
    ----------
    system_prompt:
        System message content.
    tools:
        List of OpenAI-format tool schemas (``{"type": "function", "function": {...}}``).
    tool_executor:
        Async callable ``(tool_name, args_dict) -> str``.  Called for every tool invocation.
    settings:
        Application ``Settings`` instance.
    budget_middleware:
        Optional pre-configured ``BudgetMiddleware``.  Created from ``settings`` if None.
    retry_middleware:
        Optional pre-configured ``InvalidToolCallRetryMiddleware``.  Created from
        ``settings`` if None.
    on_tool_call:
        Optional async callback ``(tool_name, args, result)`` for streaming/tracing.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], Awaitable[str]],
        settings: Any,
        *,
        budget_middleware: BudgetMiddleware | None = None,
        retry_middleware: InvalidToolCallRetryMiddleware | None = None,
        on_tool_call: ToolCallbackType = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_executor = tool_executor
        self.settings = settings
        self.on_tool_call = on_tool_call

        self._budget_mw = budget_middleware or BudgetMiddleware(
            max_tool_calls=settings.max_tool_calls_per_agent
        )
        self._retry_mw = retry_middleware or InvalidToolCallRetryMiddleware(
            max_retries=settings.invalid_tool_call_max_retries
        )
        self._schema_lookup = _build_schema_lookup(tools)

        self._client = AsyncOpenAI(
            base_url=settings.openai_compatible_api_url,
            api_key=settings.api_key,
        )

    async def run(self, messages: list[dict]) -> AgentResult:
        """Run the ReAct loop starting from *messages* (no system prompt included).

        The system prompt is prepended automatically.
        Returns an ``AgentResult`` with the final content and full message history.
        """
        self._budget_mw.reset()
        self._retry_mw.reset()

        history: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            *messages,
        ]

        tool_calls_made = 0
        iterations = 0
        total_input_tokens = 0
        total_output_tokens = 0
        tool_call_records: list[dict] = []

        for iteration in range(self.settings.max_agent_iterations):
            iterations = iteration + 1

            response = await self._client.chat.completions.create(
                model=self.settings.model_name,
                messages=history,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
            )

            # Accumulate token usage
            if response.usage:
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens or 0

            raw_message = response.choices[0].message.model_dump(exclude_unset=False)
            # Normalise: remove None keys that confuse some LLMs on next turn
            raw_message = {k: v for k, v in raw_message.items() if v is not None or k in ("content", "role")}

            # Fix tool-call JSON emitted in content field
            raw_message = _try_fix_tool_call(raw_message, self._schema_lookup)

            tool_calls = raw_message.get("tool_calls") or []

            # Check for invalid JSON in tool call args
            if tool_calls:
                correction = self._retry_mw.check_and_build_correction(tool_calls)
                if correction:
                    history.append(raw_message)
                    history.append({"role": "user", "content": correction})
                    continue  # retry without executing tools

            # No tool calls → final answer
            if not tool_calls:
                history.append(raw_message)
                return AgentResult(
                    content=raw_message.get("content") or "",
                    messages=history,
                    tool_calls_made=tool_calls_made,
                    iterations=iterations,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    tool_call_records=tool_call_records,
                )

            # Execute tool calls
            history.append(raw_message)
            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")

                try:
                    args_dict = json.loads(args_str)
                except json.JSONDecodeError:
                    args_dict = {}

                try:
                    result = await self.tool_executor(tool_name, args_dict)
                except Exception as exc:
                    result = f"Error executing {tool_name}: {exc}"

                tool_calls_made += 1
                tool_call_records.append({"name": tool_name, "args": args_dict, "result": result})

                if self.on_tool_call:
                    await self.on_tool_call(tool_name, args_dict, result)

                # Apply budget middleware
                wrapped_result = self._budget_mw.wrap_tool_result(result)

                history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": wrapped_result,
                })

            # If budget exhausted, force final answer by removing tools
            if self._budget_mw.remaining <= 0:
                # One more model call without tools to get the final answer
                final_resp = await self._client.chat.completions.create(
                    model=self.settings.model_name,
                    messages=history,
                )
                if final_resp.usage:
                    total_input_tokens += final_resp.usage.prompt_tokens or 0
                    total_output_tokens += final_resp.usage.completion_tokens or 0
                final_msg = final_resp.choices[0].message.model_dump(exclude_unset=False)
                history.append(final_msg)
                return AgentResult(
                    content=final_msg.get("content") or "",
                    messages=history,
                    tool_calls_made=tool_calls_made,
                    iterations=iterations + 1,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    tool_call_records=tool_call_records,
                )

        return AgentResult(
            content="Error: reached maximum iteration limit without a final answer.",
            messages=history,
            tool_calls_made=tool_calls_made,
            iterations=iterations,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            tool_call_records=tool_call_records,
        )
