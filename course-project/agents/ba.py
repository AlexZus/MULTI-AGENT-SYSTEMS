"""Business Analyst agent — analyses user stories and produces SpecOutput."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from agents.schemas import SpecOutput
from agentflow.agent import AgentRunner
from tools.rag import RAG_TOOL_SCHEMA, knowledge_search_async
from tools.search import SEARCH_TOOL_SCHEMA, web_search_async


def _build_tool_executor(settings: Any):
    async def execute(tool_name: str, args: dict) -> str:
        if tool_name == "web_search":
            return await web_search_async(
                args.get("query", ""),
                max_results=args.get("max_results", settings.max_search_results if hasattr(settings, "max_search_results") else 5),
            )
        elif tool_name == "knowledge_search":
            return await knowledge_search_async(
                args.get("query", ""),
                top_k=args.get("top_k", 5),
            )
        return f"Unknown tool: {tool_name}"

    return execute


def _extract_spec_json(content: str) -> dict | None:
    """Extract fenced JSON block from LLM response (structured output workaround)."""
    match = re.search(r"```json\s*([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON at end of content
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class BAAgent:
    """Business Analyst agent wrapping AgentRunner."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    async def run(
        self,
        user_story: str,
        *,
        project_name: str,
        on_tool_call=None,
        trace_store: Any = None,
        event_bus: Any = None,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> SpecOutput:
        """Analyse user_story and return a SpecOutput.

        Retries JSON extraction up to 3 times with trimmed history.
        """
        from tracevault.prompts import load_prompt

        system_prompt = await load_prompt("ba_system", project_name=project_name)
        if self.settings.structured_output_workaround:
            json_suffix = await load_prompt("ba_json_suffix")
            system_prompt = f"{system_prompt}\n\n{json_suffix}"

        tools = [SEARCH_TOOL_SCHEMA, RAG_TOOL_SCHEMA]
        executor = _build_tool_executor(self.settings)

        runner = AgentRunner(
            system_prompt=system_prompt,
            tools=tools,
            tool_executor=executor,
            settings=self.settings,
            on_tool_call=on_tool_call,
        )

        messages = [{"role": "user", "content": user_story}]

        for attempt in range(3):
            span_ctx = None
            if trace_store and trace_id and session_id:
                from tracevault.tracker import TraceContext
                span_ctx = TraceContext(
                    trace_store=trace_store,
                    event_bus=event_bus,
                    trace_id=trace_id,
                    session_id=session_id,
                    project_name=project_name,
                    agent_name="ba",
                    iteration=attempt,
                    input_messages=messages,
                )
                await span_ctx.__aenter__()

            try:
                result = await runner.run(messages)
            finally:
                if span_ctx is not None:
                    final_msg = result.messages[-1] if result.messages else {}
                    span_ctx.set_output(
                        final_msg,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                    )
                    for tc in result.tool_call_records:
                        span_ctx.add_tool_call(tc["name"], tc["args"], tc["result"])
                    await span_ctx.__aexit__(None, None, None)

            data = _extract_spec_json(result.content)
            if data:
                try:
                    return SpecOutput(**data)
                except Exception as exc:
                    logger.warning(
                        "attempt %d: SpecOutput validation failed: %s; data=%r",
                        attempt, exc, data,
                    )
            else:
                logger.warning(
                    "attempt %d: no JSON block in BA response (len=%d): %.300r",
                    attempt, len(result.content), result.content,
                )

            # Preserve full history; append a single correction so the LLM can
            # either emit the JSON spec block or continue with more research.
            # result.messages[0] is the system prompt — runner re-adds it.
            messages = result.messages[1:] + [
                {
                    "role": "user",
                    "content": (
                        "Your last response did not contain the required JSON specification block. "
                        "If you need to do more research, do so now. "
                        "Once done, end your response with the JSON block exactly "
                        "as specified in the instructions."
                    ),
                }
            ]

        raise ValueError("BAAgent failed to produce valid SpecOutput after 3 attempts")
