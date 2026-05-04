"""Developer agent — implements code based on SpecOutput and produces CodeOutput."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.schemas import CodeOutput, SpecOutput
from agentflow.agent import AgentRunner
from tools.mcp_fs import MCPFilesystem
from tools.mcp_repl import MCPRepl
from tools.rag import RAG_TOOL_SCHEMA, knowledge_search_async
from tools.search import SEARCH_TOOL_SCHEMA, web_search_async

logger = logging.getLogger(__name__)


def _extract_code_json(content: str) -> dict | None:
    match = re.search(r"```json\s*([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class DeveloperAgent:
    """Developer agent wrapping AgentRunner + filesystem/REPL MCP tools."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    async def run(
        self,
        spec: SpecOutput,
        *,
        project_name: str,
        on_tool_call=None,
        trace_store: Any = None,
        event_bus: Any = None,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> CodeOutput:
        from tracevault.prompts import load_prompt

        system_prompt = await load_prompt("developer_system", project_name=project_name)
        if self.settings.structured_output_workaround:
            json_suffix = await load_prompt("developer_json_suffix", project_name=project_name)
            system_prompt = f"{system_prompt}\n\n{json_suffix}"

        async with MCPFilesystem(
            self.settings.mcp_filesystem_url,
            project_name=project_name,
        ) as fs, MCPRepl(
            self.settings.mcp_repl_url,
            project_name=project_name,
        ) as repl:
            fs_tools = fs.get_openai_tools()
            repl_tools = repl.get_openai_tools()
            all_tools = fs_tools + repl_tools + [SEARCH_TOOL_SCHEMA, RAG_TOOL_SCHEMA]

            fs_names = {t["function"]["name"] for t in fs_tools}
            repl_names = {t["function"]["name"] for t in repl_tools}

            async def execute(tool_name: str, args: dict) -> str:
                if tool_name in fs_names:
                    return await fs.call_tool(tool_name, args)
                elif tool_name in repl_names:
                    return await repl.call_tool(tool_name, args)
                elif tool_name == "web_search":
                    return await web_search_async(args.get("query", ""), args.get("max_results", 5))
                elif tool_name == "knowledge_search":
                    return await knowledge_search_async(args.get("query", ""), args.get("top_k", 5))
                return f"Unknown tool: {tool_name}"

            runner = AgentRunner(
                system_prompt=system_prompt,
                tools=all_tools,
                tool_executor=execute,
                settings=self.settings,
                on_tool_call=on_tool_call,
            )

            spec_text = (
                f"Implement the following specification for project '{project_name}':\n\n"
                f"Title: {spec.title}\n\n"
                f"Requirements:\n" + "\n".join(f"- {r}" for r in spec.requirements) + "\n\n"
                f"Acceptance Criteria:\n" + "\n".join(f"- {c}" for c in spec.acceptance_criteria) + "\n\n"
                f"Estimated complexity: {spec.estimated_complexity}\n"
                + (f"\nNotes: {spec.notes}" if spec.notes else "")
            )

            messages = [{"role": "user", "content": spec_text}]

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
                        agent_name="developer",
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

                data = _extract_code_json(result.content)
                if data:
                    try:
                        return CodeOutput(**data)
                    except Exception as exc:
                        logger.warning(
                            "attempt %d: CodeOutput validation failed: %s; data=%r",
                            attempt, exc, data,
                        )
                else:
                    logger.warning(
                        "attempt %d: no JSON block in developer response (len=%d): %.300r",
                        attempt, len(result.content), result.content,
                    )

                # Preserve full history; append a single correction so the LLM can
                # either emit the JSON block or continue with more tool calls.
                # result.messages[0] is the system prompt — runner re-adds it.
                messages = result.messages[1:] + [
                    {
                        "role": "user",
                        "content": (
                            "Your last response did not contain the required JSON output block. "
                            "If you still need to make tool calls to finish the implementation, "
                            "do so now. Once done, end your response with the JSON block exactly "
                            "as specified in the instructions."
                        ),
                    }
                ]

        raise ValueError("DeveloperAgent failed to produce valid CodeOutput after 3 attempts")
