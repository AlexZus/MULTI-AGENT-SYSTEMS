"""QA agent — reviews code and produces ReviewOutput."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from agents.schemas import CodeOutput, ReviewOutput, SpecOutput
from agentflow.agent import AgentRunner
from tools.mcp_fs import MCPFilesystem
from tools.mcp_repl import MCPRepl


def _extract_review_json(content: str) -> dict | None:
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


class QAAgent:
    """QA agent wrapping AgentRunner + filesystem/REPL MCP tools (read-only subset)."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    async def run(
        self,
        spec: SpecOutput,
        code: CodeOutput,
        *,
        project_name: str,
        on_tool_call=None,
        trace_store: Any = None,
        event_bus: Any = None,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> ReviewOutput:
        from tracevault.prompts import load_prompt

        system_prompt = await load_prompt("qa_system", project_name=project_name)
        if self.settings.structured_output_workaround:
            json_suffix = await load_prompt("qa_json_suffix")
            system_prompt = f"{system_prompt}\n\n{json_suffix}"

        async with MCPFilesystem(
            self.settings.mcp_filesystem_url,
            project_name=project_name,
        ) as fs, MCPRepl(
            self.settings.mcp_repl_url,
            project_name=project_name,
        ) as repl:
            # QA gets all filesystem tools (read + list) and REPL tools
            fs_tools = fs.get_openai_tools()
            repl_tools = repl.get_openai_tools()
            all_tools = fs_tools + repl_tools

            fs_names = {t["function"]["name"] for t in fs_tools}
            repl_names = {t["function"]["name"] for t in repl_tools}

            async def execute(tool_name: str, args: dict) -> str:
                if tool_name in fs_names:
                    return await fs.call_tool(tool_name, args)
                elif tool_name in repl_names:
                    return await repl.call_tool(tool_name, args)
                return f"Unknown tool: {tool_name}"

            runner = AgentRunner(
                system_prompt=system_prompt,
                tools=all_tools,
                tool_executor=execute,
                settings=self.settings,
                on_tool_call=on_tool_call,
            )

            # Pre-load file contents via MCP so the LLM has code inline
            file_contents: list[str] = []
            read_tool = next(
                (t["function"]["name"] for t in fs_tools if "read" in t["function"]["name"].lower()),
                None,
            )
            if read_tool:
                for fpath in code.files_created:
                    try:
                        content = await fs.call_tool(read_tool, {"path": fpath})
                        file_contents.append(f"=== {fpath} ===\n{content}")
                    except Exception:
                        file_contents.append(f"=== {fpath} ===\n[Could not read file]")

            files_section = (
                "\n\n".join(file_contents)
                if file_contents
                else "\n".join(f"- {f}" for f in code.files_created)
            )

            review_request = (
                f"Review the implementation for project '{project_name}'.\n\n"
                f"Specification title: {spec.title}\n"
                f"Requirements:\n" + "\n".join(f"- {r}" for r in spec.requirements) + "\n\n"
                f"Acceptance criteria:\n" + "\n".join(f"- {c}" for c in spec.acceptance_criteria) + "\n\n"
                f"File contents to review:\n{files_section}\n\n"
                f"Developer notes: {code.notes or 'None'}\n"
                f"Developer says tests passed: {code.tests_passed}\n\n"
                f"IMPORTANT: Check each acceptance criterion against the code above. "
                f"If any function has `pass` or is not properly implemented, you MUST give REVISION_NEEDED."
            )

            messages = [{"role": "user", "content": review_request}]

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
                        agent_name="qa",
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

                data = _extract_review_json(result.content)
                if data:
                    try:
                        return ReviewOutput(**data)
                    except Exception as exc:
                        logger.warning(
                            "attempt %d: ReviewOutput validation failed: %s; data=%r",
                            attempt, exc, data,
                        )
                else:
                    logger.warning(
                        "attempt %d: no JSON block in QA response (len=%d): %.300r",
                        attempt, len(result.content), result.content,
                    )

                # Preserve full history; append a single correction so the LLM can
                # either emit the JSON verdict block or run more tool calls.
                # result.messages[0] is the system prompt — runner re-adds it.
                messages = result.messages[1:] + [
                    {
                        "role": "user",
                        "content": (
                            "Your last response did not contain the required JSON verdict block. "
                            "If you need to examine more files or run more tests, do so now. "
                            "Once done, end your response with the JSON block exactly "
                            "as specified in the instructions."
                        ),
                    }
                ]

        raise ValueError("QAAgent failed to produce valid ReviewOutput after 3 attempts")
