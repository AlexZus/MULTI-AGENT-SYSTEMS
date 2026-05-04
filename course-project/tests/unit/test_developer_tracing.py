"""Unit tests for DeveloperAgent — tracing, logging, and retry behaviour.

Covers bugs found in investigation:
  Bug 1+2 — TraceContext never wired: trace_store.add_span() never called
  Bug 3   — Silent JSON-extraction and Pydantic failures: no log output
  Bug 4   — Retry history grows unboundedly across attempts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.agent import AgentResult
from agents.developer import DeveloperAgent
from agents.schemas import SpecOutput

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_JSON_RESPONSE = (
    '```json\n'
    '{"summary": "Done", "files_created": ["proj/main.py"], "tests_passed": true}\n'
    '```'
)
NO_JSON_RESPONSE = "I implemented the code but forgot the JSON block."
INVALID_JSON_RESPONSE = '```json\n{"summary": "Done"}\n```'  # missing files_created


def _make_settings():
    s = MagicMock()
    s.mcp_filesystem_url = "http://localhost:8082/mcp"
    s.mcp_repl_url = "http://localhost:8083/mcp"
    s.structured_output_workaround = False  # skip suffix prompt
    s.max_agent_iterations = 10
    s.max_tool_calls_per_agent = 5
    s.invalid_tool_call_max_retries = 1
    s.openai_compatible_api_url = "http://localhost:8081/v1"
    s.model_name = "test-model"
    s.api_key = "dummy"
    return s


def _make_spec() -> SpecOutput:
    return SpecOutput(
        title="Test Task",
        requirements=["req1"],
        acceptance_criteria=["ac1"],
        estimated_complexity="simple",
    )


def _make_agent_result(content: str, *, tool_calls_in_history: int = 0,
                       input_tokens: int = 0, output_tokens: int = 0) -> AgentResult:
    """Build an AgentResult with optional fake tool-call pairs in the message history."""
    msgs: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(tool_calls_in_history):
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"/f"}'}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "file content"})
    msgs.append({"role": "assistant", "content": content})
    return AgentResult(
        content=content,
        messages=msgs,
        tool_calls_made=tool_calls_in_history,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _patch_mcp(mock_fs_instance=None, mock_repl_instance=None):
    """Return context managers that patch MCPFilesystem and MCPRepl."""
    if mock_fs_instance is None:
        mock_fs_instance = MagicMock()
        mock_fs_instance.get_openai_tools.return_value = []

    if mock_repl_instance is None:
        mock_repl_instance = MagicMock()
        mock_repl_instance.get_openai_tools.return_value = []

    fs_cm = AsyncMock()
    fs_cm.__aenter__ = AsyncMock(return_value=mock_fs_instance)
    fs_cm.__aexit__ = AsyncMock(return_value=False)

    repl_cm = AsyncMock()
    repl_cm.__aenter__ = AsyncMock(return_value=mock_repl_instance)
    repl_cm.__aexit__ = AsyncMock(return_value=False)

    return (
        patch("agents.developer.MCPFilesystem", return_value=fs_cm),
        patch("agents.developer.MCPRepl", return_value=repl_cm),
    )


# ---------------------------------------------------------------------------
# Bug 1+2 — trace_store.add_span() must be called after a successful run
# ---------------------------------------------------------------------------

class TestDeveloperAgentTracing:
    """trace_store.add_span must be called with a populated span after agent completes."""

    async def test_span_saved_on_success(self):
        """add_span() is called once with correct trace_id and agent_name."""
        agent_result = _make_agent_result(VALID_JSON_RESPONSE, input_tokens=120, output_tokens=60)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=agent_result)

        trace_store = AsyncMock()
        event_bus = AsyncMock()

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
        ):
            agent = DeveloperAgent(_make_settings())
            await agent.run(
                _make_spec(),
                project_name="proj",
                trace_store=trace_store,
                event_bus=event_bus,
                trace_id="trace-abc",
                session_id="sess-xyz",
            )

        trace_store.add_span.assert_called_once()

    async def test_span_has_correct_agent_name_and_trace_id(self):
        """The saved span carries agent_name='developer' and the supplied trace_id."""
        agent_result = _make_agent_result(VALID_JSON_RESPONSE, input_tokens=100, output_tokens=40)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=agent_result)

        trace_store = AsyncMock()
        event_bus = AsyncMock()

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
        ):
            agent = DeveloperAgent(_make_settings())
            await agent.run(
                _make_spec(),
                project_name="proj",
                trace_store=trace_store,
                event_bus=event_bus,
                trace_id="trace-abc",
                session_id="sess-xyz",
            )

        _, span_arg = trace_store.add_span.call_args[0]
        assert span_arg.agent_name == "developer"
        assert span_arg.trace_id == "trace-abc"

    async def test_span_carries_token_counts(self):
        """The saved span must include non-zero token counts from AgentResult."""
        agent_result = _make_agent_result(VALID_JSON_RESPONSE, input_tokens=200, output_tokens=80)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=agent_result)

        trace_store = AsyncMock()
        event_bus = AsyncMock()

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
        ):
            agent = DeveloperAgent(_make_settings())
            await agent.run(
                _make_spec(),
                project_name="proj",
                trace_store=trace_store,
                event_bus=event_bus,
                trace_id="t1",
                session_id="s1",
            )

        _, span_arg = trace_store.add_span.call_args[0]
        assert span_arg.input_tokens == 200
        assert span_arg.output_tokens == 80

    async def test_no_crash_when_trace_store_is_none(self):
        """Agent must succeed even without tracing parameters (tracing is optional)."""
        agent_result = _make_agent_result(VALID_JSON_RESPONSE)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=agent_result)

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
        ):
            agent = DeveloperAgent(_make_settings())
            # Should not raise — trace_store=None is valid
            result = await agent.run(_make_spec(), project_name="proj")

        assert result.files_created == ["proj/main.py"]


# ---------------------------------------------------------------------------
# Bug 3 — Silent failure: JSON extraction and Pydantic errors must be logged
# ---------------------------------------------------------------------------

class TestDeveloperAgentLogging:
    """When JSON extraction or schema validation fails, a WARNING must be emitted."""

    async def test_logs_warning_when_no_json_block(self, caplog):
        """A warning with content preview is emitted when no JSON fenced block found."""
        # First call returns no JSON; second returns valid JSON so the run succeeds.
        results = [
            _make_agent_result(NO_JSON_RESPONSE),
            _make_agent_result(VALID_JSON_RESPONSE),
        ]

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=results)

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
            caplog.at_level(logging.WARNING, logger="agents.developer"),
        ):
            agent = DeveloperAgent(_make_settings())
            await agent.run(_make_spec(), project_name="proj")

        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("json" in t.lower() or "attempt" in t.lower() for t in warning_texts), (
            f"Expected a warning about missing JSON, got: {warning_texts}"
        )

    async def test_logs_warning_when_pydantic_validation_fails(self, caplog):
        """A warning is emitted when JSON is found but fails CodeOutput validation."""
        # INVALID_JSON_RESPONSE parses as JSON but is missing 'files_created'
        results = [
            _make_agent_result(INVALID_JSON_RESPONSE),
            _make_agent_result(VALID_JSON_RESPONSE),
        ]

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=results)

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
            caplog.at_level(logging.WARNING, logger="agents.developer"),
        ):
            agent = DeveloperAgent(_make_settings())
            await agent.run(_make_spec(), project_name="proj")

        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "validation" in t.lower() or "codeouptut" in t.lower() or "attempt" in t.lower()
            for t in warning_texts
        ), f"Expected a validation warning, got: {warning_texts}"


# ---------------------------------------------------------------------------
# Retry history — must preserve full context and append one correction
# ---------------------------------------------------------------------------

class TestDeveloperAgentRetryHistory:
    """On JSON-extraction failure the retry preserves the full tool-call history
    and appends a single correction user message. The LLM can then either output
    the JSON block or continue with more tool calls — no context is discarded."""

    async def test_retry_preserves_full_history(self):
        """Second runner.run() receives the entire first-run history plus one correction."""
        result_no_json = _make_agent_result(NO_JSON_RESPONSE, tool_calls_in_history=8)
        result_valid = _make_agent_result(VALID_JSON_RESPONSE)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=[result_no_json, result_valid])

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
        ):
            agent = DeveloperAgent(_make_settings())
            await agent.run(_make_spec(), project_name="proj")

        assert mock_runner.run.call_count == 2

        second_call_messages = mock_runner.run.call_args_list[1][0][0]
        # Full history minus system prompt (runner re-adds it) plus one correction:
        # result_no_json.messages = [system, user, 8×(assistant+tool), assistant(final)]
        #   → strip system → 18 items; append correction → 19
        expected_len = len(result_no_json.messages) - 1 + 1
        assert len(second_call_messages) == expected_len, (
            f"Expected {expected_len} messages (full history + correction), "
            f"got {len(second_call_messages)}: "
            f"{[m.get('role') for m in second_call_messages]}"
        )

    async def test_retry_ends_with_correction_appended_to_history(self):
        """Retry ends with a user correction; the failed assistant response is second-to-last."""
        result_no_json = _make_agent_result(NO_JSON_RESPONSE, tool_calls_in_history=3)
        result_valid = _make_agent_result(VALID_JSON_RESPONSE)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=[result_no_json, result_valid])

        fs_patch, repl_patch = _patch_mcp()
        with (
            patch("tracevault.prompts.load_prompt", new=AsyncMock(return_value="sys")),
            patch("agents.developer.AgentRunner", return_value=mock_runner),
            fs_patch,
            repl_patch,
        ):
            agent = DeveloperAgent(_make_settings())
            await agent.run(_make_spec(), project_name="proj")

        msgs = mock_runner.run.call_args_list[1][0][0]
        # Last message: correction from user
        assert msgs[-1]["role"] == "user"
        # Second-to-last: the failed assistant response
        assert msgs[-2]["role"] == "assistant"
        assert msgs[-2]["content"] == NO_JSON_RESPONSE
        # First message: original user task (system was stripped)
        assert msgs[0]["role"] == "user"
