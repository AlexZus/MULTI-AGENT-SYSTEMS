"""Unit tests for agentflow components — no external services required."""

import json
import pytest

from agentflow.agent import _try_fix_tool_call, _build_schema_lookup
from agentflow.middleware import BudgetMiddleware, InvalidToolCallRetryMiddleware


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# _try_fix_tool_call
# ---------------------------------------------------------------------------

class TestTryFixToolCall:
    def _lookup(self):
        return _build_schema_lookup(TOOLS)

    def test_passthrough_when_tool_calls_present(self):
        """Does not modify messages that already have tool_calls."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/foo"}'}}],
        }
        result = _try_fix_tool_call(msg, self._lookup())
        assert result is msg  # same object, unchanged

    def test_passthrough_non_json_content(self):
        msg = {"role": "assistant", "content": "Hello, world!"}
        result = _try_fix_tool_call(msg, self._lookup())
        assert result["content"] == "Hello, world!"
        assert not result.get("tool_calls")

    def test_fixes_json_in_content_single_arg(self):
        """Detects read_file args in content and creates a synthetic tool_call."""
        msg = {"role": "assistant", "content": '{"path": "/workspace/foo.py"}'}
        result = _try_fix_tool_call(msg, self._lookup())
        assert result.get("tool_calls"), "Expected tool_calls to be set"
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"])["path"] == "/workspace/foo.py"
        assert result["content"] == ""

    def test_fixes_json_in_content_two_args(self):
        """Detects write_file args in content (two required params)."""
        msg = {
            "role": "assistant",
            "content": '{"path": "/workspace/out.py", "content": "print(1)"}',
        }
        result = _try_fix_tool_call(msg, self._lookup())
        assert result["tool_calls"][0]["function"]["name"] == "write_file"

    def test_no_fix_for_unknown_keys(self):
        """JSON with keys that match no known tool is left unchanged."""
        msg = {"role": "assistant", "content": '{"unknown_key": "value"}'}
        result = _try_fix_tool_call(msg, self._lookup())
        assert not result.get("tool_calls")

    def test_no_fix_for_invalid_json(self):
        msg = {"role": "assistant", "content": "{not valid json}"}
        result = _try_fix_tool_call(msg, self._lookup())
        assert not result.get("tool_calls")

    def test_no_fix_for_empty_content(self):
        msg = {"role": "assistant", "content": ""}
        result = _try_fix_tool_call(msg, self._lookup())
        assert not result.get("tool_calls")

    def test_no_fix_for_none_content(self):
        msg = {"role": "assistant", "content": None}
        result = _try_fix_tool_call(msg, self._lookup())
        assert not result.get("tool_calls")


# ---------------------------------------------------------------------------
# BudgetMiddleware
# ---------------------------------------------------------------------------

class TestBudgetMiddleware:
    def test_initial_remaining(self):
        bm = BudgetMiddleware(max_tool_calls=5)
        bm.reset()
        assert bm.remaining == 5

    def test_decrements_on_wrap(self):
        bm = BudgetMiddleware(max_tool_calls=5)
        bm.reset()
        bm.wrap_tool_result("result")
        assert bm.remaining == 4

    def test_wraps_with_xml(self):
        bm = BudgetMiddleware(max_tool_calls=10)
        bm.reset()
        wrapped = bm.wrap_tool_result("tool output")
        assert "<tool_call_output>" in wrapped
        assert "tool output" in wrapped
        assert "<tool_call_limits_info>" in wrapped

    def test_exhausted_message(self):
        bm = BudgetMiddleware(max_tool_calls=1)
        bm.reset()
        wrapped = bm.wrap_tool_result("result")
        assert bm.remaining == 0
        assert "MUST provide the final answer" in wrapped

    def test_one_remaining_message(self):
        bm = BudgetMiddleware(max_tool_calls=2)
        bm.reset()
        bm.wrap_tool_result("r1")  # remaining → 1
        wrapped = bm.wrap_tool_result("r2")  # remaining → 0
        assert "all tool call budget" in wrapped.lower() or "1 tool call remaining" in wrapped or "MUST" in wrapped

    def test_reset_restores_budget(self):
        bm = BudgetMiddleware(max_tool_calls=3)
        bm.reset()
        for _ in range(3):
            bm.wrap_tool_result("x")
        assert bm.remaining == 0
        bm.reset()
        assert bm.remaining == 3


# ---------------------------------------------------------------------------
# InvalidToolCallRetryMiddleware
# ---------------------------------------------------------------------------

class TestInvalidToolCallRetryMiddleware:
    def _valid_tc(self, name="read_file", args='{"path": "/foo"}'):
        return {
            "id": "call_abc",
            "type": "function",
            "function": {"name": name, "arguments": args},
        }

    def _invalid_tc(self, name="read_file", args="{bad json"):
        return {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": name, "arguments": args},
        }

    def test_no_correction_for_valid_json(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=3)
        assert mw.check_and_build_correction([self._valid_tc()]) is None

    def test_correction_for_invalid_json(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=3)
        correction = mw.check_and_build_correction([self._invalid_tc()])
        assert correction is not None
        assert "invalid JSON" in correction
        assert "read_file" in correction

    def test_correction_includes_raw_args(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=3)
        correction = mw.check_and_build_correction([self._invalid_tc(args="{garbage}")])
        assert "{garbage}" in correction

    def test_respects_max_retries(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=2)
        mw.check_and_build_correction([self._invalid_tc()])  # retry 1
        mw.check_and_build_correction([self._invalid_tc()])  # retry 2
        # Third call exceeds limit → None
        assert mw.check_and_build_correction([self._invalid_tc()]) is None

    def test_no_correction_for_empty_tool_calls(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=3)
        assert mw.check_and_build_correction([]) is None

    def test_reset_clears_retry_count(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=1)
        mw.check_and_build_correction([self._invalid_tc()])  # retry 1 → exhausted
        assert mw.check_and_build_correction([self._invalid_tc()]) is None
        mw.reset()
        correction = mw.check_and_build_correction([self._invalid_tc()])
        assert correction is not None

    def test_mixed_valid_invalid(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=3)
        tool_calls = [self._valid_tc(), self._invalid_tc(name="write_file")]
        correction = mw.check_and_build_correction(tool_calls)
        assert correction is not None
        assert "write_file" in correction
        assert "read_file" not in correction  # valid one shouldn't appear

    def test_retries_left_decrements(self):
        mw = InvalidToolCallRetryMiddleware(max_retries=3)
        assert mw.retries_left == 3
        mw.check_and_build_correction([self._invalid_tc()])
        assert mw.retries_left == 2


# ---------------------------------------------------------------------------
# Bug 5 — _build_schema_lookup collision: two tools sharing the same required
# param set must each still be detectable by _try_fix_tool_call
# ---------------------------------------------------------------------------

def _make_tool(name: str, required: list[str], optional: dict | None = None) -> dict:
    props = {k: {"type": "string"} for k in required}
    if optional:
        props.update({k: {"type": v} for k, v in optional.items()})
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


class TestSchemaLookupCollision:
    """Two tools with the same required params but different optional params
    must map to distinct entries so _try_fix_tool_call picks the right one."""

    def _collision_tools(self):
        # Both require only "query"; optional params differ → previously collided
        return [
            _make_tool("web_search", required=["query"], optional={"max_results": "integer"}),
            _make_tool("knowledge_search", required=["query"], optional={"top_k": "integer"}),
        ]

    def test_web_search_detected_by_full_args(self):
        """Content with query+max_results must resolve to web_search, not knowledge_search."""
        lookup = _build_schema_lookup(self._collision_tools())
        msg = {"role": "assistant", "content": '{"query": "python", "max_results": 5}'}
        result = _try_fix_tool_call(msg, lookup)
        assert result.get("tool_calls"), "Expected tool_calls to be set"
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_knowledge_search_detected_by_full_args(self):
        """Content with query+top_k must resolve to knowledge_search."""
        lookup = _build_schema_lookup(self._collision_tools())
        msg = {"role": "assistant", "content": '{"query": "fastapi", "top_k": 3}'}
        result = _try_fix_tool_call(msg, lookup)
        assert result.get("tool_calls"), "Expected tool_calls to be set"
        assert result["tool_calls"][0]["function"]["name"] == "knowledge_search"

    def test_lookup_contains_both_tools(self):
        """Schema lookup must have an entry for each distinct tool."""
        lookup = _build_schema_lookup(self._collision_tools())
        tool_names = set(lookup.values())
        assert "web_search" in tool_names
        assert "knowledge_search" in tool_names
