"""Unit tests for tracevault models — no external services required."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from tracevault.models import (
    CriterionResult,
    EvaluationModel,
    PromptModel,
    SpanModel,
    TraceModel,
)


class TestSpanModel:
    def test_valid_span(self):
        span = SpanModel(
            span_id="s1",
            trace_id="t1",
            agent_name="ba",
            input_tokens=100,
            output_tokens=200,
        )
        assert span.agent_name == "ba"
        assert span.tool_calls == []
        assert isinstance(span.timestamp, datetime)

    def test_defaults(self):
        span = SpanModel(span_id="s1", trace_id="t1", agent_name="qa")
        assert span.iteration == 0
        assert span.latency_ms == 0
        assert span.tags == []

    def test_tool_calls_list(self):
        span = SpanModel(
            span_id="s1",
            trace_id="t1",
            agent_name="developer",
            tool_calls=[{"name": "read_file", "args": {}, "result": "ok", "latency_ms": 50}],
        )
        assert len(span.tool_calls) == 1
        assert span.tool_calls[0]["name"] == "read_file"

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            SpanModel(trace_id="t1", agent_name="ba")  # missing span_id

    def test_model_dump_roundtrip(self):
        span = SpanModel(span_id="s1", trace_id="t1", agent_name="ba", input_tokens=50)
        restored = SpanModel(**span.model_dump())
        assert restored == span


class TestTraceModel:
    def test_valid_trace(self):
        trace = TraceModel(
            trace_id="t1",
            session_id="sess1",
            project_name="calculator",
            user_story="Add two numbers",
        )
        assert trace.status == "running"
        assert trace.spans == []
        assert trace.total_tokens == 0

    def test_valid_statuses(self):
        for s in ("running", "waiting_hitl", "completed", "failed"):
            t = TraceModel(trace_id="t", session_id="s", project_name="p", status=s)
            assert t.status == s

    def test_trace_with_spans(self):
        span = SpanModel(span_id="s1", trace_id="t1", agent_name="ba")
        trace = TraceModel(
            trace_id="t1", session_id="s", project_name="p", spans=[span]
        )
        assert len(trace.spans) == 1

    def test_model_dump_roundtrip(self):
        trace = TraceModel(trace_id="t1", session_id="s", project_name="p")
        restored = TraceModel(**trace.model_dump())
        assert restored == trace


class TestPromptModel:
    def test_valid_prompt(self):
        p = PromptModel(name="ba_system", template="You are a BA for {project_name}.")
        assert p.version == 1
        assert p.history == []

    def test_variables_default_empty(self):
        p = PromptModel(name="x", template="No vars here")
        assert p.variables == []

    def test_with_history(self):
        p = PromptModel(
            name="x",
            template="v2",
            version=2,
            history=[{"version": 1, "template": "v1", "updated_at": "2026-01-01T00:00:00"}],
        )
        assert len(p.history) == 1

    def test_model_dump_roundtrip(self):
        p = PromptModel(name="x", template="hello {name}", variables=["name"])
        restored = PromptModel(**p.model_dump())
        assert restored == p


class TestCriterionResult:
    def test_valid(self):
        c = CriterionResult(name="has_tests", passed=True, score=1.0, reasoning="Tests found")
        assert c.passed is True

    def test_reasoning_default(self):
        c = CriterionResult(name="x", passed=False, score=0.0)
        assert c.reasoning == ""


class TestEvaluationModel:
    def _valid(self, **overrides):
        data = {
            "eval_id": "eval1",
            "trace_id": "t1",
            "session_id": "s1",
            "agent_name": "ba",
            "overall_score": 0.85,
            "verdict": "pass",
        }
        data.update(overrides)
        return data

    def test_valid_pass(self):
        ev = EvaluationModel(**self._valid())
        assert ev.verdict == "pass"
        assert ev.overall_score == 0.85
        assert ev.evaluator == "llm-judge"

    def test_valid_fail(self):
        ev = EvaluationModel(**self._valid(verdict="fail", overall_score=0.4))
        assert ev.verdict == "fail"

    def test_with_criteria(self):
        criteria = [
            CriterionResult(name="c1", passed=True, score=1.0),
            CriterionResult(name="c2", passed=False, score=0.5),
        ]
        ev = EvaluationModel(**self._valid(criteria=criteria))
        assert len(ev.criteria) == 2

    def test_missing_required_raises(self):
        data = self._valid()
        del data["trace_id"]
        with pytest.raises(ValidationError):
            EvaluationModel(**data)

    def test_model_dump_roundtrip(self):
        ev = EvaluationModel(**self._valid())
        restored = EvaluationModel(**ev.model_dump())
        assert restored == ev

    def test_sse_serialization(self):
        """Model can be serialized to JSON (for SSE events)."""
        ev = EvaluationModel(**self._valid())
        dumped = ev.model_dump()
        serialized = json.dumps(dumped, default=str)
        assert "eval1" in serialized
