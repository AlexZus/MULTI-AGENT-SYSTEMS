"""Integration tests for tracevault stores — requires running MongoDB."""

import os
import uuid
from datetime import datetime

import motor.motor_asyncio
import pytest
import pytest_asyncio

from tracevault.models import (
    CriterionResult,
    EvaluationModel,
    PromptModel,
    SpanModel,
    TraceModel,
)
from tracevault.store import EvaluationStore, PromptStore, TraceStore

pytestmark = pytest.mark.asyncio

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin_password@172.20.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "course_project_test")


@pytest_asyncio.fixture
async def db():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    database = client[MONGODB_DB]
    yield database
    # Cleanup test collections
    await database["traces"].drop()
    await database["prompts"].drop()
    await database["evaluations"].drop()
    client.close()


@pytest_asyncio.fixture
async def trace_store(db):
    return TraceStore(db)


@pytest_asyncio.fixture
async def prompt_store(db):
    return PromptStore(db)


@pytest_asyncio.fixture
async def eval_store(db):
    return EvaluationStore(db)


# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------

class TestTraceStore:
    async def test_create_and_get(self, trace_store):
        tid = uuid.uuid4().hex
        trace = TraceModel(trace_id=tid, session_id="s1", project_name="calc")
        await trace_store.create_trace(trace)
        fetched = await trace_store.get_trace(tid)
        assert fetched is not None
        assert fetched.trace_id == tid

    async def test_get_nonexistent_returns_none(self, trace_store):
        result = await trace_store.get_trace("no-such-trace")
        assert result is None

    async def test_update_trace(self, trace_store):
        tid = uuid.uuid4().hex
        await trace_store.create_trace(TraceModel(trace_id=tid, session_id="s", project_name="p"))
        await trace_store.update_trace(tid, status="completed", verdict="APPROVED")
        fetched = await trace_store.get_trace(tid)
        assert fetched.status == "completed"
        assert fetched.verdict == "APPROVED"

    async def test_add_span(self, trace_store):
        tid = uuid.uuid4().hex
        await trace_store.create_trace(TraceModel(trace_id=tid, session_id="s", project_name="p"))
        span = SpanModel(
            span_id=uuid.uuid4().hex,
            trace_id=tid,
            agent_name="ba",
            input_tokens=50,
            output_tokens=100,
        )
        await trace_store.add_span(tid, span)
        fetched = await trace_store.get_trace(tid)
        assert len(fetched.spans) == 1
        assert fetched.spans[0].agent_name == "ba"
        assert fetched.total_tokens == 150

    async def test_list_traces_filter_status(self, trace_store):
        for status in ("running", "completed", "failed"):
            tid = uuid.uuid4().hex
            await trace_store.create_trace(
                TraceModel(trace_id=tid, session_id="s", project_name="p", status=status)
            )
        running = await trace_store.list_traces(status="running")
        assert all(t.status == "running" for t in running)

    async def test_list_traces_filter_session(self, trace_store):
        sid = uuid.uuid4().hex
        for _ in range(3):
            await trace_store.create_trace(
                TraceModel(trace_id=uuid.uuid4().hex, session_id=sid, project_name="p")
            )
        await trace_store.create_trace(
            TraceModel(trace_id=uuid.uuid4().hex, session_id="other", project_name="p")
        )
        results = await trace_store.list_traces(session_id=sid)
        assert len(results) == 3

    async def test_list_sessions(self, trace_store):
        sid = uuid.uuid4().hex
        for i in range(2):
            await trace_store.create_trace(
                TraceModel(
                    trace_id=uuid.uuid4().hex,
                    session_id=sid,
                    project_name="proj",
                    status="completed",
                )
            )
        sessions = await trace_store.list_sessions()
        session = next((s for s in sessions if s["session_id"] == sid), None)
        assert session is not None
        assert session["traces_count"] == 2


# ---------------------------------------------------------------------------
# PromptStore
# ---------------------------------------------------------------------------

class TestPromptStore:
    async def test_upsert_and_get(self, prompt_store):
        p = PromptModel(name="test_prompt", template="Hello {name}", variables=["name"])
        await prompt_store.upsert_prompt(p)
        fetched = await prompt_store.get_prompt("test_prompt")
        assert fetched is not None
        assert fetched.template == "Hello {name}"
        assert fetched.version == 1

    async def test_upsert_increments_version(self, prompt_store):
        name = f"prompt_{uuid.uuid4().hex[:6]}"
        p = PromptModel(name=name, template="v1")
        await prompt_store.upsert_prompt(p)
        p2 = PromptModel(name=name, template="v2")
        result = await prompt_store.upsert_prompt(p2)
        assert result.version == 2
        assert result.template == "v2"

    async def test_upsert_archives_history(self, prompt_store):
        name = f"prompt_{uuid.uuid4().hex[:6]}"
        await prompt_store.upsert_prompt(PromptModel(name=name, template="v1"))
        await prompt_store.upsert_prompt(PromptModel(name=name, template="v2"))
        fetched = await prompt_store.get_prompt(name)
        assert len(fetched.history) == 1
        assert fetched.history[0]["version"] == 1
        assert fetched.history[0]["template"] == "v1"

    async def test_list_prompts(self, prompt_store):
        prefix = uuid.uuid4().hex[:6]
        for i in range(3):
            await prompt_store.upsert_prompt(PromptModel(name=f"{prefix}_p{i}", template=f"t{i}"))
        prompts = await prompt_store.list_prompts()
        names = {p.name for p in prompts}
        for i in range(3):
            assert f"{prefix}_p{i}" in names

    async def test_rollback(self, prompt_store):
        name = f"prompt_{uuid.uuid4().hex[:6]}"
        await prompt_store.upsert_prompt(PromptModel(name=name, template="v1 content"))
        await prompt_store.upsert_prompt(PromptModel(name=name, template="v2 content"))
        await prompt_store.upsert_prompt(PromptModel(name=name, template="v3 content"))
        # Rollback to v1
        result = await prompt_store.rollback_prompt(name, 1)
        assert result is not None
        assert result.template == "v1 content"
        assert result.version == 4  # new version created

    async def test_rollback_invalid_version(self, prompt_store):
        name = f"prompt_{uuid.uuid4().hex[:6]}"
        await prompt_store.upsert_prompt(PromptModel(name=name, template="only one version"))
        result = await prompt_store.rollback_prompt(name, 99)
        assert result is None

    async def test_delete_prompt(self, prompt_store):
        name = f"prompt_{uuid.uuid4().hex[:6]}"
        await prompt_store.upsert_prompt(PromptModel(name=name, template="to delete"))
        deleted = await prompt_store.delete_prompt(name)
        assert deleted is True
        assert await prompt_store.get_prompt(name) is None

    async def test_delete_nonexistent(self, prompt_store):
        deleted = await prompt_store.delete_prompt("no-such-prompt")
        assert deleted is False


# ---------------------------------------------------------------------------
# EvaluationStore
# ---------------------------------------------------------------------------

class TestEvaluationStore:
    def _make_eval(self, agent_name="ba", verdict="pass", score=0.9):
        return EvaluationModel(
            eval_id=uuid.uuid4().hex,
            trace_id=uuid.uuid4().hex,
            session_id=uuid.uuid4().hex,
            agent_name=agent_name,
            overall_score=score,
            verdict=verdict,
            criteria=[
                CriterionResult(name="c1", passed=(verdict == "pass"), score=score),
            ],
        )

    async def test_save_and_list(self, eval_store):
        ev = self._make_eval()
        await eval_store.save_evaluation(ev)
        results = await eval_store.list_evaluations()
        ids = [r.eval_id for r in results]
        assert ev.eval_id in ids

    async def test_list_filter_agent(self, eval_store):
        for agent in ("ba", "developer", "qa"):
            await eval_store.save_evaluation(self._make_eval(agent_name=agent))
        ba_evals = await eval_store.list_evaluations(agent_name="ba")
        assert all(e.agent_name == "ba" for e in ba_evals)

    async def test_list_filter_session(self, eval_store):
        sid = uuid.uuid4().hex
        ev = EvaluationModel(
            eval_id=uuid.uuid4().hex,
            trace_id="t",
            session_id=sid,
            agent_name="ba",
            overall_score=0.9,
            verdict="pass",
        )
        await eval_store.save_evaluation(ev)
        results = await eval_store.list_evaluations(session_id=sid)
        assert len(results) == 1
        assert results[0].session_id == sid

    async def test_agent_stats(self, eval_store):
        agent = f"agent_{uuid.uuid4().hex[:6]}"
        for i in range(4):
            verdict = "pass" if i < 3 else "fail"
            score = 0.9 if i < 3 else 0.4
            await eval_store.save_evaluation(self._make_eval(agent_name=agent, verdict=verdict, score=score))
        stats = await eval_store.get_agent_stats(agent_name=agent)
        assert len(stats) == 1
        s = stats[0]
        assert s["count"] == 4
        assert s["pass_rate"] == pytest.approx(0.75, abs=0.01)
        assert s["avg_score"] == pytest.approx(0.775, abs=0.01)
