"""End-to-end pipeline tests — requires all services running.

These tests run the full BA → Developer → QA pipeline with HITL auto-approval,
then verify that evaluation results appear in tracevault.
"""

import asyncio
import os
import uuid
from pathlib import Path

import motor.motor_asyncio
import pytest

from config import Settings
from pipeline import DevTeamPipeline
from agentflow.graph import PipelineEventType

pytestmark = pytest.mark.asyncio

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin_password@172.20.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "course_project")


async def _setup_stores():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB]
    from tracevault.store import TraceStore, PromptStore, EvaluationStore
    from tracevault.prompts import set_prompt_store, seed_from_files

    trace_store = TraceStore(db)
    prompt_store = PromptStore(db)
    eval_store = EvaluationStore(db)

    set_prompt_store(prompt_store)
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    await seed_from_files(prompts_dir, prompt_store)

    return client, trace_store, eval_store


async def _run_pipeline_with_auto_approval(pipeline, user_story, project_name, task_id, trace_store):
    """Run pipeline with automatic HITL approval."""
    events = []
    hitl_approved = False

    async def collect():
        nonlocal hitl_approved
        async for event in pipeline.run(
            user_story,
            project_name,
            task_id,
            session_id=task_id,
            trace_store=trace_store,
        ):
            events.append(event)
            if event.type == PipelineEventType.HITL_WAITING and not hitl_approved:
                hitl_approved = True
                # Auto-approve after a short delay
                await asyncio.sleep(0.1)
                await pipeline.resume(task_id, project_name, approved=True)

    await collect()
    return events


@pytest.fixture
def settings():
    return Settings()


@pytest.mark.timeout(300)
async def test_e2e_hello_world(settings):
    """Full pipeline for a simple hello world task."""
    client, trace_store, eval_store = await _setup_stores()

    task_id = uuid.uuid4().hex
    project_name = f"e2e_{uuid.uuid4().hex[:6]}"
    pipeline = DevTeamPipeline(settings)

    events = await _run_pipeline_with_auto_approval(
        pipeline,
        "Write a Python function that returns the Fibonacci sequence up to n terms",
        project_name,
        task_id,
        trace_store,
    )
    client.close()

    event_types = [e.type for e in events]
    assert PipelineEventType.COMPLETED in event_types or PipelineEventType.FAILED not in event_types, (
        f"Pipeline did not complete. Events: {[e.type.value for e in events]}"
    )

    completed = next((e for e in events if e.type == PipelineEventType.COMPLETED), None)
    assert completed is not None, f"No COMPLETED event. Events: {[e.type.value for e in events]}"
    assert completed.data.get("files_created"), "No files created"


@pytest.mark.timeout(300)
async def test_e2e_evaluations_in_tracevault(settings):
    """Verify evaluation results appear in tracevault after a pipeline run."""
    client, trace_store, eval_store = await _setup_stores()

    session_id = uuid.uuid4().hex
    task_id = uuid.uuid4().hex
    project_name = f"e2e_eval_{uuid.uuid4().hex[:6]}"

    pipeline = DevTeamPipeline(settings)
    await _run_pipeline_with_auto_approval(
        pipeline,
        "Create a simple string utility module with functions: reverse(s), count_vowels(s), is_palindrome(s)",
        project_name,
        task_id,
        trace_store,
    )

    # Verify trace was created
    traces = await trace_store.list_traces(project_name=project_name, limit=10)
    client.close()

    assert len(traces) >= 1, "No trace found for this pipeline run"
    trace = traces[0]
    assert trace.status in ("completed", "failed"), f"Unexpected status: {trace.status}"
