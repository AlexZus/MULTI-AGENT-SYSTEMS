"""Live test for BA agent — calls real LLM and saves evaluation to tracevault."""

import asyncio
import os
import uuid
from datetime import datetime

import motor.motor_asyncio
import pytest

from agents.ba import BAAgent
from config import Settings
from tracevault.models import CriterionResult, EvaluationModel
from tracevault.store import EvaluationStore

pytestmark = pytest.mark.asyncio

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin_password@172.20.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "course_project")


async def llm_judge_spec(spec, settings: Settings) -> dict:
    """Use the local LLM to evaluate the spec output."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=settings.openai_compatible_api_url,
        api_key=settings.api_key,
    )

    prompt = f"""Evaluate this software specification:

Title: {spec.title}
Requirements: {spec.requirements}
Acceptance Criteria: {spec.acceptance_criteria}
Complexity: {spec.estimated_complexity}

Score each criterion 0.0-1.0:
1. has_enough_requirements: at least 3 requirements
2. has_acceptance_criteria: at least 2 acceptance criteria
3. mentions_validation: at least one requirement mentions validation or error handling
4. is_specific: requirements are specific and testable, not vague

Respond with JSON only:
{{"criteria": [{{"name": "...", "passed": true/false, "score": 0.0-1.0, "reasoning": "..."}}], "overall_score": 0.0-1.0}}"""

    import json
    import re

    response = await client.chat.completions.create(
        model=settings.model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        return json.loads(match.group(0))
    return {"criteria": [], "overall_score": 0.0}


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def eval_store(settings):
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB]
    return EvaluationStore(db)


async def test_ba_produces_valid_spec(settings):
    """BA agent must produce a valid SpecOutput for a simple user story."""
    agent = BAAgent(settings)

    # Seed prompts first
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB]
    from tracevault.store import PromptStore
    from tracevault.prompts import set_prompt_store, seed_from_files
    from pathlib import Path

    store = PromptStore(db)
    set_prompt_store(store)
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    await seed_from_files(prompts_dir, store)

    spec = await agent.run(
        "Build a simple calculator that supports addition, subtraction, multiplication, and division",
        project_name="calculator",
    )
    client.close()

    assert spec.title
    assert len(spec.requirements) >= 3, f"Expected >=3 requirements, got {spec.requirements}"
    assert len(spec.acceptance_criteria) >= 2, f"Expected >=2 criteria, got {spec.acceptance_criteria}"
    assert spec.estimated_complexity in ("simple", "medium", "complex")


async def test_ba_with_judge(settings, eval_store):
    """Run BA agent and save LLM-as-Judge evaluation to tracevault."""
    # Seed prompts
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB]
    from tracevault.store import PromptStore
    from tracevault.prompts import set_prompt_store, seed_from_files
    from pathlib import Path

    store = PromptStore(db)
    set_prompt_store(store)
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    await seed_from_files(prompts_dir, store)

    agent = BAAgent(settings)
    trace_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex

    spec = await agent.run(
        "Build a REST API for managing a to-do list with CRUD operations and user authentication",
        project_name="todo-api",
    )
    client.close()

    # Judge the spec
    judge_result = await llm_judge_spec(spec, settings)

    criteria = [
        CriterionResult(
            name=c["name"],
            passed=c["passed"],
            score=c["score"],
            reasoning=c.get("reasoning", ""),
        )
        for c in judge_result.get("criteria", [])
    ]
    overall_score = judge_result.get("overall_score", 0.0)

    evaluation = EvaluationModel(
        eval_id=uuid.uuid4().hex,
        trace_id=trace_id,
        session_id=session_id,
        agent_name="ba",
        evaluator="llm-judge",
        criteria=criteria,
        overall_score=overall_score,
        verdict="pass" if overall_score >= 0.7 else "fail",
    )
    await eval_store.save_evaluation(evaluation)

    assert overall_score >= 0.7, (
        f"BA agent score {overall_score:.2f} below threshold 0.7. "
        f"Criteria: {[(c.name, c.passed, c.score) for c in criteria]}"
    )
