"""Live test for Developer agent — calls real LLM + MCP tools."""

import os
import uuid
from pathlib import Path

import motor.motor_asyncio
import pytest

from agents.ba import BAAgent
from agents.developer import DeveloperAgent
from agents.schemas import SpecOutput
from config import Settings
from tracevault.models import CriterionResult, EvaluationModel
from tracevault.store import EvaluationStore

pytestmark = pytest.mark.asyncio

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin_password@172.20.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "course_project")


async def _seed_prompts(settings):
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB]
    from tracevault.store import PromptStore
    from tracevault.prompts import set_prompt_store, seed_from_files

    store = PromptStore(db)
    set_prompt_store(store)
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    await seed_from_files(prompts_dir, store)
    return client


async def llm_judge_code(code, spec, settings) -> dict:
    from openai import AsyncOpenAI
    import json, re

    client = AsyncOpenAI(
        base_url=settings.openai_compatible_api_url,
        api_key=settings.api_key,
    )
    prompt = f"""Evaluate this code implementation:

Files created: {code.files_created}
Summary: {code.summary}
Tests passed: {code.tests_passed}

Specification title: {spec.title}
Requirements: {spec.requirements}

Score each criterion 0.0-1.0:
1. files_created: at least one .py file was created
2. covers_requirements: implementation appears to address the requirements
3. has_tests: at least one test file exists
4. tests_passing: developer reports tests passed

Respond with JSON only:
{{"criteria": [{{"name": "...", "passed": true/false, "score": 0.0-1.0, "reasoning": "..."}}], "overall_score": 0.0-1.0}}"""

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


async def test_developer_creates_files(settings):
    """Developer agent must create at least one .py file."""
    client = await _seed_prompts(settings)

    spec = SpecOutput(
        title="Hello World Script",
        requirements=[
            "Create a Python script that prints 'Hello, World!'",
            "The script should accept an optional name argument",
            "If name is provided, print 'Hello, {name}!'",
        ],
        acceptance_criteria=[
            "Running the script prints a greeting",
            "Script handles missing name gracefully",
        ],
        estimated_complexity="simple",
    )

    project_name = f"test_hello_{uuid.uuid4().hex[:6]}"
    agent = DeveloperAgent(settings)
    code = await agent.run(spec, project_name=project_name)
    client.close()

    assert len(code.files_created) >= 1
    py_files = [f for f in code.files_created if f.endswith(".py")]
    assert len(py_files) >= 1, f"No .py files in {code.files_created}"


async def test_developer_with_judge(settings):
    """Run developer + LLM judge, save evaluation to tracevault."""
    client = await _seed_prompts(settings)
    db = client[MONGODB_DB]
    from tracevault.store import EvaluationStore as ES
    eval_store = ES(db)

    spec = SpecOutput(
        title="Calculator Module",
        requirements=[
            "Implement add(a, b) returning the sum",
            "Implement subtract(a, b) returning the difference",
            "Implement multiply(a, b) returning the product",
            "Implement divide(a, b) raising ValueError for b=0",
        ],
        acceptance_criteria=[
            "add(2, 3) returns 5",
            "divide(1, 0) raises ValueError",
        ],
        estimated_complexity="simple",
    )

    project_name = f"test_calc_{uuid.uuid4().hex[:6]}"
    trace_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex

    agent = DeveloperAgent(settings)
    code = await agent.run(spec, project_name=project_name)

    judge_result = await llm_judge_code(code, spec, settings)
    client.close()

    criteria = [
        CriterionResult(
            name=c["name"], passed=c["passed"], score=c["score"],
            reasoning=c.get("reasoning", ""),
        )
        for c in judge_result.get("criteria", [])
    ]
    overall_score = judge_result.get("overall_score", 0.0)

    evaluation = EvaluationModel(
        eval_id=uuid.uuid4().hex,
        trace_id=trace_id,
        session_id=session_id,
        agent_name="developer",
        evaluator="llm-judge",
        criteria=criteria,
        overall_score=overall_score,
        verdict="pass" if overall_score >= 0.7 else "fail",
    )
    # Re-open for save
    client2 = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db2 = client2[MONGODB_DB]
    await ES(db2).save_evaluation(evaluation)
    client2.close()

    assert overall_score >= 0.7, (
        f"Developer score {overall_score:.2f} below 0.7. Files: {code.files_created}"
    )
