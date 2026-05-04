"""Live test for QA agent — submit deliberately bad code, expect REVISION_NEEDED."""

import os
import uuid
from pathlib import Path

import motor.motor_asyncio
import pytest

from agents.qa import QAAgent
from agents.schemas import CodeOutput, SpecOutput
from config import Settings
from tracevault.models import CriterionResult, EvaluationModel

pytestmark = pytest.mark.asyncio

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin_password@172.20.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "course_project")


async def _seed_prompts():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB]
    from tracevault.store import PromptStore
    from tracevault.prompts import set_prompt_store, seed_from_files

    store = PromptStore(db)
    set_prompt_store(store)
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    await seed_from_files(prompts_dir, store)
    return client


async def llm_judge_review(review, settings) -> dict:
    from openai import AsyncOpenAI
    import json, re

    client = AsyncOpenAI(
        base_url=settings.openai_compatible_api_url,
        api_key=settings.api_key,
    )
    prompt = f"""Evaluate this QA review:

Verdict: {review.verdict}
Score: {review.score}
Issues: {review.issues}
Tests run: {review.tests_run}, passed: {review.tests_passed}

Score each criterion 0.0-1.0:
1. correct_verdict: verdict is REVISION_NEEDED (bad code should fail)
2. has_issues: at least 2 actionable issues identified
3. actionable_feedback: issues are specific and actionable

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


async def test_qa_flags_bad_code(settings):
    """QA agent must return REVISION_NEEDED when given deliberately broken code."""
    client = await _seed_prompts()

    # Write deliberately bad code to the workspace via MCP
    project_name = f"test_qa_{uuid.uuid4().hex[:6]}"
    bad_code = '''def divide(a, b):
    return a / b  # no zero division handling

def add(a, b):
    pass  # not implemented
'''
    from tools.mcp_fs import MCPFilesystem
    async with MCPFilesystem(settings.mcp_filesystem_url, project_name=project_name) as fs:
        tool_names = {t["function"]["name"] for t in fs.get_openai_tools()}
        mkdir = next((n for n in tool_names if "creat" in n.lower() and "dir" in n.lower()), None)
        write = next((n for n in tool_names if "write" in n.lower()), None)
        if mkdir:
            await fs.call_tool(mkdir, {"path": project_name})
        if write:
            await fs.call_tool(write, {"path": f"{project_name}/calculator.py", "content": bad_code})

    spec = SpecOutput(
        title="Calculator",
        requirements=[
            "Implement add(a, b) returning the sum",
            "Implement divide(a, b) raising ValueError for b=0",
            "All functions must be properly implemented",
        ],
        acceptance_criteria=[
            "add(2, 3) returns 5",
            "divide(1, 0) raises ValueError",
        ],
        estimated_complexity="simple",
    )
    code = CodeOutput(
        summary="Implemented calculator",
        files_created=[f"{project_name}/calculator.py"],
        tests_passed=False,
    )

    qa = QAAgent(settings)
    review = await qa.run(spec, code, project_name=project_name)
    client.close()

    assert review.verdict == "REVISION_NEEDED", (
        f"Expected REVISION_NEEDED, got {review.verdict}. Issues: {review.issues}"
    )
    assert len(review.issues) >= 2, f"Expected >=2 issues, got {review.issues}"


async def test_qa_with_judge(settings):
    """Run QA on bad code + LLM judge the review."""
    client = await _seed_prompts()
    db = client[MONGODB_DB]
    from tracevault.store import EvaluationStore

    project_name = f"test_qa2_{uuid.uuid4().hex[:6]}"
    bad_code = "def foo(): pass  # does nothing"

    from tools.mcp_fs import MCPFilesystem
    async with MCPFilesystem(settings.mcp_filesystem_url, project_name=project_name) as fs:
        tool_names = {t["function"]["name"] for t in fs.get_openai_tools()}
        mkdir = next((n for n in tool_names if "creat" in n.lower() and "dir" in n.lower()), None)
        write = next((n for n in tool_names if "write" in n.lower()), None)
        if mkdir:
            await fs.call_tool(mkdir, {"path": project_name})
        if write:
            await fs.call_tool(write, {"path": f"{project_name}/main.py", "content": bad_code})

    spec = SpecOutput(
        title="Test",
        requirements=["Implement foo() that returns 42", "Add docstring", "Write unit test"],
        acceptance_criteria=["foo() returns 42", "test passes"],
        estimated_complexity="simple",
    )
    code = CodeOutput(
        summary="stub", files_created=[f"{project_name}/main.py"], tests_passed=False
    )

    qa = QAAgent(settings)
    review = await qa.run(spec, code, project_name=project_name)

    judge_result = await llm_judge_review(review, settings)
    client.close()

    criteria = [
        CriterionResult(
            name=c["name"], passed=c["passed"], score=c["score"],
            reasoning=c.get("reasoning", ""),
        )
        for c in judge_result.get("criteria", [])
    ]
    overall_score = judge_result.get("overall_score", 0.0)

    trace_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    evaluation = EvaluationModel(
        eval_id=uuid.uuid4().hex,
        trace_id=trace_id,
        session_id=session_id,
        agent_name="qa",
        evaluator="llm-judge",
        criteria=criteria,
        overall_score=overall_score,
        verdict="pass" if overall_score >= 0.7 else "fail",
    )

    client2 = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db2 = client2[MONGODB_DB]
    await EvaluationStore(db2).save_evaluation(evaluation)
    client2.close()

    assert overall_score >= 0.7, (
        f"QA judge score {overall_score:.2f} below 0.7. Review verdict: {review.verdict}"
    )
