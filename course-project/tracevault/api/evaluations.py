"""Evaluations API routes."""

from fastapi import APIRouter, Depends, Query

from tracevault.models import EvaluationModel
from tracevault.store import EvaluationStore

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


def get_eval_store() -> EvaluationStore:
    from tracevault.server import evaluation_store
    return evaluation_store


@router.post("", response_model=EvaluationModel)
async def save_evaluation(
    evaluation: EvaluationModel,
    store: EvaluationStore = Depends(get_eval_store),
):
    await store.save_evaluation(evaluation)
    return evaluation


@router.get("", response_model=list[EvaluationModel])
async def list_evaluations(
    agent_name: str | None = Query(None),
    session_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    store: EvaluationStore = Depends(get_eval_store),
):
    return await store.list_evaluations(
        agent_name=agent_name,
        session_id=session_id,
        limit=limit,
    )


@router.get("/stats")
async def evaluation_stats(
    agent_name: str | None = Query(None),
    store: EvaluationStore = Depends(get_eval_store),
):
    return await store.get_agent_stats(agent_name=agent_name)
