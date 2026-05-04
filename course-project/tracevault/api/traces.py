"""Trace API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query

from tracevault.models import TraceModel
from tracevault.store import TraceStore

router = APIRouter(prefix="/api/traces", tags=["traces"])


def get_trace_store() -> TraceStore:
    from tracevault.server import trace_store
    return trace_store


@router.get("", response_model=list[TraceModel])
async def list_traces(
    session_id: str | None = Query(None),
    status: str | None = Query(None),
    agent: str | None = Query(None),
    project_name: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    store: TraceStore = Depends(get_trace_store),
):
    return await store.list_traces(
        agent_name=agent,
        status=status,
        session_id=session_id,
        project_name=project_name,
        limit=limit,
    )


@router.get("/{trace_id}", response_model=TraceModel)
async def get_trace(
    trace_id: str,
    store: TraceStore = Depends(get_trace_store),
):
    trace = await store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
