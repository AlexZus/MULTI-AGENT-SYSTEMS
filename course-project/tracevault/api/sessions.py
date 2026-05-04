"""Sessions API routes."""

from fastapi import APIRouter, Depends

from tracevault.store import TraceStore

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_trace_store() -> TraceStore:
    from tracevault.server import trace_store
    return trace_store


@router.get("")
async def list_sessions(store: TraceStore = Depends(get_trace_store)):
    return await store.list_sessions()
