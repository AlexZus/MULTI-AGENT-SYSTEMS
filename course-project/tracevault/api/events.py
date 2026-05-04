"""SSE events endpoint."""

import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["events"])


@router.get("/events")
async def sse_events():
    """Server-Sent Events stream — emits request_started, request_updated, request_completed."""
    from tracevault.sse import event_bus

    async def generator():
        subscription = event_bus.subscribe()
        async with subscription:
            # Send a keepalive comment immediately so the browser knows the connection is open
            yield ": keepalive\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(subscription._queue.get(), timeout=15.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
