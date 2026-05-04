"""Server-Sent Events bus for real-time tracevault dashboard."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator


class EventBus:
    """Simple async pub/sub bus for SSE events.

    Usage::

        bus = EventBus()
        # subscriber (SSE endpoint):
        async for event in bus.subscribe():
            yield f"data: {event}\\n\\n"

        # publisher (pipeline/tracker):
        await bus.publish("request_started", {"trace_id": "...", ...})
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> "_Subscription":
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(queue)
        return _Subscription(queue, self)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    async def publish(self, event_type: str, data: dict) -> None:
        """Broadcast an event to all current subscribers (non-blocking)."""
        payload = json.dumps({"type": event_type, **data})
        dead: list[asyncio.Queue] = []
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(queue)
        for queue in dead:
            self.unsubscribe(queue)


class _Subscription:
    """Async context manager + async iterator for one SSE subscriber."""

    def __init__(self, queue: asyncio.Queue, bus: EventBus) -> None:
        self._queue = queue
        self._bus = bus

    async def __aenter__(self) -> "_Subscription":
        return self

    async def __aexit__(self, *args) -> None:
        self._bus.unsubscribe(self._queue)

    def __aiter__(self) -> "_Subscription":
        return self

    async def __anext__(self) -> str:
        return await self._queue.get()


# Module-level singleton used by the tracevault server and tracker
event_bus = EventBus()
