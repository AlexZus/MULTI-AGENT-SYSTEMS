"""TraceContext — async context manager for recording agent spans."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from tracevault.models import SpanModel, TraceModel
from tracevault.sse import EventBus


class TraceContext:
    """Records a single agent span and publishes SSE events.

    Usage::

        async with TraceContext(
            trace_store=store,
            event_bus=bus,
            trace_id="...",
            session_id="...",
            agent_name="ba",
            iteration=0,
            input_messages=[...],
        ) as ctx:
            # run agent...
            ctx.set_output(output_message, input_tokens=100, output_tokens=200)
            ctx.add_tool_call("read_file", {"path": "..."}, "content...", latency_ms=120)
    """

    def __init__(
        self,
        *,
        trace_store: Any,
        event_bus: EventBus | None = None,
        trace_id: str,
        session_id: str,
        project_name: str,
        agent_name: str,
        iteration: int = 0,
        input_messages: list[dict] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._store = trace_store
        self._bus = event_bus
        self._trace_id = trace_id
        self._session_id = session_id
        self._project_name = project_name
        self._agent_name = agent_name
        self._iteration = iteration
        self._input_messages = input_messages or []
        self._tags = tags or []

        self._span_id = uuid.uuid4().hex
        self._output_message: dict = {}
        self._tool_calls: list[dict] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._start_time: float = 0.0

    def set_output(
        self,
        output_message: dict,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self._output_message = output_message
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def add_tool_call(
        self,
        name: str,
        args: dict,
        result: str,
        *,
        latency_ms: int = 0,
    ) -> None:
        self._tool_calls.append(
            {"name": name, "args": args, "result": result, "latency_ms": latency_ms}
        )

    async def __aenter__(self) -> "TraceContext":
        self._start_time = time.monotonic()
        if self._bus is not None:
            await self._bus.publish(
                "request_started",
                {
                    "trace_id": self._trace_id,
                    "session_id": self._session_id,
                    "project_name": self._project_name,
                    "agent_name": self._agent_name,
                    "iteration": self._iteration,
                },
            )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        latency_ms = int((time.monotonic() - self._start_time) * 1000)

        span = SpanModel(
            span_id=self._span_id,
            trace_id=self._trace_id,
            agent_name=self._agent_name,
            iteration=self._iteration,
            input_messages=self._input_messages,
            output_message=self._output_message,
            tool_calls=self._tool_calls,
            latency_ms=latency_ms,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            timestamp=datetime.utcnow(),
            tags=self._tags,
        )

        # Always record the span (even on error — gives partial debug info)
        await self._store.add_span(self._trace_id, span)

        if self._bus is not None:
            if exc_type is None:
                await self._bus.publish(
                    "request_completed",
                    {
                        "trace_id": self._trace_id,
                        "session_id": self._session_id,
                        "agent_name": self._agent_name,
                        "input_tokens": self._input_tokens,
                        "output_tokens": self._output_tokens,
                        "latency_ms": latency_ms,
                    },
                )
            else:
                await self._bus.publish(
                    "request_completed",
                    {
                        "trace_id": self._trace_id,
                        "session_id": self._session_id,
                        "agent_name": self._agent_name,
                        "error": str(exc_val),
                        "latency_ms": latency_ms,
                    },
                )

    async def publish_update(self, **data: Any) -> None:
        """Publish a mid-span update event (e.g. after each tool call)."""
        if self._bus is not None:
            await self._bus.publish(
                "request_updated",
                {
                    "trace_id": self._trace_id,
                    "session_id": self._session_id,
                    "agent_name": self._agent_name,
                    **data,
                },
            )
