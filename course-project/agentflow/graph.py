"""Pipeline orchestrator — BA → (HITL) → Developer ↔ QA loop.

Replaces LangGraph with a simple async generator that yields PipelineEvent
objects as the pipeline progresses through each phase.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

class PipelinePhase(str, Enum):
    INIT = "init"
    BA = "ba"
    HITL = "hitl"
    DEVELOPER = "developer"
    QA = "qa"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    session_id: str
    project_name: str
    phase: PipelinePhase = PipelinePhase.INIT
    spec: dict | None = None          # SpecOutput as dict
    code: dict | None = None          # CodeOutput as dict
    review: dict | None = None        # ReviewOutput as dict
    qa_iteration: int = 0
    hitl_approved: bool = False
    hitl_feedback: str | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline events
# ---------------------------------------------------------------------------

class PipelineEventType(str, Enum):
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    TOOL_CALL = "tool_call"
    HITL_WAITING = "hitl_waiting"
    HITL_RESUMED = "hitl_resumed"
    QA_ITERATION = "qa_iteration"
    COMPLETED = "completed"
    FAILED = "failed"
    LOG = "log"


@dataclass
class PipelineEvent:
    type: PipelineEventType
    phase: PipelinePhase
    session_id: str
    data: dict = field(default_factory=dict)

    def as_sse_dict(self) -> dict:
        return {
            "type": self.type.value,
            "phase": self.phase.value,
            "session_id": self.session_id,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Pipeline class (skeleton — agents wired in Phase 4)
# ---------------------------------------------------------------------------

class Pipeline:
    """BA → HITL → Developer ↔ QA pipeline.

    Agents are injected at construction time (set to None until Phase 4).
    The ``run`` method is an async generator yielding ``PipelineEvent`` objects.

    Usage::

        pipeline = Pipeline(settings=settings)
        async for event in pipeline.run(user_story, project_name, session_id):
            print(event.type, event.data)
    """

    def __init__(
        self,
        settings: Any,
        *,
        ba_agent: Any = None,
        developer_agent: Any = None,
        qa_agent: Any = None,
    ) -> None:
        self.settings = settings
        self.ba_agent = ba_agent
        self.developer_agent = developer_agent
        self.qa_agent = qa_agent

    async def run(
        self,
        user_story: str,
        project_name: str,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Async generator that drives the full pipeline.

        Yields PipelineEvent objects at each significant transition.
        Agents must be set before calling this method.
        """
        sid = session_id or uuid.uuid4().hex
        state = PipelineState(
            session_id=sid,
            project_name=project_name,
        )

        async for event in self._run_pipeline(state, user_story, task_id):
            yield event

    async def _run_pipeline(
        self,
        state: PipelineState,
        user_story: str,
        task_id: str,
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Internal pipeline generator — overridden/extended in Phase 4."""
        # Phase 1 skeleton: just emit phase events to verify the generator contract.
        yield PipelineEvent(
            type=PipelineEventType.PHASE_STARTED,
            phase=PipelinePhase.BA,
            session_id=state.session_id,
            data={"user_story": user_story, "task_id": task_id},
        )

        # BA agent invocation (wired in Phase 4)
        if self.ba_agent is None:
            yield PipelineEvent(
                type=PipelineEventType.LOG,
                phase=PipelinePhase.BA,
                session_id=state.session_id,
                data={"message": "BA agent not configured (Phase 4)"},
            )
            state.phase = PipelinePhase.FAILED
            state.error = "BA agent not configured"
            yield PipelineEvent(
                type=PipelineEventType.FAILED,
                phase=PipelinePhase.FAILED,
                session_id=state.session_id,
                data={"error": state.error},
            )
            return

        # Actual execution injected in Phase 4:
        # spec = await self.ba_agent.run(user_story, project_name=state.project_name)
        # state.spec = spec.model_dump()
        # ... HITL, developer, QA loop ...

        yield PipelineEvent(
            type=PipelineEventType.COMPLETED,
            phase=PipelinePhase.COMPLETED,
            session_id=state.session_id,
            data={},
        )
