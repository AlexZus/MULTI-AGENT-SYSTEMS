"""DevTeamPipeline — BA → HITL → Developer ↔ QA orchestration.

Task state is stored in tasks_state/{project_name}/tasks/{task_id}/.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from agentflow.graph import (
    Pipeline,
    PipelineEvent,
    PipelineEventType,
    PipelinePhase,
    PipelineState,
)
from agents.ba import BAAgent
from agents.developer import DeveloperAgent
from agents.qa import QAAgent
from config import Settings


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DevTeamPipeline:
    """Full BA → HITL → Developer ↔ QA pipeline with file-based task state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._hitl_events: dict[str, asyncio.Event] = {}

    def _task_dir(self, project_name: str, task_id: str) -> Path:
        return Path(self.settings.tasks_state_dir) / project_name / "tasks" / task_id

    def _latest_hitl_file(self, task_dir: Path) -> Path | None:
        files = sorted(task_dir.glob("hitl_state_*.json"))
        return files[-1] if files else None

    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text())

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2, default=str))

    async def run(
        self,
        user_story: str,
        project_name: str,
        task_id: str,
        *,
        session_id: str | None = None,
        trace_store: Any = None,
        event_bus: Any = None,
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Async generator yielding PipelineEvent objects.

        Stores state to disk so the HITL approval can resume from a separate
        HTTP request.
        """
        sid = session_id or uuid.uuid4().hex
        state = PipelineState(
            session_id=sid,
            project_name=project_name,
        )

        task_dir = self._task_dir(project_name, task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        # Setup tracevault trace
        trace_id = uuid.uuid4().hex
        if trace_store:
            from tracevault.models import TraceModel
            await trace_store.create_trace(TraceModel(
                trace_id=trace_id,
                session_id=sid,
                project_name=project_name,
                user_story=user_story,
                status="running",
            ))

        async for event in self._run(state, user_story, task_id, task_dir, trace_id, trace_store, event_bus):
            yield event

    async def _run(
        self,
        state: PipelineState,
        user_story: str,
        task_id: str,
        task_dir: Path,
        trace_id: str,
        trace_store: Any,
        event_bus: Any,
    ) -> AsyncGenerator[PipelineEvent, None]:
        sid = state.session_id
        project_name = state.project_name

        # ── BA Phase ──────────────────────────────────────────────────────────
        state.phase = PipelinePhase.BA
        yield PipelineEvent(
            type=PipelineEventType.PHASE_STARTED,
            phase=PipelinePhase.BA,
            session_id=sid,
            data={"user_story": user_story, "task_id": task_id},
        )

        ba = BAAgent(self.settings)
        tool_calls_log = []

        async def on_ba_tool(name, args, result):
            tool_calls_log.append({"name": name, "args": args, "result": result[:200]})
            yield PipelineEvent(
                type=PipelineEventType.TOOL_CALL,
                phase=PipelinePhase.BA,
                session_id=sid,
                data={"tool": name, "args": args},
            )

        try:
            spec = await ba.run(
                user_story,
                project_name=project_name,
                on_tool_call=lambda n, a, r: self._emit_tool(event_bus, sid, trace_id, PipelinePhase.BA, n, a, r),
                trace_store=trace_store,
                event_bus=event_bus,
                trace_id=trace_id,
                session_id=sid,
            )
        except Exception as exc:
            yield PipelineEvent(
                type=PipelineEventType.FAILED,
                phase=PipelinePhase.FAILED,
                session_id=sid,
                data={"error": str(exc), "phase": "ba"},
            )
            if trace_store:
                await trace_store.update_trace(trace_id, status="failed")
            return

        state.spec = spec.model_dump()
        # Save spec to disk
        self._write_json(task_dir / "spec.json", state.spec)

        yield PipelineEvent(
            type=PipelineEventType.PHASE_COMPLETED,
            phase=PipelinePhase.BA,
            session_id=sid,
            data={"spec": state.spec},
        )

        # ── HITL Phase ────────────────────────────────────────────────────────
        hitl_iteration = 1
        while True:
            state.phase = PipelinePhase.HITL
            hitl_file = task_dir / f"hitl_state_{hitl_iteration}.json"
            hitl_data = {
                "task_id": task_id,
                "project_name": project_name,
                "session_id": sid,
                "trace_id": trace_id,
                "iteration": hitl_iteration,
                "status": "waiting",
                "spec": state.spec,
                "feedback": None,
                "created_at": _utcnow(),
                "updated_at": _utcnow(),
            }
            self._write_json(hitl_file, hitl_data)

            if trace_store:
                await trace_store.update_trace(trace_id, status="waiting_hitl")
            if event_bus:
                await event_bus.publish("request_updated", {
                    "trace_id": trace_id, "session_id": sid,
                    "agent_name": "hitl", "status": "waiting_hitl",
                })

            # Register event BEFORE yielding so resume() called in the same
            # async-for loop body (e.g. in e2e tests) can set it immediately.
            event_key = f"{task_id}:{hitl_iteration}"
            self._hitl_events[event_key] = asyncio.Event()

            yield PipelineEvent(
                type=PipelineEventType.HITL_WAITING,
                phase=PipelinePhase.HITL,
                session_id=sid,
                data={"spec": state.spec, "iteration": hitl_iteration, "hitl_file": str(hitl_file)},
            )

            # Wait for approval signal (may already be set if resume() was called
            # before the generator resumed after the yield above)
            await self._hitl_events[event_key].wait()
            del self._hitl_events[event_key]

            # Read updated state from disk
            hitl_data = self._read_json(hitl_file)
            approved = hitl_data.get("status") == "approved"
            feedback = hitl_data.get("feedback")

            yield PipelineEvent(
                type=PipelineEventType.HITL_RESUMED,
                phase=PipelinePhase.HITL,
                session_id=sid,
                data={"approved": approved, "feedback": feedback},
            )

            if approved:
                state.hitl_approved = True
                break

            # Not approved — re-run BA with feedback
            state.phase = PipelinePhase.BA
            yield PipelineEvent(
                type=PipelineEventType.PHASE_STARTED,
                phase=PipelinePhase.BA,
                session_id=sid,
                data={"user_story": user_story, "feedback": feedback, "iteration": hitl_iteration + 1},
            )
            try:
                feedback_prompt = (
                    f"Original request: {user_story}\n\n"
                    f"Previous specification was rejected. User feedback:\n{feedback}\n\n"
                    f"Please revise the specification accordingly."
                )
                spec = await ba.run(
                    feedback_prompt,
                    project_name=project_name,
                    on_tool_call=lambda n, a, r: self._emit_tool(event_bus, sid, trace_id, PipelinePhase.BA, n, a, r),
                    trace_store=trace_store,
                    event_bus=event_bus,
                    trace_id=trace_id,
                    session_id=sid,
                )
            except Exception as exc:
                yield PipelineEvent(
                    type=PipelineEventType.FAILED,
                    phase=PipelinePhase.FAILED,
                    session_id=sid,
                    data={"error": str(exc), "phase": "ba_revision"},
                )
                if trace_store:
                    await trace_store.update_trace(trace_id, status="failed")
                return

            state.spec = spec.model_dump()
            self._write_json(task_dir / "spec.json", state.spec)
            hitl_iteration += 1

        if trace_store:
            await trace_store.update_trace(trace_id, status="running")

        # ── Developer + QA Loop ───────────────────────────────────────────────
        from agents.schemas import SpecOutput

        spec_obj = SpecOutput(**state.spec)
        developer = DeveloperAgent(self.settings)
        qa_agent = QAAgent(self.settings)

        for qa_iter in range(1, self.settings.max_qa_iterations + 1):
            # Developer
            state.phase = PipelinePhase.DEVELOPER
            yield PipelineEvent(
                type=PipelineEventType.PHASE_STARTED,
                phase=PipelinePhase.DEVELOPER,
                session_id=sid,
                data={"qa_iteration": qa_iter},
            )

            try:
                code = await developer.run(
                    spec_obj,
                    project_name=project_name,
                    on_tool_call=lambda n, a, r: self._emit_tool(event_bus, sid, trace_id, PipelinePhase.DEVELOPER, n, a, r),
                    trace_store=trace_store,
                    event_bus=event_bus,
                    trace_id=trace_id,
                    session_id=sid,
                )
            except Exception as exc:
                yield PipelineEvent(
                    type=PipelineEventType.FAILED,
                    phase=PipelinePhase.FAILED,
                    session_id=sid,
                    data={"error": str(exc), "phase": "developer"},
                )
                if trace_store:
                    await trace_store.update_trace(trace_id, status="failed")
                return

            state.code = code.model_dump()
            self._write_json(task_dir / f"code_{qa_iter}.json", state.code)

            yield PipelineEvent(
                type=PipelineEventType.PHASE_COMPLETED,
                phase=PipelinePhase.DEVELOPER,
                session_id=sid,
                data={"code": state.code},
            )

            # QA
            state.phase = PipelinePhase.QA
            state.qa_iteration = qa_iter
            yield PipelineEvent(
                type=PipelineEventType.QA_ITERATION,
                phase=PipelinePhase.QA,
                session_id=sid,
                data={"iteration": qa_iter, "max": self.settings.max_qa_iterations},
            )

            try:
                from agents.schemas import CodeOutput
                review = await qa_agent.run(
                    spec_obj,
                    CodeOutput(**state.code),
                    project_name=project_name,
                    on_tool_call=lambda n, a, r: self._emit_tool(event_bus, sid, trace_id, PipelinePhase.QA, n, a, r),
                    trace_store=trace_store,
                    event_bus=event_bus,
                    trace_id=trace_id,
                    session_id=sid,
                )
            except Exception as exc:
                yield PipelineEvent(
                    type=PipelineEventType.FAILED,
                    phase=PipelinePhase.FAILED,
                    session_id=sid,
                    data={"error": str(exc), "phase": "qa"},
                )
                if trace_store:
                    await trace_store.update_trace(trace_id, status="failed")
                return

            state.review = review.model_dump()
            self._write_json(task_dir / f"qa_review_{qa_iter}.json", state.review)

            if review.verdict == "APPROVED" or qa_iter == self.settings.max_qa_iterations:
                verdict = review.verdict if review.verdict == "APPROVED" else "AUTO-APPROVED"
                state.phase = PipelinePhase.COMPLETED
                if trace_store:
                    await trace_store.update_trace(
                        trace_id, status="completed", verdict=verdict
                    )
                if event_bus:
                    await event_bus.publish("request_completed", {
                        "trace_id": trace_id, "session_id": sid,
                        "verdict": verdict,
                    })
                yield PipelineEvent(
                    type=PipelineEventType.COMPLETED,
                    phase=PipelinePhase.COMPLETED,
                    session_id=sid,
                    data={
                        "verdict": verdict,
                        "score": review.score,
                        "files_created": state.code.get("files_created", []),
                        "qa_iterations": qa_iter,
                    },
                )
                return

            # QA requested revision — feed review back to developer as context
            revision_notes = "\n".join(review.issues)
            yield PipelineEvent(
                type=PipelineEventType.PHASE_STARTED,
                phase=PipelinePhase.DEVELOPER,
                session_id=sid,
                data={"qa_iteration": qa_iter + 1, "revision_notes": revision_notes},
            )
            # Inject QA issues into the developer's next run via spec notes
            spec_obj = spec_obj.model_copy(update={"notes": f"QA revision #{qa_iter}: {revision_notes}"})

    async def resume(
        self,
        task_id: str,
        project_name: str,
        *,
        approved: bool,
        feedback: str | None = None,
    ) -> bool:
        """Update HITL state file and fire the asyncio.Event to resume the pipeline."""
        task_dir = self._task_dir(project_name, task_id)
        hitl_file = self._latest_hitl_file(task_dir)
        if hitl_file is None:
            return False

        hitl_data = self._read_json(hitl_file)
        iteration = hitl_data.get("iteration", 1)
        hitl_data["status"] = "approved" if approved else "rejected"
        hitl_data["feedback"] = feedback
        hitl_data["updated_at"] = _utcnow()
        self._write_json(hitl_file, hitl_data)

        event_key = f"{task_id}:{iteration}"
        event = self._hitl_events.get(event_key)
        if event:
            event.set()
            return True
        return False

    async def _emit_tool(self, event_bus, sid, trace_id, phase, name, args, result):
        if event_bus:
            await event_bus.publish("request_updated", {
                "trace_id": trace_id,
                "session_id": sid,
                "agent_name": phase.value,
                "tool_name": name,
            })
