"""Main FastAPI application — project management UI + pipeline controller."""

from __future__ import annotations

import asyncio
import io
import json
import os
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import motor.motor_asyncio
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import Settings
from pipeline import DevTeamPipeline

settings = Settings()
_TEMPLATES_DIR = Path(__file__).parent / "ui" / "templates"
_STATIC_DIR = Path(__file__).parent / "ui" / "static"
_TASKS_STATE_DIR = Path(settings.tasks_state_dir)
_WORKSPACE_DIR = Path(settings.workspace_dir)
_TRACEVAULT_URL = f"http://localhost:{settings.tracevault_port}"

# Module-level shared state
pipeline: DevTeamPipeline = None  # type: ignore[assignment]
trace_store: Any = None
event_bus: Any = None
_service_warnings: list[str] = []
_task_sse_queues: dict[str, asyncio.Queue] = {}  # task_id → SSE queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, trace_store, event_bus, _service_warnings

    # MongoDB
    client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
    db = client[settings.mongodb_db]

    from tracevault.store import TraceStore, PromptStore
    from tracevault.prompts import set_prompt_store, seed_from_files
    from tracevault.sse import EventBus

    trace_store = TraceStore(db)
    prompt_store = PromptStore(db)
    event_bus = EventBus()

    set_prompt_store(prompt_store)

    # Seed prompts from .md files if not yet seeded
    prompts_dir = Path(__file__).parent / "prompts"
    if prompts_dir.exists():
        await seed_from_files(prompts_dir, prompt_store)

    pipeline = DevTeamPipeline(settings)

    # Ensure workspace and tasks_state dirs exist
    _WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    _TASKS_STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Service health checks (non-blocking)
    _service_warnings.clear()
    _service_warnings.extend(await _check_services())

    yield

    client.close()


app = FastAPI(title="Dev Team", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _task_dir(project_name: str, task_id: str) -> Path:
    return _TASKS_STATE_DIR / project_name / "tasks" / task_id


def _load_task_meta(project_name: str, task_id: str) -> dict | None:
    task_dir = _task_dir(project_name, task_id)
    meta_file = task_dir / "meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    return None


def _save_task_meta(project_name: str, task_id: str, meta: dict) -> None:
    task_dir = _task_dir(project_name, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _list_tasks(project_name: str) -> list[dict]:
    project_tasks_dir = _TASKS_STATE_DIR / project_name / "tasks"
    if not project_tasks_dir.exists():
        return []
    tasks = []
    for d in sorted(project_tasks_dir.iterdir(), reverse=True):
        meta = _load_task_meta(project_name, d.name)
        if meta:
            tasks.append(meta)
    return tasks


def _list_projects() -> list[dict]:
    if not _WORKSPACE_DIR.exists():
        return []
    projects = []
    for d in sorted(_WORKSPACE_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__":
            tasks = _list_tasks(d.name)
            running_or_waiting = [t for t in tasks if t.get("status") in ("running", "waiting_hitl")]
            last_activity = None
            if tasks:
                try:
                    last_activity = datetime.fromisoformat(tasks[0].get("created_at", ""))
                except Exception:
                    pass
            projects.append({
                "name": d.name,
                "tasks_count": len(tasks),
                "running_count": len(running_or_waiting),
                "last_activity": last_activity,
            })
    return projects


def _pick_read_tool(tool_names: set[str]) -> str | None:
    """Return the best text-read tool name from a set of MCP tool names.

    Prefers read_text_file > read_file over any other read-prefixed tool to
    avoid accidentally selecting read_media_file, which returns binary content
    and fails MCP's text content validation.
    """
    for preferred in ("read_text_file", "read_file"):
        if preferred in tool_names:
            return preferred
    return next((n for n in sorted(tool_names) if "read" in n.lower() and "media" not in n.lower()), None)


def _template_ctx(**extra) -> dict:
    return {
        "tracevault_url": _TRACEVAULT_URL,
        "service_warnings": _service_warnings,
        **extra,
    }


async def _check_services() -> list[str]:
    """Quick ping of all services; return list of names that are down."""
    import httpx
    warnings = []
    checks = [
        ("LLM", f"{settings.openai_compatible_api_url}/models"),
        ("Filesystem MCP", settings.mcp_filesystem_url),
        ("REPL MCP", settings.mcp_repl_url),
        ("Embedding", settings.embedding_url),
    ]
    async with httpx.AsyncClient(timeout=2.0) as client:
        for name, url in checks:
            try:
                await client.get(url)
            except Exception:
                warnings.append(name)
    # MongoDB
    try:
        client_mongo = motor.motor_asyncio.AsyncIOMotorClient(
            settings.mongodb_url, serverSelectionTimeoutMS=2000
        )
        await client_mongo.admin.command("ping")
        client_mongo.close()
    except Exception:
        warnings.append("MongoDB")
    return warnings


def _publish_task_event(task_id: str, event: dict) -> None:
    """Non-async: put event into task SSE queue if subscriber exists."""
    q = _task_sse_queues.get(task_id)
    if q:
        try:
            q.put_nowait(json.dumps(event))
        except asyncio.QueueFull:
            pass


# ---------------------------------------------------------------------------
# Routes — Home / Projects
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    projects = _list_projects()
    return templates.TemplateResponse(request, "home.html", _template_ctx(projects=projects, active="home"))


@app.post("/projects")
async def create_project(project_name: str = Form(...)):
    name = project_name.strip()
    if not name or not all(c.isalnum() or c == "-" for c in name):
        raise HTTPException(status_code=400, detail="Invalid project name")
    project_dir = _WORKSPACE_DIR / name
    project_dir.mkdir(parents=True, exist_ok=True)
    return RedirectResponse(f"/projects/{name}", status_code=303)


# ---------------------------------------------------------------------------
# Routes — Project
# ---------------------------------------------------------------------------

@app.get("/projects/{project_name}", response_class=HTMLResponse)
async def project_dashboard(request: Request, project_name: str):
    project_dir = _WORKSPACE_DIR / project_name
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    tasks = _list_tasks(project_name)[:10]  # recent 10 for dashboard
    # Convert created_at strings to datetime for template
    for t in tasks:
        try:
            t["created_at"] = datetime.fromisoformat(t["created_at"])
        except Exception:
            t["created_at"] = _utcnow()
    return templates.TemplateResponse(
        request, "project_dashboard.html",
        _template_ctx(project_name=project_name, tasks=tasks),
    )


@app.get("/projects/{project_name}/tasks")
async def project_tasks(request: Request, project_name: str, format: str = "html"):
    tasks = _list_tasks(project_name)
    for t in tasks:
        try:
            t["created_at"] = datetime.fromisoformat(t["created_at"])
        except Exception:
            t["created_at"] = _utcnow()
    if format == "json":
        return JSONResponse([{**t, "created_at": t["created_at"].isoformat()} for t in tasks])
    return templates.TemplateResponse(
        request, "task_list.html",
        _template_ctx(project_name=project_name, tasks=tasks),
    )


@app.post("/projects/{project_name}/tasks")
async def submit_task(project_name: str, user_story: str = Form(...)):
    project_dir = _WORKSPACE_DIR / project_name
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    task_id = uuid.uuid4().hex
    meta = {
        "task_id": task_id,
        "project_name": project_name,
        "user_story": user_story.strip(),
        "status": "running",
        "current_phase": "ba",
        "spec": None,
        "code": None,
        "review": None,
        "qa_iteration": 0,
        "hitl_approved": False,
        "verdict": None,
        "trace_id": None,
        "error": None,
        "created_at": _utcnow().isoformat(),
        "updated_at": _utcnow().isoformat(),
    }
    _save_task_meta(project_name, task_id, meta)

    # Launch pipeline in background
    asyncio.create_task(_run_pipeline(task_id, project_name, user_story))

    return RedirectResponse(f"/projects/{project_name}/tasks/{task_id}", status_code=303)


async def _run_pipeline(task_id: str, project_name: str, user_story: str):
    """Background task that drives the pipeline and updates task meta."""
    from agentflow.graph import PipelineEventType

    async for event in pipeline.run(
        user_story,
        project_name,
        task_id,
        session_id=task_id,
        trace_store=trace_store,
        event_bus=event_bus,
    ):
        meta = _load_task_meta(project_name, task_id) or {}
        event_dict = event.as_sse_dict()

        if event.type == PipelineEventType.PHASE_STARTED:
            meta["current_phase"] = event.phase.value
        elif event.type == PipelineEventType.PHASE_COMPLETED:
            if "spec" in event.data:
                meta["spec"] = event.data["spec"]
            if "code" in event.data:
                meta["code"] = event.data["code"]
        elif event.type == PipelineEventType.HITL_WAITING:
            meta["status"] = "waiting_hitl"
            meta["spec"] = event.data.get("spec", meta.get("spec"))
        elif event.type == PipelineEventType.HITL_RESUMED:
            meta["status"] = "running"
            meta["hitl_approved"] = event.data.get("approved", False)
        elif event.type == PipelineEventType.QA_ITERATION:
            meta["qa_iteration"] = event.data.get("iteration", 0)
        elif event.type == PipelineEventType.COMPLETED:
            meta["status"] = "completed"
            meta["verdict"] = event.data.get("verdict")
            meta["qa_iteration"] = event.data.get("qa_iterations", meta.get("qa_iteration"))
        elif event.type == PipelineEventType.FAILED:
            meta["status"] = "failed"
            meta["error"] = event.data.get("error")

        meta["updated_at"] = _utcnow().isoformat()
        _save_task_meta(project_name, task_id, meta)
        _publish_task_event(task_id, event_dict)


# ---------------------------------------------------------------------------
# Routes — Task detail
# ---------------------------------------------------------------------------

@app.get("/projects/{project_name}/tasks/{task_id}")
async def task_detail(request: Request, project_name: str, task_id: str, format: str = "html"):
    meta = _load_task_meta(project_name, task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        meta["created_at"] = datetime.fromisoformat(meta.get("created_at", _utcnow().isoformat()))
    except Exception:
        meta["created_at"] = _utcnow()

    if format == "json":
        return JSONResponse({**meta, "created_at": meta["created_at"].isoformat()})

    # Build file_contents for completed tasks
    file_contents: list[tuple[str, str]] = []
    if meta.get("status") == "completed" and meta.get("code"):
        files_created = meta["code"].get("files_created", [])
        from tools.mcp_fs import MCPFilesystem
        try:
            async with MCPFilesystem(settings.mcp_filesystem_url, project_name=project_name) as fs:
                tool_names = {t["function"]["name"] for t in fs.get_openai_tools()}
                read_tool = _pick_read_tool(tool_names)
                if read_tool:
                    for fpath in files_created:
                        try:
                            content = await fs.call_tool(read_tool, {"path": fpath})
                            file_contents.append((fpath, content))
                        except Exception as exc:
                            file_contents.append((fpath, f"[Could not read file: {exc}]"))
        except Exception:
            pass

    # Load QA review
    review = None
    if meta.get("qa_iteration"):
        review_file = _task_dir(project_name, task_id) / f"qa_review_{meta['qa_iteration']}.json"
        if review_file.exists():
            try:
                review = json.loads(review_file.read_text())
            except Exception:
                pass

    class Task:
        pass

    task_obj = Task()
    for k, v in meta.items():
        setattr(task_obj, k, v)
    task_obj.review = review  # type: ignore[attr-defined]

    return templates.TemplateResponse(
        request, "task_detail.html",
        _template_ctx(
            project_name=project_name,
            task=task_obj,
            file_contents=file_contents,
        ),
    )


class ApproveRequest(BaseModel):
    approved: bool
    feedback: str | None = None


@app.post("/projects/{project_name}/tasks/{task_id}/approve")
async def approve_task(project_name: str, task_id: str, body: ApproveRequest):
    meta = _load_task_meta(project_name, task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if meta.get("status") != "waiting_hitl":
        raise HTTPException(status_code=400, detail="Task is not waiting for approval")

    success = await pipeline.resume(
        task_id,
        project_name,
        approved=body.approved,
        feedback=body.feedback,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Could not resume pipeline — task may have expired")
    return {"ok": True}


@app.get("/projects/{project_name}/tasks/{task_id}/events")
async def task_events(project_name: str, task_id: str):
    """SSE stream for a specific task."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _task_sse_queues[task_id] = queue

    async def generator():
        yield ": keepalive\n\n"
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _task_sse_queues.pop(task_id, None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/projects/{project_name}/tasks/{task_id}/download")
async def download_task(project_name: str, task_id: str):
    """ZIP all files created by the developer for this task."""
    meta = _load_task_meta(project_name, task_id)
    if not meta or meta.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Task not completed")

    files_created = (meta.get("code") or {}).get("files_created", [])
    if not files_created:
        raise HTTPException(status_code=404, detail="No files to download")

    buf = io.BytesIO()
    from tools.mcp_fs import MCPFilesystem

    try:
        async with MCPFilesystem(settings.mcp_filesystem_url, project_name=project_name) as fs:
            tool_names = {t["function"]["name"] for t in fs.get_openai_tools()}
            read_tool = _pick_read_tool(tool_names)
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fpath in files_created:
                    if read_tool:
                        try:
                            content = await fs.call_tool(read_tool, {"path": fpath})
                            zf.writestr(fpath, content)
                        except Exception:
                            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{project_name}_{task_id[:8]}.zip"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=settings.app_port, reload=True)
