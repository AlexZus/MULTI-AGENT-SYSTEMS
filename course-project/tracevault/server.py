"""TraceVault server — FastAPI app with MongoDB-backed tracing and 6-screen UI."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import motor.motor_asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tracevault.store import EvaluationStore, PromptStore, TraceStore

# ---------------------------------------------------------------------------
# Global store references (accessed by API route modules via import)
# ---------------------------------------------------------------------------
trace_store: TraceStore = None  # type: ignore[assignment]
prompt_store: PromptStore = None  # type: ignore[assignment]
evaluation_store: EvaluationStore = None  # type: ignore[assignment]

_TEMPLATES_DIR = Path(__file__).parent / "ui" / "templates"
_STATIC_DIR = Path(__file__).parent / "ui" / "static"
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global trace_store, prompt_store, evaluation_store

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:admin_password@172.20.0.1:27017")
    mongodb_db = os.getenv("MONGODB_DB", "course_project")

    client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
    db = client[mongodb_db]

    trace_store = TraceStore(db)
    prompt_store = PromptStore(db)
    evaluation_store = EvaluationStore(db)

    # Wire prompts module
    from tracevault.prompts import set_prompt_store, seed_from_files
    set_prompt_store(prompt_store)

    # Seed prompts from .md files if collection is empty
    existing = await prompt_store.list_prompts()
    if not existing and _PROMPTS_DIR.exists():
        await seed_from_files(_PROMPTS_DIR, prompt_store)

    yield

    client.close()


app = FastAPI(title="TraceVault", lifespan=lifespan)

# Static files
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Register API routers
# ---------------------------------------------------------------------------
from tracevault.api.traces import router as traces_router
from tracevault.api.sessions import router as sessions_router
from tracevault.api.events import router as events_router
from tracevault.api.prompts import router as prompts_router
from tracevault.api.evaluations import router as evaluations_router

app.include_router(traces_router)
app.include_router(sessions_router)
app.include_router(events_router)
app.include_router(prompts_router)
app.include_router(evaluations_router)


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def ui_traces(request: Request, session_id: str | None = None):
    traces = await trace_store.list_traces(session_id=session_id, limit=100)
    projects = sorted({t.project_name for t in await trace_store.list_traces(limit=500)})
    return templates.TemplateResponse(
        request, "traces_list.html",
        {"traces": traces, "projects": projects, "active": "traces"},
    )


@app.get("/traces/{trace_id}", response_class=HTMLResponse)
async def ui_trace_detail(request: Request, trace_id: str):
    trace = await trace_store.get_trace(trace_id)
    if trace is None:
        return HTMLResponse("<h1>Trace not found</h1>", status_code=404)
    evaluations = await evaluation_store.list_evaluations(limit=50)
    trace_evals = [e for e in evaluations if e.trace_id == trace_id]
    return templates.TemplateResponse(
        request, "trace_detail.html",
        {"trace": trace, "evaluations": trace_evals, "active": "traces"},
    )


@app.get("/sessions", response_class=HTMLResponse)
async def ui_sessions(request: Request):
    sessions = await trace_store.list_sessions()
    return templates.TemplateResponse(
        request, "sessions.html",
        {"sessions": sessions, "active": "sessions"},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    cutoff = datetime.utcnow() - timedelta(hours=1)
    all_traces = await trace_store.list_traces(limit=200)
    recent = [t for t in all_traces if t.created_at >= cutoff]
    return templates.TemplateResponse(
        request, "dashboard.html",
        {"recent": recent, "active": "dashboard"},
    )


@app.get("/prompts", response_class=HTMLResponse)
async def ui_prompts(request: Request):
    prompts = await prompt_store.list_prompts()
    return templates.TemplateResponse(
        request, "prompts.html",
        {"prompts": prompts, "active": "prompts"},
    )


@app.get("/evaluations", response_class=HTMLResponse)
async def ui_evaluations(request: Request):
    evaluations = await evaluation_store.list_evaluations(limit=100)
    stats = await evaluation_store.get_agent_stats()
    sessions_data = await trace_store.list_sessions()
    sessions = [s["session_id"] for s in sessions_data]
    return templates.TemplateResponse(
        request, "evaluations.html",
        {
            "evaluations": evaluations,
            "stats": stats,
            "sessions": sessions,
            "active": "evaluations",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TRACEVAULT_PORT", "8090"))
    uvicorn.run("tracevault.server:app", host="0.0.0.0", port=port, reload=True)
