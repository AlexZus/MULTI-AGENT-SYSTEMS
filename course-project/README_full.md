# Dev Team — AI Software Development Multi-Agent System

A FastAPI-based multi-agent system that simulates an AI software development team:

```
User → Business Analyst → (HITL Approval) → Developer ↔ QA (loop, max 5)
```

Two custom packages replace external services:
- **`agentflow/`** — custom ReAct agent loop and pipeline orchestrator (replaces LangGraph)
- **`tracevault/`** — MongoDB-backed tracing, SSE dashboard, prompt versioning, LLM-as-Judge evaluations (replaces Langfuse)

All UI is plain HTML/CSS/vanilla JS. LLM backend: any OpenAI-compatible endpoint configured via `.env`.

---

## Quick Start

```bash
# 1. Verify all services are up
python scripts/check_services.py

# 2. Activate virtualenv
source .venv/bin/activate

# 3. Seed prompts into MongoDB (first time only)
python scripts/seed_prompts.py

# 4. Start the main app
uvicorn app:app --port 8000 --reload

# 5. Start the tracevault monitoring dashboard (separate terminal)
uvicorn tracevault.server:app --port 8090 --reload
```

Open **http://localhost:8000** for the project management UI.  
Open **http://localhost:8090** for the tracevault monitoring dashboard.

---

## Project Structure

```
course-project/
├── agentflow/                    # Custom pipeline package (replaces LangGraph)
│   ├── agent.py                  # AgentRunner — async ReAct loop
│   ├── graph.py                  # Pipeline orchestrator + PipelineEvent types
│   ├── middleware.py             # BudgetMiddleware, InvalidToolCallRetryMiddleware
│   └── mcp.py                   # MCP client (streamable-http transport)
├── tracevault/                   # Monitoring package (replaces Langfuse)
│   ├── models.py                 # Trace, Span, Prompt, Evaluation Pydantic models
│   ├── store.py                  # TraceStore, PromptStore, EvaluationStore (MongoDB)
│   ├── tracker.py                # TraceContext async context manager
│   ├── prompts.py                # load_prompt(), seed_from_files()
│   ├── sse.py                    # SSE EventBus
│   ├── api/                      # FastAPI route modules
│   ├── ui/templates/             # 6-screen monitoring UI
│   └── server.py                 # tracevault FastAPI app (port 8090)
├── agents/
│   ├── schemas.py                # SpecOutput, CodeOutput, ReviewOutput
│   ├── ba.py                     # Business Analyst agent (web_search + RAG)
│   ├── developer.py              # Developer agent (MCP filesystem + REPL)
│   └── qa.py                     # QA agent (reads files, runs pytest)
├── tools/
│   ├── search.py                 # DuckDuckGo web search
│   ├── rag.py                    # FAISS + BM25 hybrid retrieval (auto-rebuild)
│   ├── mcp_fs.py                 # Filesystem MCP client + PathNormalizer
│   └── mcp_repl.py               # Python REPL MCP client
├── prompts/                      # Agent system prompts as .md files (seeded to MongoDB)
│   ├── ba_system.md
│   ├── developer_system.md
│   ├── qa_system.md
│   └── *_json_suffix.md          # Structured output suffixes
├── scripts/
│   ├── check_services.py         # Ping all 5 services, print status table
│   └── seed_prompts.py           # Load prompts/*.md into tracevault MongoDB
├── rag_docs/                     # RAG knowledge base (.md files)
│   ├── coding_standards.md
│   ├── fastapi_tutorial.md
│   ├── google_python_style_guide.md
│   ├── python_stdlib.md
│   └── .index/                   # Auto-built FAISS + BM25 index
├── workspace/                    # Per-project agent-generated files
│   └── <project-name>/
├── tasks_state/                  # Per-task JSON state files
│   └── <project-name>/tasks/<task-id>/
│       ├── meta.json
│       ├── spec.json
│       ├── hitl_state_1.json
│       ├── code_1.json
│       └── qa_review_1.json
├── ui/
│   ├── templates/                # Main app Jinja2 templates
│   └── static/                   # CSS
├── tests/
│   ├── unit/                     # No external services required
│   ├── integration/              # Requires MongoDB, MCP servers, embedding
│   └── live/                     # Requires all services + LLM
├── app.py                        # Main FastAPI application (port 8000)
├── pipeline.py                   # DevTeamPipeline orchestration
├── config.py                     # Pydantic Settings (reads .env)
└── requirements.txt
```

---

## Configuration

Copy `.env` from the templates in `.metadata/project_task_files/` and fill in values:

```env
# LLM
OPENAI_COMPATIBLE_API_URL=<llm-base-url>
MODEL_NAME=<model-name>
API_KEY=<api-key-or-dummy>

# MCP servers
MCP_FILESYSTEM_URL=<filesystem-mcp-url>
MCP_REPL_URL=<repl-mcp-url>

# Embedding
EMBEDDING_URL=<embedding-url>

# MongoDB
MONGODB_URL=<mongodb-connection-string>
MONGODB_DB=<database-name>

# Ports
TRACEVAULT_PORT=<port>
APP_PORT=<port>

# Pipeline
MAX_QA_ITERATIONS=<int>
STRUCTURED_OUTPUT_WORKAROUND=<true|false>
TASKS_STATE_DIR=./tasks_state
WORKSPACE_DIR=./workspace
```

---

## UI Screens

### Main App (port 8000)

| Screen | Route | Description |
|--------|-------|-------------|
| Home | `GET /` | Project list; create new project |
| Project Dashboard | `GET /projects/{name}` | Submit task, recent task table, link to tracevault |
| Task Detail | `GET /projects/{name}/tasks/{id}` | Real-time pipeline progress, HITL approval, completed file tabs |
| Task List | `GET /projects/{name}/tasks` | Full task history with status/verdict badges |

**Task Detail states:**
- **BA Running** — spinner + live SSE event log (tool calls appear line by line)
- **Awaiting Approval** — spec card (title, requirements, acceptance criteria, complexity badge) + Approve/Request Changes buttons
- **Developer/QA Running** — phase progress bar with QA iteration counter
- **Completed** — verdict badge, QA score, file tabs with code preview, Download ZIP, link to trace

### TraceVault (port 8090)

| Screen | Route | Description |
|--------|-------|-------------|
| Dashboard | `GET /` | Total traces, token usage, pass rate, SSE live feed |
| Traces | `GET /traces` | Filterable trace list (agent, status, session) |
| Trace Detail | `GET /traces/{id}` | Per-span LLM calls, tool call log, evaluation results |
| Sessions | `GET /sessions` | Grouped by session_id with aggregate stats |
| Prompts | `GET /prompts` | Edit prompts live, version history, one-click rollback |
| Evaluations | `GET /evaluations` | LLM-as-Judge results per agent with aggregate pass rates |

---

## Pipeline

```
User Story
    │
    ▼
Business Analyst ──► SpecOutput (title, requirements, acceptance criteria, complexity)
    │
    ▼
HITL Approval ──► Approve → continue │ Reject → BA re-runs with feedback (loop)
    │
    ▼
Developer ──► CodeOutput (files_created, tests_passed, dependencies_installed)
    │
    ▼
QA ──► ReviewOutput (verdict, score, issues)
    │
    ├── REVISION_NEEDED → Developer re-runs (up to 5 iterations)
    └── APPROVED / AUTO-APPROVED (max iterations reached) → Done
```

All task state is written to `tasks_state/{project}/{task_id}/` as JSON files so the server can restart without losing in-flight tasks. HITL approval is coordinated via an `asyncio.Event` keyed on `{task_id}:{iteration}`.

---

## Agents

### Business Analyst (`agents/ba.py`)
- **Tools:** `web_search` (DuckDuckGo), `knowledge_search` (RAG over `rag_docs/`)
- **Output:** `SpecOutput` — validated Pydantic model with requirements and acceptance criteria
- **Prompt:** `prompts/ba_system.md` (editable live via tracevault)

### Developer (`agents/developer.py`)
- **Tools:** Filesystem MCP (read/write/list files in project dir) + Python REPL MCP (run code, run pytest, pip install) + `web_search`
- **Output:** `CodeOutput` — list of files created, test results, installed dependencies
- **Prompt:** `prompts/developer_system.md`

### QA (`agents/qa.py`)
- **Tools:** Filesystem MCP (read-only) + Python REPL MCP
- **Pre-loads** all `files_created` contents directly into the review message — avoids LLM skipping read tool calls
- **Output:** `ReviewOutput` — verdict (`APPROVED`/`REVISION_NEEDED`), score (0–1), issues list
- **Prompt:** `prompts/qa_system.md`

All agents use:
- `InvalidToolCallRetryMiddleware(max_retries=3)` — catches malformed JSON in tool call args
- `BudgetMiddleware(max_tool_calls)` — wraps results with remaining-budget XML
- Structured output workaround: JSON-suffix prompt → regex-extract fenced ```json block → `model_validate_json` → retry up to 3×

---

## `agentflow` Package

**`AgentRunner`** (`agentflow/agent.py`):
- Pure OpenAI SDK `chat.completions.create` loop — no LangGraph
- `_try_fix_tool_call()` — detects tool call JSON emitted in `content` field and converts it to `tool_calls`
- Configurable `max_iterations` and tool budget

**`Pipeline`** (`agentflow/graph.py`):
- Async generator yielding `PipelineEvent` objects (phase_started, phase_completed, hitl_waiting, hitl_resumed, qa_iteration, completed, failed, tool_call)
- `PipelineState` carries project_name, phase, spec, code, review, qa_iteration, hitl_approved

**`MCPClient`** (`agentflow/mcp.py`):
- Wraps `mcp.client.streamable_http` with `anyio` task group constraints
- Must be opened/closed in the same asyncio task

---

## `tracevault` Package

**Storage** (`tracevault/store.py`):
- `TraceStore` — create/update/add_span/list traces; filterable by agent, status, session, project
- `PromptStore` — upsert with full version history; `rollback_prompt(name, version)` restores old template as new version
- `EvaluationStore` — save/list LLM-as-Judge results; `get_agent_stats()` for aggregate pass rate

**Prompts** (`tracevault/prompts.py`):
- `load_prompt(name, **variables)` — loads from MongoDB at runtime; template variables injected by pipeline
- `seed_from_files(prompts_dir, store)` — idempotent seeder; run once per environment

**SSE** (`tracevault/sse.py`):
- `EventBus` — asyncio.Queue-based pub/sub; max 200 items per subscriber
- Used for real-time dashboard updates and per-task event streams in `app.py`

---

## RAG Knowledge Base

Documents in `rag_docs/`:
- `python_stdlib.md` — Python standard library reference
- `fastapi_tutorial.md` — FastAPI patterns and best practices
- `google_python_style_guide.md` — Google Python Style Guide
- `coding_standards.md` — Project-specific coding standards

The index (`rag_docs/.index/`) is built automatically on first use and **auto-rebuilt** whenever any `.md` file in `rag_docs/` is newer than the cached index (C8 pattern). To rebuild manually:

```bash
python -m tools.rag build
```

Retrieval uses hybrid **RRF (Reciprocal Rank Fusion)** over FAISS (dense, inner product) and BM25 (sparse). Embeddings come from the Docker embedding service (`all-mpnet-base-v2`, 768 dims).

---

## Path Normalization

Agents see project-relative paths (`calculator/main.py`). MCP servers expect absolute paths (`/workspace/calculator/main.py`). `PathNormalizer` in `tools/mcp_fs.py` handles the translation transparently:

```python
normalizer.to_mcp("calculator/main.py")      # → /workspace/calculator/main.py
normalizer.from_mcp("/workspace/calc/a.py")  # → calc/a.py
normalizer.normalize_result(text)            # strips /workspace/ from MCP output text
```

---

## Tests

```bash
# Unit tests (no external services)
pytest tests/unit/

# Integration tests (requires MongoDB, MCP servers, embedding service)
pytest tests/integration/

# Live agent tests (requires all services including LLM)
pytest tests/live/test_ba_agent.py
pytest tests/live/test_developer_agent.py
pytest tests/live/test_qa_agent.py

# End-to-end pipeline (full pipeline, ~5 min per test)
pytest tests/live/test_pipeline_e2e.py

# UI smoke tests (requires app running at localhost:8000 + agent-browser CLI)
pytest tests/live/test_ui.py
```

Current counts: **91 unit**, **38 integration** tests.

---

## Scripts

### `scripts/check_services.py`

Ping all 5 services before starting a session:

```bash
python scripts/check_services.py
# Service           URL                          Status
# ──────────────────────────────────────────────────────
# Local LLM         $OPENAI_COMPATIBLE_API_URL   ✓ OK
# Filesystem MCP    $MCP_FILESYSTEM_URL          ✓ OK
# Python REPL MCP   $MCP_REPL_URL                ✓ OK
# Embedding Service $EMBEDDING_URL               ✓ OK
# MongoDB           $MONGODB_URL                 ✓ OK

python scripts/check_services.py --json   # machine-readable output
```

### `scripts/seed_prompts.py`

Load all `prompts/*.md` into tracevault MongoDB. Safe to re-run (idempotent):

```bash
python scripts/seed_prompts.py
python scripts/seed_prompts.py --force   # overwrite existing prompts
```

---

## Development Notes

**Structured output workaround:** Some local LLMs do not reliably support `response_format=` combined with tool use simultaneously. When `STRUCTURED_OUTPUT_WORKAROUND=true`, a JSON-suffix is appended to the system prompt, the LLM response is regex-parsed for a fenced ` ```json ` block, then validated with `model_validate_json`. Retries up to 3× on parse failure.

**Tool call content-field fix:** Some local LLMs emit tool call JSON in the `content` field instead of `tool_calls`. `_try_fix_tool_call()` in `agentflow/agent.py` detects and converts this automatically.

**HITL asyncio coordination:** The pipeline generator registers an `asyncio.Event` *before* yielding `HITL_WAITING`, so `pipeline.resume()` called from a separate HTTP request or inside a test's `async for` body will always find the event to set.

**QA file pre-loading:** QA pre-fetches all `files_created` via the `read_file` MCP tool and embeds contents directly in the review message. This avoids relying on the LLM to proactively call tools — a common reliability problem with locally-hosted models.

---

## External Services Setup

All five services run in Docker. See sections below for Dockerfile and docker-compose configurations.

---

## 1. Local LLM

**Endpoint:** configured via `OPENAI_COMPATIBLE_API_URL`  
**Model:** configured via `MODEL_NAME`  
**Protocol:** OpenAI-compatible chat completions API

### Configuration

```env
OPENAI_COMPATIBLE_API_URL=<llm-base-url>
MODEL_NAME=<model-name>
API_KEY=<api-key-or-dummy>
```

### Verify

```bash
curl $OPENAI_COMPATIBLE_API_URL/models

curl $OPENAI_COMPATIBLE_API_URL/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":10}"
```

### Notes

- If no API key is required by the server, set `API_KEY=dummy`.
- If the model does not reliably support `response_format=` + tool use, enable `STRUCTURED_OUTPUT_WORKAROUND=true`.
- Tool calls sometimes emitted in `content` field — handled automatically by `_try_fix_tool_call`.

---

## 2. Filesystem MCP Server (Docker)

**Endpoint:** `http://localhost:8082/mcp`  
**Transport:** Streamable HTTP (MCP 2024-11-05)  
**Allowed paths:** `/workspace` (read/write), `/rag_docs` (read-only)

### File Layout (`~/mcp-filesystem/`)

**`Dockerfile`**

```dockerfile
FROM node:24-slim
WORKDIR /app
RUN npm install -g @modelcontextprotocol/server-filesystem supergateway
EXPOSE 8082
CMD supergateway \
  --outputTransport streamableHttp \
  --port 8082 \
  --stdio "mcp-server-filesystem $MCP_ROOTS"
```

**`docker-compose.yml`**

```yaml
services:
  mcp-filesystem:
    build: .
    network_mode: host
    user: "${UID:-1000}:${GID:-1000}"
    environment:
      MCP_ROOTS: "/workspace /rag_docs"
    volumes:
      - /home/alex/ClaudeProjects/MULTI-AGENT-SYSTEMS/course-project/workspace:/workspace
      - /home/alex/ClaudeProjects/MULTI-AGENT-SYSTEMS/course-project/rag_docs:/rag_docs:ro
    restart: unless-stopped
```

### Start & Verify

```bash
cd ~/mcp-filesystem && docker compose up -d --build

curl -s -X POST http://localhost:8082/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}'
```

### Available Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read full file contents |
| `write_file` | Write/overwrite a file |
| `edit_file` | Apply line-level patches |
| `create_directory` | Create directory (recursive) |
| `list_directory` | List directory contents |
| `directory_tree` | Recursive tree view |
| `move_file` | Move or rename a file |
| `search_files` | Regex search across files |
| `get_file_info` | Size, mtime, permissions |
| `list_allowed_directories` | Show accessible roots |

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on 8082 | Container not running | `docker compose up -d` |
| Container restarts | Old `--outputTransport sse` | Rebuild with `streamableHttp` |
| `Permission denied` on `/workspace` | Missing user mapping | Add `user: "${UID:-1000}:${GID:-1000}"` |
| `path must be within allowed directories` | Path outside roots | Use only `/workspace/...` or `/rag_docs/...` |

---

## 3. Python REPL MCP Server (Docker)

**Endpoint:** `http://localhost:8083/mcp`  
**Transport:** Streamable HTTP (FastMCP)  
**Purpose:** Lets Developer and QA agents run Python code, pytest, and install packages.

### File Layout (`~/python_env_mcp/`)

**`server.py`**

```python
import subprocess
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("python-repl")
WORKSPACE = Path("/workspace")
MAX_TIMEOUT = 300

def _save(output: str, output_file: str | None) -> str:
    if not output_file:
        return output
    path = WORKSPACE / output_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output)
    return output + f"\n\n[Saved to /workspace/{output_file}]"

@mcp.tool()
def python_repl(code: str, timeout: int = 10, output_file: str | None = None) -> str:
    """Execute a Python snippet."""
    result = subprocess.run(["python3", "-c", code], capture_output=True, text=True,
                            timeout=min(timeout, MAX_TIMEOUT), cwd="/workspace")
    return _save((result.stdout + result.stderr).strip() or "(no output)", output_file)

@mcp.tool()
def run_pytest(path: str = "/workspace", args: str = "-v", timeout: int = 60,
               output_file: str | None = None) -> str:
    """Run pytest on a path under /workspace."""
    if not path.startswith("/workspace"):
        return "Error: path must be under /workspace"
    result = subprocess.run(["python3", "-m", "pytest", path] + args.split(),
                            capture_output=True, text=True, timeout=min(timeout, MAX_TIMEOUT),
                            cwd="/workspace")
    return _save((result.stdout + result.stderr).strip()[:8000], output_file)

@mcp.tool()
def pip_install(packages: str) -> str:
    """Install Python packages into the persistent packages volume."""
    result = subprocess.run(["pip", "install", "--target", "/pip-packages"] + packages.split(),
                            capture_output=True, text=True, timeout=120)
    return (result.stdout + result.stderr).strip()[:2000] or "(no output)"

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8083)
```

**`Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install fastmcp pytest requests
ENV PYTHONPATH=/pip-packages:/workspace
COPY server.py .
EXPOSE 8083
CMD ["python", "server.py"]
```

**`docker-compose.yml`**

```yaml
services:
  mcp-repl:
    build: .
    network_mode: host
    user: "${UID:-1000}:${GID:-1000}"
    volumes:
      - /home/alex/ClaudeProjects/MULTI-AGENT-SYSTEMS/course-project/workspace:/workspace
      - /home/alex/Containers/python_env_mcp/pip-packages:/pip-packages
    restart: unless-stopped
```

> One-time setup: `mkdir -p ~/Containers/python_env_mcp/pip-packages`

### Start

```bash
cd ~/python_env_mcp && docker compose up -d --build
```

### Available Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `python_repl` | `code`, `timeout=10`, `output_file=None` | Execute Python snippet |
| `run_pytest` | `path`, `args="-v"`, `timeout=60`, `output_file=None` | Run pytest under `/workspace` |
| `pip_install` | `packages` | Install to persistent `/pip-packages` volume |

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on 8083 | Container not running | `docker compose up -d` |
| `ModuleNotFoundError` after install | `/pip-packages` not on PYTHONPATH | Verify `ENV PYTHONPATH=/pip-packages:/workspace` |
| Output truncated | Hard limit 4000/8000 chars | Use `output_file` to save full results |

---

## 4. Embedding Service (Docker)

**Endpoint:** `http://localhost:8084`  
**Protocol:** OpenAI-compatible `/v1/embeddings`  
**Model:** `all-mpnet-base-v2` (768 dims, GPU)

### File Layout (`~/embedding_service/`)

**`server.py`**

```python
import os, torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEVICE = f"cuda:{os.getenv('CUDA_DEVICE', '0')}" if torch.cuda.is_available() else "cpu"
app = FastAPI()
model = SentenceTransformer(MODEL_NAME, cache_folder="/models", device=DEVICE)

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME

@app.post("/v1/embeddings")
def embed(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = model.encode(texts, normalize_embeddings=True)
    return {"object": "list", "model": req.model,
            "data": [{"object": "embedding", "index": i, "embedding": v.tolist()}
                     for i, v in enumerate(vecs)]}

@app.get("/health")
def health():
    return {"model": MODEL_NAME, "dims": model.get_sentence_embedding_dimension(),
            "device": DEVICE, "cuda_available": torch.cuda.is_available()}
```

**`Dockerfile`**

```dockerfile
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ca-certificates python3 python3-pip python3.12-dev pkg-config \
  && rm -rf /var/lib/apt/lists/* \
  && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir fastapi uvicorn[standard] sentence-transformers
COPY server.py .
EXPOSE 8084
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8084"]
```

**`docker-compose.yml`**

```yaml
services:
  embedding:
    build: .
    network_mode: host
    user: "${UID:-1000}:${GID:-1000}"
    environment:
      EMBEDDING_MODEL: "all-mpnet-base-v2"
      HF_HOME: /models
      TRANSFORMERS_CACHE: /models
    volumes:
      - /home/alex/Containers/embedding_service/models:/models
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [ gpu ]
```

> One-time setup: `mkdir -p ~/Containers/embedding_service/models`

### Start & Verify

```bash
cd ~/embedding_service && docker compose up -d --build
curl http://localhost:8084/health
# {"model":"all-mpnet-base-v2","dims":768,"device":"cuda:0","cuda_available":true}
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `device` is `"cpu"` | Container not seeing GPU | Verify NVIDIA Container Toolkit installed |
| Model download stalls | HuggingFace rate limit | Set `HF_TOKEN` env var |
| Wrong GPU used | Default picks GPU 0 | Set `device_ids: ["1"]` in docker-compose |

---

## 5. MongoDB

**URL:** configured via `MONGODB_URL`  
**Database:** configured via `MONGODB_DB`

Used by tracevault to store traces, spans, prompts, and evaluations.

```env
MONGODB_URL=<mongodb-connection-string>
MONGODB_DB=<database-name>
```

### Verify

```bash
mongosh "$MONGODB_URL" --eval "db.adminCommand('ping')"
```

```python
import asyncio, os
from motor.motor_asyncio import AsyncIOMotorClient

async def verify():
    client = AsyncIOMotorClient(os.environ["MONGODB_URL"], serverSelectionTimeoutMS=3000)
    info = await client.server_info()
    print("MongoDB version:", info["version"])
    client.close()

asyncio.run(verify())
```

---

## Services Summary

| Service | Env var | Status Check |
|---------|---------|-------------|
| Local LLM | `OPENAI_COMPATIBLE_API_URL` | `python scripts/check_services.py` |
| Filesystem MCP | `MCP_FILESYSTEM_URL` | `python scripts/check_services.py` |
| Python REPL MCP | `MCP_REPL_URL` | `python scripts/check_services.py` |
| Embedding Service | `EMBEDDING_URL` | `python scripts/check_services.py` |
| MongoDB | `MONGODB_URL` | `python scripts/check_services.py` |

Run `python scripts/check_services.py` before each development session to verify all five are reachable.
