"""Regression tests for bugs found during manual testing.

Each test is designed to FAIL against the buggy code and PASS after the fix.
See .metadata/manual_testing_report.md for full context.

Tests here are fast (no browser, no LLM) and run against the real app via httpx.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Paths — must match app.py's Settings defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_WORKSPACE_DIR = _PROJECT_ROOT / "workspace"
_TASKS_STATE_DIR = _PROJECT_ROOT / "tasks_state"

APP_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique(prefix: str = "bug") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def _seed_task(
    project_name: str,
    *,
    status: str = "completed",
    code: dict | None = None,
    error: str | None = None,
    qa_iteration: int = 1,
    verdict: str | None = "APPROVED",
) -> str:
    """Write a minimal task meta.json directly to disk; return task_id."""
    task_id = uuid.uuid4().hex
    task_dir = _TASKS_STATE_DIR / project_name / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (_WORKSPACE_DIR / project_name).mkdir(parents=True, exist_ok=True)

    default_code = {
        "summary": "Created files",
        "files_created": [f"{project_name}/main.py"],
        "dependencies_installed": [],
        "tests_passed": True,
        "notes": "",
    }

    meta = {
        "task_id": task_id,
        "project_name": project_name,
        "user_story": "Test user story",
        "status": status,
        "current_phase": "done",
        "spec": {"title": "Test", "requirements": [], "acceptance_criteria": [],
                 "estimated_complexity": "simple", "notes": ""},
        "code": code if code is not None else (default_code if status == "completed" else None),
        "review": None,
        "qa_iteration": qa_iteration,
        "hitl_approved": True,
        "verdict": verdict,
        "trace_id": None,
        "error": error,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if status == "completed" and qa_iteration:
        review = {
            "verdict": verdict or "APPROVED",
            "score": 0.95,
            "issues": [],
            "suggestions": [],
            "tests_run": 1,
            "tests_passed": 1,
        }
        (task_dir / f"qa_review_{qa_iteration}.json").write_text(json.dumps(review))

    return task_id


# ===========================================================================
# Bug 1 — Starlette 1.0 TemplateResponse API break
#
# All UI routes must return HTTP 200. Before the fix they returned 500 with
# "TypeError: unhashable type: 'dict'" because TemplateResponse was called
# with the old two-argument form.
# ===========================================================================

class TestBug1StarletteSmokeRoutes:
    """HTTP smoke tests: every UI route must return 200."""

    def test_home_returns_200(self):
        r = httpx.get(f"{APP_URL}/")
        assert r.status_code == 200, f"GET / returned {r.status_code}"

    def test_home_contains_heading(self):
        r = httpx.get(f"{APP_URL}/")
        assert r.status_code == 200
        assert "Projects" in r.text

    def test_project_dashboard_returns_200(self):
        project_name = _unique("smoke-dash")
        (_WORKSPACE_DIR / project_name).mkdir(parents=True, exist_ok=True)
        r = httpx.get(f"{APP_URL}/projects/{project_name}")
        assert r.status_code == 200, f"GET /projects/{project_name} returned {r.status_code}"

    def test_task_list_returns_200(self):
        project_name = _unique("smoke-list")
        (_WORKSPACE_DIR / project_name).mkdir(parents=True, exist_ok=True)
        r = httpx.get(f"{APP_URL}/projects/{project_name}/tasks")
        assert r.status_code == 200, f"Task list returned {r.status_code}"

    def test_task_detail_returns_200(self):
        project_name = _unique("smoke-detail")
        task_id = _seed_task(project_name, status="completed")
        r = httpx.get(f"{APP_URL}/projects/{project_name}/tasks/{task_id}")
        assert r.status_code == 200, f"Task detail returned {r.status_code}"

    def test_task_detail_running_returns_200(self):
        project_name = _unique("smoke-running")
        task_id = _seed_task(project_name, status="running")
        r = httpx.get(f"{APP_URL}/projects/{project_name}/tasks/{task_id}")
        assert r.status_code == 200

    def test_task_detail_hitl_returns_200(self):
        project_name = _unique("smoke-hitl")
        task_id = _seed_task(project_name, status="waiting_hitl")
        r = httpx.get(f"{APP_URL}/projects/{project_name}/tasks/{task_id}")
        assert r.status_code == 200

    def test_task_detail_failed_returns_200(self):
        project_name = _unique("smoke-fail")
        task_id = _seed_task(project_name, status="failed", error="Something broke")
        r = httpx.get(f"{APP_URL}/projects/{project_name}/tasks/{task_id}")
        assert r.status_code == 200

    def test_tracevault_home_returns_200(self):
        r = httpx.get("http://localhost:8090/")
        assert r.status_code == 200, f"TraceVault / returned {r.status_code}"

    def test_tracevault_sessions_returns_200(self):
        r = httpx.get("http://localhost:8090/sessions")
        assert r.status_code == 200

    def test_tracevault_dashboard_returns_200(self):
        r = httpx.get("http://localhost:8090/dashboard")
        assert r.status_code == 200

    def test_tracevault_prompts_returns_200(self):
        r = httpx.get("http://localhost:8090/prompts")
        assert r.status_code == 200

    def test_tracevault_evaluations_returns_200(self):
        r = httpx.get("http://localhost:8090/evaluations")
        assert r.status_code == 200


# ===========================================================================
# Bug 2 — .pytest_cache appearing as a project card
#
# _list_projects() must not include hidden directories (names starting with '.')
# or other non-user directories (__pycache__) from the workspace.
# ===========================================================================

class TestBug2ListProjectsFiltering:
    """Unit-style tests for the _list_projects() helper."""

    def test_hidden_dir_excluded_from_api(self):
        """GET /projects/json must not include dirs starting with '.'."""
        # The workspace may already contain .pytest_cache — just check via the API
        r = httpx.get(f"{APP_URL}/", headers={"Accept": "text/html"})
        assert r.status_code == 200
        # The HTML must not contain a project card for .pytest_cache
        assert ".pytest_cache" not in r.text, (
            ".pytest_cache appears on the home page — _list_projects() is "
            "not filtering hidden directories"
        )

    def test_hidden_dir_excluded_via_workspace(self, tmp_path, monkeypatch):
        """_list_projects() must skip any directory whose name starts with '.'."""
        import app as app_module

        fake_ws = tmp_path / "workspace"
        fake_ws.mkdir()
        (fake_ws / "real-project").mkdir()
        (fake_ws / ".pytest_cache").mkdir()
        (fake_ws / "__pycache__").mkdir()

        original = app_module._WORKSPACE_DIR
        monkeypatch.setattr(app_module, "_WORKSPACE_DIR", fake_ws)
        try:
            projects = app_module._list_projects()
        finally:
            monkeypatch.setattr(app_module, "_WORKSPACE_DIR", original)

        names = [p["name"] for p in projects]
        assert ".pytest_cache" not in names, \
            f"_list_projects returned hidden dir: {names}"
        assert "__pycache__" not in names, \
            f"_list_projects returned __pycache__: {names}"
        assert "real-project" in names


# ===========================================================================
# Bug 3 — onclick attribute buttons silent no-op via agent-browser click
#
# This is an agent-browser/CDP interaction issue. We verify that the APPROVE
# endpoint actually works when called correctly (via the JS eval path), and
# that the approve API is reachable from the task detail page.
# The browser-side test is covered in tests/live/test_ui.py::test_full_pipeline.
# Here we add an HTTP-level test: approve endpoint must accept a valid payload.
# ===========================================================================

class TestBug3HITLApproveEndpoint:
    """The approve endpoint must process approval payloads correctly."""

    def test_approve_endpoint_rejects_non_hitl_task(self):
        """Approving a completed task returns 400."""
        project_name = _unique("approve")
        task_id = _seed_task(project_name, status="completed")
        r = httpx.post(
            f"{APP_URL}/projects/{project_name}/tasks/{task_id}/approve",
            json={"approved": True},
        )
        assert r.status_code == 400, (
            f"Expected 400 for approving a non-waiting_hitl task, got {r.status_code}"
        )

    def test_approve_endpoint_rejects_missing_task(self):
        """Approving a non-existent task returns 404."""
        r = httpx.post(
            f"{APP_URL}/projects/no-such-project/tasks/deadbeef/approve",
            json={"approved": True},
        )
        assert r.status_code == 404

    def test_task_detail_shows_approve_button_in_hitl_state(self):
        """Task detail HTML must contain the approve button for waiting_hitl tasks."""
        project_name = _unique("hitl-btn")
        task_id = _seed_task(project_name, status="waiting_hitl")
        r = httpx.get(f"{APP_URL}/projects/{project_name}/tasks/{task_id}")
        assert r.status_code == 200
        assert "approve" in r.text.lower(), \
            "No approve element found in waiting_hitl task detail page"


# ===========================================================================
# Bug 4 — Completed task shows [Could not read file]; ZIP download is empty
#
# When a task is completed and code.files_created is populated, AND the
# actual files exist on disk, the task detail page must show the file contents
# (not "[Could not read file]") and the download endpoint must return a
# non-empty ZIP.
# ===========================================================================

class TestBug4CompletedTaskFileDisplay:
    """File tabs must show actual content; ZIP download must be non-empty."""

    @pytest.fixture(autouse=True)
    def setup_project_with_real_files(self, tmp_path):
        """Create a completed task + write real files to workspace."""
        self.project_name = _unique("files")
        self.file_content = "# hello\ndef main():\n    print('hello world')\n"

        # Create real file in workspace
        ws_dir = _WORKSPACE_DIR / self.project_name
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "main.py").write_text(self.file_content)

        # Seed completed task pointing to the real file
        self.task_id = _seed_task(
            self.project_name,
            status="completed",
            code={
                "summary": "Created main.py",
                "files_created": [f"{self.project_name}/main.py"],
                "dependencies_installed": [],
                "tests_passed": True,
                "notes": "",
            },
        )

    def test_task_detail_no_could_not_read_file(self):
        """Task detail page must NOT show '[Could not read file]'."""
        r = httpx.get(
            f"{APP_URL}/projects/{self.project_name}/tasks/{self.task_id}",
            timeout=15.0,
        )
        assert r.status_code == 200
        assert "[Could not read file]" not in r.text, (
            "Task detail page shows '[Could not read file]' — "
            "the MCP read path inside the FastAPI handler is broken"
        )

    def test_task_detail_shows_file_content(self):
        """Task detail page must show actual file content in code tabs."""
        r = httpx.get(
            f"{APP_URL}/projects/{self.project_name}/tasks/{self.task_id}",
            timeout=15.0,
        )
        assert r.status_code == 200
        assert "def main" in r.text, (
            "Task detail page does not show the Python source code — "
            "file content is not being rendered"
        )

    def test_download_zip_not_empty(self):
        """Download ZIP must contain at least one file entry."""
        r = httpx.get(
            f"{APP_URL}/projects/{self.project_name}/tasks/{self.task_id}/download",
            timeout=15.0,
        )
        assert r.status_code == 200, f"Download returned {r.status_code}"
        assert r.headers.get("content-type") == "application/zip", \
            f"Expected application/zip, got {r.headers.get('content-type')}"

        buf = io.BytesIO(r.content)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
            assert len(names) > 0, "ZIP archive is empty — no files were included"
            assert any("main.py" in n for n in names), \
                f"main.py not found in ZIP entries: {names}"

    def test_download_zip_file_content_correct(self):
        """Files in the ZIP must have the actual content, not empty bytes."""
        r = httpx.get(
            f"{APP_URL}/projects/{self.project_name}/tasks/{self.task_id}/download",
            timeout=15.0,
        )
        assert r.status_code == 200
        buf = io.BytesIO(r.content)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
            assert names, "ZIP is empty"
            main_entry = next((n for n in names if "main.py" in n), None)
            assert main_entry is not None, f"No main.py in {names}"
            content = zf.read(main_entry).decode()
            assert "def main" in content, \
                f"ZIP main.py content is wrong: {content!r}"
