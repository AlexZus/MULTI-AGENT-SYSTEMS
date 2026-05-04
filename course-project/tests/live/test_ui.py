"""Comprehensive UI tests using the agent-browser CLI (ab.py adapter).

Requires:
  - app running at http://localhost:8000
  - agent-browser CLI installed and on PATH
  - all backend services running (for live-pipeline tests)

Run all tests:
    pytest tests/live/test_ui.py -v

Run only state-seeded tests (no LLM needed):
    pytest tests/live/test_ui.py -v -k "not test_running and not test_full_pipeline"

Run a single group:
    pytest tests/live/test_ui.py -v -k "test_hitl"
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Import AgentBrowser from the project's own tests/ copy of ab.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ab import AgentBrowser  # type: ignore[import]

APP_URL = "http://localhost:8000"

# Absolute paths to the app's state dirs (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TASKS_STATE_DIR = _PROJECT_ROOT / "tasks_state"
_WORKSPACE_DIR = _PROJECT_ROOT / "workspace"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def browser():
    """Fresh isolated browser session per test; auto-closed on teardown."""
    session = f"pytest-ui-{uuid.uuid4().hex[:10]}"
    ab = AgentBrowser(session)
    yield ab
    ab.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_name(prefix: str = "ui") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


class SeededTask:
    """Helper returned by the seed_task fixture factory."""

    def __init__(self, project_name: str, task_id: str, task_dir: Path) -> None:
        self.project_name = project_name
        self.task_id = task_id
        self.task_dir = task_dir

    @property
    def detail_url(self) -> str:
        return f"{APP_URL}/projects/{self.project_name}/tasks/{self.task_id}"

    @property
    def list_url(self) -> str:
        return f"{APP_URL}/projects/{self.project_name}/tasks"

    @property
    def dashboard_url(self) -> str:
        return f"{APP_URL}/projects/{self.project_name}"


def _seed_task(
    project_name: str,
    *,
    status: str = "completed",
    current_phase: str = "ba",
    user_story: str = "Write a hello world script",
    spec: dict | None = None,
    code: dict | None = None,
    verdict: str | None = "APPROVED",
    qa_iteration: int = 1,
    hitl_approved: bool = True,
    error: str | None = None,
) -> SeededTask:
    """Write meta.json (and optionally spec.json, qa_review_N.json) directly to disk.

    This lets tests exercise any task state without running a real pipeline.
    """
    task_id = uuid.uuid4().hex
    task_dir = _TASKS_STATE_DIR / project_name / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (_WORKSPACE_DIR / project_name).mkdir(parents=True, exist_ok=True)

    _default_spec = {
        "title": "Hello World Script",
        "requirements": ["Print hello world", "Use standard output"],
        "acceptance_criteria": ["Running the script prints 'Hello, World!'"],
        "estimated_complexity": "simple",
        "notes": "Keep it simple",
    }
    _default_code = {
        "summary": "Created a hello world script",
        "files_created": [f"{project_name}/main.py", f"{project_name}/test_main.py"],
        "dependencies_installed": [],
        "tests_passed": True,
        "notes": "",
    }

    resolved_spec = spec or _default_spec
    resolved_code = code or _default_code

    meta = {
        "task_id": task_id,
        "project_name": project_name,
        "user_story": user_story,
        "status": status,
        "current_phase": current_phase,
        "spec": resolved_spec if status in ("waiting_hitl", "completed") else None,
        "code": resolved_code if status == "completed" else None,
        "review": None,
        "qa_iteration": qa_iteration,
        "hitl_approved": hitl_approved,
        "verdict": verdict,
        "trace_id": None,
        "error": error,
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
    }
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if status in ("waiting_hitl", "completed"):
        (task_dir / "spec.json").write_text(json.dumps(resolved_spec, indent=2))

    if status == "completed" and qa_iteration:
        review = {
            "verdict": verdict or "APPROVED",
            "score": 0.92,
            "issues": [],
            "suggestions": ["Add docstrings"],
            "tests_run": 2,
            "tests_passed": 2,
        }
        (task_dir / f"qa_review_{qa_iteration}.json").write_text(
            json.dumps(review, indent=2)
        )

    return SeededTask(project_name, task_id, task_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill(browser: AgentBrowser, selector: str, value: str) -> None:
    """Type into an input/textarea via eval (avoids CLI shell-escaping issues)."""
    escaped = value.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    browser.eval(
        f"(function(){{"
        f"  var el = document.querySelector(`{selector}`);"
        f"  el.value = `{escaped}`;"
        f"  el.dispatchEvent(new Event('input', {{bubbles:true}}));"
        f"  el.dispatchEvent(new Event('change', {{bubbles:true}}));"
        f"}})();"
    )


def _submit_form(browser: AgentBrowser, action: str) -> None:
    browser.eval(f"document.querySelector('form[action=\"{action}\"]').submit();")


def _create_project(browser: AgentBrowser, project_name: str) -> None:
    """Navigate to home and create a project via the inline form."""
    browser.open(APP_URL)
    browser.wait_for_selector("h1", timeout=5_000)
    browser.eval("document.getElementById('new-project-form').style.display=''")
    browser.wait_for_selector("input[name='project_name']", timeout=3_000)
    _fill(browser, "input[name='project_name']", project_name)
    _submit_form(browser, "/projects")
    browser.wait_for_url(f"*/projects/{project_name}", timeout=10_000)


# ---------------------------------------------------------------------------
# Group 1 — Home page
# ---------------------------------------------------------------------------

def test_home_loads(browser: AgentBrowser):
    """Home page shows 'Projects' heading and nav links."""
    browser.open(APP_URL)
    browser.wait_for_selector("h1", timeout=5_000)
    assert "Projects" in browser.get_text("h1")
    # Nav brand link
    assert browser.count("a[href='/']") >= 1


def test_home_new_project_form_toggle(browser: AgentBrowser):
    """'+ New Project' button shows the form; Cancel button hides it."""
    browser.open(APP_URL)
    browser.wait_for_selector("h1", timeout=5_000)

    # Form hidden initially
    visible_before = browser.eval(
        "document.getElementById('new-project-form').style.display !== 'none' && "
        "document.getElementById('new-project-form').offsetParent !== null"
    )
    assert not visible_before

    # Show form
    browser.eval("document.getElementById('new-project-form').style.display=''")
    browser.wait_for_selector("input[name='project_name']", timeout=3_000)

    # Cancel hides it
    browser.eval("document.getElementById('new-project-form').style.display='none'")
    hidden_after = browser.eval(
        "document.getElementById('new-project-form').style.display === 'none'"
    )
    assert hidden_after


def test_home_project_cards_visible(browser: AgentBrowser):
    """After creating a project, its card appears on the home page."""
    project_name = _unique_name("home-card")
    _create_project(browser, project_name)

    browser.open(APP_URL)
    browser.wait_for_selector(".project-card", timeout=5_000)
    page_text = browser.get_text("body")
    assert project_name in page_text


def test_home_project_card_navigates(browser: AgentBrowser):
    """Clicking a project card navigates to the project dashboard."""
    project_name = _unique_name("home-nav")
    _create_project(browser, project_name)

    browser.open(APP_URL)
    browser.wait_for_selector(f"a[href='/projects/{project_name}']", timeout=5_000)
    browser.eval(
        f"document.querySelector(\"a[href='/projects/{project_name}']\").click()"
    )
    browser.wait_for_url(f"*/projects/{project_name}", timeout=8_000)
    assert project_name in browser.get_text("h1")


def test_home_empty_state(browser: AgentBrowser):
    """When no projects exist the empty-state message is shown.

    Note: we can't guarantee no projects exist in a shared environment, so this
    test only runs the page load and checks the empty-state element is present
    in the DOM (it may be hidden by the project grid).
    """
    browser.open(APP_URL)
    browser.wait_for_selector("h1", timeout=5_000)
    # Either project cards OR the empty-state div must be present
    has_projects = browser.count(".project-card") > 0
    has_empty = browser.count("div[style*='text-align:center']") > 0 or \
                "No projects yet" in browser.get_text("body")
    assert has_projects or has_empty


# ---------------------------------------------------------------------------
# Group 2 — Project dashboard
# ---------------------------------------------------------------------------

def test_dashboard_loads(browser: AgentBrowser):
    """Project dashboard shows project name, submit form, and TraceVault link."""
    project_name = _unique_name("dash")
    _create_project(browser, project_name)

    assert project_name in browser.get_text("h1")
    assert browser.count("textarea[name='user_story']") == 1
    assert browser.count("a[href*='tracevault']") >= 1 or \
           browser.count("a[href*='localhost:8090']") >= 1


def test_dashboard_breadcrumb(browser: AgentBrowser):
    """Breadcrumb contains a link back to '/'."""
    project_name = _unique_name("dash-bc")
    _create_project(browser, project_name)
    assert browser.count("a[href='/']") >= 1


def test_dashboard_empty_tasks(browser: AgentBrowser):
    """A freshly created project shows 'No tasks yet'."""
    project_name = _unique_name("dash-empty")
    _create_project(browser, project_name)
    assert "No tasks yet" in browser.get_text("body")


def test_dashboard_recent_tasks_table(browser: AgentBrowser):
    """Seeded task appears in the recent tasks table with its status badge."""
    project_name = _unique_name("dash-tasks")
    task = _seed_task(project_name, status="completed", verdict="APPROVED")

    browser.open(task.dashboard_url)
    browser.wait_for_selector("table", timeout=5_000)
    page = browser.get_text("body")
    assert task.task_id[:8] in page
    assert "completed" in page.lower() or "approved" in page.lower()


def test_dashboard_task_row_navigates(browser: AgentBrowser):
    """Clicking a task row in the dashboard table opens task detail."""
    project_name = _unique_name("dash-row")
    task = _seed_task(project_name, status="completed")

    browser.open(task.dashboard_url)
    browser.wait_for_selector("table tbody tr", timeout=5_000)
    browser.eval(
        f"document.querySelector(\"tr[data-task='{task.task_id}']\").click()"
    )
    browser.wait_for_url(f"*/tasks/{task.task_id}", timeout=8_000)


def test_dashboard_view_all_tasks_link(browser: AgentBrowser):
    """'View all tasks' link points to the task list page."""
    project_name = _unique_name("dash-all")
    _seed_task(project_name, status="completed")

    browser.open(f"{APP_URL}/projects/{project_name}")
    browser.wait_for_selector(f"a[href='/projects/{project_name}/tasks']", timeout=5_000)
    browser.eval(
        f"document.querySelector(\"a[href='/projects/{project_name}/tasks']\").click()"
    )
    browser.wait_for_url(f"*/projects/{project_name}/tasks", timeout=8_000)


def test_dashboard_submit_task_redirects(browser: AgentBrowser):
    """Submitting a user story redirects to the new task's detail page."""
    project_name = _unique_name("dash-submit")
    _create_project(browser, project_name)

    _fill(browser, "textarea[name='user_story']", "Write a hello world Python script")
    _submit_form(browser, f"/projects/{project_name}/tasks")
    browser.wait_for_url(f"*/projects/{project_name}/tasks/*", timeout=15_000)
    assert f"/projects/{project_name}/tasks/" in browser.get_url()


# ---------------------------------------------------------------------------
# Group 3 — Task list
# ---------------------------------------------------------------------------

def test_task_list_shows_all_tasks(browser: AgentBrowser):
    """Task list page shows all seeded tasks."""
    project_name = _unique_name("tlist")
    task_a = _seed_task(project_name, status="completed", verdict="APPROVED")
    task_b = _seed_task(project_name, status="failed", error="something went wrong")

    browser.open(task_a.list_url)
    browser.wait_for_selector("table", timeout=5_000)
    page = browser.get_text("body")
    assert task_a.task_id[:8] in page
    assert task_b.task_id[:8] in page


def test_task_list_status_badges(browser: AgentBrowser):
    """Status and verdict badges are rendered for each task."""
    project_name = _unique_name("tlist-badges")
    _seed_task(project_name, status="completed", verdict="APPROVED")
    _seed_task(project_name, status="failed")

    browser.open(f"{APP_URL}/projects/{project_name}/tasks")
    browser.wait_for_selector("table", timeout=5_000)
    page = browser.get_text("body")
    assert "completed" in page.lower()
    assert "failed" in page.lower()


def test_task_list_empty(browser: AgentBrowser):
    """Task list shows 'No tasks yet' when project has no tasks."""
    project_name = _unique_name("tlist-empty")
    (_WORKSPACE_DIR / project_name).mkdir(parents=True, exist_ok=True)

    browser.open(f"{APP_URL}/projects/{project_name}/tasks")
    browser.wait_for_selector("table", timeout=5_000)
    assert "No tasks yet" in browser.get_text("body")


def test_task_list_row_navigates(browser: AgentBrowser):
    """Clicking a task row navigates to task detail."""
    project_name = _unique_name("tlist-nav")
    task = _seed_task(project_name, status="completed")

    browser.open(task.list_url)
    browser.wait_for_selector("table tbody tr.clickable", timeout=5_000)
    browser.eval(
        f"document.querySelector(\"tr[onclick*='{task.task_id}']\").click()"
    )
    browser.wait_for_url(f"*/tasks/{task.task_id}", timeout=8_000)


def test_task_list_breadcrumb(browser: AgentBrowser):
    """Task list breadcrumb links to home and to project dashboard."""
    project_name = _unique_name("tlist-bc")
    task = _seed_task(project_name, status="completed")

    browser.open(task.list_url)
    browser.wait_for_selector(".breadcrumb", timeout=5_000)
    assert browser.count("a[href='/']") >= 1
    assert browser.count(f"a[href='/projects/{project_name}']") >= 1


# ---------------------------------------------------------------------------
# Group 4 — Task detail: HITL waiting state
# ---------------------------------------------------------------------------

def test_hitl_spec_card_renders(browser: AgentBrowser):
    """Spec card shows title, requirements, acceptance criteria, and complexity."""
    project_name = _unique_name("hitl")
    spec = {
        "title": "Calculator App",
        "requirements": ["Add two numbers", "Subtract two numbers"],
        "acceptance_criteria": ["add(2,3) returns 5", "sub(5,2) returns 3"],
        "estimated_complexity": "simple",
        "notes": "",
    }
    task = _seed_task(project_name, status="waiting_hitl", spec=spec, hitl_approved=False)

    browser.open(task.detail_url)
    browser.wait_for_selector("#spec-section", timeout=8_000)
    page = browser.get_text("body")

    assert "Calculator App" in page
    assert "Add two numbers" in page
    assert "add(2,3) returns 5" in page
    assert "simple" in page.lower()


def test_hitl_spec_notes_renders(browser: AgentBrowser):
    """Notes field is shown when non-empty."""
    project_name = _unique_name("hitl-notes")
    spec = {
        "title": "Test Spec",
        "requirements": ["Do something"],
        "acceptance_criteria": ["It works"],
        "estimated_complexity": "medium",
        "notes": "This is an important note about edge cases.",
    }
    task = _seed_task(project_name, status="waiting_hitl", spec=spec, hitl_approved=False)

    browser.open(task.detail_url)
    browser.wait_for_selector("#spec-section", timeout=8_000)
    assert "important note about edge cases" in browser.get_text("body")


def test_hitl_approve_button_present_and_enabled(browser: AgentBrowser):
    """Approve button is visible and not disabled in waiting_hitl state."""
    project_name = _unique_name("hitl-btn")
    task = _seed_task(project_name, status="waiting_hitl", hitl_approved=False)

    browser.open(task.detail_url)
    browser.wait_for_selector("#approve-btn", timeout=8_000)
    assert browser.is_visible("#approve-btn")
    assert browser.is_enabled("#approve-btn")


def test_hitl_feedback_section_hidden_by_default(browser: AgentBrowser):
    """Feedback textarea is hidden until 'Request Changes' is clicked."""
    project_name = _unique_name("hitl-fb")
    task = _seed_task(project_name, status="waiting_hitl", hitl_approved=False)

    browser.open(task.detail_url)
    browser.wait_for_selector("#hitl-controls", timeout=8_000)

    hidden = browser.eval(
        "document.getElementById('feedback-section').style.display === 'none'"
    )
    assert hidden


def test_hitl_request_changes_toggles_feedback(browser: AgentBrowser):
    """Clicking 'Request Changes' reveals the feedback textarea."""
    project_name = _unique_name("hitl-toggle")
    task = _seed_task(project_name, status="waiting_hitl", hitl_approved=False)

    browser.open(task.detail_url)
    browser.wait_for_selector("#hitl-controls", timeout=8_000)

    # Click "Request Changes" button (second button in hitl-controls)
    browser.eval("toggleFeedback()")
    browser.wait_for_selector("#feedback-text", timeout=3_000)

    visible = browser.eval(
        "document.getElementById('feedback-section').style.display !== 'none'"
    )
    assert visible
    assert browser.is_visible("#feedback-text")


def test_hitl_progress_bar_approval_active(browser: AgentBrowser):
    """Progress bar marks 'Approval' as the active step in waiting_hitl state."""
    project_name = _unique_name("hitl-prog")
    task = _seed_task(project_name, status="waiting_hitl", hitl_approved=False)

    browser.open(task.detail_url)
    browser.wait_for_selector("#step-hitl", timeout=8_000)

    cls = browser.get_attr("#step-hitl", "class")
    assert "active" in cls


# ---------------------------------------------------------------------------
# Group 5 — Task detail: completed state
# ---------------------------------------------------------------------------

def test_completed_verdict_badge(browser: AgentBrowser):
    """Completed task shows APPROVED verdict badge."""
    project_name = _unique_name("done")
    task = _seed_task(project_name, status="completed", verdict="APPROVED")

    browser.open(task.detail_url)
    browser.wait_for_selector(".badge", timeout=8_000)
    assert "APPROVED" in browser.get_text("body")


def test_completed_qa_score(browser: AgentBrowser):
    """QA score is displayed on a completed task."""
    project_name = _unique_name("done-score")
    task = _seed_task(project_name, status="completed", qa_iteration=1)

    browser.open(task.detail_url)
    browser.wait_for_selector(".badge", timeout=8_000)
    # The qa_review_1.json we seed has score 0.92
    assert "0.92" in browser.get_text("body")


def test_completed_qa_iterations_shown(browser: AgentBrowser):
    """QA iteration count is displayed on a completed task."""
    project_name = _unique_name("done-iter")
    task = _seed_task(project_name, status="completed", qa_iteration=2)

    browser.open(task.detail_url)
    browser.wait_for_selector(".badge", timeout=8_000)
    # "QA iterations: 2" or similar should appear
    assert "2" in browser.get_text("body")


def test_completed_download_link_present(browser: AgentBrowser):
    """Download ZIP link is present with correct href on a completed task."""
    project_name = _unique_name("done-dl")
    task = _seed_task(project_name, status="completed")

    browser.open(task.detail_url)
    browser.wait_for_selector("a[href*='/download']", timeout=8_000)
    href = browser.get_attr("a[href*='/download']", "href")
    assert task.task_id in href


def test_completed_progress_bar_all_done(browser: AgentBrowser):
    """All pipeline progress steps show as 'done' on a completed task."""
    project_name = _unique_name("done-prog")
    task = _seed_task(project_name, status="completed")

    browser.open(task.detail_url)
    browser.wait_for_selector("#step-done", timeout=8_000)

    done_cls = browser.get_attr("#step-done", "class")
    assert "done" in done_cls

    # BA step also done
    ba_cls = browser.get_attr("#step-ba", "class")
    assert "done" in ba_cls


def test_completed_file_tabs_rendered(browser: AgentBrowser):
    """Tab bar renders one tab per file in files_created."""
    project_name = _unique_name("done-tabs")
    code = {
        "summary": "Created files",
        "files_created": [f"{project_name}/main.py", f"{project_name}/test_main.py"],
        "dependencies_installed": [],
        "tests_passed": True,
        "notes": "",
    }
    task = _seed_task(project_name, status="completed", code=code)

    browser.open(task.detail_url)
    browser.wait_for_selector(".tab-btn", timeout=8_000)
    tab_count = browser.count(".tab-btn")
    assert tab_count == 2


def test_completed_tab_switching(browser: AgentBrowser):
    """Clicking a tab shows its code block and hides the others."""
    project_name = _unique_name("done-tabswitch")
    code = {
        "summary": "Created files",
        "files_created": [f"{project_name}/main.py", f"{project_name}/utils.py"],
        "dependencies_installed": [],
        "tests_passed": True,
        "notes": "",
    }
    task = _seed_task(project_name, status="completed", code=code)

    browser.open(task.detail_url)
    browser.wait_for_selector(".tab-btn", timeout=8_000)

    # First tab is active by default
    first_active = browser.eval(
        "document.querySelector('.tab-content').classList.contains('active')"
    )
    assert first_active

    # Click second tab
    browser.eval("document.querySelectorAll('.tab-btn')[1].click()")

    second_active = browser.eval(
        "document.querySelectorAll('.tab-content')[1].classList.contains('active')"
    )
    assert second_active

    first_still_active = browser.eval(
        "document.querySelectorAll('.tab-content')[0].classList.contains('active')"
    )
    assert not first_still_active


# ---------------------------------------------------------------------------
# Group 6 — Task detail: failed state
# ---------------------------------------------------------------------------

def test_failed_error_message(browser: AgentBrowser):
    """Error message from meta.error is shown on a failed task."""
    project_name = _unique_name("fail")
    task = _seed_task(
        project_name, status="failed", error="LLM returned invalid JSON after 3 retries"
    )

    browser.open(task.detail_url)
    browser.wait_for_selector(".card", timeout=8_000)
    page = browser.get_text("body")
    assert "LLM returned invalid JSON after 3 retries" in page


def test_failed_status_badge(browser: AgentBrowser):
    """Task list shows 'failed' badge for a failed task."""
    project_name = _unique_name("fail-badge")
    _seed_task(project_name, status="failed", error="Pipeline error")

    browser.open(f"{APP_URL}/projects/{project_name}/tasks")
    browser.wait_for_selector("table", timeout=5_000)
    assert "failed" in browser.get_text("body").lower()


# ---------------------------------------------------------------------------
# Group 7 — Task detail: running state (requires live pipeline)
# ---------------------------------------------------------------------------

def test_running_spinner_shown(browser: AgentBrowser):
    """After submitting a task, the running panel with a spinner is visible."""
    project_name = _unique_name("run")
    _create_project(browser, project_name)

    _fill(browser, "textarea[name='user_story']", "Write a simple Python calculator")
    _submit_form(browser, f"/projects/{project_name}/tasks")

    # Should land on task detail almost immediately
    browser.wait_for_url(f"*/projects/{project_name}/tasks/*", timeout=15_000)
    browser.wait_for_selector(".spinner", timeout=10_000)
    assert browser.is_visible(".spinner")


def test_running_progress_bar_ba_active(browser: AgentBrowser):
    """BA step is marked active immediately after task submission."""
    project_name = _unique_name("run-prog")
    _create_project(browser, project_name)

    _fill(browser, "textarea[name='user_story']", "Write a simple Python hello world")
    _submit_form(browser, f"/projects/{project_name}/tasks")
    browser.wait_for_url(f"*/projects/{project_name}/tasks/*", timeout=15_000)
    browser.wait_for_selector("#step-ba", timeout=10_000)

    ba_cls = browser.get_attr("#step-ba", "class")
    assert "done" in ba_cls or "active" in ba_cls or "pending" not in ba_cls


def test_running_event_log_present(browser: AgentBrowser):
    """Event log div is present in the DOM during running state."""
    project_name = _unique_name("run-log")
    _create_project(browser, project_name)

    _fill(browser, "textarea[name='user_story']", "Write a Python hello world script")
    _submit_form(browser, f"/projects/{project_name}/tasks")
    browser.wait_for_url(f"*/projects/{project_name}/tasks/*", timeout=15_000)
    browser.wait_for_selector("#running-panel", timeout=10_000)

    # The event-log div must exist (it may or may not have entries yet)
    assert browser.count("#event-log") == 1


# ---------------------------------------------------------------------------
# Group 8 — Navigation & cross-cutting
# ---------------------------------------------------------------------------

def test_nav_brand_link(browser: AgentBrowser):
    """The 'Dev Team' brand link navigates to home from any page."""
    project_name = _unique_name("nav-brand")
    task = _seed_task(project_name, status="completed")

    browser.open(task.detail_url)
    browser.wait_for_selector("nav", timeout=5_000)
    browser.eval("document.querySelector(\"a[href='/']\").click()")
    browser.wait_for_url(APP_URL + "/", timeout=8_000)
    assert "Projects" in browser.get_text("h1")


def test_breadcrumb_home_link(browser: AgentBrowser):
    """Home breadcrumb on task detail navigates to '/'."""
    project_name = _unique_name("bc-home")
    task = _seed_task(project_name, status="completed")

    browser.open(task.detail_url)
    browser.wait_for_selector(".breadcrumb", timeout=5_000)
    browser.eval("document.querySelector(\".breadcrumb a[href='/']\").click()")
    browser.wait_for_url(APP_URL + "/", timeout=8_000)


def test_breadcrumb_project_link(browser: AgentBrowser):
    """Project breadcrumb on task detail navigates to the project dashboard."""
    project_name = _unique_name("bc-proj")
    task = _seed_task(project_name, status="completed")

    browser.open(task.detail_url)
    browser.wait_for_selector(".breadcrumb", timeout=5_000)
    browser.eval(
        f"document.querySelector(\".breadcrumb a[href='/projects/{project_name}']\").click()"
    )
    browser.wait_for_url(f"*/projects/{project_name}", timeout=8_000)
    assert project_name in browser.get_text("h1")


# ---------------------------------------------------------------------------
# Golden path (existing test kept for completeness)
# ---------------------------------------------------------------------------

def test_full_pipeline_hello_world(browser: AgentBrowser):
    """Golden path: create project → submit task → approve spec → completed with code."""
    project_name = _unique_name("golden")
    _create_project(browser, project_name)

    _fill(browser, "textarea[name='user_story']", "Write a hello world Python script")
    _submit_form(browser, f"/projects/{project_name}/tasks")
    browser.wait_for_url(f"*/projects/{project_name}/tasks/*", timeout=15_000)

    # Wait for BA to finish and HITL approval panel to appear
    browser.wait_for_selector("#spec-section", timeout=120_000)

    # Approve
    browser.eval("document.getElementById('approve-btn').click()")

    # Wait for completion
    browser.wait_for_function(
        "document.querySelector('.badge') && "
        "(document.querySelector('.badge').textContent.includes('APPROVED') || "
        "document.querySelector('.badge').textContent.includes('completed'))",
        timeout=300_000,
    )

    assert browser.count("pre.code-block") >= 1
    assert browser.count("a[href*='/download']") >= 1
