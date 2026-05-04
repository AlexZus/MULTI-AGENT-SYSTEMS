"""
pytest adapter for the agent-browser CLI.

Wraps agent-browser commands in a Python API that:
  - Isolates each test in its own named browser session
  - Parses --json output for assertions
  - Enforces timeouts via subprocess.run(timeout=...) because
    agent-browser's wait commands do not accept a --timeout flag
  - Raises AssertionError for assertion failures (compatible with pytest)
"""
import json
import subprocess
import uuid
from typing import Any

import pytest


class AgentBrowserError(Exception):
    """Raised when an agent-browser command fails at the OS / CLI level."""


class AgentBrowser:
    """
    Thin wrapper around the agent-browser CLI.

    Each instance owns a single named session; the same browser process
    is reused across all commands within a test.
    """

    def __init__(self, session: str) -> None:
        self._session = session
        self._base = ["agent-browser", "--session", session]

    # ── Low-level runners ──────────────────────────────────────────────────

    def _run(self, *args: str, timeout: float = 30) -> None:
        """Run a command; raise AgentBrowserError on non-zero exit."""
        result = subprocess.run(
            self._base + list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise AgentBrowserError(
                f"Command failed ({' '.join(args)}): {result.stderr or result.stdout}"
            )

    def _run_json(self, *args: str, timeout: float = 30) -> dict:
        """Run a command with --json; return parsed response dict."""
        result = subprocess.run(
            self._base + ["--json"] + list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AgentBrowserError(
                f"Non-JSON output ({' '.join(args)}): {result.stdout!r}"
            ) from exc

    # ── Navigation ─────────────────────────────────────────────────────────

    def open(self, url: str) -> None:
        """Navigate to a URL."""
        self._run("open", url)

    # ── Interaction ────────────────────────────────────────────────────────

    def click(self, selector: str) -> None:
        """
        Click the first element matching a CSS selector.

        Note: for buttons that are re-created on every render (i.e. their
        click handlers are registered via addEventListener after innerHTML is
        set, not via an onclick attribute), prefer ``browser.eval(
        "document.querySelector('selector').click()")`` which fires the DOM
        event directly and is unaffected by accessibility-tree caching.
        """
        self._run("find", "first", selector, "click")

    # ── Query ──────────────────────────────────────────────────────────────

    def get_attr(self, selector: str, attr: str) -> str:
        """Return an attribute value; empty string if element or attr not found."""
        r = self._run_json("get", "attr", selector, attr)
        return r["data"]["value"] if r["success"] else ""

    def get_text(self, selector: str) -> str:
        """Return text content of an element; empty string if not found."""
        r = self._run_json("get", "text", selector)
        return r["data"]["text"] if r["success"] else ""

    def get_url(self) -> str:
        """Return the current page URL."""
        r = self._run_json("get", "url")
        return r["data"]["url"] if r["success"] else ""

    def count(self, selector: str) -> int:
        """Return count of elements matching a CSS selector."""
        r = self._run_json("get", "count", selector)
        return r["data"]["count"] if r["success"] else 0

    def is_visible(self, selector: str) -> bool:
        """Return True if the element is visible."""
        r = self._run_json("is", "visible", selector)
        return bool(r["success"] and r["data"]["visible"])

    def is_enabled(self, selector: str) -> bool:
        """Return True if the element is enabled (not disabled)."""
        r = self._run_json("is", "enabled", selector)
        return bool(r["success"] and r["data"]["enabled"])

    def eval(self, js: str) -> Any:
        """
        Evaluate a JavaScript expression in the page context and return the result.

        Uses stdin to avoid shell-escaping issues with complex expressions.
        """
        result = subprocess.run(
            self._base + ["--json", "eval", "--stdin"],
            input=js,
            capture_output=True,
            text=True,
            timeout=10,
        )
        r = json.loads(result.stdout)
        if not r["success"]:
            raise AgentBrowserError(f"eval failed: {r['error']}\nJS: {js}")
        return r["data"]["result"]

    # ── Wait ───────────────────────────────────────────────────────────────
    #
    # agent-browser's wait commands do not accept a --timeout flag (only
    # --download mode does).  We enforce timeouts by passing a subprocess
    # timeout that is slightly larger than the desired test timeout.  If the
    # process hasn't returned within that window, we kill it and raise.

    def _wait_timeout(self, timeout_ms: int) -> float:
        """Convert ms test timeout to a subprocess wall-clock timeout (seconds)."""
        return timeout_ms / 1000 + 3  # 3 s buffer for CLI startup overhead

    def wait_for_function(self, js: str, *, timeout: int = 5_000) -> None:
        """
        Wait until a JS expression becomes truthy.

        Raises AssertionError if the condition does not become true within
        *timeout* milliseconds.
        """
        try:
            r = self._run_json(
                "wait", "--fn", js, timeout=self._wait_timeout(timeout)
            )
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"wait_for_function timed out ({timeout} ms): {js}"
            )
        if not r["success"]:
            raise AssertionError(f"wait_for_function failed: {js}\n{r['error']}")

    def wait_for_selector(self, selector: str, *, timeout: int = 5_000) -> None:
        """
        Wait for a CSS selector to be present in the DOM.

        Raises AssertionError if the element does not appear within *timeout* ms.
        """
        try:
            r = self._run_json(
                "wait", selector, timeout=self._wait_timeout(timeout)
            )
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"wait_for_selector timed out ({timeout} ms): {selector}"
            )
        if not r["success"]:
            raise AssertionError(
                f"wait_for_selector failed: {selector}\n{r['error']}"
            )

    def wait_for_url(self, pattern: str, *, timeout: int = 5_000) -> None:
        """
        Wait for the current URL to match a glob pattern.

        Uses wait --fn to check window.location.href so it works whether the
        navigation already happened or is still pending (wait --url is
        event-based and misses navigations that complete before it starts).

        Raises AssertionError if the URL does not match within *timeout* ms.
        """
        # Convert glob pattern to a JS substring/prefix check.
        # Strip both leading and trailing "*" wildcards; the remainder is
        # treated as a substring that must appear in window.location.href.
        needle = pattern.strip("*")
        js = f"window.location.href.includes('{needle}')"
        self.wait_for_function(js, timeout=timeout)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the browser session."""
        subprocess.run(
            self._base + ["close"], capture_output=True, text=True, timeout=10
        )


# ── pytest fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def browser() -> AgentBrowser:
    """
    Fresh isolated browser session for each test.

    Yields an AgentBrowser instance bound to a unique session name.
    Automatically closes the session after the test (pass or fail).
    """
    session = f"pytest-{uuid.uuid4().hex[:12]}"
    ab = AgentBrowser(session)
    yield ab
    ab.close()
