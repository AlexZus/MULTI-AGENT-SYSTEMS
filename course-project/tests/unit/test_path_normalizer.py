"""Unit tests for PathNormalizer — no external services required.

Design contract:
  to_mcp()          — LLM plain relative path → absolute MCP path (adds prefix)
  from_mcp()        — absolute MCP path → LLM plain path (strips workspace+project)
  normalize_result() — strip /workspace/{project}/ from tool output text shown to LLM
  _patch_path_description() — tool docs must not expose workspace or project name
"""

import pytest

from tools.mcp_fs import PathNormalizer, _normalize_args, _patch_path_description


class TestPathNormalizerToMcp:
    """to_mcp: adds workspace+project prefix to LLM-supplied plain relative path."""

    def _n(self, project="calculator"):
        return PathNormalizer(project)

    def test_plain_file_adds_prefix(self):
        n = self._n()
        assert n.to_mcp("main.py") == "/workspace/calculator/main.py"

    def test_subdir_file(self):
        n = self._n()
        assert n.to_mcp("src/utils/helper.py") == "/workspace/calculator/src/utils/helper.py"

    def test_strips_leading_slash(self):
        n = self._n()
        assert n.to_mcp("/main.py") == "/workspace/calculator/main.py"

    def test_path_with_project_prefix_not_doubled(self):
        """If path already has project prefix, don't double it."""
        n = self._n()
        assert n.to_mcp("calculator/main.py") == "/workspace/calculator/main.py"

    def test_project_name_alone(self):
        n = self._n()
        assert n.to_mcp("calculator") == "/workspace/calculator"

    def test_different_project(self):
        n = PathNormalizer("my-webapp", "/workspace")
        assert n.to_mcp("api/routes.py") == "/workspace/my-webapp/api/routes.py"

    def test_custom_workspace_root(self):
        n = PathNormalizer("proj", "/mnt/storage")
        assert n.to_mcp("main.py") == "/mnt/storage/proj/main.py"

    def test_empty_path_resolves_to_project_root(self):
        """Empty string defaults to the project root, not the container workdir."""
        n = self._n()
        assert n.to_mcp("") == "/workspace/calculator"


class TestPathNormalizerFromMcp:
    """from_mcp: strips workspace+project prefix so LLM sees plain relative paths."""

    def _n(self, project="calculator"):
        return PathNormalizer(project)

    def test_strips_workspace_and_project_prefix(self):
        """LLM must never see workspace or project directory in returned paths."""
        n = self._n()
        assert n.from_mcp("/workspace/calculator/main.py") == "main.py"

    def test_nested_path(self):
        n = self._n()
        assert n.from_mcp("/workspace/calculator/src/app.py") == "src/app.py"

    def test_project_root_returns_empty(self):
        """Project root itself maps to empty string."""
        n = self._n()
        assert n.from_mcp("/workspace/calculator") == ""

    def test_passthrough_non_workspace_path(self):
        """Paths not starting with workspace root pass through unchanged."""
        n = self._n()
        assert n.from_mcp("some/other/path") == "some/other/path"


class TestPathNormalizerNormalizeResult:
    """normalize_result: strip /workspace/{project}/ so LLM sees plain relative paths."""

    def test_strips_full_prefix(self):
        """LLM must see 'main.py', not 'calculator/main.py'."""
        n = PathNormalizer("calculator")
        text = "File written to /workspace/calculator/main.py successfully."
        assert n.normalize_result(text) == "File written to main.py successfully."

    def test_strips_multiple_occurrences(self):
        n = PathNormalizer("calculator")
        text = "/workspace/calculator/a.py and /workspace/calculator/b.py"
        result = n.normalize_result(text)
        assert "workspace" not in result
        assert "calculator/" not in result
        assert "a.py" in result and "b.py" in result

    def test_no_workspace_in_text_unchanged(self):
        n = PathNormalizer("calculator")
        assert n.normalize_result("No paths here.") == "No paths here."

    def test_custom_root(self):
        n = PathNormalizer("proj", "/data")
        assert n.normalize_result("Created /data/proj/main.py") == "Created main.py"

    def test_pytest_output_stripped(self):
        """Typical pytest failure line shows file-relative path only."""
        n = PathNormalizer("myapp")
        text = "FAILED /workspace/myapp/tests/test_foo.py::test_bar - AssertionError"
        result = n.normalize_result(text)
        assert "myapp" not in result
        assert "workspace" not in result
        assert "tests/test_foo.py::test_bar" in result


class TestNormalizeArgs:
    def _n(self):
        return PathNormalizer("calculator")

    def test_normalizes_plain_path_arg(self):
        n = self._n()
        result = _normalize_args({"path": "main.py"}, n)
        assert result["path"] == "/workspace/calculator/main.py"

    def test_normalizes_source_and_destination(self):
        n = self._n()
        result = _normalize_args({"source": "a.py", "destination": "b.py"}, n)
        assert result["source"] == "/workspace/calculator/a.py"
        assert result["destination"] == "/workspace/calculator/b.py"

    def test_leaves_non_path_args_unchanged(self):
        n = self._n()
        result = _normalize_args({"content": "print(1)", "path": "x.py"}, n)
        assert result["content"] == "print(1)"
        assert result["path"] == "/workspace/calculator/x.py"

    def test_empty_path_normalizes_to_project_root(self):
        """Empty path must resolve to project root, not pass through as '' to MCP."""
        n = self._n()
        result = _normalize_args({"path": ""}, n)
        assert result["path"] == "/workspace/calculator"


class TestPatchPathDescription:
    def _tool(self):
        return {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The file path"},
                    },
                    "required": ["path"],
                },
            },
        }

    def test_description_mentions_relative_path(self):
        """Tool description should guide the LLM toward relative paths."""
        tool = _patch_path_description(self._tool(), "myproject")
        desc = tool["function"]["parameters"]["properties"]["path"]["description"]
        assert "relative" in desc.lower() or "project root" in desc

    def test_description_does_not_expose_project_name(self):
        """LLM must not see the project directory name in tool descriptions."""
        tool = _patch_path_description(self._tool(), "myproject")
        desc = tool["function"]["parameters"]["properties"]["path"]["description"]
        assert "myproject" not in desc

    def test_description_does_not_expose_workspace(self):
        """LLM must not see /workspace in tool descriptions."""
        tool = _patch_path_description(self._tool(), "myproject")
        desc = tool["function"]["parameters"]["properties"]["path"]["description"]
        assert "/workspace" not in desc

    def test_does_not_duplicate_if_already_patched(self):
        tool = self._tool()
        tool["function"]["parameters"]["properties"]["path"]["description"] = "Path relative to project root"
        patched = _patch_path_description(tool, "myproject")
        desc = patched["function"]["parameters"]["properties"]["path"]["description"]
        assert desc.count("project root") <= 1

    def test_deep_copy_no_mutation(self):
        tool = self._tool()
        original_desc = tool["function"]["parameters"]["properties"]["path"]["description"]
        _patch_path_description(tool, "myproject")
        assert tool["function"]["parameters"]["properties"]["path"]["description"] == original_desc
