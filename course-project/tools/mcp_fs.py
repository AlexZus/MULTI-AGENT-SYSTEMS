"""Filesystem MCP tool wrapper with project-relative path normalisation."""

from __future__ import annotations

import re
from typing import Any

from agentflow.mcp import MCPClient


class PathNormalizer:
    """Translates between LLM-visible project-relative paths and MCP absolute paths.

    The LLM sees: ``calculator/main.py``
    MCP receives: ``/workspace/calculator/main.py``
    """

    def __init__(self, project_name: str, workspace_root: str = "/workspace") -> None:
        self._root = workspace_root.rstrip("/")
        self._project = project_name

    def to_mcp(self, path: str) -> str:
        """``calculator/main.py`` → ``/workspace/calculator/main.py``

        Handles two common LLM mistakes seen in production:
        - Empty path: defaults to the project root directory.
        - ``/workspace`` or ``/workspace/project`` prefix: stripped before
          re-applying the project prefix so the path is not doubled
          (e.g. ``/workspace`` → ``/workspace/calculator`` not
          ``/workspace/calculator/workspace``).
        """
        path = path.lstrip("/")
        # Strip a bare workspace root or any workspace-prefixed path the LLM
        # may pass (e.g. "/workspace" → "", "/workspace/foo.py" → "foo.py").
        workspace_rel = self._root.lstrip("/")  # e.g. "workspace"
        if path == workspace_rel:
            path = ""
        elif path.startswith(workspace_rel + "/"):
            path = path[len(workspace_rel) + 1:]
        # Empty path (or path that became empty after stripping) defaults to project root.
        if not path or (not path.startswith(self._project + "/") and path != self._project):
            path = f"{self._project}/{path}".rstrip("/")
        return f"{self._root}/{path}"

    def from_mcp(self, path: str) -> str:
        """/workspace/calculator/main.py → main.py (strips workspace+project prefix)."""
        full_prefix = f"{self._root}/{self._project}/"
        if path.startswith(full_prefix):
            return path[len(full_prefix):]
        if path == f"{self._root}/{self._project}":
            return ""
        return path

    def normalize_result(self, text: str) -> str:
        """Strip /workspace/{project}/ from tool result text so LLM sees plain relative paths."""
        return text.replace(f"{self._root}/{self._project}/", "")


class MCPFilesystem:
    """Async context manager wrapping MCPClient for filesystem operations.

    Applies PathNormalizer to all path arguments and result text so the LLM
    always sees project-relative paths.

    Usage::

        async with MCPFilesystem(url, project_name="calculator") as fs:
            tools = fs.get_openai_tools()
            result = await fs.call_tool("read_file", {"path": "calculator/main.py"})
    """

    def __init__(
        self,
        url: str,
        project_name: str,
        workspace_root: str = "/workspace",
    ) -> None:
        self._url = url
        self._project = project_name
        self._normalizer = PathNormalizer(project_name, workspace_root)
        self._client: MCPClient | None = None
        self._raw_tools: list[dict] = []

    async def __aenter__(self) -> "MCPFilesystem":
        self._client = MCPClient(self._url)
        await self._client.__aenter__()
        self._raw_tools = await self._client.get_openai_tools()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.__aexit__(*args)

    def get_openai_tools(self) -> list[dict]:
        """Return tool schemas with path descriptions updated for project-relative paths."""
        tools = []
        for tool in self._raw_tools:
            tool = _patch_path_description(tool, self._project)
            tools.append(tool)
        return tools

    async def call_tool(self, name: str, args: dict) -> str:
        """Normalize path args, call MCP tool, normalize result text."""
        if self._client is None:
            raise RuntimeError("MCPFilesystem is not open")
        normalized_args = _normalize_args(args, self._normalizer)
        result = await self._client.call_tool(name, normalized_args)
        return self._normalizer.normalize_result(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATH_ARG_NAMES = frozenset({"path", "source", "destination", "directory"})


def _normalize_args(args: dict, normalizer: PathNormalizer) -> dict:
    out = {}
    for k, v in args.items():
        if k in _PATH_ARG_NAMES and isinstance(v, str):
            # Always normalise path args — including empty string, which to_mcp()
            # maps to the project root.  Previously the `and v` guard let empty
            # strings through to the MCP server where they resolved to the
            # container working directory (/app) and caused "Access denied".
            out[k] = normalizer.to_mcp(v)
        else:
            out[k] = v
    return out


def _patch_path_description(tool: dict, project_name: str) -> dict:
    """Update parameter descriptions to mention project-relative paths."""
    import copy
    tool = copy.deepcopy(tool)
    fn = tool.get("function", {})
    params = fn.get("parameters", {})
    properties = params.get("properties", {})
    for param_name in _PATH_ARG_NAMES:
        if param_name in properties:
            prop = properties[param_name]
            desc = prop.get("description", "")
            if "/workspace/" not in desc and "project" not in desc.lower():
                prop["description"] = (
                    "Path relative to project root, e.g. `src/main.py`. "
                    + desc
                ).strip()
    return tool
