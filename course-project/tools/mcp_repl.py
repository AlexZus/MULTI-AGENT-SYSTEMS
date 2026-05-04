"""Python REPL MCP tool wrapper with project-relative path normalisation."""

from __future__ import annotations

from typing import Any

from agentflow.mcp import MCPClient
from tools.mcp_fs import PathNormalizer, _normalize_args, _patch_path_description

# REPL tools may also have output_file path args
_REPL_PATH_ARG_NAMES = frozenset({"path", "output_file", "directory"})


class MCPRepl:
    """Async context manager wrapping MCPClient for Python REPL operations.

    Normalizes ``path`` and ``output_file`` args; strips ``/workspace/`` from
    result text so the LLM never sees absolute MCP paths.

    Usage::

        async with MCPRepl(url, project_name="calculator") as repl:
            tools = repl.get_openai_tools()
            result = await repl.call_tool("run_pytest", {"path": "calculator/tests/"})
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

    async def __aenter__(self) -> "MCPRepl":
        self._client = MCPClient(self._url)
        await self._client.__aenter__()
        self._raw_tools = await self._client.get_openai_tools()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.__aexit__(*args)

    def get_openai_tools(self) -> list[dict]:
        return [_patch_path_description(t, self._project) for t in self._raw_tools]

    async def call_tool(self, name: str, args: dict) -> str:
        if self._client is None:
            raise RuntimeError("MCPRepl is not open")
        normalized_args = _normalize_repl_args(args, self._normalizer)
        result = await self._client.call_tool(name, normalized_args)
        return self._normalizer.normalize_result(result)


def _normalize_repl_args(args: dict, normalizer: PathNormalizer) -> dict:
    out = {}
    for k, v in args.items():
        if k in _REPL_PATH_ARG_NAMES and isinstance(v, str):
            out[k] = normalizer.to_mcp(v)
        else:
            out[k] = v
    return out
