"""MCP client wrapper for streamable-HTTP transport.

Wraps the ``mcp`` library's ``streamablehttp_client`` + ``ClientSession`` into
a convenient async context manager with OpenAI-tool-schema conversion.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
try:
    from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client  # type: ignore[no-redef]


class MCPClient:
    """Async context manager that opens an MCP streamable-HTTP session.

    Usage::

        async with MCPClient("http://localhost:8082/mcp") as client:
            tools = await client.get_openai_tools()
            result = await client.call_tool("read_file", {"path": "/workspace/foo.py"})
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self) -> "MCPClient":
        transport = await self._exit_stack.enter_async_context(
            streamablehttp_client(self.url)
        )
        read, write, _ = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._exit_stack.aclose()
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPClient is not open — use as async context manager")
        return self._session

    async def get_openai_tools(self) -> list[dict]:
        """Return OpenAI-format tool schemas for all tools exposed by this MCP server."""
        result = await self.session.list_tools()
        return [_mcp_tool_to_openai(t) for t in result.tools]

    async def call_tool(self, name: str, args: dict) -> str:
        """Call an MCP tool and return the text result."""
        result = await self.session.call_tool(name, args)
        # result.content is a list of content items; prefer first text content
        if result.content:
            parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif hasattr(item, "data"):
                    parts.append(str(item.data))
            return "\n".join(parts)
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcp_tool_to_openai(tool: Any) -> dict:
    """Convert an MCP tool definition to OpenAI function-calling schema."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "parameters": schema,
        },
    }
