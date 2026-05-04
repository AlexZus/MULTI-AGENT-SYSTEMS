"""Integration tests for MCPFilesystem — requires running filesystem MCP server.

Note: MCPFilesystem must be opened/closed within the same async task because
the mcp library uses anyio task groups internally. Each test opens its own
connection to avoid cross-task teardown errors.
"""

import os
import uuid

import pytest

from tools.mcp_fs import MCPFilesystem

pytestmark = pytest.mark.asyncio

MCP_FS_URL = os.getenv("MCP_FILESYSTEM_URL", "http://localhost:8082/mcp")
PROJECT = "test_integ"


class TestMCPFilesystem:
    async def test_get_tools_returns_list(self):
        async with MCPFilesystem(MCP_FS_URL, project_name=PROJECT) as fs:
            tools = fs.get_openai_tools()
            assert isinstance(tools, list)
            assert len(tools) > 0
            names = [t["function"]["name"] for t in tools]
            assert any(
                "file" in n.lower() or "write" in n.lower() or "read" in n.lower()
                for n in names
            )

    async def test_path_description_patched(self):
        async with MCPFilesystem(MCP_FS_URL, project_name=PROJECT) as fs:
            tools = fs.get_openai_tools()
            for t in tools:
                props = t["function"]["parameters"].get("properties", {})
                if "path" in props:
                    desc = props["path"].get("description", "")
                    assert PROJECT in desc or "project root" in desc

    async def test_write_and_read_file(self):
        async with MCPFilesystem(MCP_FS_URL, project_name=PROJECT) as fs:
            filename = f"{PROJECT}/test_{uuid.uuid4().hex[:6]}.txt"
            content = "Hello from integration test!"
            tools_list = fs.get_openai_tools()
            tool_names = {t["function"]["name"] for t in tools_list}
            write_tool  = next((n for n in tool_names if "write" in n.lower() and "media" not in n.lower()), None)
            # Prefer read_text_file/read_file over read_media_file (binary content fails MCP text validation)
            read_tool = next(
                (n for pref in ("read_text_file", "read_file") for n in [pref] if n in tool_names),
                next((n for n in sorted(tool_names) if "read" in n.lower() and "media" not in n.lower()), None),
            )
            mkdir_tool  = next((n for n in tool_names if "creat" in n.lower() and "dir" in n.lower()), None)
            if not write_tool or not read_tool:
                pytest.skip("write_file/read_file tools not available")

            # Ensure project directory exists
            if mkdir_tool:
                await fs.call_tool(mkdir_tool, {"path": PROJECT})

            await fs.call_tool(write_tool, {"path": filename, "content": content})
            result = await fs.call_tool(read_tool, {"path": filename})
            assert content in result

    async def test_no_workspace_in_result(self):
        async with MCPFilesystem(MCP_FS_URL, project_name=PROJECT) as fs:
            tools = {t["function"]["name"] for t in fs.get_openai_tools()}
            list_tool = next(
                (n for n in tools if "list" in n.lower() or "dir" in n.lower()), None
            )
            if not list_tool:
                pytest.skip("directory listing tool not available")
            result = await fs.call_tool(list_tool, {"path": PROJECT})
            assert "/workspace/" not in result
