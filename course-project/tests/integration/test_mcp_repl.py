"""Integration tests for MCPRepl — requires running Python REPL MCP server.

Each test opens its own MCP connection to avoid anyio task group teardown issues.
"""

import os

import pytest

from tools.mcp_repl import MCPRepl

pytestmark = pytest.mark.asyncio

MCP_REPL_URL = os.getenv("MCP_REPL_URL", "http://localhost:8083/mcp")
PROJECT = "test_integ"


class TestMCPRepl:
    async def test_get_tools_returns_list(self):
        async with MCPRepl(MCP_REPL_URL, project_name=PROJECT) as repl:
            tools = repl.get_openai_tools()
            assert len(tools) > 0

    async def test_python_repl_executes(self):
        async with MCPRepl(MCP_REPL_URL, project_name=PROJECT) as repl:
            tools = {t["function"]["name"] for t in repl.get_openai_tools()}
            repl_tool = next(
                (n for n in tools if "python" in n.lower() or "repl" in n.lower()), None
            )
            if not repl_tool:
                pytest.skip("python_repl tool not available")
            result = await repl.call_tool(repl_tool, {"code": "print(2 + 2)"})
            assert "4" in result

    async def test_run_pytest(self):
        async with MCPRepl(MCP_REPL_URL, project_name=PROJECT) as repl:
            tools = {t["function"]["name"] for t in repl.get_openai_tools()}
            pytest_tool = next((n for n in tools if "pytest" in n.lower()), None)
            if not pytest_tool:
                pytest.skip("run_pytest tool not available")
            result = await repl.call_tool(pytest_tool, {"path": f"{PROJECT}/"})
            assert isinstance(result, str)

    async def test_no_workspace_in_result(self):
        async with MCPRepl(MCP_REPL_URL, project_name=PROJECT) as repl:
            tools = {t["function"]["name"] for t in repl.get_openai_tools()}
            repl_tool = next(
                (n for n in tools if "python" in n.lower() or "repl" in n.lower()), None
            )
            if not repl_tool:
                pytest.skip("python_repl tool not available")
            result = await repl.call_tool(repl_tool, {"code": "import os; print(os.getcwd())"})
            assert "/workspace/" not in result
