"""Helper: convert MCP tool definitions to LangChain StructuredTool objects."""

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model


def mcp_tools_to_langchain(mcp_tools, mcp_client):
    """Convert MCP tool definitions to LangChain StructuredTool objects.

    Each returned tool, when called, invokes the corresponding MCP tool on
    the given ``mcp_client`` via an async coroutine.
    """
    lc_tools = []
    for tool in mcp_tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
        fields = {}
        for name, prop in props.items():
            py_type = type_map.get(prop.get("type"), str)
            default = ... if name in required else prop.get("default")
            fields[name] = (
                py_type if name in required else Optional[py_type],
                Field(default=default, description=prop.get("description", "")),
            )

        args_model = create_model(f"{tool.name}_args", **fields) if fields else None

        _name, _client = tool.name, mcp_client

        async def _invoke(_name=_name, _client=_client, **kwargs):
            result = await _client.call_tool(_name, kwargs)
            # Prefer .data (FastMCP unwrapped result); fall back to first text content
            if result.data is not None:
                return str(result.data)
            if result.content:
                return result.content[0].text
            return ""

        lc_tools.append(StructuredTool.from_function(
            coroutine=_invoke,
            name=tool.name,
            description=tool.description or tool.name,
            args_schema=args_model,
        ))

    return lc_tools
