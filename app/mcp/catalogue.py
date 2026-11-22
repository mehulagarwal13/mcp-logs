"""Describe the tools this MCP server has actually registered.

`mcp_server.list_tools()` is the same registry the MCP protocol's `tools/list`
serves, so this is the single source of truth: nothing here lists a tool by name.
Used by the server entrypoint (publish at startup) and by the mock-catalogue export.
"""

from __future__ import annotations

from app.core.observability.schemas import McpToolSpec


async def describe_registered_tools(mcp_server) -> list[McpToolSpec]:
    tools = await mcp_server.list_tools()
    return sorted(
        (
            McpToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema or {}),
            )
            for tool in tools
        ),
        key=lambda spec: spec.name,
    )
