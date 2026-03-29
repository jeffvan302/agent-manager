"""Optional adapter for MCP (Model Context Protocol) tool servers."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterable, Mapping
from typing import Any

from agent_manager.plugins.base import Plugin
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec, normalize_tool_result


class MCPToolAdapter(BaseTool):
    """Adapt an MCP tool definition into the agent_manager tool contract.

    MCP tools expose a JSON-Schema ``inputSchema`` and are invoked through
    the MCP client's ``call_tool`` method.  This adapter normalises
    those conventions so that MCP-provided tools can be registered and
    executed through the standard tool registry.
    """

    def __init__(self, *, tool_definition: Mapping[str, Any], client: Any) -> None:
        self._client = client
        self._tool_def = dict(tool_definition)
        self.spec = ToolSpec(
            name=str(tool_definition.get("name", "mcp_tool")),
            description=str(tool_definition.get("description", "")),
            input_schema=self._extract_schema(tool_definition),
            tags=["integration", "mcp"],
        )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        result = await self._call_tool(arguments)
        return self._normalise_result(result)

    async def _call_tool(self, arguments: dict[str, Any]) -> Any:
        call_tool = getattr(self._client, "call_tool", None)
        if call_tool is None:
            raise TypeError(
                f"MCP client does not expose a 'call_tool' method "
                f"(got {type(self._client).__name__})."
            )
        value = call_tool(self.spec.name, arguments)
        if inspect.isawaitable(value):
            value = await value
        return value

    def _normalise_result(self, raw: Any) -> ToolResult:
        # MCP call_tool can return various shapes.  Handle the most common:
        # 1. A dict-like with "content" list (standard MCP CallToolResult)
        # 2. A plain dict or string
        # 3. An object with .content attribute
        if isinstance(raw, Mapping):
            content_list = raw.get("content")
            is_error = bool(raw.get("isError", False))
            if isinstance(content_list, list):
                text_parts = [
                    str(item.get("text", ""))
                    if isinstance(item, Mapping)
                    else str(getattr(item, "text", item))
                    for item in content_list
                ]
                joined = "\n".join(text_parts)
                return ToolResult(
                    tool_name=self.spec.name,
                    ok=not is_error,
                    output=joined,
                    error=joined if is_error else None,
                )
            return ToolResult(
                tool_name=self.spec.name,
                ok=not is_error,
                output=dict(raw),
                error=str(raw) if is_error else None,
            )

        # Object with .content attribute (e.g. MCP SDK CallToolResult)
        content_attr = getattr(raw, "content", None)
        is_error = bool(getattr(raw, "isError", False))
        if isinstance(content_attr, list):
            text_parts = [
                str(getattr(item, "text", item)) for item in content_attr
            ]
            joined = "\n".join(text_parts)
            return ToolResult(
                tool_name=self.spec.name,
                ok=not is_error,
                output=joined,
                error=joined if is_error else None,
            )

        return normalize_tool_result(self.spec.name, raw)

    @staticmethod
    def _extract_schema(tool_definition: Mapping[str, Any]) -> dict[str, Any]:
        schema = tool_definition.get("inputSchema") or tool_definition.get("input_schema")
        if isinstance(schema, Mapping):
            return dict(schema)
        return {"type": "object", "properties": {}}


class MCPToolsPlugin(Plugin):
    """Register MCP tools discovered from an MCP client on an AgentSession.

    Usage::

        from mcp import ClientSession
        # ... initialise mcp_client ...

        tools_response = await mcp_client.list_tools()
        plugin = MCPToolsPlugin(
            client=mcp_client,
            tool_definitions=tools_response.tools,
        )
        session = AgentSession(plugins=[plugin])
    """

    name = "mcp-tools"
    description = "Expose MCP server tools through the unified tool registry."

    def __init__(
        self,
        *,
        client: Any,
        tool_definitions: Iterable[Mapping[str, Any] | Any],
        replace: bool = True,
    ) -> None:
        self._client = client
        self._tool_defs = list(tool_definitions)
        self._replace = replace

    def register(self, target: Any) -> None:
        for tool_def in self._tool_defs:
            definition = self._as_dict(tool_def)
            target.register_tool(
                MCPToolAdapter(tool_definition=definition, client=self._client),
                replace=self._replace,
            )

    @staticmethod
    def _as_dict(tool_def: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Convert an MCP Tool object or plain dict into a plain dict."""
        if isinstance(tool_def, Mapping):
            return dict(tool_def)
        # MCP SDK Tool objects expose .name, .description, .inputSchema
        result: dict[str, Any] = {}
        for attr in ("name", "description", "inputSchema", "input_schema"):
            value = getattr(tool_def, attr, None)
            if value is not None:
                result[attr] = value
        return result
