"""Optional adapters for OpenAPI-like HTTP operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

from agent_manager.plugins.base import Plugin
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from agent_manager.tools.builtins.http import HttpRequestTool


@dataclass(slots=True)
class OpenAPIOperation:
    name: str
    description: str
    base_url: str
    path: str
    method: str = "GET"
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    headers: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class OpenAPIToolAdapter(BaseTool):
    """Expose a simple HTTP/OpenAPI operation as a runtime tool."""

    def __init__(
        self,
        operation: OpenAPIOperation,
        *,
        http_tool: HttpRequestTool | None = None,
    ) -> None:
        self.operation = operation
        self.http_tool = http_tool or HttpRequestTool()
        self.spec = ToolSpec(
            name=operation.name,
            description=operation.description,
            input_schema=dict(operation.input_schema),
            tags=["integration", "openapi", "network", *operation.tags],
            permissions=["network:request"],
        )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        url = self._build_url(arguments)
        headers = dict(self.operation.headers)
        if isinstance(arguments.get("headers"), Mapping):
            headers.update(
                {str(key): str(value) for key, value in arguments["headers"].items()}
            )

        payload: dict[str, Any] = {
            "url": url,
            "method": self.operation.method.upper(),
            "headers": headers,
        }
        if "timeout_seconds" in arguments:
            payload["timeout_seconds"] = arguments["timeout_seconds"]
        if "body" in arguments:
            if isinstance(arguments["body"], Mapping):
                payload["json"] = dict(arguments["body"])
            else:
                payload["body"] = str(arguments["body"])

        response = await self.http_tool.invoke(payload, context)
        metadata = dict(response.metadata)
        metadata["operation"] = self.operation.name
        return ToolResult(
            tool_name=self.spec.name,
            ok=response.ok,
            output=response.output,
            error=response.error,
            metadata=metadata,
        )

    def _build_url(self, arguments: dict[str, Any]) -> str:
        path = self.operation.path
        for key, value in arguments.items():
            placeholder = f"{{{key}}}"
            if placeholder in path:
                path = path.replace(placeholder, quote(str(value), safe=""))

        base_url = self.operation.base_url.rstrip("/")
        url = f"{base_url}/{path.lstrip('/')}"
        query = arguments.get("query")
        if isinstance(query, Mapping):
            cleaned = {
                str(key): value
                for key, value in query.items()
                if value is not None
            }
            if cleaned:
                url = f"{url}?{urlencode(cleaned)}"
        return url


class OpenAPIToolsPlugin(Plugin):
    """Register OpenAPI-like operations as first-class tools."""

    name = "openapi-tools"
    description = "Expose simple OpenAPI-style HTTP operations through the tool registry."

    def __init__(
        self,
        operations: Iterable[OpenAPIOperation],
        *,
        http_tool: HttpRequestTool | None = None,
        replace: bool = True,
    ) -> None:
        self._operations = list(operations)
        self._http_tool = http_tool
        self._replace = replace

    def register(self, target: Any) -> None:
        for operation in self._operations:
            target.register_tool(
                OpenAPIToolAdapter(operation, http_tool=self._http_tool),
                replace=self._replace,
            )
