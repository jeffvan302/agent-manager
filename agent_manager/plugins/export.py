"""Export bridges: convert agent_manager ToolSpec to external formats."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_manager.tools.base import ToolSpec


def to_openapi_schema(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into an OpenAPI-style operation schema.

    Returns a dict suitable for embedding in an OpenAPI paths object::

        schema = to_openapi_schema(my_tool.spec)
        # {"operationId": "...", "summary": "...", ...}
    """
    schema: dict[str, Any] = {
        "operationId": spec.name,
        "summary": spec.description,
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": dict(spec.input_schema) if spec.input_schema else {},
                },
            },
        },
    }
    if spec.output_schema:
        schema["responses"] = {
            "200": {
                "description": "Successful response",
                "content": {
                    "application/json": {
                        "schema": dict(spec.output_schema),
                    },
                },
            },
        }
    if spec.tags:
        schema["tags"] = list(spec.tags)
    return schema


def to_mcp_tool_definition(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into an MCP-compatible tool definition.

    Returns a dict matching the MCP ``Tool`` schema::

        mcp_def = to_mcp_tool_definition(my_tool.spec)
        # {"name": "...", "description": "...", "inputSchema": {...}}
    """
    definition: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": dict(spec.input_schema) if spec.input_schema else {
            "type": "object",
            "properties": {},
        },
    }
    return definition


def to_langchain_tool_definition(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into a LangChain-compatible tool definition dict.

    This produces a dict that can be passed to LangChain's
    ``StructuredTool.from_function`` or used as a schema reference::

        lc_def = to_langchain_tool_definition(my_tool.spec)
        # {"name": "...", "description": "...", "args_schema": {...}}
    """
    definition: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "args_schema": dict(spec.input_schema) if spec.input_schema else {
            "type": "object",
            "properties": {},
        },
    }
    if spec.output_schema:
        definition["return_schema"] = dict(spec.output_schema)
    return definition


def to_openai_function(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into an OpenAI function-calling definition.

    Returns a dict matching the ``tools[].function`` shape used by the
    OpenAI chat-completions API::

        fn_def = to_openai_function(my_tool.spec)
        # {"type": "function", "function": {"name": ..., "parameters": ...}}
    """
    parameters = dict(spec.input_schema) if spec.input_schema else {
        "type": "object",
        "properties": {},
    }
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": parameters,
        },
    }


def export_all(
    specs: Iterable[ToolSpec],
    *,
    format: str = "openai",
) -> list[dict[str, Any]]:
    """Batch-export multiple ToolSpec objects to the given format.

    Supported formats: ``"openai"``, ``"mcp"``, ``"langchain"``, ``"openapi"``.
    """
    exporters = {
        "openai": to_openai_function,
        "mcp": to_mcp_tool_definition,
        "langchain": to_langchain_tool_definition,
        "openapi": to_openapi_schema,
    }
    exporter = exporters.get(format)
    if exporter is None:
        raise ValueError(
            f"Unknown export format '{format}'. "
            f"Supported: {', '.join(sorted(exporters.keys()))}"
        )
    return [exporter(spec) for spec in specs]
