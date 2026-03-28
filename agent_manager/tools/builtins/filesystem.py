"""Built-in filesystem tools with working-directory scoping."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_roots(context: ToolContext) -> list[Path]:
    raw_roots = context.metadata.get("filesystem_roots")
    if isinstance(raw_roots, list) and raw_roots:
        return [Path(str(item)).resolve(strict=False) for item in raw_roots]

    if context.working_directory:
        return [Path(context.working_directory).resolve(strict=False)]

    return [Path.cwd().resolve(strict=False)]


def resolve_scoped_path(path_value: str, context: ToolContext) -> Path:
    roots = _allowed_roots(context)
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    resolved = candidate.resolve(strict=False)
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise PermissionError(f"Path '{resolved}' is outside the allowed filesystem roots.")
    return resolved


class ReadFileTool(BaseTool):
    spec = ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file from the working directory scope.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
                "max_chars": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "truncated": {"type": "boolean"},
            },
            "required": ["path", "content", "truncated"],
        },
        tags=["filesystem", "read"],
        permissions=["filesystem:read"],
    )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        path = resolve_scoped_path(str(arguments["path"]), context)
        encoding = str(arguments.get("encoding", "utf-8"))
        max_chars = max(int(arguments.get("max_chars", 20000)), 1)

        def _read() -> str:
            return path.read_text(encoding=encoding)

        content = await asyncio.to_thread(_read)
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]
        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output={
                "path": str(path),
                "content": content,
                "truncated": truncated,
            },
        )


class WriteFileTool(BaseTool):
    spec = ToolSpec(
        name="write_file",
        description="Write UTF-8 text into a file within the working directory scope.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
                "append": {"type": "boolean", "default": False},
                "create_parents": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "bytes_written": {"type": "integer"},
                "appended": {"type": "boolean"},
            },
            "required": ["path", "bytes_written", "appended"],
        },
        tags=["filesystem", "write"],
        permissions=["filesystem:write"],
    )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        path = resolve_scoped_path(str(arguments["path"]), context)
        content = str(arguments["content"])
        encoding = str(arguments.get("encoding", "utf-8"))
        append = bool(arguments.get("append", False))
        create_parents = bool(arguments.get("create_parents", False))

        def _write() -> int:
            if create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with path.open(mode, encoding=encoding) as handle:
                written = handle.write(content)
            return written

        bytes_written = await asyncio.to_thread(_write)
        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output={
                "path": str(path),
                "bytes_written": bytes_written,
                "appended": append,
            },
        )


class ListDirectoryTool(BaseTool):
    spec = ToolSpec(
        name="list_directory",
        description="List files and directories inside the working directory scope.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "recursive": {"type": "boolean", "default": False},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "entries": {"type": "array"},
            },
            "required": ["path", "entries"],
        },
        tags=["filesystem", "read"],
        permissions=["filesystem:read"],
    )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        path = resolve_scoped_path(str(arguments.get("path", ".")), context)
        recursive = bool(arguments.get("recursive", False))

        def _list() -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            iterator = path.rglob("*") if recursive else path.iterdir()
            for item in iterator:
                entries.append(
                    {
                        "path": str(item),
                        "name": item.name,
                        "is_dir": item.is_dir(),
                    }
                )
            return sorted(entries, key=lambda item: item["path"])

        entries = await asyncio.to_thread(_list)
        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output={"path": str(path), "entries": entries},
        )
