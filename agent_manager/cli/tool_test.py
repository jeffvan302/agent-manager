"""CLI for testing tools directly without a model loop."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

from agent_manager.config import load_config
from agent_manager.runtime.session import AgentSession
from agent_manager.tools.base import ToolContext, ToolSpec
from agent_manager.types import ToolCallRequest


def _load_input_source(raw: str) -> str:
    if raw.startswith("@"):
        path = Path(raw[1:])
        return path.read_text(encoding="utf-8")
    return raw


def _single_required_field(spec: ToolSpec) -> str | None:
    schema = spec.input_schema or {}
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return None
    normalized_required = [str(item) for item in required if str(item).strip()]
    if len(normalized_required) != 1:
        return None
    key = normalized_required[0]
    if key not in properties:
        return None
    return key


def parse_tool_arguments(raw: str | None, spec: ToolSpec) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}

    source = _load_input_source(raw)
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError:
        key = _single_required_field(spec)
        if key is None:
            raise ValueError(
                "Tool input must be a JSON object for this tool. "
                "Example: '{\"path\": \"requirements.md\"}'."
            ) from None
        return {key: source}

    if isinstance(parsed, dict):
        return parsed

    key = _single_required_field(spec)
    if key is None:
        raise ValueError(
            "Tool input must be a JSON object for this tool. "
            "Example: '{\"path\": \"requirements.md\"}'."
        )
    return {key: parsed}


def format_tool_schema(spec: ToolSpec) -> str:
    payload = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
        "output_schema": spec.output_schema,
        "tags": spec.tags,
        "permissions": spec.permissions,
        "timeout_seconds": spec.timeout_seconds,
        "retry_count": spec.retry_count,
        "retry_backoff_seconds": spec.retry_backoff_seconds,
    }
    return json.dumps(payload, indent=2, ensure_ascii=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tool-test",
        description="Execute one registered tool directly for testing.",
    )
    parser.add_argument("tool_name", nargs="?", help="Name of the tool to execute.")
    parser.add_argument(
        "tool_input",
        nargs="?",
        help=(
            "Tool input as JSON, a plain value for single-required-field tools, "
            "or @path/to/input.json."
        ),
    )
    parser.add_argument("--config", help="Path to a TOML, JSON, or YAML config file.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the tools available under the current config.",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Show the selected tool schema and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full normalized ToolResult as JSON.",
    )
    parser.add_argument("--task-id", default=None, help="Optional task id for the tool context.")
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Step index to include in the tool context.",
    )
    parser.add_argument(
        "--working-directory",
        help="Optional working directory for the tool context.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    session = AgentSession(
        config=config,
        working_directory=args.working_directory,
    )

    if args.list:
        for name in session.tools.names():
            print(name)
        return 0

    if not args.tool_name:
        parser.print_help()
        return 0

    if not session.tools.has(args.tool_name):
        available = ", ".join(session.tools.names())
        print(
            f"Tool '{args.tool_name}' is not registered. Available tools: {available}",
            file=sys.stderr,
        )
        return 2

    tool = session.tools.get(args.tool_name)
    if args.schema:
        print(format_tool_schema(tool.spec))
        return 0

    try:
        arguments = parse_tool_arguments(args.tool_input, tool.spec)
    except (ValueError, OSError) as exc:
        print(f"Invalid tool input: {exc}", file=sys.stderr)
        return 2

    call_id = f"tool-test-{uuid.uuid4().hex}"
    context = ToolContext(
        task_id=args.task_id or "tool-test",
        step_index=args.step_index,
        tool_call_id=call_id,
        working_directory=str(session.working_directory),
        metadata=dict(session.tool_context_metadata),
    )
    call = ToolCallRequest(
        id=call_id,
        name=args.tool_name,
        arguments=arguments,
    )

    result = session.tool_executor.execute(call, context)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    else:
        if isinstance(result.output, str):
            print(result.output)
        else:
            print(json.dumps(result.output, indent=2, ensure_ascii=True))
        if result.error:
            print(f"error: {result.error}", file=sys.stderr)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
