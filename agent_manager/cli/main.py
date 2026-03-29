"""Command-line entry point for agent-manager."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from agent_manager.config import load_config
from agent_manager.runtime.session import AgentSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-manager",
        description="Run a local-first agent-manager session.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt or task goal to execute.")
    parser.add_argument("--config", help="Path to a TOML or JSON configuration file.")
    parser.add_argument("--task-id", help="Optional task id for checkpoint continuity.")
    parser.add_argument("--provider", help="Override the configured provider name.")
    parser.add_argument("--model", help="Override the configured model name.")
    parser.add_argument(
        "--structured-schema",
        help="Path to a JSON schema file for structured output.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit runtime events as JSON lines while the run executes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full runtime result as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.prompt:
        parser.print_help()
        return 0

    config = load_config(args.config)
    if args.provider:
        config.provider.name = args.provider
    if args.model:
        config.provider.model = args.model

    structured_output = None
    if args.structured_schema:
        schema_path = Path(args.structured_schema)
        structured_output = {
            "type": "json_schema",
            "name": schema_path.stem or "response",
            "schema": json.loads(schema_path.read_text(encoding="utf-8")),
            "strict": True,
        }

    session = AgentSession(config=config)

    if args.stream:
        async def _stream() -> None:
            async for event in session.stream_async(
                args.prompt,
                task_id=args.task_id,
                structured_output=structured_output,
            ):
                print(json.dumps(event, ensure_ascii=True))

        asyncio.run(_stream())
        return 0

    result = session.run(
        args.prompt,
        task_id=args.task_id,
        structured_output=structured_output,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    else:
        print(result.output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
