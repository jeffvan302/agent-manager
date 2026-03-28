"""Command-line entry point for agent-manager."""

from __future__ import annotations

import argparse
import json
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

    session = AgentSession(config=config)
    result = session.run(args.prompt, task_id=args.task_id)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    else:
        print(result.output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

