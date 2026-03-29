"""Built-in shell execution tool."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec


_DANGEROUS_PATTERNS: set[str] = {
    "rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){", "chmod -R 777 /",
    "curl | sh", "curl | bash", "wget | sh", "wget | bash",
    "> /dev/sda", "shutdown", "reboot", "init 0", "init 6",
}

_DANGEROUS_RE = re.compile(
    r"\b(rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/(?:$|\s))"
    r"|(\bsudo\s+rm\b)"
    r"|(\bformat\s+[A-Za-z]:)"
    r"|(\bmkfs\b)",
    re.IGNORECASE,
)


class RunShellCommandTool(BaseTool):
    spec = ToolSpec(
        name="run_shell_command",
        description="Run a shell command in the current working directory and capture stdout/stderr.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
            "required": ["command"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "exit_code": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
            "required": ["command", "exit_code", "stdout", "stderr"],
        },
        tags=["shell", "process"],
        permissions=["process:execute"],
        timeout_seconds=120.0,
    )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        command = str(arguments["command"]).strip()
        if not command:
            return ToolResult(
                tool_name=self.spec.name,
                ok=False,
                output={},
                error="Command must not be empty.",
            )

        # Check for dangerous commands.
        for pattern in _DANGEROUS_PATTERNS:
            if pattern in command.lower():
                return ToolResult(
                    tool_name=self.spec.name,
                    ok=False,
                    output={},
                    error="Command blocked by safety filter: matches dangerous pattern.",
                )
        if _DANGEROUS_RE.search(command):
            return ToolResult(
                tool_name=self.spec.name,
                ok=False,
                output={},
                error="Command blocked by safety filter: matches dangerous pattern.",
            )

        timeout = float(arguments.get("timeout_seconds", self.spec.timeout_seconds))
        if timeout <= 0:
            timeout = self.spec.timeout_seconds

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=context.working_directory,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"Shell command timed out after {timeout} seconds."
            ) from exc

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        ok = process.returncode == 0
        return ToolResult(
            tool_name=self.spec.name,
            ok=ok,
            output={
                "command": command,
                "exit_code": int(process.returncode or 0),
                "stdout": stdout,
                "stderr": stderr,
            },
            error=None if ok else f"Command exited with code {process.returncode}.",
        )
