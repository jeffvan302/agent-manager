from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from agent_manager.cli.tool_test import main


class ToolTestCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_manager_tool_test_"))
        self.config_path = self.temp_dir / "tool-test.toml"
        self.config_path.write_text(
            "\n".join(
                [
                    'profile = "local-dev"',
                    'state_backend = "json"',
                    f'state_dir = "{(self.temp_dir / "state").as_posix()}"',
                    "",
                    "[provider]",
                    'name = "echo"',
                    'model = "echo-v1"',
                    "",
                    "[logging]",
                    'level = "WARNING"',
                    "json_output = false",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_outputs_builtin_tool_names(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--config", str(self.config_path), "--list"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("list_directory", output)
        self.assertIn("read_file", output)
        self.assertIn("web_search", output)

    def test_single_required_field_tool_accepts_plain_string_input(self) -> None:
        test_file = self.temp_dir / "sample.txt"
        test_file.write_text("hello tool test", encoding="utf-8")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "--config",
                    str(self.config_path),
                    "--working-directory",
                    str(self.temp_dir),
                    "read_file",
                    "sample.txt",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["content"], "hello tool test")
        self.assertNotIn("Tool execution blocked by policy", stderr.getvalue())
        self.assertNotIn("Tool execution failed:", stderr.getvalue())

    def test_policy_blocked_tool_returns_clean_error_instead_of_traceback(self) -> None:
        readonly_config = self.temp_dir / "readonly.toml"
        readonly_config.write_text(
            "\n".join(
                [
                    'profile = "readonly"',
                    'state_backend = "json"',
                    f'state_dir = "{(self.temp_dir / "state-readonly").as_posix()}"',
                    "",
                    "[provider]",
                    'name = "echo"',
                    'model = "echo-v1"',
                    "",
                    "[tools.web_search]",
                    "enabled = true",
                    'backend = "duckduckgo"',
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "--config",
                    str(readonly_config),
                    "web_search",
                    "budget GPU",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Tool execution blocked by policy", stderr.getvalue())
        self.assertIn("allowed_tools", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
