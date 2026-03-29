from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from agent_manager.cli.config_tool import (
    builtin_tool_help_text,
    config_usage_text,
    load_runtime_config_for_wizard,
    policy_fields,
    provider_connection_probe,
    runtime_config_to_toml,
    save_runtime_config_toml,
    tool_fields,
    web_search_backend_help_text,
)
from agent_manager.config import RuntimeConfig


class ConfigToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_manager_config_tool_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runtime_config_toml_roundtrip_preserves_nested_provider_settings(self) -> None:
        config = RuntimeConfig.from_dict(
            {
                "profile": "coding-agent",
                "system_prompt": "You are a config test assistant.",
                "state_backend": "sqlite",
                "state_path": ".agent_manager/checkpoints.sqlite3",
                "provider": {
                    "name": "vllm",
                    "model": "NousResearch/Meta-Llama-3-8B-Instruct",
                    "base_url": "http://localhost:8000/v1",
                    "api_key_env": "VLLM_API_KEY",
                    "settings": {
                        "api_key": "secret-test-key",
                        "request_timeout_seconds": 120,
                        "request_retries": 3,
                        "extra_body": {
                            "top_k": 40,
                            "parallel_tool_calls": False,
                        },
                    },
                },
                "context": {
                    "pre_call_functions": [
                        "collect_recent_messages",
                        "apply_token_budget",
                        "finalize_messages",
                    ]
                },
                "tools": {
                    "web_search": {
                        "enabled": True,
                        "backend": "tavily",
                        "api_key_env": "TAVILY_API_KEY",
                        "timeout_seconds": 15,
                        "max_results": 7,
                        "settings": {
                            "search_depth": "basic",
                            "topic": "general",
                        },
                    }
                },
            }
        )

        toml_text = runtime_config_to_toml(config)
        target_path = self.temp_dir / "generated.toml"
        save_runtime_config_toml(config, target_path)
        loaded = load_runtime_config_for_wizard(target_path)

        self.assertIn("[provider.settings.extra_body]", toml_text)
        self.assertIn("[tools.web_search.settings]", toml_text)
        self.assertEqual(loaded.provider.name, "vllm")
        self.assertEqual(loaded.provider.settings["api_key"], "secret-test-key")
        self.assertEqual(loaded.provider.settings["request_timeout_seconds"], 120)
        self.assertEqual(loaded.provider.settings["extra_body"]["top_k"], 40)
        self.assertEqual(loaded.tools.web_search.backend, "tavily")
        self.assertEqual(loaded.tools.web_search.api_key_env, "TAVILY_API_KEY")
        self.assertEqual(loaded.tools.web_search.timeout_seconds, 15)
        self.assertEqual(loaded.tools.web_search.max_results, 7)
        self.assertEqual(loaded.tools.web_search.settings["search_depth"], "basic")
        self.assertEqual(
            loaded.context.pre_call_functions,
            ["collect_recent_messages", "apply_token_budget", "finalize_messages"],
        )

    def test_provider_connection_probe_succeeds_for_echo(self) -> None:
        config = RuntimeConfig.from_dict(
            {
                "provider": {"name": "echo", "model": "echo-v1"},
                "runtime": {"max_output_tokens": 32},
            }
        )

        summary = provider_connection_probe(config, prompt="Reply with exactly: OK")

        self.assertTrue(summary["success"])
        self.assertEqual(summary["provider"], "echo")
        self.assertEqual(summary["output_text"], "Reply with exactly: OK")

    def test_usage_text_mentions_cli_python_and_env(self) -> None:
        usage = config_usage_text(self.temp_dir / "agent-manager.toml")

        self.assertIn("agent-manager --config", usage)
        self.assertIn("AgentSession", usage)
        self.assertIn("AGENT_MANAGER_CONFIG", usage)

    def test_tool_policy_help_lists_builtin_tools(self) -> None:
        help_text = builtin_tool_help_text()
        fields = policy_fields()

        self.assertIn("list_directory", help_text)
        self.assertIn("read_file", help_text)
        self.assertIn("write_file", help_text)
        self.assertIn("run_shell_command", help_text)
        self.assertIn("http_request", help_text)
        self.assertIn("web_search", help_text)
        self.assertIn("retrieve_documents", help_text)
        self.assertIn("list_directory", fields[0].help_text)
        self.assertIn("retrieve_documents", fields[1].help_text)

    def test_tool_fields_include_web_search_backend_help(self) -> None:
        help_text = web_search_backend_help_text()
        fields = tool_fields()

        self.assertIn("duckduckgo", help_text)
        self.assertIn("serpapi", help_text)
        self.assertIn("tavily", help_text)
        self.assertIn("brave", help_text)
        self.assertTrue(
            any(field.label == "tools.web_search.backend" for field in fields)
        )


if __name__ == "__main__":
    unittest.main()
