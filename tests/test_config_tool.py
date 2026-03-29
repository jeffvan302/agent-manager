from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from agent_manager.cli.config_tool import (
    config_usage_text,
    load_runtime_config_for_wizard,
    provider_connection_probe,
    runtime_config_to_toml,
    save_runtime_config_toml,
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
            }
        )

        toml_text = runtime_config_to_toml(config)
        target_path = self.temp_dir / "generated.toml"
        save_runtime_config_toml(config, target_path)
        loaded = load_runtime_config_for_wizard(target_path)

        self.assertIn("[provider.settings.extra_body]", toml_text)
        self.assertEqual(loaded.provider.name, "vllm")
        self.assertEqual(loaded.provider.settings["request_timeout_seconds"], 120)
        self.assertEqual(loaded.provider.settings["extra_body"]["top_k"], 40)
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


if __name__ == "__main__":
    unittest.main()
