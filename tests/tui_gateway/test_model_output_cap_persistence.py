"""Regression coverage for per-session model output-cap persistence."""

from __future__ import annotations

import json
from types import SimpleNamespace

from tui_gateway.server import (
    _restore_agent_model_runtime,
    _runtime_model_config,
    _snapshot_agent_model_runtime,
    _stored_session_runtime_overrides,
)


def _write_output_cap_policy(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: deepseek-v4-pro\n  provider: devpass\n  max_tokens: 16384\n"
        "model_overrides:\n  openai-codex:\n    gpt-5.6-sol:\n      max_output_tokens: 128000\n",
        encoding="utf-8",
    )


def test_runtime_model_config_round_trips_max_tokens() -> None:
    """A switched 128K output cap survives Desktop session resume."""
    agent = SimpleNamespace(
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        reasoning_config={"effort": "high"},
        service_tier=None,
        max_tokens=128_000,
    )

    model_config = _runtime_model_config(agent)
    assert model_config["max_tokens"] == 128_000

    restored = _stored_session_runtime_overrides(
        {"model": agent.model, "model_config": json.dumps(model_config)}
    )
    assert restored["model_override"]["max_tokens"] == 128_000


def test_legacy_16k_resume_adopts_current_per_model_policy(tmp_path, monkeypatch) -> None:
    """Derived old session metadata must not pin a model below its current explicit policy."""
    _write_output_cap_policy(tmp_path, monkeypatch)
    restored = _stored_session_runtime_overrides(
        {
            "model": "gpt-5.6-sol",
            "model_config": json.dumps(
                {
                    "provider": "openai-codex",
                    "api_mode": "codex_responses",
                    "max_tokens": 16_384,
                }
            ),
        }
    )
    assert restored["model_override"]["max_tokens"] == 128_000


def test_env_cap_overrides_per_model_policy(tmp_path, monkeypatch) -> None:
    """An explicit runtime cap wins over the model's persisted/default policy."""
    _write_output_cap_policy(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_MAX_TOKENS", "65536")

    restored = _stored_session_runtime_overrides(
        {
            "model": "gpt-5.6-sol",
            "model_config": json.dumps(
                {"provider": "openai-codex", "max_tokens": 16_384}
            ),
        }
    )

    assert restored["model_override"]["max_tokens"] == 65_536


def test_foreign_provider_resume_keeps_16k_policy(tmp_path, monkeypatch) -> None:
    """The Codex model override must not leak into an unrelated provider's session."""
    _write_output_cap_policy(tmp_path, monkeypatch)
    restored = _stored_session_runtime_overrides(
        {
            "model": "deepseek-v4-pro",
            "model_config": json.dumps(
                {
                    "provider": "devpass",
                    "api_mode": "chat_completions",
                    "max_tokens": 16_384,
                }
            ),
        }
    )
    assert restored["model_override"]["max_tokens"] == 16_384


def test_one_turn_restore_recovers_previous_output_cap() -> None:
    """A --once turn restores the prior primary runtime's local output budget."""

    class Agent:
        max_tokens = 16_384
        _primary_runtime = {"max_tokens": 16_384}
        _fallback_activated = False
        _rate_limited_until = 0

        def _restore_primary_runtime(self) -> bool:
            self.max_tokens = self._primary_runtime["max_tokens"]
            return True

    agent = Agent()
    snapshot = _snapshot_agent_model_runtime(agent)
    agent.max_tokens = 128_000
    agent._primary_runtime = {"max_tokens": 128_000}

    _restore_agent_model_runtime(agent, snapshot)

    assert agent.max_tokens == 16_384
    assert agent._primary_runtime["max_tokens"] == 16_384
