"""GWL H1 regressions for api-server loader and scoped-env guards."""

from __future__ import annotations

from agent.secret_scope import (
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from gateway.config import GatewayConfig, Platform
from gateway.config_env import _api_server
from gateway.config_loader import _API_SERVER_TUNING_KEYS, merge_platform_sections


def _merge(gateway_cfg: object) -> dict:
    gw_data: dict = {}
    platforms = merge_platform_sections({}, gateway_cfg, gw_data)
    assert gw_data["platforms"] is platforms
    return platforms


def test_tuning_only_api_server_block_is_not_a_platform() -> None:
    platforms = _merge({"api_server": {"max_concurrent_runs": 10}})

    assert _API_SERVER_TUNING_KEYS == frozenset({"max_concurrent_runs"})
    assert "api_server" not in platforms


def test_explicit_api_server_platform_block_still_merges() -> None:
    platforms = _merge(
        {"api_server": {"enabled": True, "extra": {"port": 8642}}}
    )

    assert platforms["api_server"]["enabled"] is True
    assert platforms["api_server"]["extra"]["port"] == 8642


def test_profile_scoped_key_does_not_create_missing_api_server() -> None:
    config = GatewayConfig.from_dict(
        {"platforms": _merge({"api_server": {"max_concurrent_runs": 10}})}
    )
    assert Platform.API_SERVER not in config.platforms

    set_multiplex_active(True)
    token = set_secret_scope({"API_SERVER_KEY": "x" * 32})
    try:
        _api_server(config)
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)

    assert Platform.API_SERVER not in config.platforms


def test_default_scope_key_still_enables_api_server(monkeypatch) -> None:
    config = GatewayConfig()
    monkeypatch.setenv("API_SERVER_KEY", "x" * 32)

    _api_server(config)

    assert config.platforms[Platform.API_SERVER].enabled is True
