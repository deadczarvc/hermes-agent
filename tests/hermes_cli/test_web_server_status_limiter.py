"""Status must not queue behind the shared AnyIO profile-read limiter."""
from __future__ import annotations

import asyncio
import threading
import time

import anyio
import pytest


async def _status_request_with_limiter_block():
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    from hermes_cli import web_server

    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        warm = await client.get("/api/status")
        assert warm.status_code == 200

        limiter = anyio.to_thread.current_default_thread_limiter()
        previous_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        release = threading.Event()
        started = threading.Event()

        def blocker() -> None:
            started.set()
            release.wait(5)

        task = asyncio.create_task(anyio.to_thread.run_sync(blocker))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            return await asyncio.wait_for(client.get("/api/status"), timeout=2)
        finally:
            release.set()
            await task
            limiter.total_tokens = previous_tokens


def test_status_survives_profile_read_limiter_starvation(monkeypatch):
    import hermes_cli.web_server_gateway as web_gateway
    import hermes_cli.web_server_lifecycle as web_lifecycle
    from hermes_cli import web_server

    monkeypatch.setattr(web_gateway, "_collect_profile_gateway_topology_cached", lambda: {
        "profiles": ["default"], "gateway_mode": "none", "gateways": [],
        "profile_platforms": {},
    })
    monkeypatch.setattr(web_lifecycle, "_resolve_restart_drain_timeout", lambda: 30.0)
    monkeypatch.setattr(web_server, "get_install_id", lambda: None)

    response = asyncio.run(_status_request_with_limiter_block())
    assert response.status_code == 200


def test_bounded_health_probe_returns_at_deadline(monkeypatch):
    from hermes_cli.web_routers import status

    release = threading.Event()
    started = threading.Event()

    def slow_probe():
        started.set()
        release.wait(5)
        return True, {"ok": True}

    monkeypatch.setattr(status, "_probe_gateway_health", slow_probe)
    started_at = time.perf_counter()
    try:
        assert status._bounded_health_probe() == (False, None)
        assert started.is_set()
        assert time.perf_counter() - started_at < 2.0
    finally:
        release.set()
