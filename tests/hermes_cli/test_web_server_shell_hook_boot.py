"""Desktop must register configured shell hooks before accepting work."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import shell_hooks
from hermes_cli import web_server


def test_desktop_lifespan_registers_hooks_before_other_services(monkeypatch):
    config = {'hooks': {'pre_tool_call': []}, 'hooks_auto_accept': True}
    register = Mock()
    monkeypatch.setattr(web_server, 'load_config', lambda: config)
    monkeypatch.setattr(shell_hooks, 'register_from_config', register)
    monkeypatch.setattr(web_server.threading, 'Thread', Mock())

    class ReachedServices(Exception):
        pass

    def stop_before_services():
        raise ReachedServices

    monkeypatch.setattr(web_server, '_warm_gateway_module', stop_before_services)

    async def enter():
        async with web_server._lifespan(SimpleNamespace(state=SimpleNamespace())):
            pytest.fail('should stop before starting services')

    with pytest.raises(ReachedServices):
        asyncio.run(enter())
    register.assert_called_once_with(config)
