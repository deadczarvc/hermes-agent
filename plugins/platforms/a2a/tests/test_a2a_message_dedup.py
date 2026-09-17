# ruff: noqa: I001  (the repo lint config does not select import-sorting; the sandboxed pre-write lint does — keep the repo-conventional grouping)
"""messageId idempotency for the inbound A2A adapter (A2A-DUP).

A peer that re-sends the same task — same authenticated ``peer`` AND same
``Message.messageId`` — must get the first task back instead of spawning a second
one; a message without a ``messageId`` keeps the historical behaviour. The dedup
state lives on the TaskStore records themselves (no second cache / db / file).

Deterministic: no live LLM, no external network, no sleeps — the in-flight cases
are sequenced with ``threading.Event`` and bounded joins (a timeout is reported as
a failure, never as a silent kill).
"""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import socket
import threading

import pytest

from plugins.platforms.a2a import protocol


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def sandboxed_hermes_home(tmp_path, monkeypatch):
    """Keep protocol.persist_message()/security.audit() out of the live Hermes home.

    Two levers on purpose: the context-local override covers the pytest thread, and ``HERMES_HOME``
    covers the live-POST case's HTTP worker threads, which never inherit a contextvar.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    home_override = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(home_override)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _params(text: str, message_id: str = "", context_id: str = "ctx-a2adup") -> dict:
    """``message/send`` params; ``message_id=""`` drops the field entirely (legacy peer)."""
    message = protocol.text_message(protocol.ROLE_USER, text, context_id=context_id)
    if message_id:
        message["messageId"] = message_id
    else:
        message.pop("messageId", None)
    return {"message": message}


def _dev_adapter(forward):
    """Adapter exposing one forwarded-profile agent (``dev``) whose dispatch is ``forward``.

    ``forward(call_index) -> (reply, state)``; the fixture asserts the agent really is
    non-local, so the deterministic stub is always the code path under test.
    """
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"agents": {"dev": {"profile": "dev", "tenant": "dev"}}}))
    assert adapter._agents["dev"]["local"] is False, "fixture needs a forwarded (non-local) agent"
    calls: list[dict] = []

    def _forward(_agent, peer, context_id, framed_text):
        calls.append({"peer": peer, "context_id": context_id, "text": framed_text})
        return forward(len(calls))

    adapter._forward_to_profile = _forward  # type: ignore[method-assign]
    return adapter, calls


def _completed(n: int):
    return f"reply-{n}", protocol.STATE_COMPLETED


def _task_ids(adapter) -> list[str]:
    recs, _next = adapter.tasks.list(page_size=100)
    return [rec["task_id"] for rec in recs]


class _FakeHandler:
    """Minimal A2ARequestHandler stand-in that captures the SSE bytes."""

    def __init__(self) -> None:
        self.wfile = io.BytesIO()
        self.status = 0
        self.close_connection = False

    def send_response(self, code, message=None) -> None:  # noqa: ARG002 - handler API
        self.status = code

    def send_header(self, _key, _value) -> None:
        pass

    def end_headers(self) -> None:
        pass

    @property
    def sse(self) -> str:
        return self.wfile.getvalue().decode("utf-8")

    def payloads(self) -> list[dict]:
        """Parsed StreamResponse objects — the JSON-RPC envelope is unwrapped (as in the
        existing test_a2a_phase23 ``_post_sse`` helper), so callers see bare payloads."""
        out = []
        for block in self.sse.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    raw = line[len("data: "):].strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    out.append(obj["result"] if isinstance(obj, dict) and "result" in obj else obj)
        return out


def _make_live_adapter(monkeypatch):
    """Live localhost HTTP adapter echoing inbound text (same harness as test_a2a_phase23)."""
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", "15")
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    adapter = A2AAdapter(PlatformConfig(enabled=True))

    async def fake_handle_message(event):
        await adapter.send(event.source.chat_id, "ECHO: " + event.text, metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]
    adapter._message_handler = object()
    return adapter, port


def _post_json(port: int, body: dict) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.request("POST", "/", json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
        payload = conn.getresponse().read().decode("utf-8")
    finally:
        conn.close()
    return json.loads(payload)


# --------------------------------------------------------------------------
# SendMessage dedup (I1)
# --------------------------------------------------------------------------

class TestSendMessageIdempotency:
    def test_message_send_same_messageid_dedups_to_one_task(self):
        adapter, calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first, pending = adapter._prepare_task(_params("hello", "m-1"), "peer-a", agent=agent)
        assert pending is None and first["status"]["state"] == protocol.STATE_COMPLETED

        second, pending2 = adapter._prepare_task(_params("hello", "m-1"), "peer-a", agent=agent)

        assert pending2 is None
        assert second["id"] == first["id"]
        assert second["contextId"] == first["contextId"]
        assert second["status"]["state"] == protocol.STATE_COMPLETED
        assert _task_ids(adapter) == [first["id"]]
        assert len(calls) == 1, "the agent must run once, the duplicate must not be re-dispatched"

    def test_message_send_duplicate_messageid_replays_stored_reply(self):
        adapter, _calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first, _ = adapter._prepare_task(_params("hello", "m-replay"), "peer-a", agent=agent)
        second, _ = adapter._prepare_task(_params("hello", "m-replay"), "peer-a", agent=agent)

        assert second["id"] == first["id"]
        assert protocol.extract_text(second["artifacts"][0]) == protocol.extract_text(first["artifacts"][0])
        assert protocol.extract_text(second["artifacts"][0]) == "reply-1"
        assert adapter.tasks.get(second["id"])["reply"] == "reply-1"

    def test_message_send_different_messageid_creates_second_task(self):
        """Negative control against over-dedup: a new messageId is a new task."""
        adapter, calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first, _ = adapter._prepare_task(_params("hello", "m-a"), "peer-a", agent=agent)
        second, _ = adapter._prepare_task(_params("hello", "m-b"), "peer-a", agent=agent)

        assert second["id"] != first["id"]
        assert sorted(_task_ids(adapter)) == sorted([first["id"], second["id"]])
        assert len(calls) == 2

    def test_message_send_same_messageid_different_peer_creates_second_task(self):
        """I1: the key is (peer, messageId) — another peer reusing the id is not a duplicate."""
        adapter, calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first, _ = adapter._prepare_task(_params("hello", "m-shared"), "peer-a", agent=agent)
        second, _ = adapter._prepare_task(_params("hello", "m-shared"), "peer-b", agent=agent)

        assert second["id"] != first["id"]
        assert len(_task_ids(adapter)) == 2
        assert [c["peer"] for c in calls] == ["peer-a", "peer-b"]

    def test_message_send_without_messageid_keeps_legacy_behavior(self):
        """I2: no messageId -> no idempotency key is invented, two sends are two tasks."""
        adapter, calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first, _ = adapter._prepare_task(_params("hello"), "peer-a", agent=agent)
        second, _ = adapter._prepare_task(_params("hello"), "peer-a", agent=agent)

        assert second["id"] != first["id"]
        assert len(_task_ids(adapter)) == 2
        assert len(calls) == 2

    def test_message_send_empty_text_is_still_rejected_and_deduped(self):
        """I2: validation is not weakened — a new key is still rejected, and the rejection is idempotent."""
        adapter, calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first, _ = adapter._prepare_task(_params("   ", "m-empty"), "peer-a", agent=agent)
        assert first["status"]["state"] == protocol.STATE_REJECTED
        assert first["status"].get("message"), "the rejection reason is returned on first contact"

        second, _ = adapter._prepare_task(_params("   ", "m-empty"), "peer-a", agent=agent)

        assert second["id"] == first["id"]
        assert second["status"]["state"] == protocol.STATE_REJECTED
        assert _task_ids(adapter) == [first["id"]]
        assert calls == [], "a rejected message must never reach the agent"

    def test_message_dedup_does_not_consume_anti_loop_turn(self, monkeypatch):
        """A retried (deduped) message must not burn a ping-pong turn of its context."""
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "3")
        adapter, _calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first, _ = adapter._prepare_task(_params("one", "m-1", context_id="ctx-loop"), "peer-a", agent=agent)
        second, _ = adapter._prepare_task(_params("two", "m-2", context_id="ctx-loop"), "peer-a", agent=agent)
        replay, _ = adapter._prepare_task(_params("one", "m-1", context_id="ctx-loop"), "peer-a", agent=agent)
        third, _ = adapter._prepare_task(_params("three", "m-3", context_id="ctx-loop"), "peer-a", agent=agent)

        assert replay["id"] == first["id"]
        assert second["id"] not in (first["id"], third["id"])
        # Turn 3 of 3 is still allowed: the replay above consumed no turn (a 4th turn would be rejected).
        assert third["status"]["state"] == protocol.STATE_COMPLETED, third["status"].get("message")

    def test_message_send_simultaneous_first_contact_creates_one_task(self, monkeypatch):
        """Two callers that both miss the same key must atomically share one task."""
        adapter, calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]
        params = _params("hello", "m-race")
        original_find = adapter.tasks.find_by_message
        both_looked_up = threading.Barrier(2)

        def racing_find(*args, **kwargs):
            found = original_find(*args, **kwargs)
            both_looked_up.wait(timeout=10)
            return found

        monkeypatch.setattr(adapter.tasks, "find_by_message", racing_find)
        results: list[tuple] = []
        threads = [
            threading.Thread(target=lambda: results.append(adapter._prepare_task(params, "peer-a", agent=agent)))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not any(thread.is_alive() for thread in threads)
        assert len(results) == 2
        assert len({task["id"] for task, _pending in results}) == 1
        assert len(_task_ids(adapter)) == 1
        assert len(calls) == 1, "simultaneous retries must dispatch exactly once"

    def test_message_send_duplicate_messageid_while_in_flight_reuses_task(self, monkeypatch):
        """I1 under concurrency: the second send must ride the running task, not fork one."""
        monkeypatch.setenv("A2A_REPLY_TIMEOUT", "20")
        started, gate = threading.Event(), threading.Event()

        def blocking(_n):
            started.set()
            assert gate.wait(timeout=15), "in-flight stub was never released"
            return "late reply", protocol.STATE_COMPLETED

        adapter, calls = _dev_adapter(blocking)
        agent = adapter._agents["dev"]
        params = _params("hello", "m-inflight")
        first_out: list[tuple] = []

        thread = threading.Thread(target=lambda: first_out.append(adapter._prepare_task(params, "peer-a", agent=agent)))
        thread.start()
        try:
            assert started.wait(timeout=10), "the first send never reached the agent"
            live = adapter.tasks.list(page_size=100)[0]
            assert len(live) == 1 and live[0]["state"] not in protocol.TERMINAL_STATES

            snapshot, pending = adapter._prepare_task(_params("hello", "m-inflight"), "peer-a", agent=agent)

            assert pending is None
            assert snapshot["id"] == live[0]["task_id"]
            assert snapshot["status"]["state"] not in protocol.TERMINAL_STATES
            assert len(calls) == 1, "the duplicate must not dispatch a second agent run"
        finally:
            gate.set()
            thread.join(timeout=15)
        assert not thread.is_alive(), "first send did not finish (timeout, not killed)"

        task, pending = first_out[0]
        assert pending is None
        assert task["status"]["state"] == protocol.STATE_COMPLETED
        assert adapter.tasks.get(task["id"])["reply"] == "late reply"
        assert _task_ids(adapter) == [task["id"]]


# --------------------------------------------------------------------------
# Streaming dedup
# --------------------------------------------------------------------------

class TestStreamingMessageIdempotency:
    def test_message_stream_duplicate_messageid_replays_terminal_events(self, monkeypatch):
        monkeypatch.setenv("A2A_REPLY_TIMEOUT", "20")
        adapter, calls = _dev_adapter(_completed)
        agent = adapter._agents["dev"]

        first = _FakeHandler()
        adapter._rpc_message_stream(first, "1", _params("hello", "m-stream"), "peer-a", agent=agent)
        replay = _FakeHandler()
        adapter._rpc_message_stream(replay, "2", _params("hello", "m-stream"), "peer-a", agent=agent)

        assert first.sse.endswith(": done\n\n") and replay.sse.endswith(": done\n\n")
        artifacts = [p["artifactUpdate"] for p in replay.payloads() if "artifactUpdate" in p]
        assert len(artifacts) == 1
        assert protocol.extract_text(artifacts[0]["artifact"]) == "reply-1"
        states = [p["statusUpdate"]["status"]["state"] for p in replay.payloads() if "statusUpdate" in p]
        assert states == [protocol.STATE_COMPLETED]
        assert len(_task_ids(adapter)) == 1 and len(calls) == 1

    def test_message_stream_duplicate_messageid_while_in_flight_shares_task(self, monkeypatch):
        """A duplicate stream opened while the first task runs mirrors it (no second task root)."""
        monkeypatch.setenv("A2A_REPLY_TIMEOUT", "20")
        started, gate = threading.Event(), threading.Event()

        def blocking(_n):
            started.set()
            assert gate.wait(timeout=15), "in-flight stub was never released"
            return "late reply", protocol.STATE_COMPLETED

        adapter, calls = _dev_adapter(blocking)
        agent = adapter._agents["dev"]
        params = _params("hello", "m-stream-inflight")
        handlers = [_FakeHandler(), _FakeHandler()]
        threads = [threading.Thread(target=lambda h=h: adapter._rpc_message_stream(h, "1", params, "peer-a", agent=agent))
                   for h in handlers]
        threads[0].start()
        try:
            assert started.wait(timeout=10), "the first stream never reached the agent"
            threads[1].start()
            # The duplicate is not the task creator: only one record may exist.
            assert len(adapter.tasks.list(page_size=100)[0]) == 1
        finally:
            gate.set()
            for thread in threads:
                thread.join(timeout=20)
        assert not any(t.is_alive() for t in threads), "stream threads did not finish (timeout, not killed)"

        for handler in handlers:
            assert handler.sse.endswith(": done\n\n")
            artifacts = [p["artifactUpdate"] for p in handler.payloads() if "artifactUpdate" in p]
            assert len(artifacts) == 1 and protocol.extract_text(artifacts[0]["artifact"]) == "late reply"
        assert len(adapter.tasks.list(page_size=100)[0]) == 1
        assert len(calls) == 1

    def test_message_send_http_duplicate_messageid_is_idempotent(self, monkeypatch):
        """End-to-end: two identical message/send POSTs over localhost -> one task."""
        adapter, port = _make_live_adapter(monkeypatch)
        body = {"jsonrpc": "2.0", "id": "1", "method": "message/send",
                "params": _params("hello over http", "m-http")}

        async def run():
            assert await adapter.connect() is True
            try:
                first = await asyncio.to_thread(_post_json, port, body)
                second = await asyncio.to_thread(_post_json, port, body)
            finally:
                await adapter.disconnect()
            return first, second

        first, second = asyncio.run(run())

        assert second["result"]["id"] == first["result"]["id"]
        assert second["result"]["status"]["state"] == protocol.STATE_COMPLETED
        assert protocol.extract_text(second["result"]["artifacts"][0]) == protocol.extract_text(first["result"]["artifacts"][0])
        assert len(adapter.tasks.list(page_size=100)[0]) == 1


# --------------------------------------------------------------------------
# TaskStore-level contract
# --------------------------------------------------------------------------

class TestTaskStoreMessageIdempotencyKey:
    def test_task_store_dedup_by_messageid_is_peer_scoped(self):
        store = protocol.TaskStore()
        store.create("task-1", "ctx-1", "peer-a", message_id="m-1")

        assert store.find_by_message("peer-a", "m-1")["task_id"] == "task-1"
        assert store.find_by_message("peer-b", "m-1") is None
        assert store.find_by_message("peer-a", "m-2") is None

    def test_atomic_claim_uses_exact_scope_in_both_orders(self):
        """Idempotency scope is an exact key; empty default scope is not an authz wildcard."""
        for first_scope, second_scope in ((('dev', 'team'), ('', '')), (('', ''), ('dev', 'team'))):
            store = protocol.TaskStore()
            first, first_created = store.create_or_find_by_message(
                'task-1', 'ctx-1', 'peer-a', *first_scope, message_id='m-scope'
            )
            second, second_created = store.create_or_find_by_message(
                'task-2', 'ctx-2', 'peer-a', *second_scope, message_id='m-scope'
            )
            assert first_created is True and second_created is True
            assert first['task_id'] != second['task_id']

    def test_atomic_claim_concurrent_entry_creates_one_record(self, monkeypatch):
        """Both callers reach the current claim path before the TaskStore lock decision."""
        store = protocol.TaskStore()
        original_now_iso = protocol.now_iso
        at_claim = threading.Barrier(2)

        def synchronized_now_iso():
            at_claim.wait(timeout=10)
            return original_now_iso()

        monkeypatch.setattr(protocol, 'now_iso', synchronized_now_iso)
        results: list[tuple[dict, bool]] = []
        threads = [threading.Thread(target=lambda n=n: results.append(
            store.create_or_find_by_message(f'task-{n}', f'ctx-{n}', 'peer-a', message_id='m-race')
        )) for n in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not any(thread.is_alive() for thread in threads)
        assert len(results) == 2
        assert sum(created for _rec, created in results) == 1
        assert len({rec['task_id'] for rec, _created in results}) == 1
        assert len(store.list(page_size=100)[0]) == 1

    def test_task_store_dedup_by_messageid_respects_scope_and_empty_ids(self):
        store = protocol.TaskStore()
        store.create("task-1", "ctx-1", "peer-a", agent_slug="dev", tenant="dev-team", message_id="m-1")
        store.create("task-2", "ctx-2", "peer-a", message_id="")

        assert store.find_by_message("peer-a", "m-1", "dev", "dev-team")["task_id"] == "task-1"
        assert store.find_by_message("peer-a", "m-1", "other", "dev-team") is None
        assert store.find_by_message("peer-a", "m-1", "dev", "other-team") is None
        assert store.find_by_message("peer-a", "") is None, "an absent messageId is never a key"
