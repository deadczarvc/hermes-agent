"""Plugin hook / middleware / event-bus / system-prompt-section dispatch.

Mixed into :class:`hermes_cli.plugins.PluginManager`. ``_resolve_hook_callback_timeout`` stays on
the origin (tests patch it there) and is looked up lazily.
"""

from __future__ import annotations

import contextvars
import copy
import inspect
import logging
import os
import queue
import re
import threading
import time
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger("hermes_cli.plugins")

# Allowlist of agent-turn hot-path hooks bounded by plugins.hook_callback_timeout (fail-open:
# abandon without join — joining reintroduced a shutdown hang). Unlisted hooks run synchronously.
# Intentionally unbounded: on_session_finalize/reset (last-chance flush — abandon can lose state);
# subagent_start (observer); pre_gateway_dispatch (policy gate — neither fail mode is acceptable);
# pre/post_approval_* (approval UX has its own timeout); kanban_* (own heartbeat/stale reclaim).
# The goal is to stop a hung Python plugin callback from wedging the conversation loop (#76821) without
# joining the worker (avoids the #6622 ThreadPoolExecutor shutdown hang). Hooks not listed below run
# synchronously to completion. (on_session_start/end stay bounded — they sit on the common session-boundary
# path.) - subagent_start — observer only; blocking delegation belongs in pre_tool_call. Lower frequency
# than tool/LLM hooks. Abandoning is unsafe either way (fail-open skips auth-like checks; fail-closed can
# drop legitimate messages). Prefer finish-or-exception fallthrough. - pre_approval_request /
# post_approval_response — observers only (cannot veto); the approval UX already has its own timeout; not on
# the tool loop hot path. - kanban_task_* — fire after the board DB commit, observers only, in
# dispatcher/worker processes; kanban has its own heartbeat/stale reclaim. Abandon-without-join also leaves
# a daemon thread that may still mutate shared state — safer for value-returning observers than for
# gates/flushes.
_HOOK_TIMEOUT_BOUNDED_HOOKS: Set[str] = {
    "post_tool_call", "transform_terminal_output", "transform_tool_result", "transform_llm_output",
    "pre_llm_call", "post_llm_call", "pre_api_request", "post_api_request", "api_request_error",
    "pre_verify", "on_session_start", "on_session_end",
}

# Policy hooks: timeout / still-running must fail closed (block the tool).
_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS: Set[str] = {"pre_tool_call"}
# Documented parent-thread serialization contract — never run on a timeout worker (hooks.md).
_HOOK_CALLER_THREAD_HOOKS: Set[str] = {"subagent_stop"}
# After a timeout, suppress the same callback this long so a hung hook cannot pile up threads.
_HOOK_TIMEOUT_SUPPRESSION_SECONDS = 60.0
_PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE = "pre_tool_call plugin callback timed out or is still running"

# Contention between callers of the SAME callback is not a verdict about the hook: measured
# 2026-09-14, 920 "skipped ... still running" events in 5 minutes with 4 workers — every one a
# false fail-closed block on a fast hook (12 callbacks, 0.76 s total). A caller that finds the
# callback in flight now PARKS on the gate (bounded, FIFO) and fires anyway once the wait budget
# is spent; only a post-timeout suppression still blocks (see _run_hook_callback_bounded).
# The wait-then-fire path is itself bounded: at most _HOOK_GATE_MAX_CONCURRENT_LAUNCHES (K=2)
# launches of ONE callback key may be running at once, and above that the legacy skip is kept —
# parking must not turn into a stampede, and the mass suppression measured above must go.
# Kill switch: HERMES_HOOK_GATE_WAIT=0 restores the legacy skip-on-contention behaviour exactly.
_HOOK_GATE_WAIT_ENV = "HERMES_HOOK_GATE_WAIT"
_HOOK_GATE_WAIT_MAX_SECONDS = 5.0        # budget = min(this, hook_callback_timeout / 4)
_HOOK_GATE_WAIT_TIMEOUT_FRACTION = 0.25
_HOOK_GATE_MAX_WAITERS = 8               # bounded waiting room; overflow fires without the gate
_HOOK_GATE_MAX_CONCURRENT_LAUNCHES = 2   # lava bound K: simultaneous launches allowed per key
_HOOK_GATE_WAIT_SAMPLE_CAP = 128         # wait-time samples kept for the waited_p95 metric
_HOOK_GATE_LEGACY_OFF = frozenset({"0", "false", "no", "off"})
_HOOK_GATE_INIT_LOCK = threading.Lock()  # guards lazy creation of the per-manager gate state

# System-prompt sections are tightly bounded: they become high-trust prompt bytes charged every turn.
SYSTEM_PROMPT_SECTION_POSITIONS = frozenset({"after_memory"})
DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTION_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTIONS = 32
MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS = 8_000
_SYSTEM_PROMPT_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SYSTEM_PROMPT_SECTION_HEADING_PREFIX = "## Plugin Context: "
PLUGIN_SECTIONS_START = "<!-- hermes-plugin-sections:start -->"
PLUGIN_SECTIONS_END = "<!-- hermes-plugin-sections:end -->"


def is_valid_system_prompt_section_id(value: Any) -> bool:
    """Return whether *value* is a stable, heading-safe section identifier."""
    return isinstance(value, str) and bool(_SYSTEM_PROMPT_SECTION_ID_RE.fullmatch(value))


def format_system_prompt_section(section_id: str, content: str) -> str:
    """Render an auditable, length-framed block recoverable from the full prompt."""
    return (
        f"{_SYSTEM_PROMPT_SECTION_HEADING_PREFIX}{section_id}\n"
        f"<!-- hermes-plugin-section-chars:{len(content)} -->\n\n{content}")


def format_system_prompt_sections(sections: list) -> str:
    """Render the canonical container used for persistence recovery."""
    if not sections:
        return ""
    blocks = [format_system_prompt_section(item.id, item.content) for item in sections]
    return f"{PLUGIN_SECTIONS_START}\n" + "\n\n".join(blocks) + f"\n{PLUGIN_SECTIONS_END}"


# Reserved event namespace prefix — only core may publish ``hermes:<event>``.
HERMES_EVENT_NAMESPACE = "hermes"
# Event recursion depth cap (subscribers may emit); over-deep emits are dropped with a warning.
_EVENT_EMIT_DEPTH_CAP = 8
# Max queued + running events per manager generation; emit never waits — a full budget drops.
_EVENT_PENDING_CAP = 64
_EVENT_WORKER_STOP = object()


@dataclass(frozen=True)
class PluginSystemPromptSection:
    """A plugin-owned section rendered once for each new session."""

    id: str
    content: Union[str, Callable[[Mapping[str, Any]], str]]
    position: str
    max_chars: int
    plugin: str


@dataclass(frozen=True)
class RenderedPluginSystemPromptSection:
    """Validated prompt bytes frozen on the owning AIAgent."""

    id: str
    content: str
    position: str
    plugin: str


@dataclass(frozen=True)
class _EventSubscription:
    """Host-owned subscription ledger entry."""

    owner: str
    callback: Callable


@dataclass(frozen=True)
class _QueuedPluginEvent:
    """Immutable dispatch envelope consumed by the event worker."""

    event: str
    payload: Dict[str, Any]
    subscriptions: tuple[_EventSubscription, ...]
    depth: int
    generation: int


# Hook callback timeout (non-blocking abandon). Default cap per Python hook callback; overridden by
# ``plugins.hook_callback_timeout``. Shell hooks enforce their own subprocess timeout.
_HOOK_CALLBACK_TIMEOUT_SECS = 30.0
_MAX_HOOK_CALLBACK_TIMEOUT_SECS = 600.0
_HOOK_SKIPPED = object()  # returned by _run_hook_callback_bounded on skip/timeout
_HOOK_GATE_REENTRANT = object()  # returned by _acquire_hook_gate: the caller already holds it
# Returned by _acquire_hook_gate: the callback's timeout back-off became active while this caller
# was parked (A3) — the coarse suppression contract, enforced in the wait → claim transition.
_HOOK_GATE_SUPPRESSED = object()
# Manager-local generation of the timeout bookkeeping maps. Bumped (under _hook_timeout_lock,
# together with the clear) by an unload-all, so a timeout that belongs to the generation that was
# just torn down cannot repopulate the freshly cleared maps (A2).
_HOOK_TIMEOUT_GENERATION_ATTR = "_hook_timeout_bookkeeping_generation"

# Call-identity fallback order for the GATE key (H6). Read from the payload the dispatcher
# already receives - no producer change, no new dependency. NOT api_request_id: it is coarser
# than a call (one API request carries many tool calls) and would re-collapse keys.
_HOOK_CALL_IDENTITY_KEYS = ("tool_call_id", "turn_id")


def _hook_call_identity(kwargs: Mapping[str, Any]) -> Optional[str]:
    """First non-empty call identity in *kwargs* (``tool_call_id``, then ``turn_id``), else None.

    ``""`` is not an identity: the id paths coerce ``None`` to ``""`` (model_tools._CallIds),
    so a truthiness test alone would accept the empty string and two unrelated sessionless
    calls would share a key again.
    """
    for name in _HOOK_CALL_IDENTITY_KEYS:
        value = kwargs.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _hook_uses_callback_timeout(hook_name: str, timeout: float) -> bool:
    """Whether *hook_name* should run under the non-blocking timeout path."""
    if timeout <= 0 or hook_name in _HOOK_CALLER_THREAD_HOOKS:
        return False
    return hook_name in _HOOK_TIMEOUT_BOUNDED_HOOKS or hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS


def _hook_gate_wait_enabled() -> bool:
    """Whether a caller parks on a busy callback (default) instead of skipping it.

    ``HERMES_HOOK_GATE_WAIT=0`` restores the exact legacy gate (a sibling in flight skips this
    call, and for ``pre_tool_call`` that skip is a fail-closed block). Read per call so an
    operator — or a test — can flip it without restarting the process.
    """
    return os.environ.get(_HOOK_GATE_WAIT_ENV, "").strip().lower() not in _HOOK_GATE_LEGACY_OFF


class _HookGateSlot:
    """The FIFO queue of callers parked behind the call in flight for one callback key.

    Waiters park on the manager's condition (the calling thread itself parks, so contention costs
    no new thread) and tickets keep the queue first-in-first-out, so a burst of callers cannot
    starve the one that arrived first. The holder itself is NOT duplicated here: it is the
    ``(hook, callback, tool, session)`` entry in ``_hook_running_callbacks``, so there is exactly
    one latch — and a plugin unload that clears that map also frees the gate.
    """

    __slots__ = ("next_ticket", "queue")

    def __init__(self) -> None:
        self.next_ticket = 0
        self.queue: List[int] = []


class _HookGateStats:
    """Counters and wait-time samples for one plugin manager's callback gate.

    Observability for the synchronizer: without these numbers a parked caller and a skipped one
    look identical in the log.
    """

    __slots__ = ("lock", "fired", "waited", "fail_open", "blocked_after_timeout",
                 "lava_bound", "_wait_samples")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.fired = 0
        self.waited = 0
        self.fail_open = 0
        self.blocked_after_timeout = 0
        self.lava_bound = 0  # refused: K launches of this key already running (avalanche guard)
        self._wait_samples: List[float] = []

    def record_fired(self, waited: float) -> None:
        """Count one callback launch; a non-zero *waited* also feeds the wait-time samples."""
        with self.lock:
            self.fired += 1
            if waited > 0.0:
                self.waited += 1
                self._wait_samples.append(waited)
                if len(self._wait_samples) > _HOOK_GATE_WAIT_SAMPLE_CAP:
                    del self._wait_samples[:-_HOOK_GATE_WAIT_SAMPLE_CAP]

    def record_fail_open(self) -> None:
        with self.lock:
            self.fail_open += 1

    def record_blocked(self) -> None:
        with self.lock:
            self.blocked_after_timeout += 1

    def record_lava_bound(self) -> None:
        """Count one call the lava bound refused (K launches of its key already running)."""
        with self.lock:
            self.lava_bound += 1

    def snapshot(self) -> Dict[str, Any]:
        """JSON-serializable counters: fired, waited, waited_p95, fail_open,
        blocked_after_timeout, lava_bound."""
        with self.lock:
            samples = sorted(self._wait_samples)
            p95 = 0.0
            if samples:
                # ceil(0.95 * n) - 1 in integer arithmetic: no float index rounding surprises.
                p95 = samples[max(0, (len(samples) * 95 + 99) // 100 - 1)]
            return {
                "fired": self.fired,
                "waited": self.waited,
                "waited_p95": round(p95, 4),
                "fail_open": self.fail_open,
                "blocked_after_timeout": self.blocked_after_timeout,
                "lava_bound": self.lava_bound,
            }


class _HookGateState:
    """Per-manager gate state: the shared condition, the FIFO slots and the counters.

    ``cond`` wraps the manager's existing ``_hook_timeout_lock``, so a gate decision and the
    suppression bookkeeping stay in one critical section (and the direct uses of that lock in
    ``plugins_ledger`` keep working). Created lazily by ``_hook_gate_state`` — the dispatcher
    mixin owns the gate, not ``PluginManager.__init__``.
    """

    __slots__ = ("cond", "slots", "stats", "launches", "peak_launches")

    def __init__(self, lock: Any) -> None:
        self.cond = threading.Condition(lock)
        self.slots: Dict[tuple, _HookGateSlot] = {}
        self.stats = _HookGateStats()
        self.launches: Dict[tuple, int] = {}  # key -> launches in flight right now
        self.peak_launches = 0                # highest value ever seen (observed bound)

    def slot(self, callback_key: tuple) -> _HookGateSlot:
        """Return (creating if needed) the FIFO slot for *callback_key*; caller holds the lock."""
        slot = self.slots.get(callback_key)
        if slot is None:
            slot = _HookGateSlot()
            self.slots[callback_key] = slot
        return slot

    def begin_launch(self, callback_key: tuple, *, bounded: bool) -> bool:
        """Register one launch of *callback_key*; ``False`` when the lava bound refuses it.

        ``bounded`` marks the wait-then-fire path (the caller owns no gate token): it may add a
        launch only while the key is below K, otherwise the gate stops synchronizing and becomes
        a stampede — exactly what the legacy skip protected against. The gate-holding path is
        serialized by the latch itself, so it increments unconditionally.
        """
        with self.cond:
            current = self.launches.get(callback_key, 0)
            if bounded and current >= _HOOK_GATE_MAX_CONCURRENT_LAUNCHES:
                return False
            self.launches[callback_key] = current + 1
            if current + 1 > self.peak_launches:
                self.peak_launches = current + 1
            return True

    def end_launch(self, callback_key: tuple) -> None:
        """Drop one launch of *callback_key* — from the CALLER's return path, never the worker.

        An abandoned worker never reaches its finally, and a counter that only ever grows would
        latch the lava bound shut (a permanent tool outage on the fail-closed hooks); the gate is
        released by the callback body a moment before the caller decrements, so the count may
        briefly include a finished call — that errs toward refusing, never toward an extra launch.
        """
        with self.cond:
            current = self.launches.get(callback_key, 0)
            if current <= 1:
                self.launches.pop(callback_key, None)
            else:
                self.launches[callback_key] = current - 1

    def launches_for(self, callback_key: tuple) -> int:
        """Launches of *callback_key* in flight right now."""
        with self.cond:
            return self.launches.get(callback_key, 0)

    def prune_slot(self, running: Dict[tuple, Any], callback_key: tuple) -> None:
        """Drop the FIFO slot of *callback_key* once nothing needs it; caller holds the lock.

        A per-call gate key (H6) would otherwise keep one slot per tool call for the life of
        the process. Droppable only when the queue is empty AND the key has no holder, and
        every caller holds this lock from ``slot()`` to its own queue/holder transition - so a
        slot can never be dropped from under a caller that is about to queue on it.
        """
        slot = self.slots.get(callback_key)
        if slot is not None and not slot.queue and callback_key not in running:
            del self.slots[callback_key]

    def release(self, running: Dict[tuple, Any], callback_key: tuple, token: Any) -> None:
        """Identity-checked release of *callback_key*: drop the holder and wake the waiters.

        Called on normal completion AND on timeout, so a callback that never returns cannot latch
        the gate forever: backing off a hung callback is the suppression window's job, while a
        latched gate would park every later caller (and, before G5, block their tool calls).
        """
        with self.cond:
            holder = running.get(callback_key)
            if holder is token:
                del running[callback_key]
            self.cond.notify_all()
            self.prune_slot(running, callback_key)


class _HookToken:
    """Identity token for one in-flight hook callback, carrying its start time.

    The start time lets a later skip report HOW LONG the previous call has held the gate:
    a hung callback is otherwise indistinguishable from a slow one, and the age names
    the culprit in the log.
    """

    __slots__ = ("started", "thread_id")

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.thread_id = threading.get_ident()

class PluginDispatchMixin:
    @staticmethod
    def _invoke_hook_callback(callback: Callable, payload: Dict[str, Any]) -> Any:
        """Invoke a hook while withholding additive fields from narrow legacy callbacks.

        An ``async def`` callback returns a coroutine; resolve it the way plugin slash commands
        are (loop-safe), otherwise the bare coroutine object is appended to the results and the
        plugin's body never runs (#12449).
        """
        from hermes_cli.plugins import resolve_plugin_command_result
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            return resolve_plugin_command_result(callback(**payload))  # no introspectable signature
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return resolve_plugin_command_result(callback(**payload))
        keyword_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        return resolve_plugin_command_result(callback(**{
            name: value for name, value in payload.items()
            if name in parameters and parameters[name].kind in keyword_kinds
        }))

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Call all callbacks for *hook_name*; return their non-``None`` results.

        Payloads evolve additively: ``**kwargs`` callbacks get everything, narrow signatures only
        what they declare. Each callback is isolated. Bounded hooks and ``pre_tool_call`` run under
        ``plugins.hook_callback_timeout`` (worker abandoned, never joined); a callback already in
        flight parks the caller (bounded) instead of skipping it; ``pre_tool_call`` fails closed
        with a block directive, others skip. ``_HOOK_CALLER_THREAD_HOOKS`` always run on the caller
        thread. ``pre_llm_call`` may return ``{"context": "..."}`` (or a str) to inject.
        """
        from hermes_cli.plugins import _resolve_hook_callback_timeout
        # Gateway platform events define event-local envelopes; a bus-wide version here would turn
        # unrelated adapter payloads into one monolithic compatibility contract.
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        results: List[Any] = []
        timeout = _resolve_hook_callback_timeout()
        use_timeout = _hook_uses_callback_timeout(hook_name, timeout)
        fail_closed = hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        for cb, registration_epoch in self._snapshot_hook_callbacks(hook_name):
            try:
                # A callback may DECLARE which tools it applies to; honour that before a
                # worker starts, so a slow hook for tool A cannot starve unrelated tool B.
                declared = getattr(cb, "_hermes_tool_matcher", None)
                if callable(declared):
                    _tool = kwargs.get("tool_name")
                    if not isinstance(_tool, str) or not declared(_tool):
                        continue
                if use_timeout:
                    ret = self._run_hook_callback_bounded(
                        hook_name, cb, kwargs, timeout, registration_epoch=registration_epoch
                    )
                    if ret is _HOOK_SKIPPED:
                        if fail_closed:  # policy hook: fail closed with a block directive
                            results.append({"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE})
                        continue
                else:
                    ret = self._invoke_hook_callback(cb, kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                self._report_hook_failure(hook_name, cb, kwargs, exc)
        return results

    def _snapshot_hook_callbacks(self, hook_name: str) -> tuple[tuple[Callable, Optional[int]], ...]:
        """Snapshot callbacks and their registration epochs under the lifecycle lock."""
        with self._hook_timeout_lock:
            lifetimes = getattr(self, "_hook_registration_lifetimes", {})
            return tuple(
                (
                    callback,
                    (
                        lifetimes[(hook_name, id(callback))]["epoch"]
                        if lifetimes.get((hook_name, id(callback)), {}).get("count", 0) > 0
                        else None
                    ),
                )
                for callback in self._hooks.get(hook_name, ())
            )

    def _hook_timeout_generation(self) -> int:
        """Generation of this manager's timeout bookkeeping (0 until the first unload-all).

        Read through ``getattr`` so a manager-like double that never ran
        ``PluginManager.__init__`` still answers 0, and so the counter's only writer stays in the
        unload-all path next to the maps it fences.
        """
        return getattr(self, _HOOK_TIMEOUT_GENERATION_ATTR, 0)

    def _hook_callback_still_registered(self, hook_name: str, cb: Callable) -> bool:
        """Whether *cb* is still a live registration of *hook_name* on this manager.

        The suppression window is a fact about a REGISTRATION, not about an object address: a
        callback that a dispose/unload retired must not be able to back-date it. Unknown manager
        shapes answer ``True`` (publish as before) rather than silently dropping the back-off.
        """
        lifetime = getattr(self, "_hook_registration_lifetimes", {}).get((hook_name, id(cb)))
        if lifetime is not None:
            return lifetime["count"] > 0
        hooks = getattr(self, "_hooks", None)
        if not isinstance(hooks, dict):
            return True
        return any(candidate is cb for candidate in hooks.get(hook_name, []))

    def _hook_suppression_remaining_locked(self, suppression_key: tuple) -> float:
        """Seconds of active timeout back-off for *suppression_key*; ``0.0`` when none is left.

        Caller holds ``_hook_timeout_lock`` (directly, or via the gate's condition, which wraps
        it). An expired entry is dropped here, so every reader also sweeps.
        """
        suppressed_until = self._hook_timeout_suppressed_until.get(suppression_key)
        if suppressed_until is None:
            return 0.0
        now = time.monotonic()
        if suppressed_until > now:
            return suppressed_until - now
        self._hook_timeout_suppressed_until.pop(suppression_key, None)
        return 0.0

    def _report_hook_failure(
        self, hook_name: str, cb: Callable, kwargs: Dict[str, Any], exc: Exception, *, surface: str = "Hook"
    ) -> None:
        """One WARNING per distinct (hook, callback, error); identical repeats at DEBUG.

        A callback whose signature names a parameter the hook never sends (``tool_data`` instead
        of ``tool_name``/``args``) fails identically on every tool call — ~1700 WARNING lines an
        hour that bury real signals (#111922). The first report names the fields the hook does
        provide so the plugin author can fix the signature. The key names the callback by
        module/qualname (not ``id()``, which CPython recycles across plugin reloads) and
        truncates the message so a hook that embeds tool args in its error cannot grow the set
        per call; the set is cleared on unload alongside the timeout-suppression map.
        """
        callback_name = getattr(cb, "__name__", repr(cb))
        key = (hook_name, getattr(cb, "__module__", ""), getattr(cb, "__qualname__", callback_name),
               type(exc).__name__, str(exc)[:200])
        if key in self._hook_failures_reported:
            logger.debug("%s '%s' callback %s raised again: %s", surface, hook_name, callback_name, exc)
            return
        self._hook_failures_reported.add(key)
        logger.warning(
            "%s '%s' callback %s raised: %s (%s provides: %s; identical failures are logged at DEBUG from now on)",
            surface, hook_name, callback_name, exc, surface.lower(), ", ".join(sorted(kwargs)) or "no fields")

    def _run_hook_callback_bounded(
        self, hook_name: str, cb: Callable, kwargs: Dict[str, Any], timeout: float, *,
        registration_epoch: Optional[int] = None,
    ) -> Any:
        """Run one callback on a daemon worker with a wall-clock cap; ``_HOOK_SKIPPED`` when
        suppressed after a previous timeout, re-entered from the callback's own thread, timed out
        (worker abandoned, never joined), or the worker could not be started. A callback that is
        merely *in flight* is not a skip: the caller parks on the gate (bounded, FIFO) and fires
        anyway once the wait budget is spent, so one session cannot block another. Exceptions
        propagate."""
        callback_name = getattr(cb, "__name__", repr(cb))
        # Gate one callback across tools/sessions/calls; call identity remains telemetry only.
        _tool_scope = kwargs.get("tool_name")
        callback_key = (hook_name, id(cb))
        # Suppression is callback+tool scoped; lifecycle hooks keep ``tool=None`` callback-wide.
        suppression_key = (
            hook_name,
            id(cb),
            _tool_scope if isinstance(_tool_scope, str) else None,
        )
        call_identity = _hook_call_identity(kwargs)
        identity_key = (
            (hook_name, id(cb), call_identity) if call_identity is not None else None
        )
        gate = self._hook_gate_state()
        token = _HookToken()
        blocked_for = 0.0
        with self._hook_timeout_lock:
            # A2 (W3): the generation this launch belongs to. Captured BEFORE the worker starts,
            # so a timeout that lands after an unload-all tore those maps down is recognizable as
            # a late write from the dead generation and refused at the publication below.
            generation = self._hook_timeout_generation()
            # Whether this launch came from a LIVE registration at all: a callback invoked
            # directly (diagnostic/direct dispatcher use) was never in ``_hooks``, so the
            # registration half of the A2 fence must not be applied to it.
            registered_at_launch = self._hook_callback_still_registered(hook_name, cb)
            blocked_for = self._hook_suppression_remaining_locked(suppression_key)
            running_identities = getattr(self, "_hook_running_identities", None)
            if running_identities is None:
                running_identities = {}
                self._hook_running_identities = running_identities
        if blocked_for > 0.0:
            # The one remaining blocking reason: this callback already ran away once, so a policy
            # hook stays fail-closed until the suppression window expires.
            gate.stats.record_blocked()
            logger.warning(
                "Hook '%s' callback %s skipped after previous timeout (%.1fs of suppression left)",
                hook_name, callback_name, blocked_for)
            return _HOOK_SKIPPED
        with self._hook_timeout_lock:
            if identity_key is not None and identity_key in self._hook_running_identities:
                logger.warning(
                    "Hook '%s' callback %s skipped: call identity %s is already running",
                    hook_name, callback_name, call_identity)
                return _HOOK_SKIPPED
            if getattr(self, "_hook_abandoned", {}).get(callback_key):
                logger.warning(
                    "Hook '%s' callback %s skipped while an abandoned worker is still running",
                    hook_name, callback_name)
                return _HOOK_SKIPPED
        # Contention is not a verdict: park (bounded, FIFO) on the sibling call instead of
        # skipping it, and fire anyway once the budget is spent. Without the park a fast hook in
        # another session rejects this tool call — the false block measured 2026-09-14.
        wait_budget = min(_HOOK_GATE_WAIT_MAX_SECONDS, timeout * _HOOK_GATE_WAIT_TIMEOUT_FRACTION)
        acquired = self._acquire_hook_gate(
            gate, callback_key, suppression_key, token, wait_budget
        )
        if acquired is _HOOK_SKIPPED:
            logger.warning(
                "Hook '%s' callback %s skipped: waiting is disabled (%s=0) and a sibling call "
                "holds the gate", hook_name, callback_name, _HOOK_GATE_WAIT_ENV)
            return _HOOK_SKIPPED
        if acquired is _HOOK_GATE_REENTRANT:
            logger.warning(
                "Hook '%s' callback %s skipped: re-entrant call from the thread already running it",
                hook_name, callback_name)
            return _HOOK_SKIPPED
        if acquired is _HOOK_GATE_SUPPRESSED:
            # A3 (W3): this caller was already past the suppression pre-check when the back-off
            # became active (the sibling it parked behind just timed out). Rechecked under the
            # gate's condition — the same lock the timeout publication takes — so the check and
            # the claim cannot interleave. No token was installed and the queued ticket is
            # released by _acquire_hook_gate's finally: no worker, no counter drift.
            with self._hook_timeout_lock:
                remaining = self._hook_suppression_remaining_locked(suppression_key)
            gate.stats.record_blocked()
            logger.warning(
                "Hook '%s' callback %s skipped: previous timeout back-off became active while "
                "this call was waiting (%.1fs of suppression left)", hook_name, callback_name, remaining)
            return _HOOK_SKIPPED
        holding, waited, fail_open_reason = acquired
        # Lava bound (2026-09-14, operator decision A/K=2): the wait-then-fire path adds a launch
        # only while this key is below K simultaneous launches. Above that the legacy behaviour is
        # kept — skip, which pre_tool_call turns into a fail-closed block — because past K the gate
        # no longer synchronizes anything and a stampede is what the skip existed to prevent.
        if not gate.begin_launch(callback_key, bounded=bool(fail_open_reason)):
            concurrent = gate.launches_for(callback_key)
            gate.stats.record_lava_bound()
            logger.warning(
                "Hook '%s' callback %s skipped: %d launches of this key already running (lava "
                "bound K=%d) — legacy fail-closed skip kept",
                hook_name, callback_name, concurrent, _HOOK_GATE_MAX_CONCURRENT_LAUNCHES)
            return _HOOK_SKIPPED
        with self._hook_timeout_lock:
            if identity_key is not None:
                self._hook_running_identities[identity_key] = token
        if fail_open_reason:
            logger.warning(
                "Hook '%s' callback %s fired WITHOUT the gate after waiting %.2fs (%s) — a slow "
                "sibling is not a verdict on this call",
                hook_name, callback_name, waited, fail_open_reason)

        context = contextvars.copy_context()
        done = threading.Event()
        outcome: Dict[str, Any] = {}
        failure: Dict[str, Exception] = {}
        admission = threading.Condition()
        admission_state = {"ready": False, "cancelled": False}

        def _release_token() -> None:
            gate.release(self._hook_running_callbacks, callback_key, token)
            with self._hook_timeout_lock:
                if (identity_key is not None
                        and self._hook_running_identities.get(identity_key) is token):
                    self._hook_running_identities.pop(identity_key, None)
                abandoned = getattr(self, "_hook_abandoned", {}).get(callback_key)
                if abandoned is not None:
                    abandoned.discard(token)
                    if not abandoned:
                        self._hook_abandoned.pop(callback_key, None)

        def _runner() -> None:
            try:
                with admission:
                    admission.wait_for(lambda: admission_state["ready"])
                    if admission_state["cancelled"]:
                        return
                outcome["value"] = context.run(self._invoke_hook_callback, cb, kwargs)
            except Exception as exc:
                failure["exc"] = exc
            finally:
                _release_token()
                done.set()

        thread = threading.Thread(target=_runner, name=f"hermes-hook-{callback_name}"[:40], daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            _release_token()  # the runner's finally never runs when OS thread creation fails
            gate.end_launch(callback_key)  # nor does the caller tail: give the slot back here
            logger.warning(
                "Hook '%s' callback %s worker failed to start: %s — skipping",
                hook_name, callback_name, exc)
            return _HOOK_SKIPPED
        # F2: Thread.start may be delayed after reservation.  The worker cannot enter the
        # callback body until suppression is rechecked under the publication/lifecycle lock.
        with self._hook_timeout_lock:
            suppressed_before_admission = (
                self._hook_suppression_remaining_locked(suppression_key) > 0.0
            )
            with admission:
                admission_state["cancelled"] = suppressed_before_admission
                admission_state["ready"] = True
                admission.notify_all()
        if suppressed_before_admission:
            _release_token()
            gate.end_launch(callback_key)
            gate.stats.record_blocked()
            logger.warning(
                "Hook '%s' callback %s skipped: timeout suppression became active before "
                "worker admission", hook_name, callback_name)
            return _HOOK_SKIPPED
        gate.stats.record_fired(waited)  # the callback runs now, fail-open included
        if not holding:
            gate.stats.record_fail_open()
        try:
            if not done.wait(timeout=timeout):  # do not join — that would reintroduce the hang
                stale_generation = False
                with self._hook_timeout_lock:
                    # See #6622. A2 (W3): publish a suppression ONLY for the generation that is
                    # still current, and — when this launch came from a live registration — only
                    # while that registration is still live. The unload-all clear and its
                    # generation bump happen under this same lock, so this write either precedes
                    # the clear (and is wiped by it) or follows it and is refused here: the maps
                    # an unload-all emptied can no longer be repopulated by the generation it
                    # tore down. Same guard for a registration disposed while its abandoned
                    # worker was still running (A1) — the retired callback must not back-date a
                    # deadline onto the address it is about to give up.
                    current_lifetime = getattr(
                        self, "_hook_registration_lifetimes", {}
                    ).get((hook_name, id(cb)))
                    registration_retired = (
                        registration_epoch is not None
                        and (
                            current_lifetime is None
                            or current_lifetime["count"] == 0
                            or current_lifetime["epoch"] != registration_epoch
                        )
                    )
                    if (self._hook_timeout_generation() != generation
                            or registration_retired
                            or (registration_epoch is None and registered_at_launch
                                and not self._hook_callback_still_registered(hook_name, cb))):
                        stale_generation = True
                    else:
                        self._hook_timeout_suppressed_until[suppression_key] = (
                            time.monotonic() + self._hook_timeout_suppression_seconds)
                    if self._hook_running_callbacks.get(callback_key) is token:
                        abandoned_callbacks = getattr(self, "_hook_abandoned", None)
                        if abandoned_callbacks is None:
                            abandoned_callbacks = {}
                            self._hook_abandoned = abandoned_callbacks
                        abandoned_callbacks.setdefault(callback_key, set()).add(token)
                if stale_generation:
                    logger.warning(
                        "Hook '%s' callback %s timed out after %gs but its registration is no "
                        "longer live — not repopulating the timeout bookkeeping of the cleared "
                        "generation", hook_name, callback_name, timeout)
                # 2026-09-14: the abandoned worker may never reach its finally, so the running
                # slot and the gate holder are released HERE (identity-checked); otherwise the
                # callback stays latched for the life of the process, which for pre_tool_call
                # (fail-closed) is a permanent tool outage. Backing off a hung callback is the
                # suppression window's job, not a latched slot; G5 adds the gate half so a parked
                # caller is never left waiting on a holder that will never finish. Outside the
                # with-block below on purpose: gate.release takes that same non-reentrant lock.
                gate.release(self._hook_running_callbacks, callback_key, token)
                logger.warning(
                    "Hook '%s' callback %s timed out after %gs — skipping",
                    hook_name, callback_name, timeout)
                return _HOOK_SKIPPED
            if "exc" in failure:
                raise failure["exc"]
            return outcome.get("value")
        finally:
            # The launch slot belongs to the CALLER, not the worker (see end_launch).
            gate.end_launch(callback_key)

    def _acquire_hook_gate(
        self, gate: _HookGateState, callback_key: tuple, suppression_key: tuple,
        token: _HookToken, wait_budget: float
    ) -> Any:
        """Claim the per-callback gate, parking FIFO while a sibling call holds it.

        Returns ``(token_or_None, waited_seconds, fail_open_reason)``; a ``None`` token means the
        caller must fire WITHOUT the gate (wait budget spent, or the waiting room is full) —
        fail-open, because another session's slow hook is not a verdict on this call. Returns
        ``_HOOK_SKIPPED`` only for a re-entrant call from the holder's own thread, which must not
        nest into its own callback, and ``_HOOK_GATE_SUPPRESSED`` when the callback's timeout
        back-off became active after the caller's pre-check (A3): the claim and the suppression
        recheck happen in ONE critical section — ``gate.cond`` wraps ``_hook_timeout_lock``, the
        same lock the timeout publication takes — so a parked caller can no longer launch a worker
        during the back-off. Nothing is left claimed on that path: the token is never installed
        into ``running`` and the queued ticket is dropped by ``finally``.
        """
        started = time.monotonic()
        deadline = started + wait_budget
        running = self._hook_running_callbacks
        with gate.cond:  # the same underlying lock, so the queue and the holder map move together
            # The slot is taken INSIDE this critical section, not before it: H6 keys make slots
            # numerous and droppable, and a caller must never park on a slot the release/prune
            # path has already dropped - that would give one key two FIFOs.
            slot = gate.slot(callback_key)
            holder = running.get(callback_key)
            if holder is not None and holder.thread_id == threading.get_ident():
                return _HOOK_GATE_REENTRANT
            if holder is None and not slot.queue:
                if self._hook_suppression_remaining_locked(suppression_key) > 0.0:
                    return _HOOK_GATE_SUPPRESSED
                running[callback_key] = token
                return token, 0.0, ""
            if not _hook_gate_wait_enabled():
                # Kill switch: legacy behaviour — a sibling in flight skips this call.
                return _HOOK_SKIPPED
            if len(slot.queue) >= _HOOK_GATE_MAX_WAITERS:
                # A3 (W3): the overflow path fires WITHOUT the gate — the last launch-permitting
                # exit of this method, so it rechecks the back-off too.
                if self._hook_suppression_remaining_locked(suppression_key) > 0.0:
                    return _HOOK_GATE_SUPPRESSED
                return None, time.monotonic() - started, "waiting room full (%d)" % _HOOK_GATE_MAX_WAITERS
            ticket = slot.next_ticket
            slot.next_ticket += 1
            slot.queue.append(ticket)
            try:
                while True:
                    holder = running.get(callback_key)
                    if holder is None and slot.queue and slot.queue[0] == ticket:
                        # A3 (W3): the wait → claim transition rechecks the back-off under the
                        # same lock the timeout publication takes. The sibling this caller
                        # parked behind may have timed out meanwhile: without this recheck the
                        # caller claims the freed token and launches a worker INSIDE the
                        # back-off window (measured: starts=2, waiter_launched_during_suppression).
                        if self._hook_suppression_remaining_locked(suppression_key) > 0.0:
                            return _HOOK_GATE_SUPPRESSED
                        slot.queue.pop(0)
                        running[callback_key] = token
                        return token, time.monotonic() - started, ""
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # The wait-then-fire path would launch here; a back-off that became
                        # active during the wait outranks fail-open (same coarse contract).
                        if self._hook_suppression_remaining_locked(suppression_key) > 0.0:
                            return _HOOK_GATE_SUPPRESSED
                        return None, time.monotonic() - started, "wait budget %.2fs spent" % wait_budget
                    gate.cond.wait(remaining)
            finally:  # a caller that leaves the queue early must keep the FIFO order honest
                if ticket in slot.queue:
                    slot.queue.remove(ticket)
                    gate.cond.notify_all()
                gate.prune_slot(running, callback_key)  # the caller still holds the lock here

    def _hook_gate_state(self) -> _HookGateState:
        """Return this manager's gate state, creating it on first use.

        Lazy (rather than in ``PluginManager.__init__``) because the dispatcher mixin owns the
        gate: the condition wraps the existing ``_hook_timeout_lock``, so the gate and the timeout
        bookkeeping share one critical section.
        """
        state = getattr(self, "_hook_gate_state_singleton", None)
        if state is None:
            with _HOOK_GATE_INIT_LOCK:
                state = getattr(self, "_hook_gate_state_singleton", None)
                if state is None:
                    state = _HookGateState(self._hook_timeout_lock)
                    self._hook_gate_state_singleton = state
        return state

    def hook_gate_metrics(self) -> Dict[str, Any]:
        """Gate counters as JSON-friendly data: fired, waited, waited_p95, fail_open,
        blocked_after_timeout, lava_bound, concurrent_peak, launches_in_flight, slots (live
        gate slots, pruned when idle). Makes the synchronizer observable instead of a matter
        of faith."""
        state = self._hook_gate_state()
        data = state.stats.snapshot()
        with state.cond:
            data["concurrent_peak"] = state.peak_launches
            data["launches_in_flight"] = sum(state.launches.values())
            data["slots"] = len(state.slots)
        return data

    def _subscribe_event(self, owner: str, event: str, callback: Callable) -> None:
        """Add an owner-tagged event subscription in registration order."""
        if not callable(callback):
            raise TypeError("Event subscriber callback must be callable")
        with self._event_lock:
            self._subscriptions.setdefault(event, []).append(_EventSubscription(owner, callback))

    def _remove_plugin_subscriptions(self, owner: str) -> int:
        """Remove every subscription owned by *owner*; return the count. Queued envelopes re-check
        membership per callback, so this also cancels already-snapshotted deliveries.

        TODO(#64229): when the central plugin ownership ledger / registration handles land, route this
        owner-tagged bookkeeping through that ledger so per-plugin unload cancels event subscriptions
        alongside every other registration surface. This method is the integration seam.
        """
        removed = 0
        with self._event_lock:
            for event in list(self._subscriptions):
                entries = self._subscriptions[event]
                retained = [entry for entry in entries if entry.owner != owner]
                removed += len(entries) - len(retained)
                if retained:
                    self._subscriptions[event] = retained
                else:
                    del self._subscriptions[event]
        return removed

    def _ensure_event_worker_locked(self) -> None:
        worker = self._event_worker
        if worker is not None and worker.is_alive():
            return
        worker = threading.Thread(
            target=self._event_worker_loop, args=(self._event_queue,), name="hermes-plugin-events",
            daemon=True,
        )
        self._event_worker = worker
        worker.start()

    def _event_worker_loop(self, dispatch_queue: queue.Queue[Any]) -> None:
        while True:
            item = dispatch_queue.get()
            try:
                if item is _EVENT_WORKER_STOP:
                    return
                self._deliver_event(item)
            finally:
                if item is not _EVENT_WORKER_STOP:
                    self._mark_event_done(item.generation)
                dispatch_queue.task_done()

    def _mark_event_done(self, generation: int) -> None:
        with self._event_idle:
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending > 0:
                self._event_pending_by_generation[generation] = pending - 1
            self._event_idle.notify_all()

    def _deliver_event(self, item: _QueuedPluginEvent) -> None:
        """Deliver one queued event on the host-owned worker thread."""
        from hermes_cli.plugins import resolve_plugin_command_result
        with self._event_lock:
            if item.generation != self._event_generation:
                return
        previous_depth = getattr(self._emit_depth, "value", 0)
        self._emit_depth.value = item.depth
        try:
            for subscription in item.subscriptions:
                with self._event_lock:
                    if item.generation != self._event_generation:
                        break
                    # Owner unload may have removed this entry after the event was queued.
                    if not any(cur is subscription for cur in self._subscriptions.get(item.event, [])):
                        continue
                callback = subscription.callback
                try:
                    # Fresh deep copy per subscriber: no callback can mutate what the next sees.
                    resolve_plugin_command_result(callback(**copy.deepcopy(item.payload)))
                except Exception as exc:
                    # A subscriber that fails identically on every emit is reported once (#111922).
                    self._report_hook_failure(item.event, callback, item.payload, exc, surface="Event")
        finally:
            self._emit_depth.value = previous_depth

    def _wait_for_event_dispatch(self, timeout: float = 2.0) -> bool:
        """Wait for the current event generation to become idle (test helper)."""
        with self._event_idle:
            generation = self._event_generation
            return self._event_idle.wait_for(
                lambda: self._event_pending_by_generation.get(generation, 0) == 0, timeout=timeout)

    def _dispatch_event(self, event: str, payload: Dict[str, Any]) -> int:
        """Queue *event* without blocking; return the subscriber count scheduled. Pending work is
        bounded per generation so a blocking subscriber costs one worker and later emits drop."""
        depth = getattr(self._emit_depth, "value", 0)
        if depth >= _EVENT_EMIT_DEPTH_CAP:
            logger.warning(
                "Event bus recursion cap (%d) exceeded while dispatching '%s' "
                "— dropping this emit to prevent an infinite loop", _EVENT_EMIT_DEPTH_CAP, event)
            return 0
        budget_msg = "Event bus pending budget (%d) exhausted while dispatching '%s' — dropping this emit"
        with self._event_lock:
            subscriptions = tuple(self._subscriptions.get(event, []))
            if not subscriptions:
                return 0
            generation = self._event_generation
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending >= _EVENT_PENDING_CAP:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            item = _QueuedPluginEvent(
                event=event, payload=dict(payload), subscriptions=subscriptions, depth=depth + 1,
                generation=generation)
            try:
                self._event_queue.put_nowait(item)
            except queue.Full:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            self._event_pending_by_generation[generation] = pending + 1
            self._ensure_event_worker_locked()
            return len(subscriptions)

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        return bool(self._hooks.get(hook_name))

    def iter_hook_callbacks(self, hook_name: str) -> tuple[Callable, ...]:
        """Return a stable snapshot of callbacks registered for a hook."""
        return tuple(self._hooks.get(hook_name, ()))

    def render_system_prompt_sections(
        self, session_info: Mapping[str, Any]
    ) -> List[RenderedPluginSystemPromptSection]:
        """Render all registered sections deterministically and fail open."""
        frozen_info = types.MappingProxyType(dict(session_info))
        rendered: List[RenderedPluginSystemPromptSection] = []
        total_chars = len(PLUGIN_SECTIONS_START) + len(PLUGIN_SECTIONS_END) + 2
        for _section_id, section in sorted(self._system_prompt_sections.items()):
            if len(rendered) >= MAX_SYSTEM_PROMPT_SECTIONS:
                logger.warning(
                    "Plugin system prompt section %s exceeded the section-count "
                    "budget (%d) and was skipped", section.id, MAX_SYSTEM_PROMPT_SECTIONS)
                continue
            text = self._render_prompt_section_text(section, frozen_info)
            if text is None:
                continue
            rendered_chars = len(format_system_prompt_section(section.id, text))
            if rendered:
                rendered_chars += 2  # canonical ``\n\n`` separator
            if total_chars + rendered_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded the aggregate "
                    "session budget (%d chars) and was skipped", section.id, section.plugin,
                    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS)
                continue
            rendered.append(
                RenderedPluginSystemPromptSection(
                    id=section.id, content=text, position=section.position, plugin=section.plugin))
            total_chars += rendered_chars
            logger.info(
                "Session plugin prompt section: id=%s plugin=%s position=%s chars=%d", section.id,
                section.plugin, section.position, len(text))
        return rendered

    @staticmethod
    def _render_prompt_section_text(
        section: PluginSystemPromptSection, frozen_info: Mapping[str, Any]
    ) -> Optional[str]:
        """Evaluate one section; return its stripped text or None (with a warning) when skipped."""
        def _skip(detail: str, *args: Any) -> None:
            logger.warning(
                "Plugin system prompt section %s (%s) " + detail, section.id, section.plugin, *args)

        try:
            value = section.content(frozen_info) if callable(section.content) else section.content
        except Exception as exc:
            _skip("raised and was skipped: %s", exc)
            return None
        if not isinstance(value, str):
            _skip("returned %s, not str; skipped", type(value).__name__)
            return None
        text = value.strip()
        if not text:
            return None
        if PLUGIN_SECTIONS_START in text or PLUGIN_SECTIONS_END in text:
            _skip("contained a reserved persistence marker and was skipped")
            return None
        if len(text) > section.max_chars:
            _skip("exceeded max_chars (%d > %d) and was skipped", len(text), section.max_chars)
            return None
        return text

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        return bool(self._middleware.get(kind))

    def invoke_middleware(self, kind: str, **kwargs: Any) -> List[Any]:
        """Call middleware callbacks for *kind* (each isolated); return non-``None`` results."""
        results: List[Any] = []
        for cb in self._middleware.get(kind, []):
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                # Runs once per tool call like a hook, so a mis-declared callback floods identically.
                self._report_hook_failure(kind, cb, kwargs, exc, surface="Middleware")
        return results
