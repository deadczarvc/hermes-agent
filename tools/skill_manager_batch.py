"""Atomic multi-op batch path for ``skill_manage``. Origin state
(``skill_manage``/``_find_skill``/``_skill_gate_bypass``) is reached lazily
through ``tools.skill_manager_tool`` so that module owns it.

Atomicity is enforced at this one shared boundary, in two layers:

* in-process — every touched skill is snapshotted before the first op, and an op
  that fails, or a cancellation the tool executor delivers as an interrupt, rolls
  ALL of them back;
* out-of-process — the snapshots and a transaction journal live on disk
  (``<hermes_home>/.skill-batch-txn/<pid>-<id>/``). The journal is written BEFORE
  the first op and dropped only after the last one succeeded, so a process killed
  mid-batch leaves either a complete batch (journal already gone) or a journal
  that the next entry replays. A timed-out or killed call therefore leaves every
  requested edit or none.
"""

import json
import logging
import os
import posixpath
import shutil
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger("tools.skill_manager_tool")

_BATCH_OP_ACTIONS = {"create", "patch", "write_file", "remove_file"}
_BATCH_MAX_OPS = 20
_BATCH_TXN_DIRNAME = ".skill-batch-txn"  # Hermes local patch 2026-09-19: constant lost in upstream refactor (NameError on every batch skill_manage)
# --- Hermes local patch 2026-09-19: batch-liveness/journal globals lost in the same refactor ---
_JOURNAL_NAME = "transaction.json"
_LIVE_OWNER_GRACE_S = 120.0
import threading as _threading
_ACTIVE_LOCK = _threading.Lock()
_ACTIVE_BATCHES: set = set()

# --- Per-op argument shape (checked before any effect) ---------------------------------
# action -> (arg, is_missing, error) checks run before the handler.
_MISSING, _IS_NONE = (lambda v: not v), (lambda v: v is None)
_REQUIRED_ARGS = {
    "create": [("content", _MISSING,
                "content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).")],
    "edit": [("content", _MISSING,
              "content is required for a full rewrite. Provide the full updated SKILL.md text.")],
    "write_file": [
        ("file_path", _MISSING, "file_path is required for 'write_file'. Example: 'references/api-guide.md'"),
        ("file_content", _IS_NONE, "file_content is required for 'write_file'.")],
    "remove_file": [("file_path", _MISSING, "file_path is required for 'remove_file'.")]}
# A bare "required" error is a dead end: the model retries blindly and often escapes to
# action='write_file', clobbering the whole file.
_PATCH_NEEDS_OLD_STRING = (
    "old_string is required for 'patch' and must be the EXACT text currently in the file. "
    "Read the target file first (read_file on the skill's SKILL.md, or the file named by "
    "file_path) and copy the snippet verbatim, then retry 'patch'. Do NOT fall back to "
    "action='write_file' — that rewrites the entire file and destroys unrelated content.")
_PATCH_NEEDS_NEW_STRING = "new_string is required for 'patch'. Use an empty string to delete matched text."
_PATCH_EITHER_OR = ("Pass EITHER content (full SKILL.md rewrite) OR old_string/new_string "
                    "(targeted replacement), not both.")
# Text-slot keys a model confuses: key -> the action that reads it. A 27B model that just
# used write_file's file_content re-emits it on create/patch and then replays the identical
# payload when the error only says the right key is "required" — the hint has to name where
# the text actually landed so the retry can move it.
_TEXT_SLOT_OWNER = {"content": "create (and a full-rewrite patch)",
                    "new_string": "a targeted patch (with old_string)",
                    "file_content": "write_file"}
# action -> (text slots it reads, where misfiled text belongs)
_TEXT_SLOT_FOR = {
    "create": (("content",), "'content'"),
    "edit": (("content",), "'content'"),
    "patch": (("content", "new_string"), "old_string/new_string (targeted) or 'content' (full rewrite, last resort)"),
    "write_file": (("file_content",), "'file_content'")}


def _misplaced_text_hint(action: str, args: dict) -> str:
    """Sentence naming the text-slot key(s) this op carries that ``action`` never reads, or ''."""
    if action not in _TEXT_SLOT_FOR:
        return ""  # delete/remove_file/unknown: no text slot, so no destination to point at
    reads, destination = _TEXT_SLOT_FOR[action]
    stray = [k for k in _TEXT_SLOT_OWNER if k not in reads and args.get(k) is not None]
    if not stray:
        return ""
    carried = " and ".join(f"'{k}' (that key is for {_TEXT_SLOT_OWNER[k]})" for k in stray)
    return f" Note: this op carries {carried} — move that text to {destination}."


def _op_shape_error(action: str, args: dict):
    """Argument-shape error text for one op, or None. Only shape misses carry the misplaced-text
    hint: a patch whose real problem is an unmatched old_string must not be steered to a full
    rewrite. Pure function so the batch can reject a misfiled op BEFORE any sibling is applied."""
    for arg, missing, message in _REQUIRED_ARGS.get(action, ()):
        if missing(args.get(arg)):
            return message + _misplaced_text_hint(action, args)
    if action == "patch":
        # Every patch shape miss is decided here, not in the handler, so a batch never applies
        # op[0] only to roll it back over op[1]'s missing new_string or content+old_string mix.
        if args.get("content") and (args.get("old_string") or args.get("new_string") is not None):
            return _PATCH_EITHER_OR
        if not args.get("old_string") and not args.get("content"):
            return _PATCH_NEEDS_OLD_STRING + _misplaced_text_hint(action, args)
        if not args.get("content") and args.get("new_string") is None:
            return _PATCH_NEEDS_NEW_STRING
    return None


def _validate_batch_ops(operations, default_name, tool_error):
    """Shape checks with no side effects. Returns (names, None) or (None, error_json)."""
    from tools.skill_manager_guards import _background_review_preflight
    def fail(i, msg):
        return None, tool_error(f"operations[{i}]{msg}", success=False)
    names = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict) or not op.get("action"):
            return fail(i, " needs an 'action'.")
        act = op["action"]
        if act not in _BATCH_OP_ACTIONS:
            return fail(i, f": unknown action '{act}'. Batchable: "
                           f"{', '.join(sorted(_BATCH_OP_ACTIONS))}; delete must be sole.")
        nm = op.get("name") or default_name
        if not nm:
            return fail(i, " needs a 'name' (the skill it targets).")
        # Reject a misfiled op here, before any sibling is applied: a runtime failure on
        # op[1] would first apply op[0] and then roll the whole batch back.
        if (shape_err := _op_shape_error(act, op)) is not None:
            return fail(i, f" ({act} on '{nm}'): {shape_err}")
        names.append(nm)
        if act == "create" and nm in names[:-1]:
            return fail(i, f": create for '{nm}' must precede that skill's other ops.")
        if (preflight := _background_review_preflight(act, nm)) is not None:
            return None, json.dumps(preflight, ensure_ascii=False)
    # Clobber guard: a DESTRUCTIVE op (create/write_file/remove_file/full rewrite) on
    # a file an earlier op touched would SILENTLY discard its work — reject it.
    # Additive patches are always legal. Paths are normalized against spelling variants.
    touched_files = set()
    for i, op in enumerate(operations):
        act, nm = op["action"], names[i]
        # create and full-rewrite patch (content) always hit SKILL.md.
        full_rewrite = act == "patch" and bool(op.get("content"))
        fp = (op.get("file_path") or "").strip()
        target = ("SKILL.md" if (act == "create" or full_rewrite or not fp)
                  else posixpath.normpath(fp.lstrip("/")))
        key = (nm, target)
        if (act in ("create", "write_file", "remove_file") or full_rewrite) and key in touched_files:
            return fail(i, f": {act} on '{target}' of skill '{nm}' — an earlier op in this "
                           f"batch already touched that file, and this op would silently discard its work. "
                           f"One destructive op (write_file/remove_file/full rewrite) per file per batch; put "
                           f"it first, or fold the change in. Patch chains are fine.")
        touched_files.add(key)
    return names, None


def _snapshot_skills(names, snap_root, find_skill):
    """Copy every touched skill aside. Returns (snapshots, None) or (None, error_text)."""
    snapshots = {}  # skill name -> (pre_dir or None, snapshot_dir or None)
    for nm in dict.fromkeys(names):  # ordered unique
        pre = find_skill(nm)
        pre_dir = Path(pre["path"]) if pre else None
        snap = snap_root / nm if pre_dir is not None and pre_dir.is_dir() else None
        if snap is not None:
            try:
                shutil.copytree(pre_dir, snap)
            except Exception as exc:  # noqa: BLE001 — no snapshot, no atomicity
                return None, f"Could not snapshot '{nm}' for atomic batch: {exc}"
        snapshots[nm] = (pre_dir, snap)
    return snapshots, None


def _restore_snapshot(pre_dir, snap, post_dir) -> None:
    post_exists = post_dir is not None and post_dir.is_dir()
    if snap is None:
        if post_exists:  # Batch created this skill: remove the partial result.
            shutil.rmtree(post_dir)
        return
    if not post_exists:
        shutil.copytree(snap, pre_dir)
        return
    # Move the broken state aside and delete it only after the snapshot is
    # back, so a failed copytree (disk full, locked file) can't mean total loss.
    aside = post_dir.with_name(post_dir.name + ".rollback-broken")
    shutil.rmtree(aside, ignore_errors=True)
    post_dir.rename(aside)
    try:
        shutil.copytree(snap, pre_dir)
    except Exception:
        # Restore failed: put the half-applied state back rather than nothing.
        shutil.rmtree(pre_dir, ignore_errors=True)
        aside.rename(pre_dir)
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _rollback(snapshots, find_skill):
    """Restore every snapshot. Returns (note, failed)."""
    notes = []
    for nm, (pre_dir, snap) in snapshots.items():
        try:
            post = find_skill(nm)
            _restore_snapshot(pre_dir, snap, Path(post["path"]) if post else None)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"ROLLBACK FAILED for '{nm}' ({exc})"
                         + (f"; snapshot preserved at '{snap}'" if snap is not None else ""))
    return ("; ".join(notes) if notes else "all touched skills rolled back"), bool(notes)


# --- Durable transaction journal -------------------------------------------------

def _txn_root() -> Path:
    """Transaction root for the ACTIVE profile home (multi-profile runtimes rebind per session)."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _BATCH_TXN_DIRNAME


def _pid_alive(pid) -> bool:
    """Whether another process is still running. Read-only probe: Windows has no
    ``os.kill(pid, 0)`` (a 0 signal would TerminateProcess there), so ask the kernel for the
    exit code; POSIX uses the standard 0 signal. An unreadable answer counts as gone."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001 — a liveness probe must never break a write
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _write_journal(txn_dir: Path, snapshots) -> None:
    """Record how to undo the batch that is about to run — BEFORE its first op."""
    from utils import atomic_write_text

    record = {
        "batch_id": txn_dir.name,
        "pid": os.getpid(),
        "started_at": time.time(),
        "skills": [{"name": nm,
                    "pre_dir": str(pre_dir) if pre_dir is not None else None,
                    "snap": str(snap) if snap is not None else None}
                   for nm, (pre_dir, snap) in snapshots.items()],
    }
    atomic_write_text(txn_dir / _JOURNAL_NAME,
                      json.dumps(record, ensure_ascii=False), encoding="utf-8")


def _snapshots_from_journal(record) -> dict:
    """``{name: (pre_dir or None, snap or None)}`` — the shape ``_rollback`` consumes."""
    out = {}
    for entry in record.get("skills") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        pre, snap = entry.get("pre_dir"), entry.get("snap")
        out[entry["name"]] = (Path(pre) if pre else None, Path(snap) if snap else None)
    return out


def _close_txn(txn_dir: Path) -> None:
    """Drop the transaction dir — its journal and its snapshots.

    Dropping the journal is the COMMIT POINT: a batch whose journal is gone is final. A crash
    before that point leaves the journal, and the next entry rolls the whole batch back."""
    with suppress(OSError):
        (txn_dir / _JOURNAL_NAME).unlink()
    shutil.rmtree(txn_dir, ignore_errors=True)


def _recover_one(txn_dir: Path, find_skill):
    """Undo one interrupted batch. Returns a note, or None when it is not ours to touch."""
    with _ACTIVE_LOCK:
        if txn_dir.name in _ACTIVE_BATCHES:
            return None  # this process is running that batch right now
    journal = txn_dir / _JOURNAL_NAME
    record = None
    if journal.is_file():
        try:
            record = json.loads(journal.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — an unreadable journal means a dead owner
            logger.warning("skill_manage batch journal %s unreadable (%s); treating its batch "
                           "as interrupted", journal, exc)
    owner = record.get("pid") if isinstance(record, dict) else None
    if not isinstance(owner, int):
        head = txn_dir.name.split("-", 1)[0]
        owner = int(head) if head.isdigit() else None
    if owner is not None and owner != os.getpid() and _pid_alive(owner):
        with suppress(OSError):
            if time.time() - txn_dir.stat().st_mtime < _LIVE_OWNER_GRACE_S:
                return None  # a live owner may still be mid-batch: leave it alone
    if record is None:
        # The owner died before its first (and only) journal write, i.e. before any op ran:
        # there is nothing to undo, only staging to remove.
        shutil.rmtree(txn_dir, ignore_errors=True)
        return (f"removed abandoned batch staging dir '{txn_dir.name}' "
                f"(no journal: no op had been applied)")
    snapshots = _snapshots_from_journal(record)
    note, failed = _rollback(snapshots, find_skill)
    if failed:
        logger.warning("skill_manage batch recovery incomplete for %s: %s", txn_dir, note)
        return f"recovery of interrupted batch '{txn_dir.name}' is INCOMPLETE: {note}"
    shutil.rmtree(txn_dir, ignore_errors=True)
    return (f"rolled back interrupted batch '{txn_dir.name}' "
            f"({', '.join(sorted(snapshots))}): {note}")


def _recover_interrupted_batches() -> list:
    """Replay every journal whose owner died before its commit point.

    Called at the ``skill_manage`` entry, so a call that timed out (or a process that was killed)
    leaves either all of its edits or none: the next entry undoes the torn batch before applying
    its own op. Never raises — a repair failure keeps the journal for the next entry to retry."""
    notes = []
    try:
        root = _txn_root()
        if not root.is_dir():
            return notes
        from tools import skill_manager_tool as _smt
        for txn_dir in sorted(root.iterdir()):
            if not txn_dir.is_dir():
                continue
            try:
                note = _recover_one(txn_dir, _smt._find_skill)
            except Exception as exc:  # noqa: BLE001 — never break the caller's own write
                logger.warning("skill_manage batch recovery failed for %s: %s", txn_dir, exc)
                notes.append(f"recovery of '{txn_dir.name}' failed: {exc}")
                continue
            if note:
                logger.warning("skill_manage batch recovery: %s", note)
                notes.append(note)
    except Exception:  # noqa: BLE001 — recovery is best-effort by design
        logger.debug("skill_manage batch recovery scan failed", exc_info=True)
    return notes


def _cancel_reason():
    """Why this call was cancelled, else None. The tool executor sets the WORKER thread's
    interrupt bit when it gives up on a call at its deadline (agent/tool_executor.py) and /stop
    does the same, so a batch can notice the cancellation at an op boundary and undo itself
    instead of writing ops nobody will ever read."""
    try:
        from tools.interrupt import is_interrupted
        return "cancelled (tool timeout or /stop)" if is_interrupted() else None
    except Exception:  # noqa: BLE001 — no interrupt source, no cancellation
        return None


def _cancelled_result(i, op, name, reason, note=None) -> str:
    detail = f" — batch aborted, {note}." if note else " — nothing was applied."
    return json.dumps(
        {"success": False,
         "error": (f"operations[{i}] ({op.get('action')} on '{name}') was not applied: "
                   f"{reason}{detail}"),
         "failed_index": i, "completed_before_failure": i},
        ensure_ascii=False)


def _abort_batch(txn_dir: Path, snapshots, find_skill):
    """Undo a torn batch. Returns (note, rollback_failed).

    The journal is KEPT when the undo could not complete, so the next entry retries it instead
    of losing the only record of what was applied."""
    try:
        note, failed = _rollback(snapshots, find_skill)
    except BaseException as exc:  # noqa: BLE001 — never mask the original failure
        note, failed = f"rollback raised {exc!r}", True
    if failed:
        logger.warning("skill_manage batch rollback failed; journal kept at %s for the next "
                       "entry to retry", txn_dir)
        return note, True
    _close_txn(txn_dir)
    return note, False


def _skill_manage_batch(operations, default_name: str = None, task_id: str = None,
                        session_id: str = None) -> str:
    """Apply operations atomically: every touched skill is snapshotted and journalled first, and
    any failure — or a cancellation/timeout that reaches the batch — rolls ALL of them back
    (batch-created skills are removed). ``delete`` is only legal as the SOLE op (its
    recoverable-archive path doesn't compose with rollback) and routes to the single-op handler.
    ``default_name`` is the legacy top-level ``name`` fallback (staged replay)."""
    from tools import skill_manager_tool as _smt
    from tools.registry import tool_error
    if not isinstance(operations, list) or not operations:
        return tool_error("operations must be a non-empty array.", success=False)
    if len(operations) > _BATCH_MAX_OPS:
        return tool_error(f"operations is capped at {_BATCH_MAX_OPS} ops per call.", success=False)
    if any(isinstance(op, dict) and op.get("action") == "delete" for op in operations):
        if len(operations) != 1:
            return tool_error("delete must be the SOLE op in its call — it doesn't "
                              "compose with other ops' rollback.", success=False)
        nm = operations[0].get("name") or default_name
        if not nm:
            return tool_error("operations[0] (delete) needs a 'name'.", success=False)
        return _smt.skill_manage(action="delete", name=nm, task_id=task_id, session_id=session_id,
                                 absorbed_into=operations[0].get("absorbed_into"))
    names, err = _validate_batch_ops(operations, default_name, tool_error)
    if err is not None:
        return err
    if (cancelled := _cancel_reason()) is not None:
        return _cancelled_result(0, operations[0], names[0], cancelled)
    if not _smt._skill_gate_bypass.get():
        # Approval gate for the WHOLE batch as one pending write.
        def _staging(wa):
            acts = ", ".join(op["action"] for op in operations)
            gist = f"batch({len(operations)} ops: {acts}) on {', '.join(sorted(set(names)))}"
            return {"action": "batch", "operations": operations}, gist
        staged = _smt._run_write_gate(_staging)
        if staged is not None:
            return staged
    txn_dir = _txn_root() / f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    snap_root = txn_dir / "snap"
    try:
        snap_root.mkdir(parents=True)
    except OSError as exc:
        return tool_error(f"Could not create the atomic batch transaction dir: {exc}",
                          success=False)
    with _ACTIVE_LOCK:
        _ACTIVE_BATCHES.add(txn_dir.name)
    try:
        snapshots, snap_err = _snapshot_skills(names, snap_root, _smt._find_skill)
        if snap_err is not None:
            _close_txn(txn_dir)
            return tool_error(snap_err, success=False)
        try:
            # Journal BEFORE the first mutation: from here on a killed process leaves a record.
            _write_journal(txn_dir, snapshots)
        except OSError as exc:
            _close_txn(txn_dir)
            return tool_error(f"Could not write the atomic batch journal: {exc}", success=False)
        # Single-op path with the gate bypassed (the batch already cleared/staged it).
        results = []
        token = _smt._skill_gate_bypass.set(True)
        try:
            for i, op in enumerate(operations):
                if (cancelled := _cancel_reason()) is not None:
                    note, _failed = _abort_batch(txn_dir, snapshots, _smt._find_skill)
                    return _cancelled_result(i, op, names[i], cancelled, note)
                raw = _smt._skill_manage_from({**op, "name": names[i], "operations": None},
                                              task_id=task_id, session_id=session_id)
                try:
                    parsed = json.loads(raw)
                except Exception:  # noqa: BLE001
                    parsed = {"success": False, "error": "unparseable op result"}
                if not parsed.get("success"):
                    note, rollback_failed = _abort_batch(txn_dir, snapshots, _smt._find_skill)
                    fail = {  # key order is wire-visible
                        "success": False,
                        "error": (f"operations[{i}] ({op['action']} on '{names[i]}') failed: "
                                  f"{parsed.get('error', 'unknown error')} — batch aborted, {note}."),
                        "failed_index": i, "completed_before_failure": i}
                    # Carry the failing op's teaching payload (patch's file_preview /
                    # fuzzy-match hints) through — without it the model recovers blind.
                    for k, v in parsed.items():
                        if k not in ("success", "error") and v is not None:
                            fail.setdefault(k, v)
                    return json.dumps(fail, ensure_ascii=False)
                results.append({"name": names[i], "action": op["action"],
                                "file_path": op.get("file_path"), "success": True})
        except BaseException:
            # Hard cancellation while an op was in flight (KeyboardInterrupt/SystemExit, a
            # BaseException from a hook, or a kill delivered as one): undo what landed, then
            # re-raise so the caller's own timeout/interrupt handling still sees it. The journal
            # already on disk covers the case where this undo is itself cut short.
            _abort_batch(txn_dir, snapshots, _smt._find_skill)
            raise
        finally:
            _smt._skill_gate_bypass.reset(token)
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_BATCHES.discard(txn_dir.name)
    _close_txn(txn_dir)  # commit: every op is on disk, so the journal may go
    # utf-8-sig + errors="replace": SKILL.md files are user-authored and sometimes carry a Notepad BOM or
    # stray non-UTF-8 bytes. Pinning UTF-8 with replacement keeps skill_view deterministic across platforms
    # — falling back to the machine locale (cp1252/GBK) would make the same skill render differently per
    # host (see PR #51701).
    return json.dumps(
        {"success": True, "operations_applied": len(results), "results": results},
        ensure_ascii=False)
