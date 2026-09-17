"""skill_manage operations[] batch (#95681 arc, maintainer-approved).

Memory-tool pattern: several ops on ONE skill, atomically — create + N
supporting files, or SKILL.md + the script it references, in one call.
Any failure rolls the skill directory back to its pre-batch state.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SK = (
    "---\nname: {n}\ndescription: Use when probing batch ops. Behavior.\n---\n"
    "# Probe\nStep 1.\n"
)


class TestSkillManageBatch(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="skmbatch_t_")
        os.environ["HERMES_HOME"] = self.home
        os.environ["HERMES_YOLO_MODE"] = "1"
        os.makedirs(os.path.join(self.home, "skills"), exist_ok=True)
        # Re-import against the temp home (module caches SKILLS_DIR).
        import importlib

        import tools.skill_manager_tool as smt
        importlib.reload(smt)
        self.smt = smt

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def _call(self, name, ops):
        # Inject the per-op name (tests were written per-skill; the
        # interface is name-per-op, maintainer-directed).
        for op in ops:
            op.setdefault("name", name)
        return json.loads(self.smt.skill_manage(action="", name="", operations=ops))

    def test_create_plus_files_atomic(self):
        r = self._call("probe", [
            {"action": "create", "content": SK.format(n="probe")},
            {"action": "write_file", "file_path": "references/a.md", "file_content": "a"},
            {"action": "write_file", "file_path": "scripts/r.py", "file_content": "pass"},
        ])
        self.assertTrue(r["success"], r)
        self.assertEqual(r["operations_applied"], 3)
        base = os.path.join(self.home, "skills", "probe")
        for rel in ("SKILL.md", "references/a.md", "scripts/r.py"):
            self.assertTrue(os.path.exists(os.path.join(base, rel)), rel)

    def test_midbatch_failure_rolls_back_existing_skill(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        r = self._call("probe", [
            {"action": "patch", "old_string": "Step 1.", "new_string": "Step ONE."},
            {"action": "write_file", "file_path": "bad/nope.md", "file_content": "x"},
        ])
        self.assertFalse(r["success"])
        self.assertEqual(r["failed_index"], 1)
        content = open(os.path.join(self.home, "skills", "probe", "SKILL.md")).read()
        self.assertIn("Step 1.", content)       # patch undone
        self.assertNotIn("Step ONE.", content)

    def test_failed_create_batch_removes_partial_skill(self):
        r = self._call("fresh", [
            {"action": "create", "content": SK.format(n="fresh")},
            {"action": "write_file", "file_path": "../escape.md", "file_content": "x"},
        ])
        self.assertFalse(r["success"])
        self.assertFalse(os.path.exists(os.path.join(self.home, "skills", "fresh")))

    def test_validation_rules(self):
        # delete as SOLE op routes to the real delete (works)
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        r = self._call("probe", [{"action": "delete"}])
        self.assertTrue(r["success"], r)
        self.assertFalse(os.path.exists(os.path.join(self.home, "skills", "probe")))
        # delete mixed with other ops rejected
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        r = self._call("probe", [
            {"action": "patch", "old_string": "Step 1.", "new_string": "X."},
            {"action": "delete"},
        ])
        self.assertFalse(r["success"])
        self.assertIn("SOLE", r["error"])
        # create must be first
        r = self._call("x", [
            {"action": "write_file", "file_path": "references/a.md", "file_content": "a"},
            {"action": "create", "content": SK.format(n="x")},
        ])
        self.assertFalse(r["success"])
        # empty / capped
        r = self._call("x", [])
        self.assertFalse(r["success"])
        r = self._call("x", [{"action": "patch"}] * 21)
        self.assertFalse(r["success"])
        self.assertIn("capped", r["error"])

    def test_intra_batch_conflict_guard(self):
        """Same-file double writes and post-edit full rewrites are always
        a confused plan under last-wins sequencing — rejected BEFORE any
        side effect. Patch chains and rewrite-first stay legal."""
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        # destructive op on an already-touched file: rejected — double
        # write, write+remove, patch-then-write, patch-then-remove, and a
        # path-spelling variant of the same file.
        self._call("probe", [{"action": "write_file",
                              "file_path": "references/c.md", "file_content": "seed"}])
        for ops in (
            [{"action": "write_file", "file_path": "references/a.md", "file_content": "1"},
             {"action": "write_file", "file_path": "references/a.md", "file_content": "2"}],
            [{"action": "write_file", "file_path": "references/b.md", "file_content": "x"},
             {"action": "remove_file", "file_path": "references/b.md"}],
            [{"action": "patch", "file_path": "references/c.md",
              "old_string": "seed", "new_string": "edited"},
             {"action": "write_file", "file_path": "references/c.md", "file_content": "CLOB"}],
            [{"action": "patch", "file_path": "references/c.md",
              "old_string": "seed", "new_string": "edited"},
             {"action": "remove_file", "file_path": "references/c.md"}],
            [{"action": "write_file", "file_path": "references/d.md", "file_content": "1"},
             {"action": "write_file", "file_path": "./references//d.md", "file_content": "2"}],
        ):
            r = self._call("probe", ops)
            self.assertFalse(r["success"], ops)
            self.assertIn("discard", r["error"])
        # ...and rejected pre-effect: c.md still holds its seed text.
        c_md = os.path.join(self.home, "skills", "probe", "references", "c.md")
        self.assertEqual(open(c_md).read(), "seed")
        # write-then-patch on one supporting file stays legal (additive).
        r = self._call("probe", [
            {"action": "write_file", "file_path": "references/e.md", "file_content": "base"},
            {"action": "patch", "file_path": "references/e.md",
             "old_string": "base", "new_string": "base+"},
        ])
        self.assertTrue(r["success"], r)
        # patch then full rewrite: rejected; rewrite-first: allowed
        r = self._call("probe", [
            {"action": "patch", "old_string": "Step 1.", "new_string": "P."},
            {"action": "patch", "content": SK.format(n="probe")},
        ])
        self.assertFalse(r["success"])
        self.assertIn("rewrite", r["error"])
        r = self._call("probe", [
            {"action": "patch", "content": SK.format(n="probe").replace("Step 1.", "F.")},
            {"action": "patch", "old_string": "F.", "new_string": "G."},
        ])
        self.assertTrue(r["success"], r)
        # patch chains stay legal
        r = self._call("probe", [
            {"action": "patch", "old_string": "G.", "new_string": "H."},
            {"action": "patch", "old_string": "H.", "new_string": "I."},
        ])
        self.assertTrue(r["success"], r)

    def test_cross_skill_batch_and_rollback(self):
        """Ops may target DIFFERENT skills; a late failure rolls back
        every touched skill, including removing a batch-created one."""
        self._call("alpha", [{"action": "create", "content": SK.format(n="alpha")}])
        r = json.loads(self.smt.skill_manage(action="", name="", operations=[
            {"name": "alpha", "action": "patch",
             "old_string": "Step 1.", "new_string": "Step A."},
            {"name": "beta", "action": "create", "content": SK.format(n="beta")},
            {"name": "beta", "action": "write_file",
             "file_path": "bad/nope.md", "file_content": "x"},
        ]))
        self.assertFalse(r["success"])
        self.assertEqual(r["failed_index"], 2)
        # alpha's patch undone; beta (batch-created) removed entirely.
        content = open(os.path.join(self.home, "skills", "alpha", "SKILL.md")).read()
        self.assertIn("Step 1.", content)
        self.assertNotIn("Step A.", content)
        self.assertFalse(os.path.exists(os.path.join(self.home, "skills", "beta")))

    def test_failed_restore_never_destroys_the_skill(self):
        """Rollback used to rmtree the live skill directory BEFORE
        copytree restored the snapshot. When copytree failed (disk full,
        locked file) the except only appended a note and the finally then
        deleted the snapshot too: nothing survived. The broken state must
        be moved aside and only deleted once the restore succeeded."""
        from unittest.mock import patch as _patch

        import shutil as _shutil

        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        state = {"n": 0}
        real_copytree = _shutil.copytree

        def flaky_copytree(src, dst, *a, **k):
            state["n"] += 1
            if state["n"] == 2:  # call 1 snapshots, call 2 is the restore
                raise OSError("disk full")
            return real_copytree(src, dst, *a, **k)

        with _patch("shutil.copytree", side_effect=flaky_copytree):
            r = self._call("probe", [
                {"action": "patch",
                 "old_string": "Step 1.", "new_string": "Step ONE."},
                {"action": "write_file",
                 "file_path": "bad/nope.md", "file_content": "x"},
            ])
        self.assertFalse(r["success"], r)
        self.assertIn("ROLLBACK FAILED", r["error"])
        # The skill directory was NOT destroyed by the failed rollback:
        # the half applied state survives instead of nothing at all.
        skill_md = os.path.join(self.home, "skills", "probe", "SKILL.md")
        self.assertTrue(os.path.exists(skill_md))
        content = open(skill_md).read()
        self.assertIn("Step ONE.", content)

    def test_single_op_path_unchanged(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        raw = self.smt.skill_manage(
            action="patch", name="probe",
            old_string="Step 1.", new_string="Step 1 (single).",
        )
        self.assertTrue(json.loads(raw)["success"])

    def test_batch_stages_as_one_pending_write_when_gated(self):
        """Approval gate: the whole batch stages as ONE pending record, and
        apply_skill_pending replays it (operations key round-trips)."""
        from unittest.mock import patch as _patch

        class _Decision:
            allow = False
            blocked = False
            message = "staged for review"

        staged = {}

        def fake_stage_write(area, payload, summary=None, origin=None):
            staged.update(payload=payload, summary=summary)
            return {"id": "pend_1"}

        import tools.write_approval as wa

        with _patch.object(wa, "evaluate_gate", return_value=_Decision()), \
             _patch.object(wa, "stage_write", side_effect=fake_stage_write):
            r = self._call("probe", [
                {"action": "create", "content": SK.format(n="probe")},
                {"action": "write_file", "file_path": "references/a.md",
                 "file_content": "a"},
            ])
        self.assertTrue(r.get("staged"), r)
        self.assertEqual(staged["payload"]["action"], "batch")
        self.assertEqual(len(staged["payload"]["operations"]), 2)
        self.assertIn("2 ops", staged["summary"])
        # Replay applies the batch (gate bypassed inside).
        out = json.loads(self.smt.apply_skill_pending(staged["payload"]))
        self.assertTrue(out["success"], out)
        self.assertEqual(out["operations_applied"], 2)


class TestBatchTimeoutAtomicity(unittest.TestCase):
    """A timed-out or killed batch must leave EITHER every op applied or none.

    Observed failure (logs/errors.log, 2026-09-14 21:12:25): "sequential tool skill_manage
    timed out after 420.0s". The executor abandons the worker at its deadline
    (agent/tool_executor.py, _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0) and sets its
    interrupt bit; nothing tells the batch to stop, so the ops it already wrote stay on
    disk — and a process that dies mid-loop (dispatcher SIGTERM) leaves them with no
    record of how to undo them. The transaction journal written before the first op is
    that record, and the next entry replays it.
    """

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="skmbatch_to_")
        os.environ["HERMES_HOME"] = self.home
        os.environ["HERMES_YOLO_MODE"] = "1"
        os.makedirs(os.path.join(self.home, "skills"), exist_ok=True)
        import importlib

        import tools.skill_manager_tool as smt
        importlib.reload(smt)
        self.smt = smt
        from tools import interrupt as _interrupt
        self.interrupt = _interrupt
        self.addCleanup(self.interrupt.set_interrupt, False)

    def tearDown(self):
        self.interrupt.set_interrupt(False)
        shutil.rmtree(self.home, ignore_errors=True)

    def _call(self, name, ops):
        for op in ops:
            op.setdefault("name", name)
        return json.loads(self.smt.skill_manage(action="", name="", operations=ops))

    def _skill_md(self, name="probe"):
        return open(os.path.join(self.home, "skills", name, "SKILL.md")).read()

    def _txn_dirs(self):
        root = os.path.join(self.home, ".skill-batch-txn")
        return sorted(os.listdir(root)) if os.path.isdir(root) else []

    def test_timeout_interrupt_midbatch_rolls_the_whole_batch_back(self):
        """The deadline sets this worker's interrupt bit; the batch must notice it at the
        next op boundary and undo what it already wrote instead of writing on after the
        caller stopped listening."""
        from unittest.mock import patch as _patch

        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        real = self.smt._skill_manage_from
        calls = {"n": 0}

        def dispatch(payload, **kw):
            out = real(payload, **kw)
            calls["n"] += 1
            if calls["n"] == 1:  # the 420 s deadline fires right after op 0 landed
                self.interrupt.set_interrupt(True)
            return out

        with _patch.object(self.smt, "_skill_manage_from", side_effect=dispatch):
            r = self._call("probe", [
                {"action": "patch", "old_string": "Step 1.", "new_string": "Step ONE."},
                {"action": "write_file", "file_path": "references/late.md", "file_content": "late"},
            ])
        self.assertFalse(r["success"], r)
        self.assertEqual(r["completed_before_failure"], 1)
        self.assertIn("Step 1.", self._skill_md())          # op 0 undone
        self.assertNotIn("Step ONE.", self._skill_md())
        self.assertFalse(os.path.exists(
            os.path.join(self.home, "skills", "probe", "references", "late.md")))
        self.assertEqual(self._txn_dirs(), [])              # and nothing left behind

    def test_killed_batch_leaves_a_journal_and_the_next_entry_undoes_it(self):
        """No cancellation signal reaches a killed process: the journal + snapshots on disk
        are the only record, and the next skill_manage entry must replay them."""
        from unittest.mock import patch as _patch

        import tools.skill_manager_batch as smb

        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        real = self.smt._skill_manage_from

        class _Killed(BaseException):
            pass

        def dispatch(payload, **kw):
            if payload["action"] == "write_file":  # process died between op 0 and op 1
                raise _Killed("killed mid-batch")
            return real(payload, **kw)

        with _patch.object(self.smt, "_skill_manage_from", side_effect=dispatch), \
             _patch.object(smb, "_rollback", side_effect=RuntimeError("killed before rollback")):
            with self.assertRaises(_Killed):
                self._call("probe", [
                    {"action": "patch", "old_string": "Step 1.", "new_string": "Step ONE."},
                    {"action": "write_file", "file_path": "references/late.md", "file_content": "late"},
                ])
        # torn on disk — op 0 landed and no rollback ran — but the journal survived it
        self.assertIn("Step ONE.", self._skill_md())
        txns = self._txn_dirs()
        self.assertEqual(len(txns), 1, txns)
        self.assertTrue(os.path.exists(
            os.path.join(self.home, ".skill-batch-txn", txns[0], "journal.json")))
        # the next entry repairs the torn batch before applying its own op
        r = self._call("probe", [{"action": "patch", "old_string": "Step 1.",
                                  "new_string": "Step 1 (recovered)."}])
        self.assertTrue(r["success"], r)
        self.assertTrue(r.get("recovered_interrupted_batches"), r)
        self.assertIn("Step 1 (recovered).", self._skill_md())
        self.assertNotIn("Step ONE.", self._skill_md())
        self.assertEqual(self._txn_dirs(), [])

    def test_dead_owner_journal_is_recovered_on_the_next_entry(self):
        """A batch whose owner process is gone (different pid, not alive) is reclaimed
        immediately — no waiting for a staleness window."""
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        skill_dir = os.path.join(self.home, "skills", "probe")
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as fh:  # op 0 landed, then the kill
            fh.write(SK.format(n="probe").replace("Step 1.", "Step ONE."))
        snap = os.path.join(self.home, ".skill-batch-txn", "999999999-deadbeef", "snap", "probe")
        os.makedirs(snap)
        with open(os.path.join(snap, "SKILL.md"), "w") as fh:
            fh.write(SK.format(n="probe"))
        with open(os.path.join(self.home, ".skill-batch-txn", "999999999-deadbeef",
                               "journal.json"), "w") as fh:
            json.dump({"batch_id": "999999999-deadbeef", "pid": 999999999, "started_at": 0,
                       "skills": [{"name": "probe", "pre_dir": skill_dir, "snap": snap}]}, fh)
        r = self._call("probe", [{"action": "patch", "old_string": "Step 1.",
                                  "new_string": "Step 1 (after repair)."}])
        self.assertTrue(r["success"], r)
        self.assertTrue(r.get("recovered_interrupted_batches"), r)
        content = self._skill_md()
        self.assertIn("Step 1 (after repair).", content)
        self.assertNotIn("Step ONE.", content)
        self.assertEqual(self._txn_dirs(), [])

    def test_dead_owner_recovery_removes_a_batch_created_skill(self):
        """The other half of the journal: a skill the interrupted batch CREATED has no
        pre-batch state to restore, so the repair is removing it — and a staging dir whose
        owner died before its journal write (nothing applied yet) is swept, not rolled back."""
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        torn = os.path.join(self.home, "skills", "torn")
        os.makedirs(torn)
        with open(os.path.join(torn, "SKILL.md"), "w") as fh:
            fh.write(SK.format(n="torn"))
        root = os.path.join(self.home, ".skill-batch-txn")
        os.makedirs(os.path.join(root, "999999999-badc0de"))
        with open(os.path.join(root, "999999999-badc0de", "journal.json"), "w") as fh:
            json.dump({"batch_id": "999999999-badc0de", "pid": 999999999, "started_at": 0,
                       "skills": [{"name": "torn", "pre_dir": None, "snap": None}]}, fh)
        os.makedirs(os.path.join(root, "999999999-nostage", "snap"))  # died before the journal
        r = self._call("probe2", [{"action": "create", "content": SK.format(n="probe2")}])
        self.assertTrue(r["success"], r)
        self.assertTrue(r.get("recovered_interrupted_batches"), r)
        self.assertFalse(os.path.exists(torn), "the interrupted batch's created skill must go")
        self.assertEqual(self._txn_dirs(), [])
        self.assertIn("Step 1.", open(os.path.join(
            self.home, "skills", "probe2", "SKILL.md")).read())


if __name__ == "__main__":
    unittest.main()
