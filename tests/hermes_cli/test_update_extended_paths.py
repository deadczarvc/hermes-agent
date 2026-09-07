"""Focused, isolated coverage for remaining ``hermes update`` branch seams.

These tests exercise update control-flow with collaborators patched; they never
fetch, mutate a checkout, restart a gateway, or inspect live processes.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import update_cmd


def test_windows_git_failure_routes_to_zip_without_live_update():
    """A Windows git failure selects the ZIP recovery path exactly once."""
    error = subprocess.CalledProcessError(1, ["git", "fetch", "origin", "main"])
    args = SimpleNamespace()

    with patch.object(update_cmd, "_should_zip_fallback_on_update_error", return_value=True), patch.object(
        update_cmd, "_format_update_failure_stage", return_value="Git update failed"
    ), patch.object(update_cmd, "_update_via_zip", return_value=True) as zip_update:
        update_cmd._handle_update_called_process_error(
            error, args, gateway_mode=False, had_desktop_app_before_update=False
        )

    zip_update.assert_called_once_with(args, had_desktop_app_before_update=False)


def test_current_checkout_repairs_then_applies_pending_fleet_catchup():
    """The already-current path repairs first and still catches up a prior restart."""
    plan = SimpleNamespace(
        auto_stash_ref=None,
        parked_branch_switched=False,
        switch_block_reason="",
        prompt_for_restore=False,
        upstream_checked=True,
    )
    events: list[str] = []

    with patch.object(update_cmd, "_invalidate_update_cache", side_effect=lambda: events.append("cache")), patch.object(
        update_cmd, "_repair_current_checkout", side_effect=lambda **_kwargs: events.append("repair") or True
    ), patch.object(
        update_cmd, "_resume_windows_gateways_after_update", side_effect=lambda _resume: events.append("resume")
    ), patch.object(
        update_cmd, "_apply_pending_fleet_restart_catchup", side_effect=lambda: events.append("catchup")
    ):
        update_cmd._finish_already_up_to_date(
            ["git"], "main", "main", plan, assume_yes=True, gateway_mode=False,
            gw_input_fn=None, pre_update_snapshot_id=None, desktop_dir=None,
            had_desktop_app_before_update=False, active_lazy_features=[],
            active_tool_dependencies=[], _windows_gateway_resume=None,
        )

    assert events == ["cache", "repair", "resume", "catchup"]


def test_rollback_reports_success_and_exits_after_bad_pulled_syntax(capsys, monkeypatch, tmp_path):
    """A bad pulled tree resets to the captured SHA rather than continuing."""
    monkeypatch.setattr(update_cmd, "_validate_critical_files_syntax", lambda _root: (
        False, tmp_path / "hermes_cli" / "config.py", "SyntaxError: invalid syntax"
    ))
    monkeypatch.setattr(update_cmd, "_git_run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stderr=""
    ))

    try:
        update_cmd._rollback_if_pulled_syntax_error(["git"], "0123456789abcdef")
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("syntax rollback did not exit")

    assert "Rolling back to 0123456789" in capsys.readouterr().out


def test_aborted_restart_is_incomplete_for_unknown_survivor_probe():
    """An unreadable post-abort gateway probe must fail closed."""
    assert update_cmd._restart_phase_failure_is_incomplete(None, []) is True
