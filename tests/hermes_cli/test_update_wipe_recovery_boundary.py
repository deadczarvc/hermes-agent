"""Exercise the updater boundary with real surgical recovery in a copied checkout."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hermes_cli import main as hermes_main
from hermes_cli import update_cmd, update_receipt

# Removing or moving recovery before the final mutation must break the H1 behavior checks.

def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_h1(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms import _shared

    loader = _load(root / "gateway/config_loader.py", "boundary_loader")
    assert "api_server" not in loader.merge_platform_sections(
        {}, {"api_server": {"max_concurrent_runs": 16}}, {}
    ), "tuning-only config enabled api_server before recovery"
    assert "api_server" in loader.merge_platform_sections(
        {}, {"api_server": {"enabled": True}}, {}
    )
    env = _load(root / "gateway/config_env.py", "boundary_env")
    monkeypatch.setattr(_shared, "profile_scoped", lambda: True)
    monkeypatch.setattr(env, "getenv", lambda name: "fixture" if name == "API_SERVER_KEY" else "")
    monkeypatch.setattr(env, "_has_usable_api_server_key", lambda value: True)
    config = GatewayConfig()
    env._api_server(config)
    assert Platform.API_SERVER not in config.platforms, "profile-scoped env enabled api_server"
    config.platforms[Platform.API_SERVER] = PlatformConfig(enabled=True)
    env._api_server(config)
    assert config.platforms[Platform.API_SERVER].enabled


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    source = Path(update_cmd.__file__).resolve().parents[1]
    script = source.parent / "skills/hermes-internal/update-wipe-recovery/scripts/recover.py"
    assert script.is_file(), "Install the existing update-wipe-recovery skill before this integration test"
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    recovery = _load(script, "boundary_recovery")
    root = tmp_path / "checkout with spaces"
    originals = {}
    for relative in (
        "agent/prompt_builder.py", "agent/skill_utils.py", "web/src/lib/chat-title.test.ts",
        "web/stryker.config.mjs", "gateway/config_loader.py", "gateway/config_env.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (source / relative).read_text(encoding="utf-8")
        if relative == "gateway/config_loader.py":
            text += "\n# Unrelated local edit must survive surgical recovery.\n"
        target.write_text(text, encoding="utf-8", newline="\n")
        originals[relative] = text
    state = SimpleNamespace(
        root=root, script=script, originals=originals, events=[], results=[], fault="",
        mutation="pull", fork=False, wipe=True, require_h1=True, complete=True, exit_markers=[],
    )

    def wipe() -> None:
        if not state.wipe:
            return
        loader = originals["gateway/config_loader.py"].replace(recovery.GWL_LOADER_CONSTANT, "")
        loader = loader.replace(recovery.GWL_LOADER_SWEEP_FIXED, recovery.GWL_LOADER_SWEEP)
        if state.fault == "unknown-anchor":
            loader = "UPSTREAM_LAYOUT_CHANGED = True\n"
        (root / "gateway/config_loader.py").write_text(loader, encoding="utf-8", newline="\n")
        env = originals["gateway/config_env.py"].replace(recovery.GWL_ENV_IMPORT, "")
        env = env.replace(recovery.GWL_ENV_GUARD, "")
        (root / "gateway/config_env.py").write_text(env, encoding="utf-8", newline="\n")

    def git_run(git_cmd, args, *a, **kw):
        if args == ["merge", "--ff-only", "origin/main"]:
            if state.mutation == "reset":
                return SimpleNamespace(returncode=1)
            state.events.append("pull")
            wipe()
        elif args == ["reset", "--hard", "origin/main"]:
            state.events.append("reset")
            wipe()
        elif args not in (["fetch", "origin", "main"], ["rev-parse", "HEAD"],
                          ["branch", "--show-current"], ["merge-base", "HEAD", "origin/main"]):
            raise AssertionError(f"Unexpected Git operation: {args}")
        return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

    def upstream(*a, **kw) -> None:
        state.events.append("upstream")
        wipe()

    def consume(name: str) -> None:
        state.events.append(name)
        if state.require_h1:
            _assert_h1(root, monkeypatch)

    def begin(*a):
        update_receipt.begin_update_receipt()
        return {}

    def restart(*a):
        consume("restart")
        return SimpleNamespace(incomplete=False)

    def verify(*a, **kw) -> None:
        state.events.append("verify")
        assert kw["update_complete"] is state.complete
        update_receipt.finalize_update_receipt("success" if kw["update_complete"] else "partial")

    real_run = subprocess.run

    def recover_run(cmd, **kw):
        assert cmd == [sys.executable, str(script), "--repo", str(root), "--apply", "--json"]
        assert kw["timeout"] == 30
        assert kw["capture_output"] is True
        assert kw["stdin"] == subprocess.DEVNULL
        assert kw.get("shell", False) is False
        state.events.append("recovery")
        if state.fault == "timeout":
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        if state.fault == "spawn":
            raise OSError("fixture spawn failure")
        if state.fault in {"malformed", "nonzero"}:
            return subprocess.CompletedProcess(
                cmd, 9 if state.fault == "nonzero" else 0,
                stdout=json.dumps({"changed": 0, "errors": [], "missing": []}) if state.fault == "nonzero" else "not-json",
                stderr="fixture stderr must not be copied to receipts",
            )
        result = real_run(cmd, **kw)
        state.results.append(json.loads(result.stdout))
        return result

    opts = SimpleNamespace(
        assume_yes=True, gw_input_fn=None, switch_branch=False, discard_local_changes=False,
        keep_stash=False, active_lazy_features=set(), active_tool_dependencies={}, pre_update_version="fixture",
    )
    plan = update_cmd._CheckoutPlan("fixture-stash", 1, False, False, False, None, False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.delenv("UPDATE_WIPE_RECOVERY_DISABLE", raising=False)
    monkeypatch.setenv("UPDATE_WIPE_RECOVERY_SCRIPT", str(script))
    monkeypatch.setenv("UPDATE_WIPE_RECOVERY_REPO", str(tmp_path / "wrong-checkout"))
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(update_receipt, "_current", None)
    monkeypatch.setattr(update_receipt, "_code_identity", lambda **kw: {})
    monkeypatch.setattr(update_receipt, "_receipt_dir", lambda: tmp_path / "receipts")
    for name in ("_run_pre_update_backup", "_pause_windows_gateways_for_update",
                 "_warn_orphaned_update_autostashes", "_build_web_ui"):
        monkeypatch.setattr(hermes_main, name, lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", lambda *a, **kw: state.events.append("stash"))
    monkeypatch.setattr(hermes_main, "_resolve_update_branch", lambda *a: "main")
    monkeypatch.setattr(hermes_main, "_sync_with_upstream_if_needed", upstream)
    for name in ("_invalidate_update_cache", "_write_fleet_restart_pending_marker",
                 "_sweep_bytecode_after_update", "_clear_windows_venv_holders_or_exit",
                 "_resume_windows_gateways_and_merge_outcome"):
        monkeypatch.setattr(update_cmd, name, lambda *a, **kw: None)
    monkeypatch.setattr(update_cmd, "_desktop_app_present", lambda *a: False)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *a: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", begin)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], state.fork))
    monkeypatch.setattr(update_cmd, "_current_branch_name", lambda *a, **kw: "main")
    monkeypatch.setattr(update_cmd, "_prepare_checkout_for_update", lambda *a, **kw: plan)
    monkeypatch.setattr(update_cmd, "_verify_head_after_pull", lambda *a, **kw: "fixture-after")
    monkeypatch.setattr(update_cmd, "_git_run", git_run)
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda *a: "")
    monkeypatch.setattr(update_cmd, "_sync_python_dependencies_after_pull", lambda *a, **kw: consume("deps"))
    monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(update_cmd, "_rebuild_desktop_after_update", lambda *a, **kw: True)
    monkeypatch.setattr(update_cmd, "_run_post_update_maintenance", lambda **kw: state.complete)
    monkeypatch.setattr(update_cmd, "_write_gateway_update_exit_code", state.exit_markers.append)
    monkeypatch.setattr(update_cmd, "_restart_gateway_fleet_after_update", restart)
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", verify)
    monkeypatch.setattr(subprocess, "run", recover_run)
    state.run = lambda: update_cmd._cmd_update_impl(SimpleNamespace(force_venv=True), gateway_mode=True)
    state.receipt = lambda: json.loads((tmp_path / "receipts/latest.json").read_text(encoding="utf-8"))
    return state


@pytest.mark.parametrize("mutation", ["pull", "reset"])
@pytest.mark.parametrize("fork", [False, True])
def test_update_repairs_h1_before_consumers(
    sandbox: SimpleNamespace, mutation: str, fork: bool,
) -> None:
    sandbox.mutation, sandbox.fork = mutation, fork
    sandbox.run()
    assert sandbox.events == [mutation, "stash"] + (["upstream"] if fork else []) + [
        "recovery", "deps", "restart", "verify",
    ]
    assert sandbox.results[-1]["changed"] > 0
    for relative, original in sandbox.originals.items():
        assert (sandbox.root / relative).read_text(encoding="utf-8") == original
    receipt = sandbox.receipt()
    step = next(s for s in receipt["steps"] if s["name"] == "update_wipe_recovery")
    assert step["ok"] is True
    assert receipt["outcome"] == "success"
    sandbox.wipe = False
    sandbox.run()
    assert sandbox.results[-1] == {"changed": 0, "missing": [], "errors": []}
    assert sandbox.exit_markers == [True, True]


@pytest.mark.parametrize("fault", ["unknown-anchor", "nonzero", "malformed", "timeout", "spawn", "disabled", "missing"])
@pytest.mark.parametrize("complete", [False, True])
def test_recovery_advisory_preserves_update_outcome(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, fault: str, complete: bool,
) -> None:
    sandbox.fault, sandbox.complete, sandbox.require_h1 = fault, complete, False
    if fault == "disabled":
        monkeypatch.setenv("UPDATE_WIPE_RECOVERY_DISABLE", "1")
    if fault == "missing":
        monkeypatch.setenv("UPDATE_WIPE_RECOVERY_SCRIPT", str(sandbox.root / "missing-recover.py"))
    sandbox.run()
    receipt = sandbox.receipt()
    step = next(s for s in receipt["steps"] if s["name"] == "update_wipe_recovery")
    assert step["ok"] is (fault in {"disabled", "missing"})
    assert step["detail"]
    assert "fixture stderr" not in step["detail"]
    if fault in {"disabled", "missing"}:
        assert "skipped" in step["detail"]
        assert "recovery" not in sandbox.events
    else:
        assert sandbox.events.index("recovery") < sandbox.events.index("deps")
    if fault == "nonzero":
        assert "rc=9" in step["detail"]
    if fault == "timeout":
        assert "TimeoutExpired" in step["detail"]
    if fault == "spawn":
        assert "OSError" in step["detail"]
    assert sandbox.events[-3:] == ["deps", "restart", "verify"]
    assert sandbox.exit_markers == [complete]
    assert receipt["outcome"] == ("success" if complete else "partial")
    if fault == "unknown-anchor":
        assert "rc=2" in step["detail"]
        assert sandbox.results[-1]["errors"]
        assert sandbox.results[-1]["changed"] == 0
        assert (sandbox.root / "gateway/config_loader.py").read_text(encoding="utf-8") == "UPSTREAM_LAYOUT_CHANGED = True\n"
