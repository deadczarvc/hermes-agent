"""TOCTOU-патч _ensure_private: гонка параллельного старта 2 бэкендов.
Урок 2026-09-16: WinError 183 -> is_dir() -> stat(5) — при НАЛИЧИИ каталога.
Негативный контроль: когда目录а нет и создать нельзя — вызов не должен маскировать проблему."""
import pytest
from hermes_cli.observability.shared_metrics import _ensure_private


def test_ensure_private_idempotent_when_dir_exists(tmp_path):
    p = tmp_path / "outbox"
    p.mkdir()
    _ensure_private(p, 0o700)
    _ensure_private(p, 0o700)
    assert p.is_dir()


def test_ensure_private_raises_when_dir_is_file(tmp_path):
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    with pytest.raises(OSError):
        _ensure_private(f, 0o700)
    assert f.read_text() == "x"


def test_ensure_private_creates_missing_dir(tmp_path):
    p = tmp_path / "fresh"
    _ensure_private(p, 0o700)
    assert p.is_dir()


def test_ensure_private_swallows_stat_denied(tmp_path, monkeypatch):
    """WinError 5 on stat: dir exists but unreadable - must not raise.

    Zhivoy incident 2026-09-17: elevated plugin install created dirs owned by
    BUILTIN-Admins group; non-elevated process got Access Denied on
    mkdir/stat. The store must not hard-fail init."""
    from pathlib import Path

    p = tmp_path / "denied"
    p.mkdir()

    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Access denied", str(self))

    monkeypatch.setattr(Path, "mkdir", denied)
    monkeypatch.setattr(Path, "is_dir", denied)
    _ensure_private(p, 0o700)  # must not raise
