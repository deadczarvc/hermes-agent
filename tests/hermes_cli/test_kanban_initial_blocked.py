"""Creation-time blocking must survive dispatch until explicit unblock."""
from pathlib import Path

import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.profiles import _get_profiles_root


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(home / 'kanban.db'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    (_get_profiles_root() / 'worker-coder-luna').mkdir(parents=True)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn, home


def test_initial_block_fences_dispatch_until_explicit_unblock(isolated):
    conn, home = isolated
    tid = kb.create_task(conn, title='creation fence', assignee='worker-coder-luna',
                         initial_status='blocked', workspace_kind='dir', workspace_path=str(home))
    spawned = []

    def spawn(*args, **kwargs):
        spawned.append(args)
        return 424242

    for _ in range(2):
        result = dispatch.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
        assert result.promoted == 0 and not result.spawned and not spawned
        assert kb.get_task(conn, tid).status == 'blocked'
    assert kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == 'ready'
    result = dispatch.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
    assert len(result.spawned) == len(spawned) == 1


@pytest.mark.parametrize('payload', [None, '{}', '[]', 'broken', '{"status":"ready"}'])
def test_legacy_or_malformed_creation_keeps_recovery(isolated, payload):
    conn, _ = isolated
    tid = kb.create_task(conn, title='legacy recovery')
    conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
    conn.execute("UPDATE task_events SET payload=? WHERE task_id=? AND kind='created'", (payload, tid))
    conn.commit()
    assert kb.recompute_ready(conn) == 1
