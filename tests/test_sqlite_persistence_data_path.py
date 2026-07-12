"""Regression test for PR D: SQLitePersistence should honour DATA_PATH.

Before PR D, ``SQLitePersistence.__init__`` hard-coded ``data_dir='data'``.
On Render the persistent-disk mount lives at ``/opt/render/project/src/data``
(same string, coincidentally), but any deployment that relocates DATA_PATH
would silently write to the wrong file. This test pins the new behaviour.
"""
from __future__ import annotations

from pathlib import Path

from components.sqlite_persistence import SQLitePersistence


def test_data_path_env_var_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    sp = SQLitePersistence()
    assert sp.data_dir == tmp_path
    assert sp.db_path == tmp_path / "dmai_knowledge.db"
    assert sp.db_path.parent.exists()


def test_explicit_data_dir_still_wins(tmp_path, monkeypatch):
    """Explicit data_dir arg must override the env var — needed so unit tests
    can safely point at a tmpdir even when DATA_PATH is set elsewhere."""
    other = tmp_path / "other"
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "envpath"))
    sp = SQLitePersistence(data_dir=str(other))
    assert sp.data_dir == other
    assert sp.db_path == other / "dmai_knowledge.db"


def test_default_falls_back_to_data_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    sp = SQLitePersistence()
    # Default is 'data' relative to cwd — matches every legacy caller.
    assert sp.data_dir == Path("data")
