"""Tests for SQLite source auto-discovery (PR R.1).

The migration endpoint used to hardcode <data_dir>/dmai_knowledge.db, but the
live DB is <DATA_PATH>/dmai.db (see components/sqlite_storage.py). These tests
lock the discovery order (dmai.db first, then a glob scan for whatever holds
admin_api_keys) and the list-sqlite-sources diagnostic endpoint.

DATA_PATH is pointed at a temp dir *before* importing the app so boot side
effects stay isolated. Env overrides go through monkeypatch only.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="migrate_disc_")
os.environ["DATA_PATH"] = _TMP

import dmai_core_complete  # noqa: E402
from dmai_core_complete import _discover_sqlite_source, app  # noqa: E402

_MASTER_PW = "test-master-pw"
_AUTH = {"X-Master-Password": _MASTER_PW}


def _make_db(path, with_admin_keys, rows=None):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE misc (id INTEGER PRIMARY KEY, v TEXT)")
    if with_admin_keys:
        conn.execute(
            "CREATE TABLE admin_api_keys ("
            "provider_id TEXT PRIMARY KEY, api_key TEXT, updated_at TIMESTAMP)"
        )
        for pid, key in (rows or {}).items():
            conn.execute(
                "INSERT INTO admin_api_keys (provider_id, api_key, updated_at) "
                "VALUES (?,?,?)", (pid, key, "2026-07-14"))
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_discover_finds_dmai_db_first(monkeypatch, tmp_path):
    d = str(tmp_path)
    monkeypatch.setenv("DATA_PATH", d)
    monkeypatch.delenv("DATA_DIR", raising=False)
    # Both present and both contain admin_api_keys — dmai.db must win.
    _make_db(os.path.join(d, "dmai.db"), True, {"groq": "sk-g"})
    _make_db(os.path.join(d, "dmai_knowledge.db"), True, {"openai": "sk-o"})

    found = _discover_sqlite_source()
    assert found == os.path.join(d, "dmai.db")


def test_discover_falls_back_to_scan(monkeypatch, tmp_path):
    d = str(tmp_path)
    monkeypatch.setenv("DATA_PATH", d)
    monkeypatch.delenv("DATA_DIR", raising=False)
    # No dmai.db / dmai_knowledge.db — only a random-named DB has the table.
    _make_db(os.path.join(d, "empty.db"), False)
    _make_db(os.path.join(d, "legacy_state.db"), True, {"cohere": "sk-c"})

    found = _discover_sqlite_source()
    assert found == os.path.join(d, "legacy_state.db")


def test_discover_returns_none_when_absent(monkeypatch, tmp_path):
    d = str(tmp_path)
    monkeypatch.setenv("DATA_PATH", d)
    monkeypatch.delenv("DATA_DIR", raising=False)
    _make_db(os.path.join(d, "nothing.db"), False)
    assert _discover_sqlite_source() is None


def test_list_sources_endpoint(client, monkeypatch, tmp_path):
    d = str(tmp_path)
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.setenv("DATA_PATH", d)
    monkeypatch.delenv("DATA_DIR", raising=False)
    _make_db(os.path.join(d, "dmai.db"), True, {"groq": "sk-g", "openai": "sk-o"})
    _make_db(os.path.join(d, "dmai_knowledge.db"), False)

    resp = client.get("/api/admin/list-sqlite-sources", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["data_dir"] == d
    by_path = {os.path.basename(s["path"]): s for s in body["sources"]}
    assert by_path["dmai.db"]["has_admin_api_keys"] is True
    assert by_path["dmai.db"]["admin_api_keys_rows"] == 2
    assert "admin_api_keys" in by_path["dmai.db"]["tables"]
    assert by_path["dmai_knowledge.db"]["has_admin_api_keys"] is False
    assert by_path["dmai_knowledge.db"]["admin_api_keys_rows"] == 0


def test_list_sources_requires_auth(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    resp = client.get("/api/admin/list-sqlite-sources")
    assert resp.status_code == 401
