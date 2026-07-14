"""Tests for POST /api/admin/migrate-sqlite-to-postgres (PR R).

The endpoint lifts rows from the on-disk SQLite DB into the attached Postgres
during the backend cutover. It must be auth-gated, refuse to run when Postgres
is not the active backend, upsert idempotently (ON CONFLICT), and never mutate
the SQLite source. These tests lock that contract.

DATA_PATH is pointed at a temp dir *before* importing the app so boot side
effects stay isolated. Env/component overrides go through monkeypatch only, so
nothing leaks into the wider pytest session (the PR-N leak class of bug).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="migrate_pg_")
os.environ["DATA_PATH"] = _TMP

import dmai_core_complete  # noqa: E402
from dmai_core_complete import app  # noqa: E402

_MASTER_PW = "test-master-pw"
_AUTH = {"X-Master-Password": _MASTER_PW}


class _FakePG:
    """In-memory stand-in for PGStorage.

    Rows are keyed by their primary key (first INSERT value). _exec recognises
    the exact statements the migration endpoint emits and nothing more.
    """

    def __init__(self):
        self.rows: dict = {}          # pk -> full value tuple
        self.columns = [
            {"column_name": "provider_id"},
            {"column_name": "api_key"},
            {"column_name": "updated_at"},
        ]

    def is_available(self) -> bool:
        return True

    # Hydration reads this; return None so no real env vars get mutated.
    def get_api_key(self, provider_id):
        return None

    def _exec(self, sql, params=(), fetch="none"):
        s = sql.strip()
        up = s.upper()
        if "INFORMATION_SCHEMA.COLUMNS" in up:
            return list(self.columns)
        if up.startswith("CREATE TABLE"):
            return None
        if "COUNT(*)" in up:
            return {"c": len(self.rows)}
        if up.startswith("SELECT"):
            # Existing-PK probe: SELECT "provider_id" FROM admin_api_keys
            return [{"provider_id": pk} for pk in self.rows]
        if up.startswith("INSERT"):
            self.rows[params[0]] = tuple(params)
            return None
        return None


class _FakeActivator:
    def scan_and_activate(self):
        return {"activated": ["groq"], "invalid": [], "pending": ["openai"],
                "total_active": 1, "timestamp": "2026-07-14T00:00:00Z"}


def _make_sqlite_source(dirpath, keys):
    """Create <dir>/dmai_knowledge.db with an admin_api_keys table + rows."""
    path = os.path.join(dirpath, "dmai_knowledge.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE admin_api_keys ("
        "provider_id TEXT PRIMARY KEY, api_key TEXT, updated_at TIMESTAMP)"
    )
    conn.executemany(
        "INSERT INTO admin_api_keys (provider_id, api_key, updated_at) VALUES (?,?,?)",
        [(pid, key, "2026-07-14") for pid, key in keys.items()],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_endpoint_requires_auth(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    resp = client.post("/api/admin/migrate-sqlite-to-postgres")
    assert resp.status_code == 401


def test_migration_when_postgres_not_active(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    resp = client.post("/api/admin/migrate-sqlite-to-postgres", headers=_AUTH)
    assert resp.status_code == 400
    assert "postgres not active" in resp.get_json()["error"]


def test_migration_happy_path(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    src_dir = str(tmp_path)
    _make_sqlite_source(src_dir, {"groq": "sk-g", "openai": "sk-o", "cohere": "sk-c"})
    monkeypatch.setenv("DATA_PATH", src_dir)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setitem(dmai_core_complete.components, "db_storage", _FakePG())
    monkeypatch.setitem(dmai_core_complete.components, "api_activator", _FakeActivator())

    resp = client.post("/api/admin/migrate-sqlite-to-postgres", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["backend"] == "postgres"
    t = body["tables"]["admin_api_keys"]
    assert t["sqlite_rows_read"] == 3
    assert t["inserted"] == 3
    assert t["updated"] == 0
    assert t["errors"] == []
    assert t["pg_rows_after"] == 3
    assert body["hydration"]["db_ready"] is True
    assert body["post_scan"]["active"] == ["groq"]


def test_migration_idempotent(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    src_dir = str(tmp_path)
    _make_sqlite_source(src_dir, {"groq": "sk-g", "openai": "sk-o", "cohere": "sk-c"})
    monkeypatch.setenv("DATA_PATH", src_dir)
    monkeypatch.delenv("DATA_DIR", raising=False)
    fake = _FakePG()
    monkeypatch.setitem(dmai_core_complete.components, "db_storage", fake)
    monkeypatch.setitem(dmai_core_complete.components, "api_activator", _FakeActivator())

    first = client.post("/api/admin/migrate-sqlite-to-postgres", headers=_AUTH).get_json()
    assert first["tables"]["admin_api_keys"]["inserted"] == 3
    assert first["tables"]["admin_api_keys"]["updated"] == 0

    # Second run: rows already present -> everything takes the UPDATE path.
    second = client.post("/api/admin/migrate-sqlite-to-postgres", headers=_AUTH).get_json()
    t = second["tables"]["admin_api_keys"]
    assert t["inserted"] == 0
    assert t["updated"] == 3
    assert t["pg_rows_after"] == 3


def test_migration_sqlite_source_missing(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("DATA_PATH", str(tmp_path))  # empty dir, no db file
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setitem(dmai_core_complete.components, "db_storage", _FakePG())
    resp = client.post("/api/admin/migrate-sqlite-to-postgres", headers=_AUTH)
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "no sqlite source found"
