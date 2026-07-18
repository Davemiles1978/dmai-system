"""PR CCC-1a: /api/external/* auth + rate-limit regression tests.

Uses an in-memory SQLite so we exercise the real _require_external_key
decorator against a real DB path (no mocks). Each test provisions the
CCC-1a schema, seeds one or more api_keys rows, and asserts the
decorator returns the right (status_code, error) pair.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import time

import pytest

from flask import Flask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Provision a fresh SQLite DB with the CCC-1a schema and point the
    external_api module at it via DMAI_DB_PATH."""
    db_path = tmp_path / "test_ccc1a.db"
    monkeypatch.setenv("DMAI_DB_PATH", str(db_path))
    # Ensure we do NOT accidentally hit prod
    monkeypatch.delenv("DATABASE_URL", raising=False)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE api_keys (
            key                 TEXT PRIMARY KEY,
            service             TEXT,
            source              TEXT,
            validated           INTEGER DEFAULT 0,
            created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
            last_used           TEXT,
            key_hash            TEXT,
            scope               TEXT DEFAULT '',
            rate_limit_per_min  INTEGER DEFAULT 60,
            revoked             INTEGER DEFAULT 0,
            label               TEXT
        );
        CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
        CREATE TABLE external_api_calls (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash     TEXT NOT NULL,
            service      TEXT,
            endpoint     TEXT NOT NULL,
            status_code  INTEGER,
            ts           TEXT DEFAULT CURRENT_TIMESTAMP,
            duration_ms  INTEGER
        );
        CREATE INDEX idx_ext_calls_key_ts ON external_api_calls(key_hash, ts DESC);
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def flask_client(temp_db):
    """Build a minimal Flask app with the external_api blueprint mounted."""
    from components.external_api import external_api_bp
    app = Flask(__name__)
    app.register_blueprint(external_api_bp)
    return app.test_client()


def _seed_key(db_path, plaintext, *, service="test", scope="",
              rate_limit=60, validated=1, revoked=0, label="test-label"):
    from components.external_api.auth import hash_key
    key_hash = hash_key(plaintext)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT INTO api_keys (key, service, source, validated, key_hash,
                                 scope, rate_limit_per_min, revoked, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (plaintext, service, "ccc1a-test", validated, key_hash,
         scope, rate_limit, revoked, label),
    )
    conn.commit()
    conn.close()
    return key_hash


# ---------------------------------------------------------------------------
# Ping - unauthenticated liveness
# ---------------------------------------------------------------------------
def test_ping_is_unauthenticated(flask_client):
    r = flask_client.get("/api/external/ping")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["service"] == "dmai-external-api"


# ---------------------------------------------------------------------------
# 401 family
# ---------------------------------------------------------------------------
def test_status_401_missing_key(flask_client):
    r = flask_client.get("/api/external/status")
    assert r.status_code == 401
    assert r.get_json()["error"] == "missing_key"


def test_status_401_malformed_key(flask_client):
    r = flask_client.get(
        "/api/external/status",
        headers={"X-DMAI-Api-Key": "not-a-dmai-key"},
    )
    assert r.status_code == 401
    assert r.get_json()["error"] == "malformed_key"


def test_status_401_unknown_key(flask_client):
    r = flask_client.get(
        "/api/external/status",
        headers={"X-DMAI-Api-Key": "dmai_ext_" + "a" * 32},
    )
    assert r.status_code == 401
    assert r.get_json()["error"] == "unknown_key"


def test_status_401_unvalidated_key(flask_client, temp_db):
    _seed_key(temp_db, "dmai_ext_" + "b" * 32, validated=0)
    r = flask_client.get(
        "/api/external/status",
        headers={"X-DMAI-Api-Key": "dmai_ext_" + "b" * 32},
    )
    assert r.status_code == 401
    assert r.get_json()["error"] == "unvalidated_key"


# ---------------------------------------------------------------------------
# 403 family
# ---------------------------------------------------------------------------
def test_status_403_revoked_key(flask_client, temp_db):
    _seed_key(temp_db, "dmai_ext_" + "c" * 32, validated=1, revoked=1)
    r = flask_client.get(
        "/api/external/status",
        headers={"X-DMAI-Api-Key": "dmai_ext_" + "c" * 32},
    )
    assert r.status_code == 403
    assert r.get_json()["error"] == "revoked_key"


def test_scope_grants_helper():
    from components.external_api.auth import scope_grants
    assert scope_grants("insight:write signal:read", "insight:write") is True
    assert scope_grants("insight:write", "signal:write") is False
    assert scope_grants("", "insight:write") is False
    assert scope_grants("insight:write", "") is True  # empty required = any key


# ---------------------------------------------------------------------------
# 200 happy path
# ---------------------------------------------------------------------------
def test_status_200_valid_key_returns_metadata(flask_client, temp_db):
    key = "dmai_ext_" + "d" * 32
    _seed_key(
        temp_db, key,
        service="test-partner",
        scope="insight:write signal:read",
        rate_limit=99,
        label="Test Partner Key",
    )
    r = flask_client.get(
        "/api/external/status",
        headers={"X-DMAI-Api-Key": key},
    )
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["ok"] is True
    assert body["service"] == "test-partner"
    assert body["label"] == "Test Partner Key"
    assert body["scope"] == ["insight:write", "signal:read"]
    assert body["rate_limit_per_min"] == 99
    # Plaintext key and key_hash must NEVER leak to caller
    assert "key" not in body
    assert "key_hash" not in body


def test_status_200_records_call_audit(flask_client, temp_db):
    key = "dmai_ext_" + "e" * 32
    _seed_key(temp_db, key, service="audit-test", rate_limit=60)
    flask_client.get(
        "/api/external/status",
        headers={"X-DMAI-Api-Key": key},
    )
    conn = sqlite3.connect(str(temp_db))
    row = conn.execute(
        "SELECT service, endpoint, status_code FROM external_api_calls"
    ).fetchone()
    conn.close()
    assert row == ("audit-test", "/api/external/status", 200)


# ---------------------------------------------------------------------------
# 429 rate limiting
# ---------------------------------------------------------------------------
def test_status_429_when_rate_limit_exceeded(flask_client, temp_db):
    key = "dmai_ext_" + "f" * 32
    key_hash = _seed_key(temp_db, key, rate_limit=2)
    # Pre-fill the last-minute counter to the limit
    conn = sqlite3.connect(str(temp_db))
    for _ in range(2):
        conn.execute(
            """INSERT INTO external_api_calls (key_hash, service, endpoint, status_code)
               VALUES (?, ?, ?, ?)""",
            (key_hash, "test", "/api/external/status", 200),
        )
    conn.commit()
    conn.close()
    r = flask_client.get(
        "/api/external/status",
        headers={"X-DMAI-Api-Key": key},
    )
    assert r.status_code == 429
    body = r.get_json()
    assert body["error"] == "rate_limited"
    assert body["limit_per_min"] == 2


# ---------------------------------------------------------------------------
# Hash sanity
# ---------------------------------------------------------------------------
def test_hash_key_stable_and_sha256():
    from components.external_api.auth import hash_key
    h1 = hash_key("dmai_ext_" + "z" * 32)
    h2 = hash_key("dmai_ext_" + "z" * 32)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex
    # Different plaintext -> different hash
    assert h1 != hash_key("dmai_ext_" + "y" * 32)
