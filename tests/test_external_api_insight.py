"""PR CCC-1b (2/3): tests for POST /api/external/insight.

Verifies auth (unauth, missing/malformed/unknown key), scope
enforcement, validation guards, happy-path insert, and provenance
defaulting to external_api:<label>.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3

import pytest
from flask import Flask

KEY_PREFIX = "dmai_ext_"


def _mint():
    return KEY_PREFIX + secrets.token_hex(16)


def _hash(pt: str) -> str:
    return hashlib.sha256(pt.encode()).hexdigest()


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Sqlite with api_keys + insights + external_api_calls."""
    db_path = tmp_path / "test_insight.db"
    monkeypatch.setenv("DMAI_DB_PATH", str(db_path))
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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash    TEXT,
            service     TEXT,
            endpoint    TEXT,
            status_code INTEGER,
            duration_ms INTEGER,
            ts          TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE insights (
            id                TEXT PRIMARY KEY,
            insight_text      TEXT,
            entity_type       TEXT,
            entities          TEXT,
            relationship      TEXT,
            confidence        REAL DEFAULT 0.5,
            source_topic      TEXT,
            target_topic      TEXT,
            source_url        TEXT,
            source_title      TEXT,
            source_type       TEXT DEFAULT 'web',
            created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
            occurrence_count  INTEGER DEFAULT 1,
            last_used         TEXT,
            neuron_level      TEXT DEFAULT 'micro',
            parent_macro_id   TEXT,
            domain            TEXT,
            provenance        TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def valid_key(temp_db):
    """Provision a key with scope 'insight:write signal:read'."""
    plaintext = _mint()
    kh = _hash(plaintext)
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """INSERT INTO api_keys
           (key, service, source, validated, key_hash, scope,
            rate_limit_per_min, revoked, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (plaintext, "test-svc", "test", 1, kh,
         "insight:write signal:read", 60, 0, "TestPartner"),
    )
    conn.commit()
    conn.close()
    return plaintext


@pytest.fixture
def read_only_key(temp_db):
    """Key with scope insight:read only - lacks insight:write."""
    plaintext = _mint()
    kh = _hash(plaintext)
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """INSERT INTO api_keys
           (key, service, source, validated, key_hash, scope,
            rate_limit_per_min, revoked, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (plaintext, "test-svc", "test", 1, kh, "insight:read", 60, 0, "ReadOnly"),
    )
    conn.commit()
    conn.close()
    return plaintext


@pytest.fixture
def client(temp_db):
    from components.external_api import external_insight_bp
    app = Flask(__name__)
    app.register_blueprint(external_insight_bp)
    return app.test_client()


def _post(client, key, **body_overrides):
    body = {
        "insight_text": "Poulton-le-Fylde is in Lancashire.",
        "entity_type": "location",
        "entities": "Poulton-le-Fylde, Lancashire",
        "relationship": "part_of",
        "confidence": 0.9,
    }
    body.update(body_overrides)
    headers = {}
    if key is not None:
        headers["X-DMAI-Api-Key"] = key
    return client.post("/api/external/insight", json=body, headers=headers)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_401_missing_key(client):
    r = _post(client, None)
    assert r.status_code == 401
    assert r.get_json()["error"] == "missing_key"


def test_401_malformed_key(client):
    r = _post(client, "not-a-key")
    assert r.status_code == 401
    assert r.get_json()["error"] == "malformed_key"


def test_401_unknown_key(client):
    r = _post(client, _mint())
    assert r.status_code == 401
    assert r.get_json()["error"] == "unknown_key"


def test_403_insufficient_scope(client, read_only_key):
    r = _post(client, read_only_key)
    assert r.status_code == 403
    body = r.get_json()
    assert body["error"] == "insufficient_scope"
    assert body["required"] == "insight:write"


# ---------------------------------------------------------------------------
# Validation - data-quality rule 1 (no empty rows)
# ---------------------------------------------------------------------------
def test_400_missing_insight_text(client, valid_key):
    r = _post(client, valid_key, insight_text="")
    assert r.status_code == 400
    assert r.get_json()["error"] == "insight_text_required"


def test_400_missing_entity_type(client, valid_key):
    r = _post(client, valid_key, entity_type="")
    assert r.status_code == 400
    assert r.get_json()["error"] == "entity_type_required"


def test_400_bad_confidence_out_of_range(client, valid_key):
    r = _post(client, valid_key, confidence=1.5)
    assert r.status_code == 400
    assert r.get_json()["error"] == "confidence_out_of_range"
    r = _post(client, valid_key, confidence=-0.1)
    assert r.status_code == 400


def test_400_bad_confidence_type(client, valid_key):
    r = _post(client, valid_key, confidence="not-a-float")
    assert r.status_code == 400
    assert r.get_json()["error"] == "bad_confidence"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_201_creates_insight_row(client, valid_key, temp_db):
    r = _post(client, valid_key)
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    assert body["ok"] is True
    assert re.match(r"^[0-9a-f-]{36}$", body["id"])
    assert body["confidence"] == 0.9
    # Provenance defaulted to external_api:<label>
    assert body["provenance"] == "external_api:TestPartner"

    conn = sqlite3.connect(str(temp_db))
    row = conn.execute(
        "SELECT insight_text, entity_type, confidence, provenance, source_type "
        "FROM insights WHERE id = ?", (body["id"],)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "Poulton-le-Fylde is in Lancashire."
    assert row[1] == "location"
    assert abs(row[2] - 0.9) < 1e-6
    assert row[3] == "external_api:TestPartner"
    assert row[4] == "external-api"  # default source_type


def test_201_caller_provided_provenance_wins(client, valid_key, temp_db):
    r = _post(client, valid_key, provenance="crawler:seed-lancs-v2")
    assert r.status_code == 201
    assert r.get_json()["provenance"] == "crawler:seed-lancs-v2"


def test_201_confidence_defaults_to_0_5_when_omitted(client, valid_key):
    # Build the body without a confidence key at all so the default fires.
    r = client.post(
        "/api/external/insight",
        json={"insight_text": "Test default confidence",
              "entity_type": "fact"},
        headers={"X-DMAI-Api-Key": valid_key},
    )
    assert r.status_code == 201, r.get_json()
    assert r.get_json()["confidence"] == 0.5


def test_201_confidence_null_uses_default(client, valid_key):
    # confidence:null in the payload should also default to 0.5, not crash.
    r = client.post(
        "/api/external/insight",
        json={"insight_text": "Null confidence test",
              "entity_type": "fact",
              "confidence": None},
        headers={"X-DMAI-Api-Key": valid_key},
    )
    assert r.status_code == 201, r.get_json()
    assert r.get_json()["confidence"] == 0.5


def test_field_truncation_never_produces_empty_row(client, valid_key, temp_db):
    """Very long text is truncated, but the resulting row is still valid."""
    long_text = "x" * 10000  # 2x the 5000-char cap
    r = _post(client, valid_key, insight_text=long_text)
    assert r.status_code == 201
    conn = sqlite3.connect(str(temp_db))
    row = conn.execute(
        "SELECT LENGTH(insight_text) FROM insights WHERE id = ?",
        (r.get_json()["id"],)
    ).fetchone()
    conn.close()
    assert row[0] == 5000
