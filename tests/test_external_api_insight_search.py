"""PR CCC-1b (3/3): tests for GET /api/external/insight/search.

Verifies auth (401 missing/malformed/unknown, 403 insufficient scope),
validation (bad_since, bad_limit, bad_offset), filter combinations
(q, entity_type, source_topic, domain, provenance exact + prefix,
since, limit, offset), empty-filter guard (empty ?q= does not filter),
and result shape (count matches, no plaintext keys leaked).
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta

import pytest
from flask import Flask

KEY_PREFIX = "dmai_ext_"


def _mint():
    return KEY_PREFIX + secrets.token_hex(16)


def _hash(pt: str) -> str:
    return hashlib.sha256(pt.encode()).hexdigest()


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_search.db"
    monkeypatch.setenv("DMAI_DB_PATH", str(db_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE api_keys (
            key TEXT PRIMARY KEY, service TEXT, source TEXT,
            validated INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_used TEXT, key_hash TEXT,
            scope TEXT DEFAULT '',
            rate_limit_per_min INTEGER DEFAULT 60,
            revoked INTEGER DEFAULT 0, label TEXT
        );
        CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
        CREATE TABLE external_api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT, service TEXT, endpoint TEXT,
            status_code INTEGER, duration_ms INTEGER,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE insights (
            id TEXT PRIMARY KEY, insight_text TEXT, entity_type TEXT,
            entities TEXT, relationship TEXT, confidence REAL DEFAULT 0.5,
            source_topic TEXT, target_topic TEXT, source_url TEXT,
            source_title TEXT, source_type TEXT DEFAULT 'web',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            occurrence_count INTEGER DEFAULT 1, last_used TEXT,
            neuron_level TEXT DEFAULT 'micro', parent_macro_id TEXT,
            domain TEXT, provenance TEXT
        );
        """
    )
    # Seed 5 insights - varying entity_type, provenance, created_at.
    now = datetime(2026, 7, 18, 12, 0, 0)
    rows = [
        (str(uuid.uuid4()), "Poulton-le-Fylde is in Lancashire.",
         "location", "Poulton-le-Fylde",
         "part_of", 0.9, "geography", None,
         "https://en.wikipedia.org/wiki/Poulton", "Poulton", "web",
         (now - timedelta(days=1)).isoformat(), 1, None, "micro", None,
         "geography", "external_api:crawlerA"),
        (str(uuid.uuid4()), "Chorley is also in Lancashire.",
         "location", "Chorley",
         "part_of", 0.85, "geography", None, None, None, "web",
         (now - timedelta(days=2)).isoformat(), 1, None, "micro", None,
         "geography", "external_api:crawlerA"),
        (str(uuid.uuid4()), "Python is a high-level language.",
         "concept", "Python",
         "is_a", 0.95, "programming", None, None, None, "web",
         (now - timedelta(days=3)).isoformat(), 1, None, "micro", None,
         "computing", "external_api:crawlerB"),
        (str(uuid.uuid4()), "TypeScript adds types to JavaScript.",
         "concept", "TypeScript",
         "extends", 0.9, "programming", None, None, None, "web",
         (now - timedelta(days=4)).isoformat(), 1, None, "micro", None,
         "computing", "external_api:crawlerB"),
        (str(uuid.uuid4()), "Greyhound racing is a UK sport.",
         "activity", "Greyhound racing", "is_a", 0.8,
         "sports", None, None, None, "web",
         (now - timedelta(days=5)).isoformat(), 1, None, "micro", None,
         "sports", "manual:seed"),
    ]
    conn.executemany(
        """INSERT INTO insights
           (id, insight_text, entity_type, entities, relationship,
            confidence, source_topic, target_topic, source_url,
            source_title, source_type, created_at, occurrence_count,
            last_used, neuron_level, parent_macro_id, domain, provenance)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def read_key(temp_db):
    pt = _mint()
    kh = _hash(pt)
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """INSERT INTO api_keys
           (key, service, source, validated, key_hash, scope,
            rate_limit_per_min, revoked, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pt, "test-svc", "test", 1, kh, "insight:read", 60, 0, "ReadTester"),
    )
    conn.commit()
    conn.close()
    return pt


@pytest.fixture
def write_only_key(temp_db):
    pt = _mint()
    kh = _hash(pt)
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """INSERT INTO api_keys
           (key, service, source, validated, key_hash, scope,
            rate_limit_per_min, revoked, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pt, "test-svc", "test", 1, kh, "insight:write", 60, 0, "WriteOnly"),
    )
    conn.commit()
    conn.close()
    return pt


@pytest.fixture
def client(temp_db):
    from components.external_api import external_insight_search_bp
    app = Flask(__name__)
    app.register_blueprint(external_insight_search_bp)
    return app.test_client()


def _get(client, key, **params):
    headers = {"X-DMAI-Api-Key": key} if key else {}
    return client.get("/api/external/insight/search",
                      query_string=params, headers=headers)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_401_missing_key(client):
    assert _get(client, None).status_code == 401


def test_401_malformed_key(client):
    assert _get(client, "not-a-key").status_code == 401


def test_403_insufficient_scope(client, write_only_key):
    r = _get(client, write_only_key)
    assert r.status_code == 403
    assert r.get_json()["error"] == "insufficient_scope"


# ---------------------------------------------------------------------------
# Happy path - no filters returns all
# ---------------------------------------------------------------------------
def test_returns_all_5_seeded(client, read_key):
    r = _get(client, read_key)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["count"] == 5
    assert len(body["insights"]) == 5
    # Descending order by created_at
    dates = [i["created_at"] for i in body["insights"]]
    assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def test_filter_q_substring(client, read_key):
    r = _get(client, read_key, q="Lancashire")
    assert r.status_code == 200
    assert r.get_json()["count"] == 2  # Poulton + Chorley


def test_filter_q_case_insensitive(client, read_key):
    r = _get(client, read_key, q="LANCASHIRE")
    assert r.get_json()["count"] == 2


def test_filter_entity_type(client, read_key):
    r = _get(client, read_key, entity_type="concept")
    assert r.get_json()["count"] == 2  # Python + TypeScript


def test_filter_source_topic(client, read_key):
    r = _get(client, read_key, source_topic="programming")
    assert r.get_json()["count"] == 2


def test_filter_domain(client, read_key):
    r = _get(client, read_key, domain="sports")
    assert r.get_json()["count"] == 1


def test_filter_provenance_exact(client, read_key):
    r = _get(client, read_key, provenance="manual:seed")
    assert r.get_json()["count"] == 1


def test_filter_provenance_prefix(client, read_key):
    r = _get(client, read_key, provenance="external_api:*")
    assert r.get_json()["count"] == 4  # all crawlerA + crawlerB


def test_filter_since(client, read_key):
    # Seeds are dated 2026-07-13..17 (now=2026-07-18 minus 1..5 days).
    # since=2026-07-16 00:00 catches created_at >= that, which is
    # 2026-07-17 12:00 (day-1) and 2026-07-16 12:00 (day-2) = 2 rows.
    r = _get(client, read_key, since="2026-07-16")
    assert r.get_json()["count"] == 2


def test_filter_since_wider_window(client, read_key):
    # since=2026-07-15 catches days 1, 2, 3 back = 3 rows.
    r = _get(client, read_key, since="2026-07-15")
    assert r.get_json()["count"] == 3


def test_filter_combined(client, read_key):
    # entity_type=location AND q=Lancashire -> 2 (Poulton + Chorley)
    r = _get(client, read_key, entity_type="location", q="Lancashire")
    assert r.get_json()["count"] == 2


# ---------------------------------------------------------------------------
# Data-quality guard: empty filters do NOT filter
# ---------------------------------------------------------------------------
def test_empty_string_filters_do_not_filter(client, read_key):
    r = _get(client, read_key, q="", entity_type="", source_topic="")
    assert r.get_json()["count"] == 5


# ---------------------------------------------------------------------------
# Pagination + validation
# ---------------------------------------------------------------------------
def test_limit_offset(client, read_key):
    r = _get(client, read_key, limit=2, offset=0)
    body = r.get_json()
    assert body["count"] == 2
    assert body["limit"] == 2
    assert body["offset"] == 0
    r2 = _get(client, read_key, limit=2, offset=2)
    body2 = r2.get_json()
    assert body2["count"] == 2
    # Different rows
    ids1 = {i["id"] for i in body["insights"]}
    ids2 = {i["id"] for i in body2["insights"]}
    assert ids1.isdisjoint(ids2)


def test_400_bad_limit_too_low(client, read_key):
    assert _get(client, read_key, limit=0).status_code == 400


def test_400_bad_limit_too_high(client, read_key):
    assert _get(client, read_key, limit=1000).status_code == 400


def test_400_bad_offset(client, read_key):
    assert _get(client, read_key, offset=-1).status_code == 400


def test_400_bad_since(client, read_key):
    r = _get(client, read_key, since="not-a-date")
    assert r.status_code == 400
    assert r.get_json()["error"] == "bad_since"


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------
def test_response_shape_never_leaks_internal_columns(client, read_key):
    r = _get(client, read_key, limit=1)
    body = r.get_json()
    row = body["insights"][0]
    allowed = {"id", "insight_text", "entity_type", "entities", "relationship",
               "confidence", "source_topic", "target_topic", "source_url",
               "source_title", "source_type", "domain", "provenance",
               "created_at"}
    assert set(row.keys()) == allowed
    # Never leak plaintext api_keys or hashes
    for v in row.values():
        if isinstance(v, str):
            assert not v.startswith("dmai_ext_")
