"""PR CCC-1b (1/3): tests for /api/admin/external-keys provisioning.

Verifies:
  * Auth: 401 without master password, 200/201 with it.
  * Validation: label + scope required, scope tokens must be
    resource:action shape, rate_limit bounded.
  * Create: mints a dmai_ext_<32-hex>, stores hash + scope + label,
    returns plaintext once.
  * List: shows all keys without plaintext.
  * Revoke: idempotent, 404 for unknown hash, updates revoked=1.
  * Data-quality guards: never inserts empty scope/label rows.
"""
from __future__ import annotations

import re
import sqlite3

import pytest
from flask import Flask


MASTER = "test-master-password"


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Sqlite DB with the CCC-1a schema."""
    db_path = tmp_path / "test_admin.db"
    monkeypatch.setenv("DMAI_DB_PATH", str(db_path))
    monkeypatch.setenv("MASTER_PASSWORD", MASTER)
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
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def client(temp_db, monkeypatch):
    """Build a minimal app with the admin blueprint AND a stubbed
    _require_auth that reads X-Master-Password against MASTER_PASSWORD."""
    from components.external_api import external_admin_bp
    app = Flask(__name__)
    app.register_blueprint(external_admin_bp)

    # Stub the lazy-import target inside admin._require_admin so the
    # test client works without loading dmai_core_complete.
    import components.external_api.admin as admin_mod

    def fake_require():
        from flask import request
        return request.headers.get("X-Master-Password", "") == MASTER

    monkeypatch.setattr(admin_mod, "_require_admin", fake_require)
    return app.test_client()


def _post_create(client, **overrides):
    body = {
        "label": "Test Partner",
        "scope": "insight:write signal:read",
        "rate_limit_per_min": 30,
        "service": "test-service",
    }
    body.update(overrides)
    return client.post(
        "/api/admin/external-keys",
        json=body,
        headers={"X-Master-Password": MASTER},
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_create_401_without_master_password(client):
    r = client.post(
        "/api/admin/external-keys",
        json={"label": "x", "scope": "insight:write"},
    )
    assert r.status_code == 401


def test_list_401_without_master_password(client):
    r = client.get("/api/admin/external-keys")
    assert r.status_code == 401


def test_revoke_401_without_master_password(client):
    r = client.post("/api/admin/external-keys/" + "a" * 64 + "/revoke")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Validation - honours data-quality rule 1 (no empty rows)
# ---------------------------------------------------------------------------
def test_create_400_empty_label(client):
    r = _post_create(client, label="")
    assert r.status_code == 400
    assert r.get_json()["error"] == "label_required"


def test_create_400_empty_scope(client):
    r = _post_create(client, scope="")
    assert r.status_code == 400
    assert r.get_json()["error"] == "scope_required"


def test_create_400_whitespace_only_scope(client):
    r = _post_create(client, scope="   ")
    assert r.status_code == 400
    assert r.get_json()["error"] == "scope_required"


def test_create_400_malformed_scope_token(client):
    r = _post_create(client, scope="justword")
    assert r.status_code == 400
    assert r.get_json()["error"].startswith("malformed_scope_token")


def test_create_400_bad_rate_limit(client):
    r = _post_create(client, rate_limit_per_min=0)
    assert r.status_code == 400
    r = _post_create(client, rate_limit_per_min=200_000)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Create happy path
# ---------------------------------------------------------------------------
def test_create_201_returns_plaintext_key_once(client, temp_db):
    r = _post_create(client)
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    assert body["ok"] is True
    # Plaintext must match the dmai_ext_<32-hex> shape
    assert re.match(r"^dmai_ext_[0-9a-f]{32}$", body["key"])
    assert len(body["key_hash"]) == 64
    assert body["label"] == "Test Partner"
    assert body["scope"] == "insight:write signal:read"
    assert body["rate_limit_per_min"] == 30
    # Warning must be present so the operator knows to copy now
    assert "warning" in body

    # DB row must have hash + scope + label populated
    conn = sqlite3.connect(str(temp_db))
    row = conn.execute(
        "SELECT key_hash, scope, label, rate_limit_per_min, revoked, validated "
        "FROM api_keys"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == body["key_hash"]
    assert row[1] == "insight:write signal:read"
    assert row[2] == "Test Partner"
    assert row[3] == 30
    assert row[4] == 0
    assert row[5] == 1  # auto-validated


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------
def test_list_returns_provisioned_keys_without_plaintext(client):
    _post_create(client, label="k1")
    _post_create(client, label="k2", scope="signal:write")
    r = client.get(
        "/api/admin/external-keys",
        headers={"X-Master-Password": MASTER},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == 2
    labels = sorted(k["label"] for k in body["keys"])
    assert labels == ["k1", "k2"]
    # No plaintext should ever appear in a list response
    for k in body["keys"]:
        assert "key" not in k or k["key"] is None
        assert "key_hash" in k
        assert "revoked" in k


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------
def test_revoke_404_unknown_hash(client):
    r = client.post(
        "/api/admin/external-keys/" + "a" * 64 + "/revoke",
        headers={"X-Master-Password": MASTER},
    )
    assert r.status_code == 404


def test_revoke_400_bad_hash(client):
    r = client.post(
        "/api/admin/external-keys/tooshort/revoke",
        headers={"X-Master-Password": MASTER},
    )
    assert r.status_code == 400


def test_revoke_200_sets_revoked_flag(client, temp_db):
    create = _post_create(client).get_json()
    kh = create["key_hash"]
    r = client.post(
        f"/api/admin/external-keys/{kh}/revoke",
        headers={"X-Master-Password": MASTER},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["key_hash"] == kh
    conn = sqlite3.connect(str(temp_db))
    revoked = conn.execute(
        "SELECT revoked FROM api_keys WHERE key_hash = ?", (kh,)
    ).fetchone()[0]
    conn.close()
    assert revoked == 1


def test_revoke_is_idempotent(client, temp_db):
    kh = _post_create(client).get_json()["key_hash"]
    for _ in range(3):
        r = client.post(
            f"/api/admin/external-keys/{kh}/revoke",
            headers={"X-Master-Password": MASTER},
        )
        assert r.status_code == 200
    # Still just one row, still revoked
    conn = sqlite3.connect(str(temp_db))
    n, rev = conn.execute(
        "SELECT COUNT(*), MAX(revoked) FROM api_keys WHERE key_hash = ?", (kh,)
    ).fetchone()
    conn.close()
    assert n == 1
    assert rev == 1
