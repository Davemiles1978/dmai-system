"""Tests for the CRON_SECRET auth path (PR M).

A dedicated ``X-Cron-Secret`` header authorises scheduled callers against the
``CRON_SECRET`` env var (constant-time compare, fail-closed). These endpoints
never accept the master password or a JWT.

DATA_PATH is pointed at a temp dir *before* importing the app so the app's
boot side effects stay isolated.
"""
from __future__ import annotations

import hmac
import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="cron_auth_")
os.environ["DATA_PATH"] = _TMP

import dmai_core_complete  # noqa: E402
from dmai_core_complete import _require_cron_auth, app  # noqa: E402

_SECRET = "test-cron-secret-value"


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


# ── _require_cron_auth unit tests ─────────────────────────────────────────────

def test_require_cron_auth_false_when_secret_unset(monkeypatch):
    monkeypatch.delenv("CRON_SECRET", raising=False)
    with app.test_request_context(headers={"X-Cron-Secret": "anything"}):
        assert _require_cron_auth() is False


def test_require_cron_auth_false_when_header_missing(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    with app.test_request_context():
        assert _require_cron_auth() is False


def test_require_cron_auth_false_when_header_mismatch(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    with app.test_request_context(headers={"X-Cron-Secret": "wrong"}):
        assert _require_cron_auth() is False


def test_require_cron_auth_true_when_header_matches(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    with app.test_request_context(headers={"X-Cron-Secret": _SECRET}):
        assert _require_cron_auth() is True


def test_require_cron_auth_uses_compare_digest(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    calls = []
    _real = hmac.compare_digest

    def _spy(a, b):
        calls.append((a, b))
        return _real(a, b)

    monkeypatch.setattr(dmai_core_complete.hmac, "compare_digest", _spy)
    with app.test_request_context(headers={"X-Cron-Secret": _SECRET}):
        assert _require_cron_auth() is True
    assert calls, "hmac.compare_digest was not used for the comparison"


# ── endpoint tests ────────────────────────────────────────────────────────────

def test_cron_integrity_run_401_without_header(client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    resp = client.post("/api/cron/integrity/run")
    assert resp.status_code == 401
    body = resp.get_json()
    assert body["error"] == "cron auth required"
    assert "X-Cron-Secret" in body["hint"]


def test_cron_integrity_run_200_with_header(client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    ran = {"count": 0}
    from components.knowledge.integrity_checker import KnowledgeIntegrityChecker
    monkeypatch.setattr(KnowledgeIntegrityChecker, "run",
                        lambda self, *a, **k: ran.__setitem__("count", ran["count"] + 1))
    resp = client.post("/api/cron/integrity/run",
                       headers={"X-Cron-Secret": _SECRET})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "started"


def test_cron_providers_health_check_401_without_header(client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    resp = client.post("/api/cron/providers/health-check")
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "cron auth required"


def test_cron_providers_health_check_200_with_header(client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)

    class _FakeActivator:
        def get_status(self):
            return {
                "timestamp": "2026-07-14T00:00:00Z",
                "providers": {
                    "prov_a": {"status": "active"},
                    "prov_b": {"status": "pending_api_key"},
                    "prov_c": {"status": "invalid"},
                },
            }

    monkeypatch.setitem(dmai_core_complete.components, "api_activator",
                        _FakeActivator())
    resp = client.post("/api/cron/providers/health-check",
                       headers={"X-Cron-Secret": _SECRET})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["total_providers"] == 3
    assert body["active_count"] == 1
    assert body["active"] == ["prov_a"]
    assert body["pending_key"] == ["prov_b"]
    assert body["invalid"] == ["prov_c"]
    assert body["healthy"] is True


def test_cron_status_200_with_header(client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _SECRET)
    resp = client.get("/api/cron/status",
                      headers={"X-Cron-Secret": _SECRET})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "auth": "cron"}
