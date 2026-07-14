"""Backend contract tests for the save+activate UI feedback (PR N).

The UI (static/admin.html, static/mobile.html) reads two response shapes to
show inline activation status. These tests lock those shapes so a backend
change can't silently break the dashboard feedback. No JS test runner exists in
this repo, so UI behaviour is covered indirectly via the contract.

DATA_PATH is pointed at a temp dir *before* importing the app so boot side
effects stay isolated.
"""
from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="admin_keys_ui_")
os.environ["DATA_PATH"] = _TMP

import dmai_core_complete  # noqa: E402
from dmai_core_complete import app  # noqa: E402

_MASTER_PW = "test-master-pw"
_AUTH = {"X-Master-Password": _MASTER_PW}


class _FakeActivator:
    """Deterministic stand-in for AutoAPIActivator."""

    def __init__(self, provider_statuses):
        self._statuses = provider_statuses

    def scan_and_activate(self):
        active  = [p for p, s in self._statuses.items() if s == "active"]
        pending = [p for p, s in self._statuses.items() if s == "pending_api_key"]
        invalid = [p for p, s in self._statuses.items() if s == "invalid"]
        return {
            "providers":    {p: {"status": s} for p, s in self._statuses.items()},
            "activated":    active,
            "pending":      pending,
            "invalid":      invalid,
            "total_active": len(active),
            "timestamp":    "2026-07-14T00:00:00Z",
        }

    def get_status(self):
        return {
            "providers": {p: {"status": s, "error": ("bad key" if s == "invalid" else None)}
                          for p, s in self._statuses.items()},
            "timestamp": "2026-07-14T00:00:00Z",
        }


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_keys_post_requires_auth(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    resp = client.post("/api/admin/keys",
                       json={"provider_id": "groq", "key": "sk-x"})
    assert resp.status_code == 401


def test_keys_post_returns_activator_provider_status(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.setitem(dmai_core_complete.components, "api_activator",
                        _FakeActivator({"groq": "active"}))
    monkeypatch.delitem(dmai_core_complete.components, "db_storage", raising=False)
    resp = client.post("/api/admin/keys",
                       headers=_AUTH,
                       json={"provider_id": "groq", "key": "sk-live-123"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert "sinks" in body
    assert body["sinks"]["activator"]["provider_status"] == "active"
    # Render sink degrades gracefully when RENDER_API_KEY is absent.
    monkeypatch.delenv("RENDER_API_KEY", raising=False)


def test_keys_post_reports_invalid_status(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.setitem(dmai_core_complete.components, "api_activator",
                        _FakeActivator({"openai": "invalid"}))
    monkeypatch.delitem(dmai_core_complete.components, "db_storage", raising=False)
    resp = client.post("/api/admin/keys",
                       headers=_AUTH,
                       json={"provider_id": "openai", "key": "sk-bad"})
    assert resp.status_code == 200
    assert resp.get_json()["sinks"]["activator"]["provider_status"] == "invalid"


def test_harvester_scan_response_well_formed(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    monkeypatch.setitem(
        dmai_core_complete.components, "api_activator",
        _FakeActivator({"groq": "active", "openai": "invalid", "cohere": "pending_api_key"}))
    resp = client.post("/api/harvester/scan", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    for key in ("active_count", "activated", "pending", "invalid"):
        assert key in body
    assert body["activated"] == ["groq"]
    assert body["invalid"] == ["openai"]
    assert body["pending"] == ["cohere"]
    assert body["active_count"] == 1
