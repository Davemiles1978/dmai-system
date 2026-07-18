"""PR DDD-3: /api/cron/promoter-drift/email endpoint tests.

Uses monkeypatched send_resend/send_slack so the tests don't hit the
network. Verifies:

  * CRON_SECRET auth (401 without, 200 with)
  * Bad payload rejected (400)
  * Happy path (Resend succeeds -> delivered_via=resend)
  * Fallback path (Resend fails -> Slack tried -> delivered_via=slack)
  * Total failure (both fail -> delivered_via=none, ok:false, 200)
  * Ping is unauthenticated and reports configuration
"""
from __future__ import annotations

import pytest
from flask import Flask


@pytest.fixture
def client(monkeypatch):
    from components.cron_email import cron_email_bp
    app = Flask(__name__)
    app.register_blueprint(cron_email_bp)
    monkeypatch.setenv("CRON_SECRET", "test-secret")
    return app.test_client()


@pytest.fixture
def force_resend_ok(monkeypatch):
    from components.cron_email import sender
    def fake(*_a, **_kw):
        return {"ok": True, "status_code": 200, "id": "test-id"}
    monkeypatch.setattr(sender, "send_resend", fake)


@pytest.fixture
def force_resend_fail(monkeypatch):
    from components.cron_email import sender
    def fake(*_a, **_kw):
        return {"ok": False, "status_code": 503, "error": "resend down"}
    monkeypatch.setattr(sender, "send_resend", fake)


@pytest.fixture
def force_slack_ok(monkeypatch):
    from components.cron_email import sender
    def fake(*_a, **_kw):
        return {"ok": True, "status_code": 200}
    monkeypatch.setattr(sender, "send_slack", fake)


@pytest.fixture
def force_slack_fail(monkeypatch):
    from components.cron_email import sender
    def fake(*_a, **_kw):
        return {"ok": False, "status_code": 500, "error": "slack down"}
    monkeypatch.setattr(sender, "send_slack", fake)


VALID_PAYLOAD = {
    "subject": "DMAI Promoter Drift Alert - test",
    "html_body": "<p>Drift dropped 15%</p>",
    "text_body": "Drift dropped 15%",
    "slack_summary": "Drift -15%",
}


def test_email_401_without_secret(client):
    r = client.post("/api/cron/promoter-drift/email", json=VALID_PAYLOAD)
    assert r.status_code == 401
    assert r.get_json()["error"] == "unauthorized"


def test_email_401_wrong_secret(client):
    r = client.post(
        "/api/cron/promoter-drift/email",
        json=VALID_PAYLOAD,
        headers={"X-Cron-Secret": "wrong"},
    )
    assert r.status_code == 401


def test_email_400_bad_payload(client):
    r = client.post(
        "/api/cron/promoter-drift/email",
        json={"subject": "no body"},
        headers={"X-Cron-Secret": "test-secret"},
    )
    assert r.status_code == 400
    assert r.get_json()["error"] == "bad_payload"


def test_email_delivered_via_resend(client, force_resend_ok, force_slack_fail):
    """When Resend succeeds, Slack must NOT be attempted."""
    r = client.post(
        "/api/cron/promoter-drift/email",
        json=VALID_PAYLOAD,
        headers={"X-Cron-Secret": "test-secret"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["delivered_via"] == "resend"
    # Slack must be recorded as not attempted
    assert body["slack"]["error"] == "not_attempted"


def test_email_fallback_to_slack(client, force_resend_fail, force_slack_ok):
    """When Resend fails, Slack fallback must fire and succeed."""
    r = client.post(
        "/api/cron/promoter-drift/email",
        json=VALID_PAYLOAD,
        headers={"X-Cron-Secret": "test-secret"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["delivered_via"] == "slack"
    assert body["resend"]["ok"] is False
    assert body["slack"]["ok"] is True


def test_email_both_fail(client, force_resend_fail, force_slack_fail):
    """Both fail -> 200 with ok:false and delivered_via:none."""
    r = client.post(
        "/api/cron/promoter-drift/email",
        json=VALID_PAYLOAD,
        headers={"X-Cron-Secret": "test-secret"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert body["delivered_via"] == "none"


def test_email_synthesises_text_from_html_if_missing(client, force_resend_ok):
    """Payload with only html_body must still succeed."""
    payload = {
        "subject": "Test",
        "html_body": "<p>Hi</p>",
    }
    r = client.post(
        "/api/cron/promoter-drift/email",
        json=payload,
        headers={"X-Cron-Secret": "test-secret"},
    )
    assert r.status_code == 200
    assert r.get_json()["delivered_via"] == "resend"


def test_ping_is_unauthenticated(client):
    r = client.get("/api/cron/promoter-drift/ping")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["endpoint"] == "/api/cron/promoter-drift/email"
    assert "resend_configured" in body
    assert "slack_configured" in body


def test_send_resend_no_api_key_returns_no_api_key(monkeypatch):
    """When RESEND_API_KEY is unset the sender must return a clear error
    without attempting a network call."""
    from components.cron_email.sender import send_resend
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    result = send_resend(
        to=["x@example.com"], subject="s", html_body="", text_body="t",
    )
    assert result["ok"] is False
    assert result["error"] == "no_api_key"


def test_send_slack_no_webhook_returns_no_webhook(monkeypatch):
    """Same for Slack: unset SLACK_WEBHOOK_URL must fail fast."""
    from components.cron_email.sender import send_slack
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    result = send_slack("hi")
    assert result["ok"] is False
    assert result["error"] == "no_webhook"
