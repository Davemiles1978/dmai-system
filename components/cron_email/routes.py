"""PR DDD-3: /api/cron/promoter-drift/email endpoint.

CRON_SECRET-protected (same constant-time compare as the other
/api/cron/* routes). Accepts a JSON body from the GitHub Actions cron:

    {
      "subject":     "DMAI Promoter Drift ...",
      "html_body":   "<html>...</html>",
      "text_body":   "plain text version",
      "slack_summary": "one-line summary for Slack fallback",
      "to":          ["milesd040@gmail.com"]     # optional
    }

Returns {ok, delivered_via, resend, slack, elapsed_ms}. Returns 200
even if delivery failed (the response body carries the failure detail
so the cron can log it); we only return 401/400/500 for auth/shape
errors so the cron's exit code reliably signals scheduler-level bugs.
"""
from __future__ import annotations

import hmac
import logging
import os
import time
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from .sender import send_with_fallback

logger = logging.getLogger(__name__)

cron_email_bp = Blueprint("cron_email", __name__, url_prefix="/api/cron")

CRON_SECRET_HEADER = "X-Cron-Secret"
DEFAULT_TO = ["milesd040@gmail.com"]


def _authenticate(req) -> bool:
    """Constant-time compare against CRON_SECRET env var."""
    presented = req.headers.get(CRON_SECRET_HEADER, "")
    expected = os.environ.get("CRON_SECRET", "")
    if not expected:
        # Fail closed: if the secret isn't configured, refuse.
        return False
    return hmac.compare_digest(presented, expected)


@cron_email_bp.route("/promoter-drift/email", methods=["POST"])
def promoter_drift_email():
    """Deliver a pre-composed drift email via Resend + Slack fallback."""
    t0 = time.time()

    if not _authenticate(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    subject = (payload.get("subject") or "").strip()
    html_body = payload.get("html_body") or ""
    text_body = payload.get("text_body") or ""
    slack_summary = payload.get("slack_summary")
    to = payload.get("to") or DEFAULT_TO

    if not subject or not (html_body or text_body):
        return jsonify({
            "ok": False,
            "error": "bad_payload",
            "required": ["subject", "html_body or text_body"],
        }), 400

    # If only html_body is given, generate a bare-bones text_body so
    # Resend has both mime parts (better deliverability). Cheap fallback.
    if html_body and not text_body:
        text_body = "See HTML version. Subject: " + subject

    result = send_with_fallback(
        to=to,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        slack_summary=slack_summary,
    )
    result["ok"] = result["delivered_via"] != "none"
    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    result["to"] = to
    return jsonify(result), 200


@cron_email_bp.route("/promoter-drift/ping", methods=["GET"])
def promoter_drift_ping():
    """Public liveness probe. Confirms the blueprint is mounted and
    tells you whether Resend + Slack are configured (without revealing
    the actual credentials)."""
    return jsonify({
        "ok": True,
        "endpoint": "/api/cron/promoter-drift/email",
        "resend_configured": bool(os.environ.get("RESEND_API_KEY", "").strip()),
        "slack_configured": bool(os.environ.get("SLACK_WEBHOOK_URL", "").strip()),
    }), 200
