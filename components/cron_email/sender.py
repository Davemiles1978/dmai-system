"""PR DDD-3: two-hop email sender - Resend primary, Slack fallback.

Design goals:
  * NEVER raise from send_with_fallback(); external cron jobs must not
    receive 500s just because we couldn't email.
  * Always attempt Slack if Resend fails, even for a transient network
    blip - the whole point of the fallback is redundancy.
  * Return a rich status dict so the caller can log which path fired.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib import request as _urlreq
from urllib import error as _urlerr

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"
RESEND_TIMEOUT_SEC = 20
SLACK_TIMEOUT_SEC = 15


def send_resend(
    to: List[str],
    subject: str,
    html_body: str,
    text_body: str,
    from_addr: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """POST an email to Resend. Returns {ok, status_code, id?, error?}.

    Never raises - all failure modes flow through the return value so
    the caller can decide whether to trigger the Slack fallback.
    """
    api_key = api_key or os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "status_code": 0, "error": "no_api_key"}
    from_addr = (
        from_addr
        or os.environ.get("RESEND_FROM", "").strip()
        or "DMAI Alerts <onboarding@resend.dev>"
    )
    payload = {
        "from": from_addr,
        "to": to,
        "subject": subject,
        "html": html_body,
        "text": text_body,
    }
    req = _urlreq.Request(
        RESEND_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=RESEND_TIMEOUT_SEC) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body) if body else {}
            return {
                "ok": 200 <= resp.status < 300,
                "status_code": resp.status,
                "id": data.get("id"),
            }
    except _urlerr.HTTPError as e:
        # Resend returns detailed JSON errors; keep them for triage.
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        logger.warning("Resend HTTPError %s: %s", e.code, body[:400])
        return {"ok": False, "status_code": e.code, "error": body[:400]}
    except Exception as e:  # noqa: BLE001
        logger.warning("Resend send failed (transport): %s", e)
        return {"ok": False, "status_code": 0, "error": str(e)[:200]}


def send_slack(
    text: str,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """POST a Slack incoming-webhook message. Never raises.

    Uses the already-configured SLACK_WEBHOOK_URL env var so we share
    the same channel (#dmaitalk) as the rest of DMAI's alerts.
    """
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return {"ok": False, "status_code": 0, "error": "no_webhook"}
    req = _urlreq.Request(
        webhook_url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=SLACK_TIMEOUT_SEC) as resp:
            return {"ok": 200 <= resp.status < 300, "status_code": resp.status}
    except _urlerr.HTTPError as e:
        return {"ok": False, "status_code": e.code, "error": "http_error"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status_code": 0, "error": str(e)[:200]}


def send_with_fallback(
    to: List[str],
    subject: str,
    html_body: str,
    text_body: str,
    slack_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Send via Resend; if it fails, send a compact summary via Slack.

    Slack is always attempted when Resend fails - the fallback path is
    the whole point. Returns:

        {"resend": {...}, "slack": {...}, "delivered_via": "resend"|"slack"|"none"}
    """
    resend_result = send_resend(to, subject, html_body, text_body)
    slack_result: Dict[str, Any] = {"ok": False, "status_code": 0, "error": "not_attempted"}
    delivered_via = "none"

    if resend_result.get("ok"):
        delivered_via = "resend"
    else:
        summary = slack_summary or (
            f"DMAI email delivery FAILED via Resend (status="
            f"{resend_result.get('status_code')}): {subject}"
        )
        slack_result = send_slack(summary)
        if slack_result.get("ok"):
            delivered_via = "slack"

    return {
        "resend": resend_result,
        "slack": slack_result,
        "delivered_via": delivered_via,
    }
