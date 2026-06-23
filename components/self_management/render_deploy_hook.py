"""
RenderDeployHook — listens for GitHub webhook events.
When a PR labelled `auto-generated` is merged to main, it triggers
a Render deploy via the Render Deploy Hook URL.

Setup:
1. Set RENDER_DEPLOY_HOOK_URL env var (from Render dashboard → your service → Deploy Hook)
2. Set GITHUB_WEBHOOK_SECRET env var (shared secret for HMAC validation)
3. Register this endpoint as a GitHub webhook on your repo:
   URL: https://dmai-complete.onrender.com/api/webhook/github
   Content-Type: application/json
   Events: Pull requests

This module registers the Flask route on the given app instance.
"""

import os
import hmac
import hashlib
import logging
import threading

import requests
from flask import Blueprint, request, jsonify

logger = logging.getLogger("RenderDeployHook")

RENDER_DEPLOY_HOOK_URL = os.getenv("RENDER_DEPLOY_HOOK_URL", "")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")

render_webhook_bp = Blueprint("render_webhook", __name__)


def _verify_signature(payload_bytes: bytes, sig_header: str) -> bool:
    """Validate GitHub's HMAC-SHA256 webhook signature."""
    if not GITHUB_WEBHOOK_SECRET:
        logger.warning("GITHUB_WEBHOOK_SECRET not set — skipping signature verification")
        return True  # permissive if secret not configured

    if not sig_header or not sig_header.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, sig_header)


def _trigger_render_deploy(service_name: str = "dmai-complete"):
    """POST to the Render deploy hook URL to trigger a new deploy."""
    if not RENDER_DEPLOY_HOOK_URL:
        logger.warning("RENDER_DEPLOY_HOOK_URL not set — cannot trigger deploy")
        return False

    try:
        r = requests.post(RENDER_DEPLOY_HOOK_URL, timeout=15)
        if r.status_code in (200, 201, 204):
            logger.info("Render deploy triggered successfully for %s", service_name)
            return True
        else:
            logger.error("Render deploy hook returned %s: %s", r.status_code, r.text[:200])
            return False
    except Exception as e:
        logger.error("Failed to trigger Render deploy: %s", e)
        return False


def _handle_pr_event(payload: dict) -> dict:
    """
    Process a pull_request webhook payload.
    Triggers deploy when an auto-generated PR is merged to main.
    """
    action = payload.get("action")
    pr = payload.get("pull_request", {})
    merged = pr.get("merged", False)
    base_branch = pr.get("base", {}).get("ref", "")
    labels = [lbl.get("name", "") for lbl in pr.get("labels", [])]

    if action != "closed" or not merged or base_branch != "main":
        return {"status": "skipped", "reason": "not a merge to main"}

    if "auto-generated" not in labels:
        return {"status": "skipped", "reason": "PR not labelled auto-generated"}

    pr_number = pr.get("number")
    pr_title = pr.get("title", "")
    logger.info("Auto-generated PR #%s merged to main: %s — triggering Render deploy", pr_number, pr_title)

    # Trigger in a background thread so we respond to GitHub quickly
    threading.Thread(
        target=_trigger_render_deploy,
        args=("dmai-complete",),
        daemon=True,
        name="RenderDeployTrigger",
    ).start()

    return {
        "status": "deploy_triggered",
        "pr_number": pr_number,
        "pr_title": pr_title,
    }


@render_webhook_bp.route("/api/webhook/github", methods=["POST"])
def github_webhook():
    payload_bytes = request.get_data()
    sig = request.headers.get("X-Hub-Signature-256", "")

    if not _verify_signature(payload_bytes, sig):
        logger.warning("Invalid webhook signature from %s", request.remote_addr)
        return jsonify({"error": "invalid signature"}), 401

    event_type = request.headers.get("X-GitHub-Event", "")
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    if event_type == "pull_request":
        result = _handle_pr_event(payload)
        return jsonify(result), 200
    elif event_type == "ping":
        return jsonify({"status": "pong"}), 200
    else:
        return jsonify({"status": "ignored", "event": event_type}), 200


def register(app):
    """Register the webhook blueprint on a Flask app."""
    app.register_blueprint(render_webhook_bp)
    logger.info("RenderDeployHook registered at /api/webhook/github")
