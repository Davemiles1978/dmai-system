"""PR CCC-1a: /api/external/* blueprint.

v1 ships one read-only endpoint - /api/external/status - so we can
verify the auth + rate-limit + audit-log path end to end in prod
before adding any write endpoints. Subsequent PRs (CCC-1b, CCC-1c)
layer insight write/search, signal write, and HMAC webhooks on top.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify

from .auth import (
    _require_external_key,
    count_calls_last_minute,
)

logger = logging.getLogger(__name__)

external_api_bp = Blueprint(
    "external_api", __name__, url_prefix="/api/external",
)


@external_api_bp.route("/status", methods=["GET"])
@_require_external_key(scope="")  # any valid key works
def external_status():
    """Return the caller's key metadata + current rate-limit window.

    Safe first endpoint: no side effects, useful for a partner to test
    their credentials end-to-end. Never returns the plaintext key or
    key_hash to the caller.
    """
    key = g.dmai_key  # populated by _require_external_key
    used_last_min = count_calls_last_minute(key["key_hash"])
    return jsonify({
        "ok": True,
        "service": key["service"],
        "label": key["label"],
        "scope": key["scope"].split() if key["scope"] else [],
        "rate_limit_per_min": key["rate_limit_per_min"],
        "used_last_min": used_last_min,
        "ts": datetime.now(timezone.utc).isoformat(),
    }), 200


@external_api_bp.route("/ping", methods=["GET"])
def external_ping():
    """Unauthenticated liveness probe for external partners.

    Returns {'ok': true, 'service': 'dmai-external-api', 'ts': ...}.
    Explicitly unauthenticated so partners can sanity-check the
    endpoint is reachable before they provision a key.
    """
    return jsonify({
        "ok": True,
        "service": "dmai-external-api",
        "version": "1.0",
        "ts": datetime.now(timezone.utc).isoformat(),
    }), 200
