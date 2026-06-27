"""
DMAI HMAC Webhook Validator
============================
Validates incoming payment webhook signatures using HMAC-SHA256.
Supports Stripe, generic, and custom webhook formats.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Webhook secret configuration
# ---------------------------------------------------------------------------

WEBHOOK_SECRET: str = os.environ.get("WEBHOOK_SECRET", "")
if not WEBHOOK_SECRET:
    logger.warning(
        "WEBHOOK_SECRET env var is not set. "
        "All webhook signature verifications will be rejected."
    )

# ---------------------------------------------------------------------------
# Core HMAC helpers
# ---------------------------------------------------------------------------


def compute_hmac(payload: bytes, secret: str) -> str:
    """Compute the HMAC-SHA256 hex digest of a payload.

    Args:
        payload: Raw bytes of the request body.
        secret: Shared secret string used as the HMAC key.

    Returns:
        Lowercase hexadecimal digest string.
    """
    return hmac.new(
        secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()


def _parse_stripe_header(signature_header: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a Stripe-Signature header into its timestamp and v1 signature.

    Stripe format: ``t=1492774577,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a05bd539ba74379a9516d28``

    Args:
        signature_header: Raw value of the Stripe-Signature HTTP header.

    Returns:
        A tuple of (timestamp_str, v1_signature), both None if parsing fails.
    """
    parts = {}
    for item in signature_header.split(","):
        kv = item.split("=", 1)
        if len(kv) == 2:
            parts[kv[0].strip()] = kv[1].strip()

    timestamp = parts.get("t")
    v1_sig = parts.get("v1")
    return timestamp, v1_sig


def verify_webhook_signature(
    payload: bytes,
    signature_header: str,
    secret: Optional[str] = None,
) -> bool:
    """Verify a webhook payload against its signature header.

    Supports two formats:

    * **Stripe format**: ``t=<timestamp>,v1=<hex_sig>``
      The signed string is ``<timestamp>.<payload_body>``.
    * **Plain hex format**: the header contains just the hex digest.

    Args:
        payload: Raw request body bytes.
        signature_header: Value of the signature HTTP header.
        secret: Override the module-level WEBHOOK_SECRET for this call.
            Defaults to the env-configured WEBHOOK_SECRET.

    Returns:
        True if the signature is valid; False otherwise.
    """
    effective_secret = secret if secret is not None else WEBHOOK_SECRET

    if not effective_secret:
        logger.warning("verify_webhook_signature: secret is empty; rejecting signature.")
        return False

    if not signature_header:
        logger.warning("verify_webhook_signature: empty signature header.")
        return False

    # Try Stripe format first
    if "v1=" in signature_header:
        timestamp_str, v1_sig = _parse_stripe_header(signature_header)
        if timestamp_str is None or v1_sig is None:
            logger.warning("verify_webhook_signature: malformed Stripe header.")
            return False

        # Optionally enforce timestamp tolerance (300 seconds)
        try:
            ts = int(timestamp_str)
            skew = abs(int(time.time()) - ts)
            if skew > 300:
                logger.warning(
                    "verify_webhook_signature: Stripe timestamp skew %ds exceeds 300s tolerance.",
                    skew,
                )
                return False
        except ValueError:
            logger.warning("verify_webhook_signature: non-integer Stripe timestamp.")
            return False

        signed_payload = ("%s." % timestamp_str).encode("utf-8") + payload
        expected = hmac.new(
            effective_secret.encode("utf-8"),
            msg=signed_payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, v1_sig.lower())

    # Plain hex signature
    expected = compute_hmac(payload, effective_secret)
    return hmac.compare_digest(expected, signature_header.lower().strip())


def require_webhook_auth(f):
    """Flask route decorator that validates HMAC signatures on webhook endpoints.

    Reads the raw request body and checks the signature from one of these
    headers (in priority order):

    1. ``Stripe-Signature``
    2. ``X-Hub-Signature-256`` (GitHub/generic)
    3. ``X-Webhook-Signature``

    Returns HTTP 401 JSON if the signature is missing or invalid.
    """
    from flask import jsonify, request

    @wraps(f)
    def decorated(*args, **kwargs):
        """Perform HMAC verification before delegating to the route handler."""
        raw_body = request.get_data()

        sig_header = (
            request.headers.get("Stripe-Signature")
            or request.headers.get("X-Hub-Signature-256")
            or request.headers.get("X-Webhook-Signature")
            or ""
        )

        if not sig_header:
            return (
                jsonify({"error": "Unauthorized", "detail": "Missing webhook signature header"}),
                401,
            )

        if not verify_webhook_signature(raw_body, sig_header):
            return (
                jsonify({"error": "Unauthorized", "detail": "Invalid webhook signature"}),
                401,
            )

        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# WebhookPayload dataclass
# ---------------------------------------------------------------------------


@dataclass
class WebhookPayload:
    """Parsed and validated representation of an incoming payment webhook.

    Attributes:
        event_type: Event name string, e.g. 'payment_intent.succeeded'.
        payment_status: Payment status string, e.g. 'paid', 'failed'.
        amount: Transaction amount in the smallest currency unit (e.g. pence).
        currency: ISO 4217 currency code, e.g. 'gbp'.
        invoice_id: Invoice or charge identifier string.
        raw_data: The full deserialized webhook payload as a dict.
        verified: True if the HMAC signature was validated successfully.
    """

    event_type: str
    payment_status: str
    amount: int
    currency: str
    invoice_id: str
    raw_data: dict = field(default_factory=dict)
    verified: bool = False


# ---------------------------------------------------------------------------
# High-level parser
# ---------------------------------------------------------------------------


def parse_payment_webhook(request) -> Tuple[Optional[WebhookPayload], str]:
    """Parse and validate an incoming payment webhook in one call.

    Performs HMAC verification, then extracts standard fields from the JSON
    body. Supports Stripe event envelopes; falls back to generic flat JSON.

    Args:
        request: A Flask Request object.  The raw body is read via
            ``request.get_data()``.

    Returns:
        A tuple of (WebhookPayload | None, error_message).  On success,
        the first element is populated and ``error_message`` is an empty
        string.  On failure, the first element is None and ``error_message``
        explains the reason.
    """
    raw_body: bytes = request.get_data()

    # Determine signature header
    sig_header = (
        request.headers.get("Stripe-Signature")
        or request.headers.get("X-Hub-Signature-256")
        or request.headers.get("X-Webhook-Signature")
        or ""
    )

    verified = False
    if sig_header:
        verified = verify_webhook_signature(raw_body, sig_header)
    else:
        logger.warning("parse_payment_webhook: no signature header present.")

    # Deserialise JSON
    try:
        data = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, "Invalid JSON payload: %s" % exc

    if not isinstance(data, dict):
        return None, "Payload root must be a JSON object."

    # Extract fields -- handle Stripe envelope vs. flat format
    event_type = data.get("type") or data.get("event_type") or "unknown"
    invoice_id = (
        data.get("id")
        or (data.get("data", {}).get("object", {}).get("id") if isinstance(data.get("data"), dict) else None)
        or data.get("invoice_id")
        or ""
    )

    # Stripe puts amount/currency inside data.object
    stripe_obj = {}
    if isinstance(data.get("data"), dict):
        stripe_obj = data["data"].get("object", {})

    amount_raw = stripe_obj.get("amount_received") or stripe_obj.get("amount") or data.get("amount") or 0
    try:
        amount = int(amount_raw)
    except (TypeError, ValueError):
        amount = 0

    currency = (
        stripe_obj.get("currency")
        or data.get("currency")
        or "unknown"
    ).lower()

    payment_status = (
        stripe_obj.get("status")
        or data.get("payment_status")
        or data.get("status")
        or "unknown"
    )

    webhook = WebhookPayload(
        event_type=str(event_type),
        payment_status=str(payment_status),
        amount=amount,
        currency=str(currency),
        invoice_id=str(invoice_id),
        raw_data=data,
        verified=verified,
    )
    return webhook, ""
