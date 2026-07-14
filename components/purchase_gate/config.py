"""Config + constants for the purchase-approval gate (PR L).

Module constants are the *defaults*. Runtime overrides live in the
``config_kv`` table and are read via
:meth:`components.purchase_gate.purchase_ledger.PurchaseGateStore.config_kv_get`
(which falls back to these constants). Env vars (``DMAI_AUTO_CHECKOUT_*``)
can seed the overrides at boot.
"""
from __future__ import annotations

import os
from pathlib import Path

PURCHASE_GATE_DB_FILENAME = "dmai_purchase_gate.db"

# Operator contact for proposal notifications.
OPERATOR_EMAIL = "milesd040@gmail.com"
SLACK_CHANNEL = "#dmaitalk"

# ── Trigger / cadence ────────────────────────────────────────────────────────
TRIGGER_MULTIPLIER = 1.2               # balance >= 1.2 * top-1 capex → propose
PROPOSAL_POLL_INTERVAL_SECONDS = 1800  # 30 minutes

# ── Auto-checkout scaffold (FEATURE-FLAGGED OFF) ──────────────────────────────
# Invariants (see checkout_adapter.py + workspace/pr_l_notes.md):
#  1. AUTO_CHECKOUT_ENABLED defaults False.
#  2. AUTO_CHECKOUT_DRY_RUN defaults True even when enabled.
#  3. No adapter implements execute_checkout — no live purchase path exists.
#  4. AUTO_CHECKOUT_MAX_GBP is a hard cap; proposals above it never auto-checkout.
#  5. Treasury must show REQUIRE_STREAK_DAYS consecutive net-positive days.
#  6. Enabling requires a confirm_token = sha256("enable-auto-checkout-"+install_ts).
AUTO_CHECKOUT_ENABLED = False
AUTO_CHECKOUT_DRY_RUN = True
AUTO_CHECKOUT_MAX_GBP = 750.0
AUTO_CHECKOUT_REQUIRE_STREAK_DAYS = 30

# config_kv keys
KV_AUTO_CHECKOUT_ENABLED = "auto_checkout_enabled"
KV_AUTO_CHECKOUT_DRY_RUN = "auto_checkout_dry_run"
KV_AUTO_CHECKOUT_MAX_GBP = "auto_checkout_max_gbp"
KV_INSTALL_TS = "install_ts"

# Treasury kinds that count as *realised P&L* for the positive-day streak.
REALISED_PNL_KINDS = ("trade_realised", "bet_settled")


def default_purchase_gate_path() -> str:
    """Return the default purchase-gate DB path (respects DATA_PATH)."""
    base = os.environ.get("DATA_PATH", "data")
    p = Path(base) / PURCHASE_GATE_DB_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def env_auto_checkout_enabled() -> bool | None:
    return _env_bool("DMAI_AUTO_CHECKOUT_ENABLED")


def env_auto_checkout_dry_run() -> bool | None:
    return _env_bool("DMAI_AUTO_CHECKOUT_DRY_RUN")


def env_auto_checkout_max_gbp() -> float | None:
    return _env_float("DMAI_AUTO_CHECKOUT_MAX_GBP")


def confirm_token(install_ts: str) -> str:
    """Token the operator must present to enable auto-checkout."""
    import hashlib
    return hashlib.sha256(
        ("enable-auto-checkout-" + str(install_ts)).encode()
    ).hexdigest()


__all__ = [
    "PURCHASE_GATE_DB_FILENAME",
    "OPERATOR_EMAIL",
    "SLACK_CHANNEL",
    "TRIGGER_MULTIPLIER",
    "PROPOSAL_POLL_INTERVAL_SECONDS",
    "AUTO_CHECKOUT_ENABLED",
    "AUTO_CHECKOUT_DRY_RUN",
    "AUTO_CHECKOUT_MAX_GBP",
    "AUTO_CHECKOUT_REQUIRE_STREAK_DAYS",
    "KV_AUTO_CHECKOUT_ENABLED",
    "KV_AUTO_CHECKOUT_DRY_RUN",
    "KV_AUTO_CHECKOUT_MAX_GBP",
    "KV_INSTALL_TS",
    "REALISED_PNL_KINDS",
    "default_purchase_gate_path",
    "env_auto_checkout_enabled",
    "env_auto_checkout_dry_run",
    "env_auto_checkout_max_gbp",
    "confirm_token",
]
