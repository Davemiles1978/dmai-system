"""DDL for the purchase-gate DB (PR L)."""
from __future__ import annotations

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS purchase_proposals (
        id                        SERIAL PRIMARY KEY,
        created_ts                TEXT    NOT NULL,
        hardware_id               INTEGER NOT NULL,
        hw_name                   TEXT    NOT NULL,
        hw_source                 TEXT    NOT NULL,
        hw_url                    TEXT    NOT NULL,
        capex_gbp                 REAL    NOT NULL,
        tco_gbp_3yr               REAL    NOT NULL,
        treasury_at_proposal_gbp  REAL    NOT NULL,
        trigger_multiplier        REAL    NOT NULL,
        state                     TEXT    NOT NULL DEFAULT 'pending',
        operator_action_ts        TEXT,
        operator_note             TEXT,
        actual_price_gbp          REAL,
        auto_debit_entry_id       INTEGER,
        cancel_credit_entry_id    INTEGER,
        reconcile_entry_id        INTEGER,
        channels_notified         TEXT,
        auto_checkout_attempted   INTEGER DEFAULT 0,
        auto_checkout_result      TEXT,
        notes                     TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_prop_state ON purchase_proposals(state)",
    """CREATE TABLE IF NOT EXISTS config_kv (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_ts TEXT NOT NULL
    )""",
]

__all__ = ["SCHEMA"]
