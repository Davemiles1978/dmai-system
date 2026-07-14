"""SQLite schema for the procurement DB (PR K).

Two tables, kept in an isolated ``data/dmai_procurement.db`` file so
procurement writes never contend with knowledge / ledger / treasury /
workload DBs:

* ``hardware_catalog`` — normalised candidate boxes/CPUs, one row per
  (source, url/name), priced in GBP with the original currency + FX kept
  for provenance.
* ``procurement_shortlist`` — the ranked output of one research run.
  ``hardware_id`` references ``hardware_catalog(id)``.
"""
from __future__ import annotations

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS hardware_catalog (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        source         TEXT    NOT NULL,
        url            TEXT,
        name           TEXT    NOT NULL,
        cpu            TEXT,
        cpu_passmark   INTEGER,
        tdp_w          REAL,
        idle_w         REAL,
        ram_gb         INTEGER,
        storage_gb     INTEGER,
        price_gbp      REAL,
        currency_orig  TEXT,
        price_orig     REAL,
        fx_used        REAL,
        fetched_ts     TEXT    NOT NULL,
        raw_json       TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS ix_catalog_source ON hardware_catalog(source)",
    "CREATE INDEX IF NOT EXISTS ix_catalog_name ON hardware_catalog(name)",
    """CREATE TABLE IF NOT EXISTS procurement_shortlist (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        run_ts         TEXT    NOT NULL,
        rank           INTEGER NOT NULL,
        hardware_id    INTEGER NOT NULL
                           REFERENCES hardware_catalog(id),
        tco_gbp_3yr    REAL    NOT NULL,
        capex_gbp      REAL    NOT NULL,
        opex_3yr_gbp   REAL    NOT NULL,
        headroom_ram_x REAL,
        headroom_cpu_x REAL,
        verdict        TEXT    NOT NULL CHECK (verdict IN
                           ('affordable', 'stretch', 'aspirational')),
        notes          TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS ix_shortlist_run ON procurement_shortlist(run_ts)",
    """CREATE TABLE IF NOT EXISTS procurement_state (
        key        TEXT PRIMARY KEY,
        value      TEXT,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )""",
]

__all__ = ["SCHEMA"]
