"""Treasury ledger — banked revenue and infra spend, reported in GBP.

PR I of the self-hosting roadmap. Reads realised P&L from the two
existing per-domain ledgers in ``data/dmai_ledger.db``:

* ``trades_ledger`` where ``mode='live'`` AND ``status='closed'``
  AND ``closed_at >= install_ts`` — reported in USD, converted to
  GBP using ``fx_rate_usd_gbp``.
* ``bets_ledger`` where ``outcome IN ('win','loss','void')`` AND
  ``settled_at >= install_ts`` — reported in GBP (no conversion).

Plus a manual ``infra_spend`` table for Render + OpenRouter monthly
bills that the user records by hand until those services expose
their own APIs.

All rows live in ``data/dmai_treasury.db`` — deliberately kept off
the knowledge DB and off the ledger DB so treasury writes never
contend with trading or bet-settling writes.

Zero-start rule: on first init, ``install_ts`` is stamped to
``now()``. Every realised P&L or bet-settle event dated earlier
than that is invisible to the treasury.

Sync strategy: the trades and bets ledgers are the source of
truth; the treasury just mirrors realised rows into
``treasury_ledger`` with idempotent inserts keyed on
``(source_table, source_id)``. That means we can call ``sync()``
as often as we like without double-counting.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────

TREASURY_DB_FILENAME = "dmai_treasury.db"

# Conservative default USD/GBP - overridden by admin PATCH or env.
# 2026-07 mid-market has been near 0.78 GBP per 1 USD; keep default
# pessimistic so a stale rate slightly under-counts USD revenue
# rather than over-counting it (safer for funding decisions).
DEFAULT_USD_TO_GBP = 0.77

# State keys
STATE_INSTALL_TS   = "treasury:install_ts"
STATE_FX_USD_GBP   = "treasury:fx_usd_gbp"
STATE_LAST_SYNC_TS = "treasury:last_sync_ts"


# ── Paths ─────────────────────────────────────────────────────────────────

def default_treasury_path() -> str:
    data_path = os.environ.get("DATA_PATH", "data/")
    p = Path(data_path) / TREASURY_DB_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def default_ledger_path() -> str:
    """Path to the trades / bets source ledger. Kept in sync with
    :func:`components.ledger.ledger_db.default_ledger_path` but we
    duplicate the constant instead of importing to avoid a
    hard-dependency on that module at treasury import time."""
    from components.ledger.ledger_db import default_ledger_path as _p
    return _p()


# ── Schema ────────────────────────────────────────────────────────────────

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS treasury_ledger (
        id             SERIAL PRIMARY KEY,
        ts             TEXT    NOT NULL,       -- event time (closed_at / settled_at / manual)
        kind           TEXT    NOT NULL CHECK (kind IN
                            ('trade_realised', 'bet_settled',
                             'infra_spend', 'manual_credit',
                             'manual_debit')),
        source_table   TEXT,                   -- 'trades_ledger' / 'bets_ledger' / 'manual'
        source_id      INTEGER,                -- id in the source table (nullable for manual)
        amount_gbp     REAL    NOT NULL,       -- signed; +revenue, -spend
        amount_native  REAL,                   -- pre-FX amount
        currency       TEXT    NOT NULL DEFAULT 'GBP',
        fx_rate        REAL,                   -- rate applied at conversion
        description    TEXT,
        created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
    )""",
    "CREATE INDEX IF NOT EXISTS ix_treasury_ts ON treasury_ledger(ts DESC)",
    "CREATE INDEX IF NOT EXISTS ix_treasury_kind ON treasury_ledger(kind)",
    # Idempotency: no duplicate mirror-rows for the same source row.
    """CREATE UNIQUE INDEX IF NOT EXISTS ux_treasury_source
       ON treasury_ledger(source_table, source_id)
       WHERE source_table IS NOT NULL AND source_id IS NOT NULL""",
    """CREATE TABLE IF NOT EXISTS treasury_state (
        key        TEXT PRIMARY KEY,
        value      TEXT,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )""",
]


# ── Connections ───────────────────────────────────────────────────────────

def _conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or default_treasury_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=30.0)
    c.row_factory = sqlite3.Row
    return c


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── State helpers ─────────────────────────────────────────────────────────

def _state_get(c: sqlite3.Connection, key: str) -> Optional[str]:
    row = c.execute(
        "SELECT value FROM treasury_state WHERE key = ?", (key,),
    ).fetchone()
    return row["value"] if row else None


def _state_set(c: sqlite3.Connection, key: str, value: str) -> None:
    c.execute(
        "INSERT INTO treasury_state (key, value, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = excluded.updated_at",
        (key, value, _now_iso()),
    )


# ── Init ──────────────────────────────────────────────────────────────────

def init_treasury_db(db_path: Optional[str] = None,
                     *,
                     install_ts: Optional[str] = None,
                     fx_usd_gbp: Optional[float] = None,
                     ) -> Dict[str, Any]:
    """Create schema (idempotent) and stamp install_ts on first run.

    Returns the effective state (install_ts + fx_usd_gbp) after the
    call. Callers may pre-supply ``install_ts`` (tests) or
    ``fx_usd_gbp`` (config); otherwise defaults are used.
    """
    with _conn(db_path) as c:
        for ddl in SCHEMA:
            c.execute(ddl)
        existing_install = _state_get(c, STATE_INSTALL_TS)
        if not existing_install:
            _state_set(c, STATE_INSTALL_TS, install_ts or _now_iso())
        existing_fx = _state_get(c, STATE_FX_USD_GBP)
        if not existing_fx:
            _state_set(c, STATE_FX_USD_GBP,
                       str(fx_usd_gbp
                           if fx_usd_gbp is not None
                           else DEFAULT_USD_TO_GBP))
        c.commit()
        return {
            "install_ts":  _state_get(c, STATE_INSTALL_TS),
            "fx_usd_gbp":  float(_state_get(c, STATE_FX_USD_GBP)
                                 or DEFAULT_USD_TO_GBP),
        }


# ── FX ────────────────────────────────────────────────────────────────────

def get_fx_usd_gbp(db_path: Optional[str] = None) -> float:
    with _conn(db_path) as c:
        raw = _state_get(c, STATE_FX_USD_GBP)
    try:
        return float(raw) if raw else DEFAULT_USD_TO_GBP
    except (TypeError, ValueError):
        return DEFAULT_USD_TO_GBP


def set_fx_usd_gbp(rate: float, db_path: Optional[str] = None) -> None:
    """Manual FX override. Non-positive rates are rejected."""
    r = float(rate)
    if not (r > 0):
        raise ValueError(f"fx_usd_gbp must be > 0, got {rate!r}")
    with _conn(db_path) as c:
        _state_set(c, STATE_FX_USD_GBP, str(r))
        c.commit()


def get_install_ts(db_path: Optional[str] = None) -> str:
    with _conn(db_path) as c:
        raw = _state_get(c, STATE_INSTALL_TS)
    return raw or _now_iso()


# ── Sync from source ledgers ──────────────────────────────────────────────

@dataclass
class SyncReport:
    trades_mirrored: int = 0
    bets_mirrored:   int = 0
    skipped_no_pnl:  int = 0
    fx_used:         float = 0.0
    install_ts:      str = ""
    balance_gbp:     float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trades_mirrored": self.trades_mirrored,
            "bets_mirrored":   self.bets_mirrored,
            "skipped_no_pnl":  self.skipped_no_pnl,
            "fx_used":         round(self.fx_used, 6),
            "install_ts":      self.install_ts,
            "balance_gbp":     round(self.balance_gbp, 2),
        }


def _open_source_ledger(ledger_path: Optional[str]) -> sqlite3.Connection:
    path = ledger_path or default_ledger_path()
    c = sqlite3.connect(path, timeout=15.0)
    c.row_factory = sqlite3.Row
    return c


def sync_from_ledger(*,
                     treasury_db_path: Optional[str] = None,
                     ledger_db_path:   Optional[str] = None,
                     ) -> SyncReport:
    """Idempotently mirror realised trades + settled bets into the
    treasury ledger. Returns a report + the current balance.
    """
    init_treasury_db(treasury_db_path)  # cheap; ensures schema
    install_ts = get_install_ts(treasury_db_path)
    fx         = get_fx_usd_gbp(treasury_db_path)
    report = SyncReport(fx_used=fx, install_ts=install_ts)

    # Trades — realised P&L in USD.
    try:
        with _open_source_ledger(ledger_db_path) as src:
            trade_rows = src.execute(
                "SELECT id, pnl, closed_at, symbol, mode, status "
                "FROM trades_ledger "
                "WHERE mode = 'live' AND status = 'closed' "
                "  AND closed_at IS NOT NULL "
                "  AND closed_at >= ? "
                "  AND pnl IS NOT NULL",
                (install_ts,),
            ).fetchall()
    except sqlite3.OperationalError:
        trade_rows = []

    # Bets — settled P&L in GBP.
    try:
        with _open_source_ledger(ledger_db_path) as src:
            bet_rows = src.execute(
                "SELECT id, pnl, settled_at, event, selection, outcome "
                "FROM bets_ledger "
                "WHERE outcome IN ('win','loss','void') "
                "  AND settled_at IS NOT NULL "
                "  AND settled_at >= ? "
                "  AND pnl IS NOT NULL",
                (install_ts,),
            ).fetchall()
    except sqlite3.OperationalError:
        bet_rows = []

    with _conn(treasury_db_path) as c:
        for r in trade_rows:
            pnl_usd = r["pnl"]
            if pnl_usd is None:
                report.skipped_no_pnl += 1
                continue
            pnl_gbp = float(pnl_usd) * fx
            cur = c.execute(
                "INSERT OR IGNORE INTO treasury_ledger "
                "(ts, kind, source_table, source_id, amount_gbp, "
                " amount_native, currency, fx_rate, description) "
                "VALUES (?, 'trade_realised', 'trades_ledger', ?, ?, "
                "        ?, 'USD', ?, ?)",
                (r["closed_at"], int(r["id"]), pnl_gbp, float(pnl_usd),
                 fx, f"{r['symbol']} live trade realised"),
            )
            if cur.rowcount:
                report.trades_mirrored += 1

        for r in bet_rows:
            pnl_gbp_raw = r["pnl"]
            if pnl_gbp_raw is None:
                report.skipped_no_pnl += 1
                continue
            cur = c.execute(
                "INSERT OR IGNORE INTO treasury_ledger "
                "(ts, kind, source_table, source_id, amount_gbp, "
                " amount_native, currency, fx_rate, description) "
                "VALUES (?, 'bet_settled', 'bets_ledger', ?, ?, "
                "        ?, 'GBP', 1.0, ?)",
                (r["settled_at"], int(r["id"]), float(pnl_gbp_raw),
                 float(pnl_gbp_raw),
                 f"{r['event']} / {r['selection']} settled {r['outcome']}"),
            )
            if cur.rowcount:
                report.bets_mirrored += 1

        _state_set(c, STATE_LAST_SYNC_TS, _now_iso())
        c.commit()
        report.balance_gbp = _balance(c)
    return report


# ── Manual entries (infra spend / credits / debits) ───────────────────────

def record_manual(*,
                  kind: str,
                  amount_gbp: float,
                  description: str = "",
                  ts: Optional[str] = None,
                  db_path: Optional[str] = None) -> int:
    """Record a manual credit or debit. ``kind`` must be one of
    ``infra_spend``, ``manual_credit``, ``manual_debit``. Amount is
    signed by the caller; the treasury does not flip signs itself
    (an ``infra_spend`` of +50 will *increase* the balance — the
    caller is expected to pass -50 for a Render bill).
    """
    if kind not in ("infra_spend", "manual_credit", "manual_debit"):
        raise ValueError(f"unsupported manual kind: {kind!r}")
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO treasury_ledger "
            "(ts, kind, source_table, amount_gbp, amount_native, "
            " currency, description) "
            "VALUES (?, ?, 'manual', ?, ?, 'GBP', ?)",
            (ts or _now_iso(), kind, float(amount_gbp),
             float(amount_gbp), description),
        )
        c.commit()
        return int(cur.lastrowid)


# ── Reporting ─────────────────────────────────────────────────────────────

def _balance(c: sqlite3.Connection) -> float:
    row = c.execute(
        "SELECT COALESCE(SUM(amount_gbp), 0.0) AS bal "
        "FROM treasury_ledger"
    ).fetchone()
    return float(row["bal"] or 0.0)


def get_balance(db_path: Optional[str] = None) -> float:
    with _conn(db_path) as c:
        return _balance(c)


def get_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    with _conn(db_path) as c:
        by_kind = {
            r["kind"]: {
                "count":    int(r["n"]),
                "total_gbp": round(float(r["s"] or 0.0), 2),
            }
            for r in c.execute(
                "SELECT kind, COUNT(*) AS n, SUM(amount_gbp) AS s "
                "FROM treasury_ledger GROUP BY kind"
            ).fetchall()
        }
        last_row = c.execute(
            "SELECT ts, kind, amount_gbp, description "
            "FROM treasury_ledger ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "balance_gbp":     round(_balance(c), 2),
            "install_ts":      _state_get(c, STATE_INSTALL_TS) or "",
            "last_sync_ts":    _state_get(c, STATE_LAST_SYNC_TS) or "",
            "fx_usd_gbp":      float(_state_get(c, STATE_FX_USD_GBP)
                                     or DEFAULT_USD_TO_GBP),
            "by_kind":         by_kind,
            "last_entry":      dict(last_row) if last_row else None,
        }


def list_entries(*, limit: int = 50, offset: int = 0,
                 kind: Optional[str] = None,
                 db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    q = "SELECT * FROM treasury_ledger"
    params: List[Any] = []
    if kind:
        q += " WHERE kind = ?"
        params.append(kind)
    q += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    with _conn(db_path) as c:
        rows = c.execute(q, params).fetchall()
    return [dict(r) for r in rows]


__all__ = [
    "TREASURY_DB_FILENAME",
    "DEFAULT_USD_TO_GBP",
    "STATE_INSTALL_TS",
    "STATE_FX_USD_GBP",
    "STATE_LAST_SYNC_TS",
    "SyncReport",
    "default_treasury_path",
    "init_treasury_db",
    "get_fx_usd_gbp",
    "set_fx_usd_gbp",
    "get_install_ts",
    "sync_from_ledger",
    "record_manual",
    "get_balance",
    "get_summary",
    "list_entries",
]
