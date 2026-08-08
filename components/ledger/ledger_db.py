"""Isolated performance ledger for DMAI trades and greyhound bets.

Why a separate file (`data/dmai_ledger.db`) and NOT `dmai_knowledge.db`:
PRs #150–#164 peeled writers off the shared knowledge DB to end the
write-mutex/`database is locked` storm. Adding two new high-frequency writers
(trader + tipster) back onto that file would undo that work, so the ledger
lives in its own SQLite file. Connections still go through
``components.db.safe_open_kdb`` so they inherit the exact same per-connection
PRAGMAs (WAL, busy_timeout=30000, foreign_keys ON, synchronous=NORMAL) and the
per-path write mutex — but keyed on a different path, so ledger writes never
contend with knowledge-DB writes.

Two tables:
  trades_ledger  — one row per autonomous-trader position (open → closed).
  bets_ledger    — one row per greyhound tip (manual: user places + settles).

All money is single-currency per row (USD for trades, GBP for bets); no FX.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

LEDGER_DB_FILENAME = "dmai_ledger.db"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS trades_ledger (
        id             SERIAL PRIMARY KEY,
        symbol         TEXT NOT NULL,
        side           TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
        qty            REAL NOT NULL,
        entry_price    REAL,
        exit_price     REAL,
        stake          REAL,
        pnl            REAL,
        mode           TEXT NOT NULL CHECK (mode IN ('paper', 'live')),
        status         TEXT NOT NULL CHECK (status IN ('open', 'closed', 'cancelled', 'error')),
        opened_at      TEXT NOT NULL,
        closed_at      TEXT,
        source         TEXT NOT NULL DEFAULT 'autonomous_trader',
        confidence     REAL,
        notes          TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS ix_trades_opened ON trades_ledger(opened_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_trades_mode_status ON trades_ledger(mode, status)",
    """CREATE TABLE IF NOT EXISTS bets_ledger (
        id             SERIAL PRIMARY KEY,
        event          TEXT NOT NULL,
        selection      TEXT NOT NULL,
        odds           REAL,
        stake          REAL,
        outcome        TEXT CHECK (outcome IN ('win', 'loss', 'void', 'pending', NULL)),
        pnl            REAL,
        tipped_at      TEXT NOT NULL,
        placed_at      TEXT,
        settled_at     TEXT,
        source         TEXT NOT NULL DEFAULT 'greyhound_runner',
        ev             REAL,
        confidence     REAL,
        notes          TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS ix_bets_tipped ON bets_ledger(tipped_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_bets_outcome ON bets_ledger(outcome)",
]


# ── path + connection ─────────────────────────────────────────────────────────
def default_ledger_path() -> str:
    """`<DATA_PATH>/dmai_ledger.db`, ensuring the parent dir exists."""
    data_path = os.environ.get("DATA_PATH", "data/")
    p = Path(data_path) / LEDGER_DB_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or default_ledger_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = safe_open_kdb(path, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_ledger_db(db_path: Optional[str] = None) -> None:
    """Create tables + indexes if absent. Idempotent — safe to call repeatedly."""
    with _conn(db_path) as c:
        for ddl in SCHEMA:
            c.execute(ddl)
        c.commit()


# ── P&L helper ──────────────────────────────────────────────────────────────
def compute_bet_pnl(odds: Optional[float], stake: Optional[float],
                    outcome: Optional[str]) -> Optional[float]:
    """(odds-1)*stake for a win, -stake for a loss, 0 for void, else None."""
    if outcome == "void":
        return 0.0
    if stake is None or outcome in (None, "pending"):
        return None
    try:
        stake_f = float(stake)
    except (TypeError, ValueError):
        return None
    if outcome == "win":
        if odds is None:
            return None
        return (float(odds) - 1.0) * stake_f
    if outcome == "loss":
        return -stake_f
    return None


# ── trades ────────────────────────────────────────────────────────────────────
def insert_trade(*, symbol: str, side: str, qty: float, mode: str,
                 entry_price: Optional[float] = None,
                 stake: Optional[float] = None,
                 confidence: Optional[float] = None,
                 status: str = "open",
                 opened_at: Optional[str] = None,
                 source: str = "autonomous_trader",
                 notes: Optional[str] = None,
                 db_path: Optional[str] = None) -> int:
    """Insert a trade row; returns its id."""
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO trades_ledger(symbol, side, qty, entry_price, stake, "
            "mode, status, opened_at, source, confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, side, qty, entry_price, stake, mode, status,
             opened_at or _now_iso(), source, confidence, notes),
        )
        c.commit()
        return int(cur.lastrowid)


def close_trade(trade_id: int, *, exit_price: Optional[float] = None,
                pnl: Optional[float] = None,
                closed_at: Optional[str] = None,
                status: str = "closed",
                notes: Optional[str] = None,
                db_path: Optional[str] = None) -> None:
    """Mark a trade closed/cancelled/errored, recording exit price + P&L."""
    with _conn(db_path) as c:
        c.execute(
            "UPDATE trades_ledger SET exit_price = ?, pnl = ?, closed_at = ?, "
            "status = ?, notes = COALESCE(?, notes) WHERE id = ?",
            (exit_price, pnl, closed_at or _now_iso(), status, notes, trade_id),
        )
        c.commit()


def close_open_trade_for_symbol(symbol: str, *, exit_price: Optional[float],
                                pnl: Optional[float],
                                closed_at: Optional[str] = None,
                                notes: Optional[str] = None,
                                db_path: Optional[str] = None) -> Optional[int]:
    """Close the most recent still-open trade for ``symbol``. Returns its id
    (or None when there's no open row to match)."""
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT id FROM trades_ledger WHERE symbol = ? AND status = 'open' "
            "ORDER BY id DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if not row:
            return None
        tid = int(row["id"])
        c.execute(
            "UPDATE trades_ledger SET exit_price = ?, pnl = ?, closed_at = ?, "
            "status = 'closed', notes = COALESCE(?, notes) WHERE id = ?",
            (exit_price, pnl, closed_at or _now_iso(), notes, tid),
        )
        c.commit()
        return tid


def get_trade(trade_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM trades_ledger WHERE id = ?", (trade_id,)
        ).fetchone()
    return dict(row) if row else None


def list_trades(*, mode: Optional[str] = None, status: Optional[str] = None,
                limit: int = 100, offset: int = 0,
                db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    clauses, params = [], []
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.extend([int(limit), int(offset)])
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM trades_ledger" + where +
            " ORDER BY id DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


# ── bets ────────────────────────────────────────────────────────────────────
def insert_bet(*, event: str, selection: str,
               odds: Optional[float] = None,
               stake: Optional[float] = None,
               outcome: Optional[str] = "pending",
               pnl: Optional[float] = None,
               tipped_at: Optional[str] = None,
               placed_at: Optional[str] = None,
               settled_at: Optional[str] = None,
               source: str = "greyhound_runner",
               ev: Optional[float] = None,
               confidence: Optional[float] = None,
               notes: Optional[str] = None,
               db_path: Optional[str] = None) -> int:
    """Insert a bet row; returns its id."""
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO bets_ledger(event, selection, odds, stake, outcome, "
            "pnl, tipped_at, placed_at, settled_at, source, ev, confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event, selection, odds, stake, outcome, pnl,
             tipped_at or _now_iso(), placed_at, settled_at, source,
             ev, confidence, notes),
        )
        c.commit()
        return int(cur.lastrowid)


def update_bet(bet_id: int, *, stake: Optional[float] = None,
               outcome: Optional[str] = None,
               placed_at: Optional[str] = None,
               settled_at: Optional[str] = None,
               notes: Optional[str] = None,
               db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Apply a user-supplied bet outcome. P&L is recomputed server-side from the
    stored odds and the new stake/outcome — never trusted from the caller."""
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM bets_ledger WHERE id = ?", (bet_id,)
        ).fetchone()
        if not row:
            return None
        cur = dict(row)
        new_stake = cur["stake"] if stake is None else stake
        new_outcome = cur["outcome"] if outcome is None else outcome
        pnl = compute_bet_pnl(cur["odds"], new_stake, new_outcome)
        c.execute(
            "UPDATE bets_ledger SET stake = ?, outcome = ?, pnl = ?, "
            "placed_at = COALESCE(?, placed_at), "
            "settled_at = COALESCE(?, settled_at), "
            "notes = COALESCE(?, notes) WHERE id = ?",
            (new_stake, new_outcome, pnl, placed_at, settled_at, notes, bet_id),
        )
        c.commit()
        row = c.execute(
            "SELECT * FROM bets_ledger WHERE id = ?", (bet_id,)
        ).fetchone()
    return dict(row) if row else None


def get_bet(bet_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM bets_ledger WHERE id = ?", (bet_id,)
        ).fetchone()
    return dict(row) if row else None


def list_bets(*, outcome: Optional[str] = None, limit: int = 100, offset: int = 0,
              db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    clauses, params = [], []
    if outcome:
        clauses.append("outcome = ?")
        params.append(outcome)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.extend([int(limit), int(offset)])
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM bets_ledger" + where +
            " ORDER BY id DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


# ── summary ────────────────────────────────────────────────────────────────
def summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate stats across both ledgers for the dashboard."""
    with _conn(db_path) as c:
        trades_by_mode = {
            r["mode"]: int(r["n"])
            for r in c.execute(
                "SELECT mode, COUNT(*) AS n FROM trades_ledger GROUP BY mode"
            ).fetchall()
        }
        closed = c.execute(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins, "
            "COALESCE(SUM(pnl), 0) AS pnl "
            "FROM trades_ledger WHERE status = 'closed'"
        ).fetchone()
        bets_by_outcome = {
            (r["outcome"] or "none"): int(r["n"])
            for r in c.execute(
                "SELECT outcome, COUNT(*) AS n FROM bets_ledger GROUP BY outcome"
            ).fetchall()
        }
        bet_agg = c.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS pnl, AVG(odds) AS avg_odds "
            "FROM bets_ledger"
        ).fetchone()
        settled_bets = c.execute(
            "SELECT outcome FROM bets_ledger "
            "WHERE outcome IN ('win', 'loss') ORDER BY id ASC"
        ).fetchall()

    closed_n = int(closed["n"] or 0)
    wins = int(closed["wins"] or 0)
    longest_streak = _longest_win_streak([r["outcome"] for r in settled_bets])

    return {
        "trades": {
            "total": sum(trades_by_mode.values()),
            "by_mode": trades_by_mode,
            "closed": closed_n,
            "win_rate": (wins / closed_n) if closed_n else 0.0,
            "total_pnl": float(closed["pnl"] or 0.0),
        },
        "bets": {
            "total": sum(bets_by_outcome.values()),
            "by_outcome": bets_by_outcome,
            "total_pnl": float(bet_agg["pnl"] or 0.0),
            "avg_odds": float(bet_agg["avg_odds"]) if bet_agg["avg_odds"] is not None else None,
            "longest_win_streak": longest_streak,
        },
    }


def _longest_win_streak(outcomes: List[Optional[str]]) -> int:
    best = cur = 0
    for o in outcomes:
        if o == "win":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
