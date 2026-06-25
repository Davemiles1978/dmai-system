"""
RevenueAllocator: credits every income event, splits 60/40, audits to SQLite.

Wallets:
  - dmai_operating  (60%): pays bills/subscriptions/hardware reserve
  - david_wealth    (40%): deployed via WealthAllocator

Schema (data/dmai_knowledge.db):
  mon_income_events   (id, source, amount, currency, ts, raw_json)
  mon_wallet_ledger   (id, wallet, delta, balance_after, event_id, reason, ts)
  mon_wallets         (name PK, balance, currency, updated_at)
"""
from __future__ import annotations
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()

DMAI_SPLIT = 0.60
DAVID_SPLIT = 0.40

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mon_wallets (
    name TEXT PRIMARY KEY,
    balance REAL NOT NULL DEFAULT 0.0,
    currency TEXT NOT NULL DEFAULT 'GBP',
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mon_income_events (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    ts REAL NOT NULL,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS mon_wallet_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet TEXT NOT NULL,
    delta REAL NOT NULL,
    balance_after REAL NOT NULL,
    event_id TEXT,
    reason TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mon_ledger_wallet ON mon_wallet_ledger(wallet, ts DESC);
"""


class RevenueAllocator:
    """Splits every income event 60/40 between DMAI and David wallets."""

    DMAI_WALLET = "dmai_operating"
    DAVID_WALLET = "david_wealth"

    def __init__(self, db_path: str = "data/dmai_knowledge.db", currency: str = "GBP"):
        self.db_path = db_path
        self.currency = currency
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()
        self._ensure_wallet(self.DMAI_WALLET)
        self._ensure_wallet(self.DAVID_WALLET)

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with _LOCK, self._conn() as c:
            c.executescript(_SCHEMA)

    def _ensure_wallet(self, name: str):
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO mon_wallets (name, balance, currency, updated_at) VALUES (?,?,?,?)",
                (name, 0.0, self.currency, time.time()),
            )

    # ---- public API ----

    def credit_income(self, source: str, amount: float, *,
                      currency: str = "GBP", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Credit a new income event. Splits 60/40 and returns the result."""
        if amount <= 0:
            return {"error": "amount must be positive"}
        event_id = uuid.uuid4().hex[:16]
        dmai_share = round(amount * DMAI_SPLIT, 2)
        david_share = round(amount - dmai_share, 2)  # avoid rounding drift
        ts = time.time()

        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mon_income_events (id, source, amount, currency, ts, raw_json) VALUES (?,?,?,?,?,?)",
                (event_id, source, amount, currency, ts, json.dumps(metadata or {})),
            )
            for wallet, delta, label in [
                (self.DMAI_WALLET, dmai_share, f"60% split from {source}"),
                (self.DAVID_WALLET, david_share, f"40% split from {source}"),
            ]:
                self._apply_delta_locked(c, wallet, delta, event_id, label, ts)

        logger.info("RevenueAllocator: credited %s %.2f from %s (DMAI %.2f / David %.2f)",
                    currency, amount, source, dmai_share, david_share)
        return {
            "event_id": event_id,
            "source": source,
            "amount": amount,
            "currency": currency,
            "dmai_share": dmai_share,
            "david_share": david_share,
            "ts": ts,
        }

    def debit(self, wallet: str, amount: float, reason: str,
              event_id: Optional[str] = None) -> Dict[str, Any]:
        """Debit a wallet (used by BillPayer and WealthAllocator)."""
        if amount <= 0:
            return {"error": "amount must be positive"}
        bal = self.get_balance(wallet)
        if bal < amount:
            return {"error": "insufficient_funds", "balance": bal, "requested": amount}
        ts = time.time()
        with _LOCK, self._conn() as c:
            new_bal = self._apply_delta_locked(c, wallet, -amount, event_id, reason, ts)
        return {"wallet": wallet, "debit": amount, "balance_after": new_bal, "ts": ts}

    def _apply_delta_locked(self, c: sqlite3.Connection, wallet: str, delta: float,
                            event_id: Optional[str], reason: str, ts: float) -> float:
        row = c.execute("SELECT balance FROM mon_wallets WHERE name=?", (wallet,)).fetchone()
        cur = float(row["balance"]) if row else 0.0
        new_bal = round(cur + delta, 2)
        c.execute("UPDATE mon_wallets SET balance=?, updated_at=? WHERE name=?", (new_bal, ts, wallet))
        c.execute(
            "INSERT INTO mon_wallet_ledger (wallet, delta, balance_after, event_id, reason, ts) VALUES (?,?,?,?,?,?)",
            (wallet, delta, new_bal, event_id, reason, ts),
        )
        return new_bal

    # ---- reads ----

    def get_balance(self, wallet: str) -> float:
        try:
            with self._conn() as c:
                row = c.execute("SELECT balance FROM mon_wallets WHERE name=?", (wallet,)).fetchone()
            return float(row["balance"]) if row else 0.0
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                logger.warning("mon_wallets missing — re-creating schema")
                self._init_schema()
                self._ensure_wallet(self.DMAI_WALLET)
                self._ensure_wallet(self.DAVID_WALLET)
                return 0.0
            raise

    def get_wallets(self) -> List[Dict[str, Any]]:
        try:
            with self._conn() as c:
                rows = c.execute("SELECT name, balance, currency, updated_at FROM mon_wallets").fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                logger.warning("mon_wallets missing — re-creating schema")
                self._init_schema()
                self._ensure_wallet(self.DMAI_WALLET)
                self._ensure_wallet(self.DAVID_WALLET)
                return []
            raise

    def get_ledger(self, wallet: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as c:
            if wallet:
                rows = c.execute(
                    "SELECT * FROM mon_wallet_ledger WHERE wallet=? ORDER BY ts DESC LIMIT ?",
                    (wallet, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM mon_wallet_ledger ORDER BY ts DESC LIMIT ?", (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_income_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM mon_income_events ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.pop("raw_json") or "{}")
            except Exception:
                d["metadata"] = {}
            out.append(d)
        return out

    def get_summary(self) -> Dict[str, Any]:
        wallets = {w["name"]: w["balance"] for w in self.get_wallets()}
        with self._conn() as c:
            total_in = c.execute("SELECT COALESCE(SUM(amount),0) AS s FROM mon_income_events").fetchone()["s"]
            total_out = c.execute("SELECT COALESCE(SUM(-delta),0) AS s FROM mon_wallet_ledger WHERE delta<0").fetchone()["s"]
        return {
            "wallets": wallets,
            "lifetime_income": round(float(total_in), 2),
            "lifetime_outflow": round(float(total_out), 2),
            "currency": self.currency,
            "split_policy": {"dmai": DMAI_SPLIT, "david": DAVID_SPLIT},
        }
