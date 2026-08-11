"""
BillPayer: auto-pays recurring DMAI operating expenses from the 60% wallet.

Categories (locked by user):
  - infrastructure   (Render, domains, SSL)
  - ai_credits       (OpenRouter, Cerebras, Groq, etc.)
  - data_apis        (market data, news, sports odds)
  - hardware_reserve (accumulating pot for Mac/GPU upgrades)

Notes:
  - This component does NOT actually move money outside DMAI; it debits the
    internal dmai_operating wallet and credits an internal sub-ledger that tracks
    what has been allocated/paid. Real provider top-ups remain manual until you
    explicitly wire payment APIs (Stripe/Render billing API/etc.).
  - Hardware reserve is a virtual sub-wallet that accumulates monthly.
"""
from __future__ import annotations
import json
import logging
import os
from components.db import safe_open_kdb  # was sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mon_bills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    cadence TEXT NOT NULL DEFAULT 'monthly',
    next_due REAL,
    auto_pay INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mon_bill_payments (
    id SERIAL PRIMARY KEY,
    bill_id TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    ts REAL NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_mon_bills_active ON mon_bills(active, next_due);
"""

DEFAULT_BILLS = [
    # name,                   category,            amount, cadence
    ("Render hosting",        "infrastructure",     19.00, "monthly"),
    ("Domain renewal",        "infrastructure",      1.00, "monthly"),  # ~12/yr amortised
    ("OpenRouter credits",    "ai_credits",         25.00, "monthly"),
    ("Cerebras credits",      "ai_credits",         10.00, "monthly"),
    ("Market data API",       "data_apis",          15.00, "monthly"),
    ("News/sentiment API",    "data_apis",          10.00, "monthly"),
    ("Hardware reserve",      "hardware_reserve",   50.00, "monthly"),
]

_CADENCE_SECONDS = {
    "daily":   86400,
    "weekly":  86400 * 7,
    "monthly": 86400 * 30,
    "yearly":  86400 * 365,
}


class BillPayer:
    def __init__(self, allocator, db_path: str = "data/dmai_knowledge.db",
                 currency: str = "GBP"):
        self.allocator = allocator
        self.db_path = db_path
        self.currency = currency
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()
        self._seed_defaults()

    def _conn(self):
        # Integrity check + quarantine removed: it was destroying shared tables
        # (at_state, capabilities, system_state) created by boot bootstrap.
        # Boot bootstrap is now authoritative; per-component _ensure_tables
        # handles missing tables on demand.
        c = safe_open_kdb(self.db_path, timeout=30.0)

        
        return c

    def _init_schema(self):
        with _LOCK, self._conn() as c:
            c.executescript(_SCHEMA)

    def _seed_defaults(self):
        with self._conn() as c:
            existing = c.execute("SELECT COUNT(*) AS n FROM mon_bills").fetchone()["n"]
        if existing:
            return
        now = time.time()
        for name, cat, amt, cad in DEFAULT_BILLS:
            self.add_bill(name, cat, amt, cadence=cad, auto_pay=True,
                          next_due=now + _CADENCE_SECONDS.get(cad, 86400 * 30))
        logger.info("BillPayer: seeded %d default bills", len(DEFAULT_BILLS))

    # ---- bill management ----

    def add_bill(self, name: str, category: str, amount: float, *,
                 cadence: str = "monthly", auto_pay: bool = True,
                 next_due: Optional[float] = None) -> Dict[str, Any]:
        bid = uuid.uuid4().hex[:12]
        nd = next_due if next_due is not None else (time.time() + _CADENCE_SECONDS.get(cadence, 86400 * 30))
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mon_bills (id, name, category, amount, currency, cadence, next_due, auto_pay, active, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (bid, name, category, amount, self.currency, cadence, nd, int(auto_pay), 1, time.time()),
            )
        return {"id": bid, "name": name, "category": category, "amount": amount,
                "cadence": cadence, "next_due": nd}

    def update_bill(self, bill_id: str, **fields) -> bool:
        allowed = {"name", "category", "amount", "cadence", "next_due", "auto_pay", "active"}
        sets = []
        vals = []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(int(v) if k in ("auto_pay", "active") else v)
        if not sets:
            return False
        vals.append(bill_id)
        with _LOCK, self._conn() as c:
            c.execute(f"UPDATE mon_bills SET {', '.join(sets)} WHERE id=?", vals)
        return True

    def list_bills(self, active_only: bool = True) -> List[Dict[str, Any]]:
        with self._conn() as c:
            q = "SELECT * FROM mon_bills"
            args: tuple = ()
            if active_only:
                q += " WHERE active=1"
            q += " ORDER BY next_due ASC"
            rows = c.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    # ---- payment engine ----

    def due_bills(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM mon_bills WHERE active=1 AND auto_pay=1 AND next_due<=?",
                (now,),
            ).fetchall()
        return [dict(r) for r in rows]

    def pay_due(self) -> Dict[str, Any]:
        """Auto-pay all currently-due bills from the DMAI operating wallet."""
        results = {"paid": [], "skipped_insufficient": [], "errors": []}
        for bill in self.due_bills():
            res = self._pay_bill(bill)
            if res.get("status") == "paid":
                results["paid"].append(res)
            elif res.get("status") == "insufficient_funds":
                results["skipped_insufficient"].append(res)
            else:
                results["errors"].append(res)
        return results

    def _pay_bill(self, bill: Dict[str, Any]) -> Dict[str, Any]:
        try:
            debit = self.allocator.debit(
                wallet=self.allocator.DMAI_WALLET,
                amount=float(bill["amount"]),
                reason=f"bill_pay:{bill['category']}:{bill['name']}",
            )
            if "error" in debit:
                self._record_payment(bill["id"], bill["amount"], "insufficient_funds", debit.get("error"))
                return {"status": "insufficient_funds", "bill": bill["name"], **debit}
            # advance next_due
            new_due = bill["next_due"] + _CADENCE_SECONDS.get(bill.get("cadence", "monthly"), 86400 * 30)
            with _LOCK, self._conn() as c:
                c.execute("UPDATE mon_bills SET next_due=? WHERE id=?", (new_due, bill["id"]))
            self._record_payment(bill["id"], bill["amount"], "paid",
                                 f"category={bill['category']}; balance_after={debit['balance_after']}")
            return {"status": "paid", "bill": bill["name"], "amount": bill["amount"],
                    "category": bill["category"], "next_due": new_due,
                    "balance_after": debit["balance_after"]}
        except Exception as e:
            logger.exception("BillPayer._pay_bill failed for %s", bill.get("name"))
            self._record_payment(bill["id"], bill["amount"], "error", str(e))
            return {"status": "error", "bill": bill.get("name"), "error": str(e)}

    def _record_payment(self, bill_id: str, amount: float, status: str, notes: str = ""):
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mon_bill_payments (bill_id, amount, status, ts, notes) VALUES (?,?,?,?,?)",
                (bill_id, amount, status, time.time(), notes),
            )

    def payment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT bp.*, b.name AS bill_name, b.category AS bill_category "
                "FROM mon_bill_payments bp LEFT JOIN mon_bills b ON b.id=bp.bill_id "
                "ORDER BY bp.ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> Dict[str, Any]:
        bills = self.list_bills(active_only=True)
        by_cat: Dict[str, float] = {}
        monthly_total = 0.0
        for b in bills:
            cad_factor = {"daily": 30, "weekly": 4.33, "monthly": 1, "yearly": 1 / 12}.get(b["cadence"], 1)
            monthly = float(b["amount"]) * cad_factor
            by_cat[b["category"]] = round(by_cat.get(b["category"], 0.0) + monthly, 2)
            monthly_total += monthly
        return {
            "active_bill_count": len(bills),
            "monthly_total": round(monthly_total, 2),
            "by_category": by_cat,
            "next_due": bills[0]["next_due"] if bills else None,
            "currency": self.currency,
        }
