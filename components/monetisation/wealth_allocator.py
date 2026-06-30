"""
WealthAllocator: deploys David's 40% wealth share via aggressive growth basket.

Default basket (user choice "Aggressive growth"):
  - 60% ETFs (SPY 25, QQQ 25, IVV 10)
  - 40% Individual equities (NVDA 15, MSFT 13, AAPL 12)

Behaviour:
  - Watches david_wealth wallet; when balance >= deploy_threshold, allocates
    proportionally and routes orders through AggressiveTrader (paper unless
    TRADING_LIVE=true env var is set).
  - Each deployment is logged so you can audit every basket buy.
  - Does NOT auto-sell. Manual liquidation only.
"""
from __future__ import annotations
import json
import logging
import os
import sqlite3
from components.db import safe_open_kdb
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mon_wealth_deployments (
    id TEXT PRIMARY KEY,
    total_amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    basket_name TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    status TEXT NOT NULL,
    ts REAL NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_mon_wealth_ts ON mon_wealth_deployments(ts DESC);
"""

# Aggressive growth basket (locked by user choice)
DEFAULT_BASKET = {
    "name": "aggressive_growth",
    "weights": {
        # ETF core (60%)
        "SPY":  0.25,
        "QQQ":  0.25,
        "IVV":  0.10,
        # Individual equities (40%)
        "NVDA": 0.15,
        "MSFT": 0.13,
        "AAPL": 0.12,
    },
}


class WealthAllocator:
    def __init__(self, allocator, trader=None,
                 db_path: str = "data/dmai_knowledge.db",
                 basket: Optional[Dict[str, Any]] = None,
                 deploy_threshold: float = 100.0,
                 currency: str = "GBP"):
        self.allocator = allocator
        self.trader = trader  # AggressiveTrader (optional; if None, only logs)
        self.db_path = db_path
        self.basket = basket or DEFAULT_BASKET
        self.deploy_threshold = deploy_threshold
        self.currency = currency
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    def _conn(self):
        # Integrity check + quarantine removed (was destroying shared tables).
        c = safe_open_kdb(self.db_path, timeout=30.0)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with _LOCK, self._conn() as c:
            c.executescript(_SCHEMA)

    # ---- core ----

    def pending_amount(self) -> float:
        if not self.allocator:
            return 0.0
        return self.allocator.get_balance(self.allocator.DAVID_WALLET)

    def deploy(self, force: bool = False, amount: Optional[float] = None) -> Dict[str, Any]:
        """Deploy david_wealth balance (or a specified amount) into the basket."""
        if not self.allocator:
            return {"error": "allocator_unavailable"}
        available = self.pending_amount()
        target = amount if amount is not None else available
        if target <= 0:
            return {"status": "nothing_to_deploy", "balance": available}
        if not force and target < self.deploy_threshold:
            return {"status": "below_threshold", "balance": available,
                    "threshold": self.deploy_threshold}
        if target > available:
            return {"error": "insufficient_balance", "requested": target, "balance": available}

        breakdown = []
        for symbol, weight in self.basket["weights"].items():
            alloc = round(target * weight, 2)
            order_result = self._place_order(symbol, alloc)
            breakdown.append({"symbol": symbol, "weight": weight, "amount": alloc,
                              "order": order_result})

        # Debit the wealth wallet
        deployment_id = uuid.uuid4().hex[:12]
        self.allocator.debit(
            wallet=self.allocator.DAVID_WALLET, amount=target,
            reason=f"wealth_deploy:{self.basket['name']}:{deployment_id}",
        )
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mon_wealth_deployments (id, total_amount, currency, basket_name, "
                "breakdown_json, status, ts, notes) VALUES (?,?,?,?,?,?,?,?)",
                (deployment_id, target, self.currency, self.basket["name"],
                 json.dumps(breakdown), "deployed", time.time(),
                 f"trader_paper={getattr(self.trader, 'paper', True) if self.trader else 'no_trader'}"),
            )
        logger.info("WealthAllocator: deployed %s%.2f into %s basket (%d positions)",
                    self.currency, target, self.basket["name"], len(breakdown))
        return {
            "id": deployment_id,
            "total_amount": target,
            "basket": self.basket["name"],
            "breakdown": breakdown,
            "status": "deployed",
        }

    def _place_order(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Place a notional-amount buy through AggressiveTrader if available.
        Note: AggressiveTrader.execute_buy uses confidence-based sizing internally
        from account equity; here we just trigger a buy and log the intended
        notional. Paper-by-default is enforced inside AggressiveTrader."""
        if not self.trader:
            return {"status": "logged_only", "reason": "no_trader", "amount": amount}
        try:
            # Use a high-confidence signal (0.9) since this is portfolio rebalance,
            # not speculation. AggressiveTrader handles paper-mode enforcement.
            res = self.trader.execute_buy(symbol, confidence=0.9)
            return {"status": "ordered", "amount": amount, "trader_result": res,
                    "paper": getattr(self.trader, "paper", True)}
        except Exception as e:
            logger.warning("WealthAllocator._place_order failed for %s: %s", symbol, e)
            return {"status": "error", "amount": amount, "error": str(e)}

    # ---- config ----

    def set_basket(self, name: str, weights: Dict[str, float]) -> Dict[str, Any]:
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            return {"error": "weights_must_sum_to_1.0", "actual_total": total}
        self.basket = {"name": name, "weights": weights}
        return {"basket": self.basket}

    def get_basket(self) -> Dict[str, Any]:
        return dict(self.basket)

    # ---- reads ----

    def list_deployments(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM mon_wealth_deployments ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["breakdown"] = json.loads(d.pop("breakdown_json") or "[]")
            except Exception:
                d["breakdown"] = []
            out.append(d)
        return out

    def summary(self) -> Dict[str, Any]:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(total_amount),0) AS s FROM mon_wealth_deployments"
            ).fetchone()
        return {
            "basket": self.basket,
            "pending_balance": self.pending_amount(),
            "deploy_threshold": self.deploy_threshold,
            "lifetime_deployments": int(row["n"]),
            "lifetime_deployed": round(float(row["s"]), 2),
            "currency": self.currency,
            "trader_attached": self.trader is not None,
            "trader_paper": getattr(self.trader, "paper", None) if self.trader else None,
        }
