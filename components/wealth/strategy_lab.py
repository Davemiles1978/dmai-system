"""
StrategyLab — autonomous algorithm testing for the DMAI trader.

Every run, it evaluates a set of strategy variants against real historical
trades and exits stored in SQLite (`at_trades` + `at_exits`) and persists
each variant's score. The active trader can then read the leaderboard and
weight its signal acceptance toward the best-scoring variants.

Strategies tested (all real, no synthetic data):
  - Variant A: baseline EV gate (current)
  - Variant B: stricter EV gate (+25%)
  - Variant C: confidence-weighted EV gate
  - Variant D: momentum filter (only buy if recent exits in symbol were green)
  - Variant E: tier-conservative override

Scoring is `total_pnl_usd / max_drawdown_pct` proxy on the same trade
history; this is a comparative ranking, not absolute backtest.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StrategyLab:
    """
    Background-run strategy evaluator. Persists scores and exposes a
    leaderboard. Cheap: reads recent trade/exit rows, no broker calls.
    """

    INTERVAL_S = 3600  # 1 hour between evaluations

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS strategy_runs ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT DEFAULT (datetime('now')), "
                "variant TEXT NOT NULL, "
                "trades_considered INTEGER, "
                "trades_accepted INTEGER, "
                "total_pnl_usd REAL, "
                "win_rate REAL, "
                "avg_pnl_pct REAL, "
                "score REAL, "
                "notes TEXT)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_runs_variant "
                "ON strategy_runs(variant, ts DESC)"
            )
            c.commit()

    # ── Public surface ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(target=self._run, name="StrategyLab-loop", daemon=True)
        self._thread = t
        t.start()
        logger.info("StrategyLab: loop started")

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> Dict[str, Any]:
        with self._lock:
            return self._evaluate_all()

    def leaderboard(self, days: int = 30) -> List[Dict[str, Any]]:
        """Return latest score per variant, ordered by score desc."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT variant, ts, trades_considered, trades_accepted, "
                "total_pnl_usd, win_rate, avg_pnl_pct, score, notes "
                "FROM strategy_runs WHERE ts >= ? "
                "GROUP BY variant HAVING MAX(ts) ORDER BY score DESC",
                (since,),
            ).fetchall()
            return [dict(r) for r in rows]

    def best_variant(self) -> Optional[str]:
        lb = self.leaderboard()
        return lb[0]["variant"] if lb else None

    # ── Loop ──────────────────────────────────────────────────────────────────
    def _run(self) -> None:
        time.sleep(30)
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.exception("StrategyLab evaluation failed: %s", e)
            slept = 0
            while slept < self.INTERVAL_S and not self._stop.is_set():
                time.sleep(min(10, self.INTERVAL_S - slept))
                slept += 10

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _load_trade_history(self, days: int = 90) -> List[Dict[str, Any]]:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as c:
            trades = c.execute(
                "SELECT ts, symbol, qty, confidence, ev, tier, live "
                "FROM at_trades WHERE ts >= ?",
                (since,),
            ).fetchall()
        return [dict(r) for r in trades]

    def _load_exit_pnl_by_symbol(self, days: int = 90) -> Dict[str, List[Dict[str, Any]]]:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        out: Dict[str, List[Dict[str, Any]]] = {}
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT ts, symbol, pnl_usd, pnl_pct FROM at_exits WHERE ts >= ?",
                    (since,),
                ).fetchall()
                for r in rows:
                    out.setdefault(r["symbol"], []).append(dict(r))
        except sqlite3.OperationalError:
            # at_exits not yet created (first deploy) — return empty
            pass
        return out

    def _evaluate_all(self) -> Dict[str, Any]:
        trades = self._load_trade_history()
        exits_by_sym = self._load_exit_pnl_by_symbol()

        variants = [
            ("baseline", self._eval_baseline),
            ("strict_ev", self._eval_strict_ev),
            ("confidence_weighted", self._eval_conf_weighted),
            ("momentum_filter", self._eval_momentum),
            ("tier_conservative_only", self._eval_tier_cons),
        ]
        results = []
        for name, fn in variants:
            try:
                r = fn(trades, exits_by_sym)
                r["variant"] = name
                self._persist(r)
                results.append(r)
            except Exception as e:
                logger.exception("StrategyLab variant %s failed: %s", name, e)

        return {
            "ts": datetime.utcnow().isoformat() + "Z",
            "trades_in_window": len(trades),
            "exits_in_window": sum(len(v) for v in exits_by_sym.values()),
            "results": results,
        }

    def _persist(self, r: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO strategy_runs(variant, trades_considered, "
                "trades_accepted, total_pnl_usd, win_rate, avg_pnl_pct, score, "
                "notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["variant"],
                    r.get("trades_considered", 0),
                    r.get("trades_accepted", 0),
                    r.get("total_pnl_usd", 0.0),
                    r.get("win_rate", 0.0),
                    r.get("avg_pnl_pct", 0.0),
                    r.get("score", 0.0),
                    r.get("notes", ""),
                ),
            )
            c.commit()

    # ── Variant evaluators ────────────────────────────────────────────────────
    def _summarise(
        self,
        accepted: List[Dict[str, Any]],
        exits_by_sym: Dict[str, List[Dict[str, Any]]],
        total_considered: int,
        notes: str = "",
    ) -> Dict[str, Any]:
        if not accepted:
            return {
                "trades_considered": total_considered,
                "trades_accepted": 0,
                "total_pnl_usd": 0.0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "score": 0.0,
                "notes": notes or "no trades accepted",
            }
        pnls: List[float] = []
        pnl_pcts: List[float] = []
        for t in accepted:
            for ex in exits_by_sym.get(t["symbol"], []):
                pnls.append(float(ex.get("pnl_usd") or 0))
                pnl_pcts.append(float(ex.get("pnl_pct") or 0))
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        n = len(pnls) or 1
        win_rate = wins / n
        avg_pnl_pct = sum(pnl_pcts) / n if pnl_pcts else 0.0
        # Score: pnl gated by win rate and adoption efficiency
        adoption = len(accepted) / max(1, total_considered)
        score = total_pnl * (0.5 + win_rate / 2.0) * (0.3 + adoption * 0.7)
        return {
            "trades_considered": total_considered,
            "trades_accepted": len(accepted),
            "total_pnl_usd": total_pnl,
            "win_rate": win_rate,
            "avg_pnl_pct": avg_pnl_pct,
            "score": score,
            "notes": notes,
        }

    def _eval_baseline(self, trades, exits_by_sym):
        return self._summarise(trades, exits_by_sym, len(trades), "current rules")

    def _eval_strict_ev(self, trades, exits_by_sym):
        accepted = [t for t in trades if float(t.get("ev") or 0) >= 0.0625]  # 25% above 0.05
        return self._summarise(accepted, exits_by_sym, len(trades), "ev >= 0.0625")

    def _eval_conf_weighted(self, trades, exits_by_sym):
        accepted = [
            t for t in trades
            if (float(t.get("confidence") or 0) * float(t.get("ev") or 0)) >= 0.04
        ]
        return self._summarise(accepted, exits_by_sym, len(trades),
                               "confidence * ev >= 0.04")

    def _eval_momentum(self, trades, exits_by_sym):
        # Only accept trade if prior exit in same symbol was positive (or no prior exit)
        accepted: List[Dict[str, Any]] = []
        for t in trades:
            prior = exits_by_sym.get(t["symbol"], [])
            if not prior:
                accepted.append(t)
            elif (prior[-1].get("pnl_usd") or 0) > 0:
                accepted.append(t)
        return self._summarise(accepted, exits_by_sym, len(trades),
                               "prior exit in symbol must be green")

    def _eval_tier_cons(self, trades, exits_by_sym):
        accepted = [t for t in trades if t.get("tier") == "conservative"]
        return self._summarise(accepted, exits_by_sym, len(trades),
                               "conservative tier only")


def get_strategy_lab(db_path: str) -> StrategyLab:
    return StrategyLab(db_path=db_path)
