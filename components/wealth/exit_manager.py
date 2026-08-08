"""
ExitManager — autonomous position-exit logic for AutonomousTrader.

Evaluates every open position on each tick and closes any that hit:
  - stop-loss (default -4% from entry)
  - take-profit (default +8% from entry)
  - trailing stop (gives back 3% from session high)
  - max hold age (default 10 trading days)
  - sentiment / confidence flip (model now predicts down)

All exits are logged to SQLite (`at_exits` table) with entry price, exit price,
qty, P&L USD, P&L pct, and exit reason. Real broker prices only — no estimates.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Defaults — tier-aware overrides live in TIERS dict in autonomous_trader.py
DEFAULT_STOP_LOSS_PCT = 0.04          # 4% below entry
DEFAULT_TAKE_PROFIT_PCT = 0.08        # 8% above entry
DEFAULT_TRAILING_GIVEBACK_PCT = 0.03  # close if 3% off intraday high
DEFAULT_MAX_HOLD_DAYS = 10            # close stale positions
CONFIDENCE_FLIP_THRESHOLD = -0.10     # model now predicts >10% chance of fall


class ExitManager:
    """
    Manages exits for an AutonomousTrader. One instance per trader.
    """

    def __init__(
        self,
        db_path: str,
        trader: Any,                  # AggressiveTrader-shaped (get_positions, execute_sell)
        prediction_engine: Any = None,
        notifier: Any = None,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        trailing_giveback_pct: float = DEFAULT_TRAILING_GIVEBACK_PCT,
        max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
    ) -> None:
        self.db_path = db_path
        self.trader = trader
        self.prediction_engine = prediction_engine
        self.notifier = notifier
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_giveback_pct = trailing_giveback_pct
        self.max_hold_days = max_hold_days
        self._lock = threading.RLock()
        self._init_db()

    # ── DB ────────────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        from components.db import safe_open_kdb
        c = safe_open_kdb(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS at_exits ("
                "id SERIAL PRIMARY KEY, "
                "ts TEXT DEFAULT (datetime('now')), "
                "symbol TEXT NOT NULL, "
                "qty REAL, "
                "entry_avg REAL, "
                "exit_price REAL, "
                "pnl_usd REAL, "
                "pnl_pct REAL, "
                "hold_days REAL, "
                "reason TEXT NOT NULL, "
                "live INTEGER, "
                "result_json TEXT)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS at_position_high ("
                "symbol TEXT PRIMARY KEY, "
                "session_high REAL, "
                "updated_at TEXT DEFAULT (datetime('now')))"
            )
            c.commit()

    # ── Position price tracking ───────────────────────────────────────────────
    def _update_session_high(self, symbol: str, current_price: float) -> float:
        """Track intraday peak per symbol. Returns the new session high."""
        with self._conn() as c:
            row = c.execute(
                "SELECT session_high FROM at_position_high WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO at_position_high(symbol, session_high) VALUES (?, ?)",
                    (symbol, current_price),
                )
                c.commit()
                return current_price
            prev = float(row["session_high"] or 0)
            new_high = max(prev, current_price)
            if new_high > prev:
                c.execute(
                    "UPDATE at_position_high SET session_high = ?, "
                    "updated_at = datetime('now') WHERE symbol = ?",
                    (new_high, symbol),
                )
                c.commit()
            return new_high

    def _clear_session_high(self, symbol: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM at_position_high WHERE symbol = ?", (symbol,))
            c.commit()

    # ── Exit evaluation ───────────────────────────────────────────────────────
    def evaluate(
        self,
        tier_caps: Optional[Dict[str, float]] = None,
        live: bool = False,
    ) -> Dict[str, Any]:
        """
        Examine every open position and close any that trip an exit condition.
        Returns a summary dict.
        """
        with self._lock:
            return self._evaluate_inner(tier_caps or {}, live)

    def _evaluate_inner(self, tier_caps: Dict[str, float], live: bool) -> Dict[str, Any]:
        try:
            positions = self.trader.get_positions() or []
        except Exception as e:
            logger.warning("ExitManager: get_positions failed: %s", e)
            return {"checked": 0, "closed": 0, "reasons": [], "error": str(e)}

        stop_loss_pct = tier_caps.get("stop_loss_pct", self.stop_loss_pct)
        take_profit_pct = tier_caps.get("take_profit_pct", self.take_profit_pct)
        trailing_giveback_pct = tier_caps.get(
            "trailing_giveback_pct", self.trailing_giveback_pct
        )
        max_hold_days = tier_caps.get("max_hold_days", self.max_hold_days)

        closed: List[Dict[str, Any]] = []
        checked = 0

        for pos in positions:
            try:
                sym = pos.get("symbol")
                if not sym:
                    continue
                checked += 1
                qty = float(pos.get("qty") or 0)
                avg_entry = float(
                    pos.get("avg_entry_price") or pos.get("avg_entry") or 0
                )
                current_price = float(
                    pos.get("current_price")
                    or pos.get("market_price")
                    or pos.get("last")
                    or 0
                )
                if not avg_entry or not current_price or qty <= 0:
                    continue

                pnl_pct = (current_price - avg_entry) / avg_entry
                session_high = self._update_session_high(sym, current_price)
                trailing_pct = (
                    (session_high - current_price) / session_high
                    if session_high
                    else 0.0
                )
                hold_days = self._hold_days_for(sym)

                reason: Optional[str] = None
                if pnl_pct <= -stop_loss_pct:
                    reason = f"stop_loss({pnl_pct:.2%})"
                elif pnl_pct >= take_profit_pct:
                    reason = f"take_profit({pnl_pct:.2%})"
                elif trailing_pct >= trailing_giveback_pct and pnl_pct > 0:
                    reason = f"trailing_stop(-{trailing_pct:.2%} from high)"
                elif hold_days is not None and hold_days >= max_hold_days:
                    reason = f"max_hold({hold_days:.1f}d)"
                else:
                    # Confidence flip: ask predictor if it now expects down move
                    flip = self._confidence_flip(sym)
                    if flip is not None and flip <= CONFIDENCE_FLIP_THRESHOLD:
                        reason = f"confidence_flip({flip:+.2f})"

                if reason:
                    result = self._close_position(
                        sym, qty, avg_entry, current_price, reason, hold_days, live
                    )
                    closed.append(result)

            except Exception as e:
                logger.exception("ExitManager: error on %s: %s", pos, e)

        summary = {
            "checked": checked,
            "closed": len(closed),
            "exits": closed,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        return summary

    # ── Close + log ───────────────────────────────────────────────────────────
    def _close_position(
        self,
        symbol: str,
        qty: float,
        avg_entry: float,
        exit_price: float,
        reason: str,
        hold_days: Optional[float],
        live: bool,
    ) -> Dict[str, Any]:
        try:
            result = self.trader.execute_sell(symbol)
        except Exception as e:
            result = {"error": str(e)}
            if self.notifier:
                try:
                    self.notifier.error("execute_sell", f"{symbol}: {e}")
                except Exception:
                    pass

        pnl_usd = (exit_price - avg_entry) * qty
        pnl_pct = (exit_price - avg_entry) / avg_entry if avg_entry else 0.0

        sell_ok = isinstance(result, dict) and "error" not in result
        # Ledger mirror (PR #166 B.3): close the matching open row in the isolated
        # dmai_ledger.db. Best-effort — a ledger failure must never block an exit.
        if sell_ok:
            try:
                from components.ledger import ledger_db
                ledger_db.close_open_trade_for_symbol(
                    symbol, exit_price=exit_price, pnl=pnl_usd, notes=reason,
                )
            except Exception as le:
                logger.warning("ExitManager: ledger close-record failed: %s", le)

        with self._conn() as c:
            c.execute(
                "INSERT INTO at_exits(symbol, qty, entry_avg, exit_price, "
                "pnl_usd, pnl_pct, hold_days, reason, live, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    qty,
                    avg_entry,
                    exit_price,
                    pnl_usd,
                    pnl_pct,
                    hold_days,
                    reason,
                    1 if live else 0,
                    json.dumps(result)[:4000],
                ),
            )
            c.commit()
        self._clear_session_high(symbol)

        logger.info(
            "ExitManager: closed %s qty=%s pnl=%.2f (%.2f%%) reason=%s live=%s",
            symbol, qty, pnl_usd, pnl_pct * 100, reason, live,
        )
        if self.notifier:
            try:
                self.notifier.exit({
                    "symbol": symbol,
                    "qty": qty,
                    "entry": avg_entry,
                    "exit": exit_price,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "live": live,
                })
            except Exception as e:
                logger.debug("notifier.exit failed: %s", e)
        return {
            "symbol": symbol,
            "qty": qty,
            "entry_avg": avg_entry,
            "exit_price": exit_price,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "reason": reason,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _hold_days_for(self, symbol: str) -> Optional[float]:
        """Look up earliest buy ts for this symbol in at_trades — returns hold-days."""
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT ts FROM at_trades WHERE symbol = ? AND side = 'buy' "
                    "ORDER BY id ASC LIMIT 1",
                    (symbol,),
                ).fetchone()
                if not row:
                    return None
                ts = row["ts"]
                # SQLite ts is 'YYYY-MM-DD HH:MM:SS'
                try:
                    bought_at = datetime.fromisoformat(ts.replace(" ", "T"))
                except Exception:
                    return None
                return (datetime.utcnow() - bought_at).total_seconds() / 86400.0
        except Exception:
            return None

    def _confidence_flip(self, symbol: str) -> Optional[float]:
        """Ask the predictor whether the symbol still looks bullish.
        Returns a confidence delta in [-1, 1] or None if unavailable."""
        if not self.prediction_engine:
            return None
        try:
            pred = self.prediction_engine.predict(
                requirement=(
                    f"Will {symbol} close LOWER than today's price within the "
                    "next 5 trading days?"
                )
            )
            if not isinstance(pred, dict):
                return None
            # Models output prob of the requirement being TRUE (i.e. down move)
            prob_down = float(pred.get("probability") or pred.get("p") or 0)
            # Return delta from neutral 0.5 — positive means model expects DOWN
            return -(prob_down - 0.5) * 2  # in [-1, +1], negative = bearish
        except Exception as e:
            logger.debug("ExitManager: prediction failed for %s: %s", symbol, e)
            return None

    # ── Reporting ─────────────────────────────────────────────────────────────
    def recent_exits(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, symbol, qty, entry_avg, exit_price, pnl_usd, "
                "pnl_pct, hold_days, reason, live FROM at_exits "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def stats(self, days: int = 30) -> Dict[str, Any]:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT pnl_usd, pnl_pct, reason FROM at_exits WHERE ts >= ?",
                (since,),
            ).fetchall()
        if not rows:
            return {
                "exits": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "total_pnl_usd": 0.0, "avg_pnl_pct": 0.0,
                "reasons": {},
            }
        wins = sum(1 for r in rows if (r["pnl_usd"] or 0) > 0)
        total_pnl = sum(float(r["pnl_usd"] or 0) for r in rows)
        avg_pct = sum(float(r["pnl_pct"] or 0) for r in rows) / len(rows)
        reasons: Dict[str, int] = {}
        for r in rows:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        return {
            "exits": len(rows),
            "wins": wins,
            "losses": len(rows) - wins,
            "win_rate": wins / len(rows) if rows else 0.0,
            "total_pnl_usd": total_pnl,
            "avg_pnl_pct": avg_pct,
            "reasons": reasons,
        }


def get_exit_manager(db_path, trader, prediction_engine=None, notifier=None,
                     **kwargs) -> ExitManager:
    return ExitManager(
        db_path=db_path,
        trader=trader,
        prediction_engine=prediction_engine,
        notifier=notifier,
        **kwargs,
    )
