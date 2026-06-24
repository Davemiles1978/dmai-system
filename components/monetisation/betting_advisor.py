"""
BettingAdvisor: Microfish-driven +EV tip generator (notify-only).

Pipeline per candidate event:
  1. Microfish PredictionEngine -> {probability, confidence, rationale}
  2. EV gate: only emit tip if (probability * decimal_odds) - 1 >= ev_threshold (default 5%)
  3. Kelly fraction sizing (quarter-Kelly default) capped to stake_cap_pct of bankroll
  4. Notify user via send_notification + write tip to SQLite for the UI

User then places the bet manually and marks the tip as 'placed'/'skipped' in UI.
Outcome is recorded later to track ROI.

Bankroll source: RevenueAllocator.dmai_operating wallet's "betting" sub-allocation
(default: 5% of operating wallet, configurable).
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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mon_tips (
    id TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    market TEXT,
    selection TEXT NOT NULL,
    bookmaker TEXT,
    decimal_odds REAL NOT NULL,
    model_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_value REAL NOT NULL,
    kelly_fraction REAL NOT NULL,
    recommended_stake REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    rationale TEXT,
    prediction_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|placed|skipped|won|lost|void
    placed_at REAL,
    settled_at REAL,
    actual_stake REAL,
    profit_loss REAL,
    notes TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mon_tips_status ON mon_tips(status, created_at DESC);
"""


class BettingAdvisor:
    """
    Generates betting tips via Microfish. NEVER places bets.
    User confirms placement manually through UI -> mark_placed/mark_skipped/settle.
    """

    def __init__(self, prediction_engine=None, allocator=None,
                 db_path: str = "data/dmai_knowledge.db",
                 ev_threshold: float = 0.05,
                 kelly_multiplier: float = 0.25,   # quarter-Kelly
                 stake_cap_pct: float = 0.02,      # max 2% of bankroll per bet
                 bankroll_pct: float = 0.05,       # 5% of DMAI operating wallet
                 max_stake_absolute: float = 50.0, # hard ceiling per bet
                 currency: str = "GBP"):
        self.prediction_engine = prediction_engine
        self.allocator = allocator
        self.db_path = db_path
        self.ev_threshold = ev_threshold
        self.kelly_multiplier = kelly_multiplier
        self.stake_cap_pct = stake_cap_pct
        self.bankroll_pct = bankroll_pct
        self.max_stake_absolute = max_stake_absolute
        self.currency = currency
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with _LOCK, self._conn() as c:
            c.executescript(_SCHEMA)

    # ---- bankroll ----

    def get_bankroll(self) -> float:
        """Notional betting bankroll = bankroll_pct of DMAI operating wallet."""
        if not self.allocator:
            return 0.0
        op_balance = self.allocator.get_balance(self.allocator.DMAI_WALLET)
        return round(max(0.0, op_balance * self.bankroll_pct), 2)

    # ---- analysis ----

    @staticmethod
    def kelly_fraction(probability: float, decimal_odds: float) -> float:
        """Standard Kelly: f* = (bp - q) / b where b=odds-1, p=win prob, q=1-p."""
        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - probability
        f = (b * probability - q) / b
        return max(0.0, f)

    def analyse_candidate(self, event_name: str, selection: str,
                          decimal_odds: float, *,
                          market: str = "match_winner",
                          bookmaker: str = "",
                          seed_data: str = "",
                          max_rounds: int = 2,
                          agent_count: int = 4) -> Dict[str, Any]:
        """Run Microfish on a candidate bet and return analysis. Does NOT persist."""
        if not self.prediction_engine:
            return {"error": "prediction_engine unavailable"}
        if decimal_odds <= 1.0:
            return {"error": "decimal_odds must be > 1.0"}

        requirement = (
            f"Will the selection '{selection}' win in the market '{market}' "
            f"for the event '{event_name}'?"
        )
        seed = (
            f"Event: {event_name}\nMarket: {market}\nSelection: {selection}\n"
            f"Bookmaker offered decimal odds: {decimal_odds}\n"
            f"Implied probability from odds: {1.0/decimal_odds:.3f}\n"
            f"Additional context:\n{seed_data}"
        )
        verdict = self.prediction_engine.predict(
            requirement=requirement, seed_data=seed,
            max_rounds=max_rounds, agent_count=agent_count,
        )
        p = float(verdict.get("probability", 0.5))
        conf = float(verdict.get("confidence", 0.5))
        ev = (p * decimal_odds) - 1.0
        kelly = self.kelly_fraction(p, decimal_odds) * self.kelly_multiplier
        bankroll = self.get_bankroll()
        cap = bankroll * self.stake_cap_pct
        # confidence dampens stake: low confidence -> smaller bet
        stake = min(bankroll * kelly * conf, cap, self.max_stake_absolute)
        stake = round(max(0.0, stake), 2)
        passes_gate = (ev >= self.ev_threshold) and (stake > 0)
        return {
            "event_name": event_name, "market": market, "selection": selection,
            "bookmaker": bookmaker, "decimal_odds": decimal_odds,
            "model_probability": round(p, 3),
            "confidence": round(conf, 3),
            "expected_value": round(ev, 3),
            "kelly_fraction": round(kelly, 4),
            "recommended_stake": stake,
            "bankroll": bankroll,
            "passes_ev_gate": passes_gate,
            "rationale": verdict.get("rationale", ""),
            "prediction_id": verdict.get("id"),
        }

    def generate_tip(self, **kwargs) -> Dict[str, Any]:
        """Analyse and, if it passes EV gate, persist as a pending tip."""
        analysis = self.analyse_candidate(**{k: v for k, v in kwargs.items()
                                             if k in ("event_name", "selection", "decimal_odds",
                                                      "market", "bookmaker", "seed_data",
                                                      "max_rounds", "agent_count")})
        if "error" in analysis:
            return analysis
        if not analysis["passes_ev_gate"]:
            return {"status": "rejected_no_edge", **analysis}
        tip_id = uuid.uuid4().hex[:16]
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mon_tips (id, event_name, market, selection, bookmaker, decimal_odds, "
                "model_probability, confidence, expected_value, kelly_fraction, recommended_stake, "
                "currency, rationale, prediction_id, status, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tip_id, analysis["event_name"], analysis["market"], analysis["selection"],
                 analysis["bookmaker"], analysis["decimal_odds"], analysis["model_probability"],
                 analysis["confidence"], analysis["expected_value"], analysis["kelly_fraction"],
                 analysis["recommended_stake"], self.currency, analysis["rationale"],
                 analysis["prediction_id"], "pending", time.time()),
            )
        analysis["id"] = tip_id
        analysis["status"] = "pending"
        self._notify_tip(analysis)
        return analysis

    def _notify_tip(self, tip: Dict[str, Any]):
        """Send in-app notification with a Place-manually call to action."""
        try:
            # Lazy import to avoid circular deps; this is a Flask app, send_notification
            # is provided by the agent runtime, not the app. We log + persist as the
            # primary signal; the UI polls /api/monetisation/tips.
            logger.info(
                "[TIP] %s — %s @ %.2f | EV=%+.1f%% | stake %s%.2f | conf=%.2f",
                tip.get("event_name"), tip.get("selection"), tip.get("decimal_odds"),
                tip.get("expected_value", 0) * 100, self.currency,
                tip.get("recommended_stake", 0), tip.get("confidence", 0),
            )
        except Exception:
            pass

    # ---- lifecycle ----

    def mark_placed(self, tip_id: str, actual_stake: Optional[float] = None,
                    notes: str = "") -> Dict[str, Any]:
        with _LOCK, self._conn() as c:
            row = c.execute("SELECT * FROM mon_tips WHERE id=?", (tip_id,)).fetchone()
            if not row:
                return {"error": "tip_not_found"}
            stake = actual_stake if actual_stake is not None else row["recommended_stake"]
            c.execute(
                "UPDATE mon_tips SET status='placed', placed_at=?, actual_stake=?, notes=? WHERE id=?",
                (time.time(), stake, notes, tip_id),
            )
        return {"id": tip_id, "status": "placed", "actual_stake": stake}

    def mark_skipped(self, tip_id: str, notes: str = "") -> Dict[str, Any]:
        with _LOCK, self._conn() as c:
            c.execute("UPDATE mon_tips SET status='skipped', notes=? WHERE id=? AND status='pending'",
                      (notes, tip_id))
        return {"id": tip_id, "status": "skipped"}

    def settle(self, tip_id: str, outcome: str, actual_return: float = 0.0,
               notes: str = "") -> Dict[str, Any]:
        """outcome in {won, lost, void}. actual_return is the gross return (incl. stake) on a win."""
        outcome = outcome.lower()
        if outcome not in ("won", "lost", "void"):
            return {"error": "invalid_outcome"}
        with _LOCK, self._conn() as c:
            row = c.execute("SELECT * FROM mon_tips WHERE id=?", (tip_id,)).fetchone()
            if not row:
                return {"error": "tip_not_found"}
            stake = float(row["actual_stake"] or 0.0)
            if outcome == "won":
                pl = round(actual_return - stake, 2)
            elif outcome == "lost":
                pl = -stake
            else:  # void
                pl = 0.0
            c.execute(
                "UPDATE mon_tips SET status=?, settled_at=?, profit_loss=?, notes=? WHERE id=?",
                (outcome, time.time(), pl, notes, tip_id),
            )
            # If won, credit the profit (not stake) as new income; if lost, debit
            if self.allocator and outcome == "won" and pl > 0:
                self.allocator.credit_income(
                    source=f"betting_win:{row['event_name']}",
                    amount=pl,
                    metadata={"tip_id": tip_id, "selection": row["selection"]},
                )
            elif self.allocator and outcome == "lost" and stake > 0:
                # Debit dmai_operating to reflect the lost stake
                self.allocator.debit(
                    wallet=self.allocator.DMAI_WALLET, amount=stake,
                    reason=f"betting_loss:{row['event_name']}",
                )
        return {"id": tip_id, "status": outcome, "profit_loss": pl}

    # ---- reads ----

    def list_tips(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as c:
            if status:
                rows = c.execute(
                    "SELECT * FROM mon_tips WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM mon_tips ORDER BY created_at DESC LIMIT ?", (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT status, COUNT(*) AS n, COALESCE(SUM(profit_loss),0) AS pl, "
                "COALESCE(SUM(actual_stake),0) AS turnover FROM mon_tips GROUP BY status"
            ).fetchall()
        by_status = {r["status"]: {"count": r["n"], "pl": round(float(r["pl"]), 2),
                                   "turnover": round(float(r["turnover"]), 2)} for r in rows}
        settled = sum(by_status.get(s, {}).get("count", 0) for s in ("won", "lost", "void"))
        won = by_status.get("won", {}).get("count", 0)
        total_pl = sum(by_status.get(s, {}).get("pl", 0) for s in ("won", "lost", "void"))
        total_turnover = sum(by_status.get(s, {}).get("turnover", 0) for s in ("won", "lost", "void"))
        return {
            "by_status": by_status,
            "settled_count": settled,
            "win_rate": round(won / settled, 3) if settled else None,
            "total_pl": round(total_pl, 2),
            "total_turnover": round(total_turnover, 2),
            "roi_pct": round((total_pl / total_turnover) * 100, 2) if total_turnover else None,
            "bankroll": self.get_bankroll(),
            "currency": self.currency,
        }
