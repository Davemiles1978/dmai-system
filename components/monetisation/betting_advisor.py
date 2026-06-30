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
import atexit
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

# ── Greyhound tracking-pick write batching (2026-06-30, lock-storm reduction) ──
# The GreyhoundRunner records one top pick per race via record_tracking_pick,
# which previously wrote a row (INSERT + COMMIT) per call at race-tick
# frequency — the dominant remaining source of `database is locked` after
# PR #154. We buffer picks and flush them in a single executemany transaction.
#
# Unlike the vocabulary ingester's count-only flush, picks must reach the DB
# quickly so the trader/admin UI sees them within ~1s, so the trigger is
# TIME-BOUNDED as well: flush after GREYHOUND_BATCH_SIZE picks OR
# GREYHOUND_FLUSH_MS since the first buffered pick — whichever comes first.
# A daemon flusher thread wakes every 250ms to honour the time bound. Both
# knobs are optional env overrides; no schema change.
_DEFAULT_GREYHOUND_BATCH_SIZE = 50
_DEFAULT_GREYHOUND_FLUSH_MS = 1000
_GREYHOUND_FLUSH_POLL_SECONDS = 0.25

# INSERT OR IGNORE so the UNIQUE(event_name, market) constraint silently
# dedupes re-buffered picks (the runner re-records the same race every cycle
# until it starts) instead of failing the whole executemany batch.
_TRACKING_PICK_INSERT_SQL = (
    "INSERT OR IGNORE INTO mon_tracking_picks (id, event_name, market, selection, "
    "decimal_odds, model_probability, confidence, expected_value, rationale, "
    "prediction_id, outcome, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
)

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

CREATE TABLE IF NOT EXISTS mon_user_bets (
    id TEXT PRIMARY KEY,
    tip_id TEXT,
    placed_at REAL NOT NULL,
    event_name TEXT NOT NULL,
    market TEXT,
    selection TEXT NOT NULL,
    actual_odds REAL NOT NULL,
    actual_stake REAL NOT NULL,
    bookmaker TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    settled_at REAL,
    actual_return REAL,
    profit_loss REAL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    notes TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_bets_status ON mon_user_bets(status, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_bets_tip ON mon_user_bets(tip_id);

-- Tracking picks: the model's top pick per race, recorded regardless of EV
-- gate, so we can score prediction accuracy over 2-7 days before going live.
-- One row per (event_name, market). No money implied.
CREATE TABLE IF NOT EXISTS mon_tracking_picks (
    id TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    market TEXT NOT NULL DEFAULT 'trap_winner',
    selection TEXT NOT NULL,
    decimal_odds REAL NOT NULL,
    model_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_value REAL NOT NULL,
    rationale TEXT,
    prediction_id TEXT,
    -- outcome: pending | won | lost | void  (settled by runner against GBGB)
    outcome TEXT NOT NULL DEFAULT 'pending',
    settled_at REAL,
    -- paper P/L if you had staked 1 unit at decimal_odds (informational only)
    paper_pl REAL,
    notes TEXT,
    created_at REAL NOT NULL,
    UNIQUE(event_name, market)
);
CREATE INDEX IF NOT EXISTS idx_tracking_outcome ON mon_tracking_picks(outcome, created_at DESC);
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
                 currency: str = "GBP",
                 greyhound_model=None,
                 notifier=None):
        self.prediction_engine = prediction_engine
        # Optional Slack-style notifier. The DMAI runtime sets this via
        # `advisor.notifier = components['notifier']` after both are built,
        # so we keep the kwarg purely as a wiring escape hatch.
        self.notifier = notifier
        # StatisticalGreyhoundModel — used for greyhound markets only.
        # Microfish is unsuitable for sports betting (per user decision).
        if greyhound_model is None:
            try:
                from components.monetisation.statistical_greyhound_model import (
                    StatisticalGreyhoundModel,
                )
                greyhound_model = StatisticalGreyhoundModel(db_path=db_path)
            except Exception as _e:
                logger.warning("StatisticalGreyhoundModel init failed: %s", _e)
                greyhound_model = None
        self.greyhound_model = greyhound_model
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

        # ── Greyhound tracking-pick write buffer ───────────────────────────
        try:
            self._pick_batch_size = max(
                1, int(os.environ.get("GREYHOUND_BATCH_SIZE",
                                      _DEFAULT_GREYHOUND_BATCH_SIZE))
            )
        except (TypeError, ValueError):
            self._pick_batch_size = _DEFAULT_GREYHOUND_BATCH_SIZE
        try:
            self._pick_flush_ms = float(
                os.environ.get("GREYHOUND_FLUSH_MS", _DEFAULT_GREYHOUND_FLUSH_MS)
            )
        except (TypeError, ValueError):
            self._pick_flush_ms = float(_DEFAULT_GREYHOUND_FLUSH_MS)
        self._pick_buffer: List[tuple] = []     # pending INSERT row tuples
        self._pick_keys: set = set()            # (event_name, market) in-buffer dedup
        self._pick_buffer_lock = threading.Lock()
        self._pick_first_ts: Optional[float] = None  # monotonic ts of first buffered pick
        self._pick_flusher: Optional[threading.Thread] = None
        self._pick_stop = threading.Event()
        atexit.register(self.shutdown)

    def _conn(self):
        # Integrity check + quarantine removed: it was destroying shared tables
        # (at_state, capabilities, system_state) created by boot bootstrap.
        # Boot bootstrap is now authoritative; per-component _ensure_tables
        # handles missing tables on demand.
        c = safe_open_kdb(self.db_path, timeout=30.0)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")
            c.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
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
        """Analyse a candidate bet and return analysis. Does NOT persist.

        Routes to StatisticalGreyhoundModel for greyhound markets,
        Microfish PredictionEngine for everything else.
        """
        if decimal_odds <= 1.0:
            return {"error": "decimal_odds must be > 1.0"}

        # Route greyhound markets to the deterministic statistical model.
        # Microfish extrapolates curves and is unsuitable for sports.
        _is_greyhound = (
            (market or "").startswith("trap_")
            or (market or "") == "greyhound_winner"
        )
        engine = self.greyhound_model if _is_greyhound else self.prediction_engine
        if engine is None:
            return {"error": (
                "greyhound_model unavailable" if _is_greyhound
                else "prediction_engine unavailable"
            )}

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
        verdict = engine.predict(
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
        """Send a HOT-TIP notification + log line. Best-effort, never raises.

        Order of preference:
          1. SlackNotifier (if wired) -> loud Slack alert via .hot_tip()
          2. Always log a [HOT TIP] line that the health-check loop relays.
        """
        try:
            logger.warning(
                "[HOT TIP] %s \u2014 %s @ %.2f | EV=%+.1f%% | conf=%.2f | stake %s%.2f | id=%s",
                tip.get("event_name"), tip.get("selection"), tip.get("decimal_odds"),
                float(tip.get("expected_value", 0)) * 100,
                float(tip.get("confidence", 0)),
                self.currency, float(tip.get("recommended_stake", 0)),
                tip.get("id", "?"),
            )
        except Exception:
            pass
        try:
            if self.notifier and hasattr(self.notifier, "hot_tip"):
                # Make sure currency is present on the dict so the notifier can format it.
                if "currency" not in tip:
                    tip = {**tip, "currency": self.currency}
                self.notifier.hot_tip(tip)
        except Exception as e:
            logger.warning("hot_tip notifier failed: %s", e)

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

    # ---- tracking-mode reads/writes (used by GreyhoundRunner) ----

    def record_tracking_pick(self, *, event_name: str, market: str,
                             selection: str, decimal_odds: float,
                             model_probability: float, confidence: float,
                             expected_value: float, rationale: str = "",
                             prediction_id: Optional[str] = None) -> Dict[str, Any]:
        """Persist the model's top pick for a race regardless of EV gate.

        Returns {id, event_name, selection, status} or {error, ...}.
        Inserts are idempotent by (event_name, market): if the race already
        has a tracked pick, this is a no-op and the existing row is returned.
        """
        if decimal_odds <= 1.0:
            return {"error": "decimal_odds must be > 1.0"}
        pick_id = uuid.uuid4().hex[:16]
        row = (pick_id, event_name, market, selection, decimal_odds,
               model_probability, confidence, expected_value, rationale,
               prediction_id, "pending", time.time())
        # Buffer the write (single executemany transaction on flush) rather than
        # writing one row per pick. DB-level idempotency on (event_name, market)
        # is preserved by INSERT OR IGNORE against the UNIQUE constraint.
        self._maybe_start_pick_flusher()
        flush_now = False
        status = "tracked"
        key = (event_name, market)
        with self._pick_buffer_lock:
            if key in self._pick_keys:
                status = "already_tracked"
            else:
                self._pick_keys.add(key)
                if not self._pick_buffer:
                    self._pick_first_ts = time.monotonic()
                self._pick_buffer.append(row)
                if len(self._pick_buffer) >= self._pick_batch_size:
                    flush_now = True
        if flush_now:
            self._flush_picks()
        if status == "already_tracked":
            return {"id": pick_id, "event_name": event_name,
                    "selection": selection, "status": "already_tracked"}
        # HOT-TIP notification for STRONG-tier tracking picks (manual-bet phase).
        # Criteria mirror SlackNotifier.hot_tip 'STRONG': EV >= 0.20 AND confidence >= 0.70.
        try:
            if expected_value >= 0.20 and confidence >= 0.70:
                self._notify_tip({
                    "id": pick_id,
                    "event_name": event_name,
                    "market": market,
                    "selection": selection,
                    "decimal_odds": decimal_odds,
                    "model_probability": model_probability,
                    "confidence": confidence,
                    "expected_value": expected_value,
                    "recommended_stake": 0,  # tracking mode = no auto stake
                    "rationale": rationale,
                    "prediction_id": prediction_id,
                    "mode": "TRACKING",
                })
        except Exception as e:
            logger.warning("tracking hot-tip notify failed: %s", e)
        return {"id": pick_id, "event_name": event_name, "selection": selection,
                "status": "tracked"}

    # ---- tracking-pick write batching ----

    def _maybe_start_pick_flusher(self) -> None:
        """Lazily start the daemon thread that flushes the pick buffer once it
        has been waiting ``GREYHOUND_FLUSH_MS`` since its first buffered pick."""
        if self._pick_flusher is not None or self._pick_flush_ms <= 0:
            return
        self._pick_stop.clear()
        t = threading.Thread(
            target=self._pick_flush_loop, name="greyhound-pick-flush", daemon=True
        )
        self._pick_flusher = t
        t.start()

    def _pick_flush_loop(self) -> None:
        """Wake every 250ms; flush when the time bound has elapsed. The
        count bound is handled inline in record_tracking_pick."""
        while not self._pick_stop.wait(_GREYHOUND_FLUSH_POLL_SECONDS):
            try:
                with self._pick_buffer_lock:
                    due = bool(
                        self._pick_buffer
                        and self._pick_first_ts is not None
                        and (time.monotonic() - self._pick_first_ts) * 1000.0
                        >= self._pick_flush_ms
                    )
                if due:
                    self._flush_picks()
            except Exception as e:
                logger.debug("greyhound pick idle flush failed: %s", e)

    def flush_tracking_picks(self) -> int:
        """Flush any buffered tracking picks in a single transaction.

        Returns the number of rows written. Safe to call from any thread and
        at shutdown."""
        return self._flush_picks()

    def _flush_picks(self) -> int:
        with self._pick_buffer_lock:
            if not self._pick_buffer:
                return 0
            rows = self._pick_buffer
            self._pick_buffer = []
            self._pick_keys = set()
            self._pick_first_ts = None
        return self._write_picks(rows)

    def _write_picks(self, rows) -> int:
        """Write buffered picks: acquire the write lock once, executemany in a
        single transaction. On batch failure, fall back to one-by-one so a
        single bad row doesn't drop the rest (mirrors the vocab ingester)."""
        start = time.monotonic()
        try:
            with _LOCK, self._conn() as c:
                c.executemany(_TRACKING_PICK_INSERT_SQL, rows)
            written = len(rows)
        except Exception as e:
            logger.warning(
                "greyhound tracking-pick batch insert of %d rows failed (%s) "
                "— falling back to one-by-one", len(rows), e,
            )
            written = self._write_picks_individually(rows)
        logger.info(
            "greyhound_tracking_pick_flush rows=%d elapsed_ms=%d source=greyhound_runner",
            written, int((time.monotonic() - start) * 1000),
        )
        return written

    def _write_picks_individually(self, rows) -> int:
        written = 0
        for row in rows:
            try:
                with _LOCK, self._conn() as c:
                    c.execute(_TRACKING_PICK_INSERT_SQL, row)
                written += 1
            except Exception as e:
                logger.warning(
                    "greyhound tracking-pick row insert failed (event=%r): %s",
                    row[1] if len(row) > 1 else None, e,
                )
        return written

    def shutdown(self) -> int:
        """Stop the flusher thread and flush remaining buffered picks."""
        self._pick_stop.set()
        return self._flush_picks()

    def list_tracking_picks(self, outcome: Optional[str] = None,
                            limit: int = 200) -> List[Dict[str, Any]]:
        with self._conn() as c:
            if outcome:
                rows = c.execute(
                    "SELECT * FROM mon_tracking_picks WHERE outcome=? "
                    "ORDER BY created_at DESC LIMIT ?", (outcome, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM mon_tracking_picks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def settle_tracking_pick(self, *, event_name: str, market: str,
                             winning_selection: str) -> Dict[str, Any]:
        """Mark a tracked pick won/lost based on the actual race winner.

        paper_pl assumes a notional 1-unit stake at the recorded decimal_odds.
        """
        with _LOCK, self._conn() as c:
            row = c.execute(
                "SELECT * FROM mon_tracking_picks WHERE event_name=? AND market=? "
                "AND outcome='pending'", (event_name, market),
            ).fetchone()
            if not row:
                return {"status": "no_pending_pick"}
            won = (row["selection"].strip().lower() ==
                   (winning_selection or "").strip().lower())
            paper_pl = round(float(row["decimal_odds"]) - 1.0, 4) if won else -1.0
            outcome = "won" if won else "lost"
            c.execute(
                "UPDATE mon_tracking_picks SET outcome=?, settled_at=?, paper_pl=? "
                "WHERE id=?", (outcome, time.time(), paper_pl, row["id"]),
            )
        return {"id": row["id"], "outcome": outcome, "paper_pl": paper_pl}

    def tracking_performance(self) -> Dict[str, Any]:
        """Aggregate accuracy metrics across all settled tracking picks."""
        with self._conn() as c:
            total   = c.execute("SELECT COUNT(*) FROM mon_tracking_picks").fetchone()[0]
            pending = c.execute("SELECT COUNT(*) FROM mon_tracking_picks WHERE outcome='pending'").fetchone()[0]
            settled = c.execute("SELECT COUNT(*) FROM mon_tracking_picks WHERE outcome IN ('won','lost')").fetchone()[0]
            won     = c.execute("SELECT COUNT(*) FROM mon_tracking_picks WHERE outcome='won'").fetchone()[0]
            pl_row  = c.execute("SELECT COALESCE(SUM(paper_pl),0) FROM mon_tracking_picks WHERE paper_pl IS NOT NULL").fetchone()
            brier_row = c.execute(
                "SELECT AVG((model_probability - CASE outcome WHEN 'won' THEN 1 ELSE 0 END) * "
                "(model_probability - CASE outcome WHEN 'won' THEN 1 ELSE 0 END)) "
                "FROM mon_tracking_picks WHERE outcome IN ('won','lost')"
            ).fetchone()
        hit_rate = round(won / settled, 4) if settled else 0.0
        return {
            "total": total,
            "pending": pending,
            "settled": settled,
            "won": won,
            "lost": settled - won,
            "hit_rate": hit_rate,
            "brier_score": round(brier_row[0], 4) if brier_row and brier_row[0] is not None else None,
            "paper_pl_units": round(float(pl_row[0] or 0), 4),
        }

    # ---- upcoming + history reads (for Tip Tracking dashboard) ----

    def list_upcoming(self, days: int = 7, limit: int = 200) -> List[Dict[str, Any]]:
        """Pending tips for upcoming races (event_name suffix '(YYYY-MM-DD)')."""
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        import re as _re
        cutoff_lo = _dt.now(_tz.utc).date()
        cutoff_hi = cutoff_lo + _td(days=max(days, 1))
        date_pat = _re.compile(r"\((\d{4})-(\d{2})-(\d{2})\)")
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM mon_tips WHERE status='pending' "
                "ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            ev = r["event_name"] or ""
            m = date_pat.search(ev)
            if not m:
                out.append(dict(r))
                continue
            try:
                d = _dt(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
            except Exception:
                out.append(dict(r))
                continue
            if cutoff_lo <= d <= cutoff_hi:
                row = dict(r)
                row["race_date"] = d.isoformat()
                out.append(row)
        return out

    def list_history(self, limit: int = 200, paper_only: bool = False,
                     live_only: bool = False) -> List[Dict[str, Any]]:
        """Settled tips (won/lost/void/skipped)."""
        clauses = ["status IN ('won','lost','void','skipped')"]
        params: List[Any] = []
        if paper_only:
            clauses.append("COALESCE(notes,'') LIKE '%paper%'")
        elif live_only:
            clauses.append("COALESCE(notes,'') LIKE '%live%'")
        sql = (
            "SELECT * FROM mon_tips WHERE " + " AND ".join(clauses)
            + " ORDER BY COALESCE(settled_at, created_at) DESC LIMIT ?"
        )
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def performance(self, window: int = 100) -> Dict[str, Any]:
        """Model accuracy over the last `window` settled (won/lost) tips."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT model_probability, confidence, expected_value, decimal_odds, "
                "actual_stake, recommended_stake, profit_loss, status, notes, settled_at "
                "FROM mon_tips WHERE status IN ('won','lost') "
                "ORDER BY COALESCE(settled_at, created_at) DESC LIMIT ?",
                (window,),
            ).fetchall()
        try:
            with self._conn() as c:
                pending_count = c.execute(
                    "SELECT COUNT(*) AS n FROM mon_tips WHERE status='pending'"
                ).fetchone()["n"]
        except Exception:
            pending_count = 0
        if not rows:
            return {
                "window": window, "settled_count": 0,
                "win_rate": None, "hit_rate": None,
                "roi_pct": None, "brier": None, "pending": int(pending_count),
                "turnover": 0.0, "total_pl": 0.0,
                "total_paper_stake": 0.0, "paper_pl": 0.0,
                "calibration": [],
                "by_confidence_bucket": {},
                "mode_breakdown": {
                    "paper": {"count": 0, "pl": 0.0, "win_rate": None},
                    "live":  {"count": 0, "pl": 0.0, "win_rate": None},
                },
            }
        n = len(rows)
        won = sum(1 for r in rows if r["status"] == "won")
        stakes = [float(r["actual_stake"] or r["recommended_stake"] or 0) for r in rows]
        turnover = sum(stakes)
        total_pl = sum(float(r["profit_loss"] or 0) for r in rows)
        buckets: List[Dict[str, Any]] = []
        for i in range(10):
            lo, hi = i / 10.0, (i + 1) / 10.0
            in_bkt = [r for r in rows
                      if lo <= float(r["model_probability"] or 0) < hi
                      or (i == 9 and float(r["model_probability"] or 0) >= 1.0)]
            if not in_bkt:
                buckets.append({"range": [round(lo, 2), round(hi, 2)],
                                "count": 0, "predicted_mean": None,
                                "actual_win_rate": None})
                continue
            pred = sum(float(r["model_probability"] or 0) for r in in_bkt) / len(in_bkt)
            actual = sum(1 for r in in_bkt if r["status"] == "won") / len(in_bkt)
            buckets.append({
                "range": [round(lo, 2), round(hi, 2)],
                "count": len(in_bkt),
                "predicted_mean": round(pred, 3),
                "actual_win_rate": round(actual, 3),
            })
        def _cb(lo: float, hi: float):
            sub = [r for r in rows if lo <= float(r["confidence"] or 0) < hi]
            if not sub:
                return {"count": 0, "win_rate": None, "pl": 0.0}
            return {"count": len(sub),
                    "win_rate": round(sum(1 for r in sub if r["status"] == "won") / len(sub), 3),
                    "pl": round(sum(float(r["profit_loss"] or 0) for r in sub), 2)}
        def _mode(label: str):
            sub = [r for r in rows if label in (r["notes"] or "").lower()]
            return {"count": len(sub),
                    "pl": round(sum(float(r["profit_loss"] or 0) for r in sub), 2),
                    "win_rate": (round(sum(1 for r in sub if r["status"] == "won") / len(sub), 3)
                                 if sub else None)}
        # Brier score: lower is better calibration (0=perfect, 0.25=random).
        brier_terms = [
            (float(r["model_probability"] or 0)
             - (1.0 if r["status"] == "won" else 0.0)) ** 2
            for r in rows
        ]
        brier = round(sum(brier_terms) / n, 4) if brier_terms else None
        paper_rows = [r for r in rows if "paper" in (r["notes"] or "").lower()]
        paper_stake = (sum(float(r["actual_stake"] or r["recommended_stake"] or 0)
                           for r in paper_rows) or turnover)
        paper_pl = (sum(float(r["profit_loss"] or 0) for r in paper_rows)
                    or total_pl)
        # Reshape calibration buckets to the {band,n,predicted,actual} shape
        # the Tip Tracking UI consumes.
        calib_for_ui = [
            {
                "band": f"{int(b['range'][0]*100)}-{int(b['range'][1]*100)}%",
                "n": b["count"],
                "predicted": b["predicted_mean"],
                "actual": b["actual_win_rate"],
            }
            for b in buckets if b["count"]
        ]
        return {
            "window": window, "settled_count": n,
            "win_rate": round(won / n, 3),
            "hit_rate": round(won / n, 3),
            "roi_pct": round((total_pl / turnover) * 100, 2) if turnover else None,
            "brier": brier,
            "pending": int(pending_count),
            "turnover": round(turnover, 2), "total_pl": round(total_pl, 2),
            "total_paper_stake": round(paper_stake, 2),
            "paper_pl": round(paper_pl, 2),
            "calibration": calib_for_ui,
            "calibration_raw": buckets,
            "by_confidence_bucket": {
                "low":  _cb(0.0, 0.5),
                "mid":  _cb(0.5, 0.75),
                "high": _cb(0.75, 1.01),
            },
            "mode_breakdown": {"paper": _mode("paper"), "live": _mode("live")},
        }

    # ---- user bets (real bets the user actually places) ----

    def record_user_bet(self, *, tip_id: Optional[str] = None,
                        event_name: str = "", market: str = "",
                        selection: str = "", actual_odds: float = 0.0,
                        actual_stake: float = 0.0, bookmaker: str = "",
                        notes: str = "") -> Dict[str, Any]:
        """Record a real bet the user placed. Optionally linked to a model tip."""
        if actual_odds <= 1.0:
            return {"error": "actual_odds must be > 1.0"}
        if actual_stake <= 0:
            return {"error": "actual_stake must be > 0"}
        with _LOCK, self._conn() as c:
            if tip_id:
                tip_row = c.execute(
                    "SELECT event_name, market, selection, bookmaker FROM mon_tips WHERE id=?",
                    (tip_id,),
                ).fetchone()
                if tip_row:
                    event_name = event_name or tip_row["event_name"]
                    market = market or (tip_row["market"] or "")
                    selection = selection or tip_row["selection"]
                    bookmaker = bookmaker or (tip_row["bookmaker"] or "")
            bet_id = uuid.uuid4().hex[:16]
            c.execute(
                "INSERT INTO mon_user_bets (id, tip_id, placed_at, event_name, market, "
                "selection, actual_odds, actual_stake, bookmaker, status, currency, notes, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (bet_id, tip_id, time.time(), event_name, market, selection,
                 float(actual_odds), float(actual_stake), bookmaker,
                 "pending", self.currency, notes, time.time()),
            )
            if tip_id:
                c.execute(
                    "UPDATE mon_tips SET status='placed', placed_at=?, "
                    "actual_stake=COALESCE(actual_stake, ?) WHERE id=? AND status='pending'",
                    (time.time(), float(actual_stake), tip_id),
                )
        return {"id": bet_id, "tip_id": tip_id, "status": "pending"}

    def settle_user_bet(self, bet_id: str, outcome: str,
                        actual_return: float = 0.0,
                        notes: str = "") -> Dict[str, Any]:
        outcome = (outcome or "").lower()
        if outcome not in ("won", "lost", "void", "cashed_out"):
            return {"error": "invalid_outcome"}
        with _LOCK, self._conn() as c:
            row = c.execute(
                "SELECT actual_stake FROM mon_user_bets WHERE id=?", (bet_id,)
            ).fetchone()
            if not row:
                return {"error": "bet_not_found"}
            stake = float(row["actual_stake"] or 0)
            if outcome == "won":
                pl = round(float(actual_return) - stake, 2)
            elif outcome == "lost":
                pl = -stake
            elif outcome == "cashed_out":
                pl = round(float(actual_return) - stake, 2)
            else:  # void
                pl = 0.0
            c.execute(
                "UPDATE mon_user_bets SET status=?, settled_at=?, actual_return=?, "
                "profit_loss=?, notes=COALESCE(?, notes) WHERE id=?",
                (outcome, time.time(), float(actual_return), pl,
                 notes or None, bet_id),
            )
        return {"id": bet_id, "status": outcome, "profit_loss": pl}

    def list_user_bets(self, status: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with self._conn() as c:
                if status:
                    rows = c.execute(
                        "SELECT * FROM mon_user_bets WHERE status=? "
                        "ORDER BY placed_at DESC LIMIT ?", (status, limit),
                    ).fetchall()
                else:
                    rows = c.execute(
                        "SELECT * FROM mon_user_bets ORDER BY placed_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                self._init_schema()
                return []
            raise

    def user_bet_performance(self) -> Dict[str, Any]:
        """User real-bet performance + delta vs model recommendations."""
        empty = {"total_bets": 0, "settled_bets": 0, "settled_count": 0,
                 "hit_rate": None, "win_rate": None,
                 "profit_loss": 0.0, "total_pl": 0.0,
                 "roi_pct": None, "turnover": 0.0,
                 "model_pl_at_recommended_stake": None,
                 "delta_vs_model": None}
        try:
            with self._conn() as c:
                total_bets = c.execute(
                    "SELECT COUNT(*) AS n FROM mon_user_bets"
                ).fetchone()["n"]
                rows = c.execute(
                    "SELECT b.status AS b_status, b.actual_stake, b.actual_odds, "
                    "b.profit_loss AS user_pl, "
                    "t.recommended_stake, t.decimal_odds, t.model_probability "
                    "FROM mon_user_bets b LEFT JOIN mon_tips t ON b.tip_id = t.id "
                    "WHERE b.status IN ('won','lost','void','cashed_out')"
                ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                self._init_schema()
                return empty
            raise
        if not rows:
            return {**empty, "total_bets": int(total_bets)}
        won = sum(1 for r in rows if r["b_status"] == "won")
        n = len(rows)
        turnover = sum(float(r["actual_stake"] or 0) for r in rows)
        total_pl = sum(float(r["user_pl"] or 0) for r in rows)
        model_pl = 0.0
        for r in rows:
            rec = float(r["recommended_stake"] or 0)
            if rec <= 0:
                continue
            if r["b_status"] == "won":
                model_pl += rec * (float(r["actual_odds"] or 0) - 1)
            elif r["b_status"] == "lost":
                model_pl -= rec
        return {
            "total_bets": int(total_bets),
            "settled_bets": n,
            "settled_count": n,
            "win_rate": round(won / n, 3),
            "hit_rate": round(won / n, 3),
            "turnover": round(turnover, 2),
            "total_pl": round(total_pl, 2),
            "profit_loss": round(total_pl, 2),
            "roi_pct": round((total_pl / turnover) * 100, 2) if turnover else None,
            "model_pl_at_recommended_stake": round(model_pl, 2),
            "delta_vs_model": round(total_pl - model_pl, 2),
        }

    def stats(self) -> Dict[str, Any]:
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT status, COUNT(*) AS n, COALESCE(SUM(profit_loss),0) AS pl, "
                    "COALESCE(SUM(actual_stake),0) AS turnover FROM mon_tips GROUP BY status"
                ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                import logging as _lg
                _lg.getLogger("dmai.monetisation").warning("mon_tips missing — re-creating schema")
                self._init_schema()
                rows = []
            else:
                raise
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
