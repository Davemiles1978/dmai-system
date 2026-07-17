"""Training ledger for paper bets and paper trades.

Records EVERY analysed pick and EVERY signal — including ones that would
be filtered out for live money — so we can measure end-to-end tip/signal
quality against a fixed paper bankroll before switching to live.

Separate from mon_tips/at_trades to keep the "for-money" tables clean.

Data-quality rules:
  * Never persist rows with empty/None/zero recommended_stake — the paper
    bankroll fallback guarantees a real positive stake.
  * Unsettled tips/trades stay pending; only settled rows contribute to
    win-rate / ROI totals.
"""
from __future__ import annotations
import logging
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

SCHEMA = [
    # Every analysed greyhound pick (day's best or not, EV gate passed or not).
    """CREATE TABLE IF NOT EXISTS training_paper_tips (
        id                  TEXT PRIMARY KEY,
        created_ts          REAL NOT NULL,
        event_name          TEXT NOT NULL,
        market              TEXT NOT NULL DEFAULT 'trap_winner',
        selection           TEXT NOT NULL,
        decimal_odds        REAL NOT NULL,
        model_probability   REAL NOT NULL,
        confidence          REAL NOT NULL,
        expected_value      REAL NOT NULL,
        kelly_fraction      REAL NOT NULL,
        passes_ev_gate      INTEGER NOT NULL DEFAULT 0,
        recommended_stake   REAL NOT NULL,      -- always > 0
        paper_bankroll      REAL NOT NULL,      -- snapshot at tip time
        rationale           TEXT,
        prediction_id       TEXT,
        outcome             TEXT NOT NULL DEFAULT 'pending',
        settled_ts          REAL,
        profit_loss         REAL,
        source              TEXT NOT NULL DEFAULT 'greyhound_runner'
    )""",
    "CREATE INDEX IF NOT EXISTS ix_tpt_created ON training_paper_tips(created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS ix_tpt_outcome ON training_paper_tips(outcome)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_tpt_dedup ON training_paper_tips(event_name, market, selection, DATE(created_ts, 'unixepoch'))",

    # Every trader signal (whether EV-gated or not).
    """CREATE TABLE IF NOT EXISTS training_paper_trades (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        created_ts          REAL NOT NULL,
        symbol              TEXT NOT NULL,
        side                TEXT NOT NULL,
        entry_price         REAL NOT NULL,
        exit_price          REAL,
        qty                 REAL NOT NULL,
        stake               REAL NOT NULL,       -- always > 0
        paper_bankroll      REAL NOT NULL,
        confidence          REAL NOT NULL DEFAULT 0,
        expected_value      REAL NOT NULL DEFAULT 0,
        tier                TEXT NOT NULL,
        passed_ev_gate      INTEGER NOT NULL DEFAULT 0,
        outcome             TEXT NOT NULL DEFAULT 'pending',
        settled_ts          REAL,
        profit_loss         REAL,
        source              TEXT NOT NULL DEFAULT 'autonomous_trader'
    )""",
    "CREATE INDEX IF NOT EXISTS ix_tpr_created ON training_paper_trades(created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS ix_tpr_outcome ON training_paper_trades(outcome)",
    "CREATE INDEX IF NOT EXISTS ix_tpr_symbol ON training_paper_trades(symbol, created_ts DESC)",
]


def _db_path() -> str:
    return os.environ.get("DATA_PATH", "data/").rstrip("/") + "/dmai_knowledge.db"


def init_schema(db_path: Optional[str] = None) -> None:
    """Create tables if not present. Safe to call at import + on every run."""
    path = db_path or _db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with safe_open_kdb(path, timeout=30.0) as c:
        c.execute("PRAGMA busy_timeout=30000")
        for stmt in SCHEMA:
            c.execute(stmt)
        c.commit()


def paper_bankroll() -> float:
    """Fixed paper bankroll used for stake sizing when the real wallet is £0.

    Overridable via BETTING_PAPER_BANKROLL. Defaults to £100.
    Never returns <= 0 — that would violate the never-zero-stake rule.
    """
    try:
        val = float(os.environ.get("BETTING_PAPER_BANKROLL", "100") or "100")
    except (TypeError, ValueError):
        val = 100.0
    return val if val > 0 else 100.0


# ── Paper tips (bets) ─────────────────────────────────────────────────────────

def record_paper_tip(
    *,
    event_name: str,
    market: str,
    selection: str,
    decimal_odds: float,
    model_probability: float,
    confidence: float,
    expected_value: float,
    kelly_fraction: float,
    passes_ev_gate: bool,
    recommended_stake: float,
    paper_bankroll_amt: float,
    rationale: str = "",
    prediction_id: Optional[str] = None,
    source: str = "greyhound_runner",
    db_path: Optional[str] = None,
) -> Optional[str]:
    """Persist a single analysed pick.

    Returns the tip id, or None if the (event, market, selection, date) row
    already exists (safe idempotent behaviour — one tip per race per day).

    Never persists with recommended_stake <= 0 or missing required fields.
    """
    # Never-zero-stake rule.
    if recommended_stake is None or float(recommended_stake) <= 0:
        logger.warning(
            "training_ledger.record_paper_tip: refusing zero/None stake for %s / %s",
            event_name, selection,
        )
        return None
    if not event_name or not selection or float(decimal_odds) <= 1.0:
        logger.warning(
            "training_ledger.record_paper_tip: refusing incomplete row for %r/%r odds=%r",
            event_name, selection, decimal_odds,
        )
        return None

    path = db_path or _db_path()
    init_schema(path)
    tid = "tpt_" + uuid.uuid4().hex[:16]
    now = time.time()
    try:
        with safe_open_kdb(path, timeout=30.0) as c:
            c.execute("PRAGMA busy_timeout=30000")
            c.execute(
                "INSERT INTO training_paper_tips ("
                "id, created_ts, event_name, market, selection, decimal_odds, "
                "model_probability, confidence, expected_value, kelly_fraction, "
                "passes_ev_gate, recommended_stake, paper_bankroll, rationale, "
                "prediction_id, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tid, now, event_name, market, selection, float(decimal_odds),
                    float(model_probability), float(confidence), float(expected_value),
                    float(kelly_fraction), 1 if passes_ev_gate else 0,
                    float(recommended_stake), float(paper_bankroll_amt),
                    rationale or "", prediction_id, source,
                ),
            )
            c.commit()
        return tid
    except sqlite3.IntegrityError:
        # Dedup constraint hit — already recorded today. That's fine.
        return None
    except Exception as e:
        logger.warning("record_paper_tip DB write failed: %s", e)
        return None


def settle_paper_tip(
    tip_id: str, outcome: str, *, db_path: Optional[str] = None,
) -> bool:
    """Settle one paper tip.

    outcome ∈ {'won', 'lost', 'void'}. P/L computed against decimal_odds and
    recommended_stake.
    """
    if outcome not in ("won", "lost", "void"):
        raise ValueError(f"bad outcome: {outcome!r}")
    path = db_path or _db_path()
    with safe_open_kdb(path, timeout=30.0) as c:
        c.execute("PRAGMA busy_timeout=30000")
        row = c.execute(
            "SELECT decimal_odds, recommended_stake, outcome FROM training_paper_tips "
            "WHERE id = ?", (tip_id,),
        ).fetchone()
        if not row:
            return False
        odds, stake, cur_outcome = float(row[0]), float(row[1]), row[2]
        if cur_outcome != "pending":
            return False  # already settled
        if outcome == "won":
            pl = stake * (odds - 1.0)
        elif outcome == "lost":
            pl = -stake
        else:  # void
            pl = 0.0
        c.execute(
            "UPDATE training_paper_tips SET outcome=?, settled_ts=?, profit_loss=? "
            "WHERE id=?",
            (outcome, time.time(), pl, tip_id),
        )
        c.commit()
    return True


# ── Paper trades ──────────────────────────────────────────────────────────────

def record_paper_trade(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    qty: float,
    stake: float,
    paper_bankroll_amt: float,
    confidence: float,
    expected_value: float,
    tier: str,
    passed_ev_gate: bool,
    source: str = "autonomous_trader",
    db_path: Optional[str] = None,
) -> Optional[int]:
    """Persist a single trader signal as a paper trade.

    Never persists with stake <= 0 or entry_price <= 0.
    """
    if stake is None or float(stake) <= 0:
        logger.warning(
            "training_ledger.record_paper_trade: refusing zero/None stake for %s/%s",
            symbol, side,
        )
        return None
    if not symbol or side not in ("buy", "sell", "long", "short") or float(entry_price) <= 0:
        logger.warning(
            "training_ledger.record_paper_trade: refusing incomplete row %r/%r/%r",
            symbol, side, entry_price,
        )
        return None

    path = db_path or _db_path()
    init_schema(path)
    try:
        with safe_open_kdb(path, timeout=30.0) as c:
            c.execute("PRAGMA busy_timeout=30000")
            cur = c.execute(
                "INSERT INTO training_paper_trades ("
                "created_ts, symbol, side, entry_price, qty, stake, paper_bankroll, "
                "confidence, expected_value, tier, passed_ev_gate, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(), symbol, side, float(entry_price), float(qty),
                    float(stake), float(paper_bankroll_amt), float(confidence),
                    float(expected_value), tier, 1 if passed_ev_gate else 0, source,
                ),
            )
            c.commit()
            return cur.lastrowid
    except Exception as e:
        logger.warning("record_paper_trade DB write failed: %s", e)
        return None


def settle_paper_trade(
    trade_id: int, exit_price: float, *, db_path: Optional[str] = None,
) -> bool:
    """Settle one paper trade using its stored qty and side."""
    if exit_price is None or float(exit_price) <= 0:
        return False
    path = db_path or _db_path()
    with safe_open_kdb(path, timeout=30.0) as c:
        c.execute("PRAGMA busy_timeout=30000")
        row = c.execute(
            "SELECT side, entry_price, qty, outcome FROM training_paper_trades "
            "WHERE id = ?", (trade_id,),
        ).fetchone()
        if not row or row[3] != "pending":
            return False
        side = row[0]
        entry = float(row[1])
        qty = float(row[2])
        if side in ("buy", "long"):
            pl = (float(exit_price) - entry) * qty
        else:
            pl = (entry - float(exit_price)) * qty
        outcome = "won" if pl > 0 else ("lost" if pl < 0 else "void")
        c.execute(
            "UPDATE training_paper_trades SET exit_price=?, outcome=?, settled_ts=?, "
            "profit_loss=? WHERE id=?",
            (float(exit_price), outcome, time.time(), pl, trade_id),
        )
        c.commit()
    return True


# ── Read models ───────────────────────────────────────────────────────────────

def list_paper_tips(
    *, limit: int = 100, outcome: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = db_path or _db_path()
    init_schema(path)
    q = (
        "SELECT id, created_ts, event_name, market, selection, decimal_odds, "
        "model_probability, confidence, expected_value, kelly_fraction, "
        "passes_ev_gate, recommended_stake, paper_bankroll, rationale, "
        "prediction_id, outcome, settled_ts, profit_loss, source "
        "FROM training_paper_tips"
    )
    args: List[Any] = []
    if outcome:
        q += " WHERE outcome = ?"
        args.append(outcome)
    q += " ORDER BY created_ts DESC LIMIT ?"
    args.append(int(limit))
    cols = [
        "id", "created_ts", "event_name", "market", "selection", "decimal_odds",
        "model_probability", "confidence", "expected_value", "kelly_fraction",
        "passes_ev_gate", "recommended_stake", "paper_bankroll", "rationale",
        "prediction_id", "outcome", "settled_ts", "profit_loss", "source",
    ]
    with safe_open_kdb(path, timeout=30.0) as c:
        c.execute("PRAGMA busy_timeout=30000")
        rows = c.execute(q, args).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def list_paper_trades(
    *, limit: int = 100, outcome: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = db_path or _db_path()
    init_schema(path)
    cols = [
        "id", "created_ts", "symbol", "side", "entry_price", "exit_price",
        "qty", "stake", "paper_bankroll", "confidence", "expected_value",
        "tier", "passed_ev_gate", "outcome", "settled_ts", "profit_loss", "source",
    ]
    q = "SELECT " + ", ".join(cols) + " FROM training_paper_trades"
    args: List[Any] = []
    if outcome:
        q += " WHERE outcome = ?"
        args.append(outcome)
    q += " ORDER BY created_ts DESC LIMIT ?"
    args.append(int(limit))
    with safe_open_kdb(path, timeout=30.0) as c:
        c.execute("PRAGMA busy_timeout=30000")
        rows = c.execute(q, args).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def performance(*, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate performance across both paper tips and trades.

    Returns win rate, ROI, running P/L, and readiness flags per stream.
    """
    path = db_path or _db_path()
    init_schema(path)
    out: Dict[str, Any] = {"paper_bankroll": paper_bankroll()}

    with safe_open_kdb(path, timeout=30.0) as c:
        c.execute("PRAGMA busy_timeout=30000")

        # Bets
        rows = c.execute(
            "SELECT outcome, COUNT(*) n, COALESCE(SUM(profit_loss),0) pl, "
            "COALESCE(SUM(recommended_stake),0) turnover "
            "FROM training_paper_tips GROUP BY outcome"
        ).fetchall()
        bets = {"won": 0, "lost": 0, "void": 0, "pending": 0,
                "total_pl": 0.0, "turnover": 0.0}
        for r in rows:
            oc, n, pl, turn = r[0], r[1], r[2], r[3]
            bets[oc] = n
            bets["total_pl"] += float(pl or 0)
            if oc in ("won", "lost", "void"):
                bets["turnover"] += float(turn or 0)
        settled = bets["won"] + bets["lost"] + bets["void"]
        bets["settled_count"] = settled
        bets["total_count"] = settled + bets["pending"]
        bets["win_rate"] = (bets["won"] / (bets["won"] + bets["lost"])) if (bets["won"] + bets["lost"]) > 0 else None
        bets["roi_pct"] = ((bets["total_pl"] / bets["turnover"]) * 100.0) if bets["turnover"] > 0 else None
        out["bets"] = bets

        # Trades
        rows = c.execute(
            "SELECT outcome, COUNT(*) n, COALESCE(SUM(profit_loss),0) pl, "
            "COALESCE(SUM(stake),0) turnover "
            "FROM training_paper_trades GROUP BY outcome"
        ).fetchall()
        trades = {"won": 0, "lost": 0, "void": 0, "pending": 0,
                  "total_pl": 0.0, "turnover": 0.0}
        for r in rows:
            oc, n, pl, turn = r[0], r[1], r[2], r[3]
            trades[oc] = n
            trades["total_pl"] += float(pl or 0)
            if oc in ("won", "lost", "void"):
                trades["turnover"] += float(turn or 0)
        settled = trades["won"] + trades["lost"] + trades["void"]
        trades["settled_count"] = settled
        trades["total_count"] = settled + trades["pending"]
        trades["win_rate"] = (trades["won"] / (trades["won"] + trades["lost"])) if (trades["won"] + trades["lost"]) > 0 else None
        trades["roi_pct"] = ((trades["total_pl"] / trades["turnover"]) * 100.0) if trades["turnover"] > 0 else None
        out["trades"] = trades

    out["ready_for_live"] = _readiness(out)
    return out


# Live-readiness thresholds. Deliberately conservative — user's revenue path.
READY_THRESHOLDS = {
    "min_settled_bets": 50,
    "min_settled_trades": 30,
    "min_bet_win_rate": 0.20,     # greyhound win rate benchmark
    "min_bet_roi_pct": 5.0,
    "min_trade_roi_pct": 2.0,
}


def _readiness(perf: Dict[str, Any]) -> Dict[str, Any]:
    """Per-stream go/no-go against ready-for-live thresholds.

    Returns both the legacy per-stream shape (bets/trades) and a flat
    ``checks`` array the /admin/training UI (ZZ-3) renders as a checklist.
    ``overall_ok`` is the AND of every check.
    """
    bets = perf.get("bets", {})
    trades = perf.get("trades", {})
    bet_wr = bets.get("win_rate")
    bet_roi = bets.get("roi_pct")
    tr_roi = trades.get("roi_pct")

    checks = [
        {
            "label": f"\u2265 {READY_THRESHOLDS['min_settled_bets']} settled bets",
            "current": bets.get("settled_count", 0),
            "threshold": READY_THRESHOLDS["min_settled_bets"],
            "ok": bets.get("settled_count", 0) >= READY_THRESHOLDS["min_settled_bets"],
        },
        {
            "label": f"Bet win rate \u2265 {int(READY_THRESHOLDS['min_bet_win_rate']*100)}%",
            "current": None if bet_wr is None else round(bet_wr * 100, 1),
            "threshold": int(READY_THRESHOLDS["min_bet_win_rate"] * 100),
            "ok": (bet_wr or 0) >= READY_THRESHOLDS["min_bet_win_rate"],
        },
        {
            "label": f"Bet ROI \u2265 +{READY_THRESHOLDS['min_bet_roi_pct']:.1f}%",
            "current": None if bet_roi is None else round(bet_roi, 2),
            "threshold": READY_THRESHOLDS["min_bet_roi_pct"],
            "ok": (bet_roi or -1e9) >= READY_THRESHOLDS["min_bet_roi_pct"],
        },
        {
            "label": f"\u2265 {READY_THRESHOLDS['min_settled_trades']} settled trades",
            "current": trades.get("settled_count", 0),
            "threshold": READY_THRESHOLDS["min_settled_trades"],
            "ok": trades.get("settled_count", 0) >= READY_THRESHOLDS["min_settled_trades"],
        },
        {
            "label": f"Trade ROI \u2265 +{READY_THRESHOLDS['min_trade_roi_pct']:.1f}%",
            "current": None if tr_roi is None else round(tr_roi, 2),
            "threshold": READY_THRESHOLDS["min_trade_roi_pct"],
            "ok": (tr_roi or -1e9) >= READY_THRESHOLDS["min_trade_roi_pct"],
        },
    ]
    bet_ready = all(c["ok"] for c in checks[:3])
    trade_ready = all(c["ok"] for c in checks[3:])
    return {
        "bets": {
            "ok": bool(bet_ready),
            "settled_count": bets.get("settled_count", 0),
            "win_rate": bet_wr,
            "roi_pct": bet_roi,
        },
        "trades": {
            "ok": bool(trade_ready),
            "settled_count": trades.get("settled_count", 0),
            "win_rate": trades.get("win_rate"),
            "roi_pct": tr_roi,
        },
        "checks": checks,
        "overall_ok": bool(bet_ready and trade_ready),
        "thresholds": dict(READY_THRESHOLDS),
    }
