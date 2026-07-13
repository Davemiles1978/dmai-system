"""Go-live readiness engine for the trader and betting systems.

Purpose
=======
Autonomously decide when a paper-trading system has demonstrated
enough sustained edge to be considered "ready for live money".

This module is intentionally pure Python + sqlite: no framework
dependency, no network calls. It's called from a background loop
and from the /api/records/readiness endpoint.

Design notes
------------
- **Volume-only window** (not calendar-based). We look at the most
  recent N settled records and require every metric to clear its
  threshold. Rationale: an autonomous engine that fires 20 picks
  in an hour then goes quiet for 3 days makes calendar windows
  misleading. Volume is the honest measure of "the system has
  been given enough chances to prove itself".
- Trader: last 100 settled trades.
- Betting: last 200 settled bets (higher variance = more samples).
- Thresholds mirror Investopedia guidance on Sharpe (>=1 good) and
  standard prop-firm eval expectations, plus Pinnacle's yield
  guidance for sharp bettors.

Readiness signal is *edge-detected* by the monitor loop, which
persists the previous overall pass/fail state and only notifies
on state change.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("dmai.monitoring.readiness")

# ---------------------------------------------------------------------------
# Configuration ("locked in" from the user 2026-07-13). Kept as module-level
# constants so tests can monkeypatch and the endpoint can echo them back.
# ---------------------------------------------------------------------------
TRADER_WINDOW_SIZE  = 100
BETTING_WINDOW_SIZE = 200
COUNTDOWN_WITHIN    = 3   # notify daily when this close to sample requirement

TRADER_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "settled_count":  {"op": ">=", "value": TRADER_WINDOW_SIZE,
                       "label": "Settled trades",           "unit": "count"},
    "roi_pct":        {"op": ">=", "value": 8.0,
                       "label": "ROI %",                    "unit": "pct"},
    "win_rate_pct":   {"op": ">=", "value": 52.0,
                       "label": "Win rate",                 "unit": "pct"},
    "sharpe_daily":   {"op": ">=", "value": 1.0,
                       "label": "Sharpe (per-trade proxy)", "unit": "ratio"},
    "max_drawdown_pct": {"op": "<=", "value": 15.0,
                       "label": "Max drawdown",             "unit": "pct"},
    "profit_factor":  {"op": ">=", "value": 1.5,
                       "label": "Profit factor",            "unit": "ratio"},
}

BETTING_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "settled_count":  {"op": ">=", "value": BETTING_WINDOW_SIZE,
                       "label": "Settled bets",     "unit": "count"},
    "yield_pct":      {"op": ">=", "value": 3.0,
                       "label": "Yield (ROI on stake)", "unit": "pct"},
    "win_rate_pct":   {"op": ">=", "value": 53.0,
                       "label": "Win rate",         "unit": "pct"},
    "max_drawdown_pct": {"op": "<=", "value": 20.0,
                       "label": "Max drawdown",     "unit": "pct"},
    "positive_ev_ratio_pct": {"op": ">=", "value": 60.0,
                       "label": "Positive-EV bets", "unit": "pct"},
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    key:       str
    label:     str
    unit:      str          # "count" | "pct" | "ratio"
    value:     Optional[float]
    threshold: float
    op:        str          # ">=" or "<="
    passed:    bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReadinessSnapshot:
    system:            str                   # "trader" | "betting"
    window_size:       int
    sample_size:       int
    ready:             bool
    metrics:           List[MetricResult] = field(default_factory=list)
    ts:                float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system":      self.system,
            "window_size": self.window_size,
            "sample_size": self.sample_size,
            "ready":       self.ready,
            "metrics":     [m.to_dict() for m in self.metrics],
            "ts":          self.ts,
        }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _settled_only(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to rows that have a definitive outcome (win/loss/void)."""
    settled = []
    for r in rows:
        outcome = (r.get("outcome") or "").lower()
        if outcome in ("win", "loss", "won", "lost", "void"):
            settled.append(r)
    return settled


def _stake_of(row: Dict[str, Any]) -> float:
    """Best-effort stake basis. Mirrors dmai_core_complete._stake_basis
    but kept local to avoid circular imports."""
    stake = row.get("stake")
    if stake is not None:
        try:
            v = float(stake)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    entry = row.get("entry_price")
    qty = row.get("qty")
    if entry is not None and qty is not None:
        try:
            v = abs(float(entry) * float(qty))
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return 0.0


def _pnl_of(row: Dict[str, Any]) -> Optional[float]:
    p = row.get("pnl")
    if p is None:
        return None
    try:
        return float(p)
    except (TypeError, ValueError):
        return None


def _max_drawdown_pct(pnls: Sequence[float], stakes: Sequence[float]) -> float:
    """Peak-to-trough drawdown as a % of the peak cumulative stake.

    This is a *deployed-capital* drawdown proxy: it asks "how much of
    what we committed did we lose from our best point?". It's not a
    NAV drawdown (which would need an equity curve of a fixed bankroll)
    but it's the honest number for a paper-trader without a fixed
    starting balance.
    """
    if not pnls:
        return 0.0
    cum = 0.0
    peak = 0.0
    peak_stake = 0.0
    running_stake = 0.0
    worst_dd_pct = 0.0
    for pnl, stake in zip(pnls, stakes):
        cum += pnl
        running_stake += stake
        if cum > peak:
            peak = cum
            peak_stake = running_stake
        # Only measure drawdown against a positive peak; otherwise
        # a system in early losses would show 0% DD misleadingly.
        if peak > 0 and cum < peak:
            dd = peak - cum
            base = max(peak_stake, 1e-9)
            dd_pct = dd / base * 100.0
            if dd_pct > worst_dd_pct:
                worst_dd_pct = dd_pct
    return round(worst_dd_pct, 4)


def _sharpe_per_trade(pnl_pcts: Sequence[float]) -> Optional[float]:
    """Per-trade Sharpe (mean/stdev of per-trade % returns).

    We compute over per-trade returns rather than daily returns because
    the training regime has variable firing cadence; per-trade is the
    honest unit of risk for this engine. Investopedia's ">1 good, >2
    great" thresholds apply to any consistent Sharpe unit.
    """
    if len(pnl_pcts) < 2:
        return None
    n = len(pnl_pcts)
    mean = sum(pnl_pcts) / n
    variance = sum((x - mean) ** 2 for x in pnl_pcts) / (n - 1)
    stdev = math.sqrt(variance)
    if stdev == 0:
        # All returns identical - degenerate; treat as no signal.
        return None
    return round(mean / stdev, 4)


def _profit_factor(pnls: Sequence[float]) -> Optional[float]:
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    if gross_loss == 0:
        return None if gross_win == 0 else float("inf")
    return round(gross_win / gross_loss, 4)


def _positive_ev_ratio(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    counted = 0
    positive = 0
    for r in rows:
        ev = r.get("ev")
        if ev is None:
            continue
        try:
            v = float(ev)
        except (TypeError, ValueError):
            continue
        counted += 1
        if v > 0:
            positive += 1
    if counted == 0:
        return None
    return round(positive / counted * 100.0, 4)


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------
def _apply(op: str, value: Optional[float], threshold: float) -> bool:
    if value is None:
        return False
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(f"unknown op {op!r}")


def _build_snapshot(
    *, system: str, window_size: int,
    thresholds: Dict[str, Dict[str, Any]],
    computed_values: Dict[str, Optional[float]],
    sample_size: int,
) -> ReadinessSnapshot:
    metrics: List[MetricResult] = []
    for key, cfg in thresholds.items():
        v = computed_values.get(key)
        passed = _apply(cfg["op"], v, cfg["value"])
        metrics.append(MetricResult(
            key=key, label=cfg["label"], unit=cfg["unit"],
            value=v, threshold=cfg["value"], op=cfg["op"], passed=passed,
        ))
    ready = all(m.passed for m in metrics)
    return ReadinessSnapshot(
        system=system, window_size=window_size, sample_size=sample_size,
        ready=ready, metrics=metrics,
    )


def evaluate_trader(all_rows: Sequence[Dict[str, Any]]) -> ReadinessSnapshot:
    """Evaluate trader readiness from a list of trade rows.

    Rows may be newest-first or oldest-first; we sort by ts to ensure
    the drawdown/sharpe walk is chronological. Only settled rows
    contribute to the window.
    """
    settled = _settled_only(all_rows)
    # Sort chronologically. Rows lacking a ts fall to the end.
    settled_sorted = sorted(settled, key=lambda r: r.get("ts") or "")
    window = settled_sorted[-TRADER_WINDOW_SIZE:]
    sample_size = len(window)

    pnls   = [p for p in (_pnl_of(r) for r in window) if p is not None]
    stakes = [_stake_of(r) for r in window]
    # Per-trade % returns, only where we have a positive stake basis.
    pnl_pcts: List[float] = []
    wins = 0
    losses = 0
    for r in window:
        p = _pnl_of(r)
        s = _stake_of(r)
        if p is not None and s > 0:
            pnl_pcts.append(p / s * 100.0)
        outcome = (r.get("outcome") or "").lower()
        if outcome in ("win", "won"):
            wins += 1
        elif outcome in ("loss", "lost"):
            losses += 1

    total_pnl = sum(pnls)
    total_stake = sum(stakes)
    roi_pct = (total_pnl / total_stake * 100.0) if total_stake > 0 else None
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else None
    sharpe = _sharpe_per_trade(pnl_pcts)
    max_dd = _max_drawdown_pct(pnls, stakes) if pnls else None
    pf = _profit_factor(pnls)

    computed = {
        "settled_count":    float(sample_size),
        "roi_pct":          (round(roi_pct, 4)  if roi_pct  is not None else None),
        "win_rate_pct":     (round(win_rate, 4) if win_rate is not None else None),
        "sharpe_daily":     sharpe,
        "max_drawdown_pct": max_dd,
        "profit_factor":    pf,
    }
    return _build_snapshot(
        system="trader", window_size=TRADER_WINDOW_SIZE,
        thresholds=TRADER_THRESHOLDS, computed_values=computed,
        sample_size=sample_size,
    )


def evaluate_betting(all_rows: Sequence[Dict[str, Any]]) -> ReadinessSnapshot:
    settled = _settled_only(all_rows)
    settled_sorted = sorted(settled, key=lambda r: r.get("ts") or "")
    window = settled_sorted[-BETTING_WINDOW_SIZE:]
    sample_size = len(window)

    pnls   = [p for p in (_pnl_of(r) for r in window) if p is not None]
    stakes = [_stake_of(r) for r in window]
    wins = 0
    losses = 0
    for r in window:
        outcome = (r.get("outcome") or "").lower()
        if outcome in ("win", "won"):
            wins += 1
        elif outcome in ("loss", "lost"):
            losses += 1

    total_pnl = sum(pnls)
    total_stake = sum(stakes)
    # For betting, "yield" and "ROI" are the same thing: return per unit staked.
    yield_pct = (total_pnl / total_stake * 100.0) if total_stake > 0 else None
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else None
    max_dd = _max_drawdown_pct(pnls, stakes) if pnls else None
    pos_ev = _positive_ev_ratio(window)

    computed = {
        "settled_count":         float(sample_size),
        "yield_pct":             (round(yield_pct, 4) if yield_pct is not None else None),
        "win_rate_pct":          (round(win_rate, 4)  if win_rate  is not None else None),
        "max_drawdown_pct":      max_dd,
        "positive_ev_ratio_pct": pos_ev,
    }
    return _build_snapshot(
        system="betting", window_size=BETTING_WINDOW_SIZE,
        thresholds=BETTING_THRESHOLDS, computed_values=computed,
        sample_size=sample_size,
    )


# ---------------------------------------------------------------------------
# Persistence + state-change detection
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS readiness_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    system       TEXT NOT NULL,
    ready        INTEGER NOT NULL,
    sample_size  INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_readiness_history_system_ts
    ON readiness_history(system, ts DESC);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(_SCHEMA)


def load_last(conn: sqlite3.Connection, system: str) -> Optional[Dict[str, Any]]:
    """Return the most recent history row for `system`, or None."""
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT ts, system, ready, sample_size, payload_json "
        "FROM readiness_history WHERE system = ? "
        "ORDER BY id DESC LIMIT 1",
        (system,),
    ).fetchone()
    if not row:
        return None
    return {
        "ts": row[0], "system": row[1],
        "ready": bool(row[2]), "sample_size": int(row[3]),
        "payload_json": row[4],
    }


def record(conn: sqlite3.Connection, snap: ReadinessSnapshot) -> None:
    """Persist a snapshot to readiness_history."""
    import json
    _ensure_schema(conn)
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(snap.ts))
    with conn:
        conn.execute(
            "INSERT INTO readiness_history (ts, system, ready, sample_size, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts_iso, snap.system, int(snap.ready), snap.sample_size,
             json.dumps(snap.to_dict())),
        )


def classify_transition(
    prev: Optional[Dict[str, Any]], snap: ReadinessSnapshot,
) -> str:
    """Classify what happened between the previous history entry and now.

    Returns one of:
      - "first_ready"       first-ever transition into ready state
      - "regained_ready"    was previously ready, dropped, now ready again
      - "lost_ready"        was ready, now not
      - "approaching"       sample_size within COUNTDOWN_WITHIN of window
      - "no_change"         nothing worth notifying
    """
    if prev is None:
        return "first_ready" if snap.ready else "no_change"
    was_ready = bool(prev["ready"])
    if snap.ready and not was_ready:
        return "regained_ready"
    if not snap.ready and was_ready:
        return "lost_ready"
    if not snap.ready:
        # Approaching iff we're inside the countdown zone on sample size.
        # Only surface if all *other* metrics are passing, so we don't
        # spam when the system is nowhere near ready.
        non_sample_metrics = [m for m in snap.metrics
                              if m.key not in ("settled_count",)]
        if non_sample_metrics and all(m.passed for m in non_sample_metrics):
            deficit = snap.window_size - snap.sample_size
            if 0 < deficit <= COUNTDOWN_WITHIN * (snap.window_size // 100 or 1):
                # `deficit` is measured in records; for trader window=100
                # this means <=3 records to go. For betting window=200 we
                # scale so "within 3 windows" == within ~6 bets.
                return "approaching"
    return "no_change"


__all__ = [
    "TRADER_THRESHOLDS", "BETTING_THRESHOLDS",
    "TRADER_WINDOW_SIZE", "BETTING_WINDOW_SIZE",
    "MetricResult", "ReadinessSnapshot",
    "evaluate_trader", "evaluate_betting",
    "load_last", "record", "classify_transition",
]
