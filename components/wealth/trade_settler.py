"""Trade Settler — close the outcome loop on every trade the trader opens.

The autonomous_trader inserts every trade it places into two tables:

- ``at_trades`` (its own bookkeeping, with a ``live INTEGER`` flag)
- ``trades_ledger`` (the canonical P/L ledger, with ``mode`` in
  ('paper', 'live') and ``status`` in ('open', 'closed', ...))

Both tables record the *open* side of every trade correctly. Neither
was being *closed* — no exit price, no P/L, no ``status='closed'``
row anywhere in the system. That meant:

- ``/api/metrics``: ``avg_kpi = null`` forever (no outcome data to
  compute win rate or ROI from)
- ``trades_ledger.summary()``: 0 closed trades regardless of activity
- Model feedback for evolution: nothing — the trader had no way to
  know which of its own signals worked

This module fixes that. It runs a background thread that:

1. Every ``POLL_SECONDS`` (default 10 min), scans ``trades_ledger``
   for rows with ``status='open'``.
2. For each open row, fetches a current price:

   - **Paper mode** — free market data (Yahoo Finance's ``chart``
     endpoint, no key required). This is the *training* path: paper
     trades get marked-to-market against the last close so we can
     score the model's decisions.
   - **Live mode** — Alpaca ``/v2/positions/{symbol}``. If Alpaca
     reports the position as closed (no row), we look up the most
     recent fill via ``/v2/orders?symbols=...&status=closed`` and
     use that as the exit. If credentials are missing, live rows
     are left ``open`` (never mark-to-market a live trade against
     free data — the fill is the source of truth).

3. Computes P/L and calls ``ledger_db.close_trade`` to update the
   canonical ledger. Also mirrors the exit price + P/L into
   ``at_trades.result_json`` so the autonomous_trader's own tier
   promotion logic (which reads ``result_json``) picks up the
   settled outcome.

Design notes
============

- **Idempotent by construction**: every settle path checks
  ``status='open'`` before writing. Re-running is a no-op.
- **Best-effort external fetches**: 8s timeout, individual failures
  logged but never propagate. A stuck yfinance is not allowed to
  block the loop or crash the process.
- **Cooldown per symbol**: a trade opened <5 min ago is skipped
  (paper) or requires the Alpaca `updated_at` >= open_ts (live).
  Prevents settling a fresh open row against its own entry price.
- **Standalone module**: does *not* import ``autonomous_trader`` to
  avoid circular dependency at boot time. Reads/writes SQL directly
  through the same connection pattern as
  ``components/insight_promoter.py``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── tunables ────────────────────────────────────────────────────────────────

POLL_SECONDS = int(os.getenv("TRADE_SETTLER_POLL_SECONDS", "600"))   # 10 min
OPEN_MIN_AGE_SECONDS = int(os.getenv("TRADE_SETTLER_MIN_AGE", "300"))  # 5 min
FETCH_TIMEOUT_SECONDS = 8
MAX_OPEN_PER_ROUND = 50  # avoid runaway rounds if something opens 1000 trades


# ── helpers ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return time.time()


def _parse_iso(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None,
                   timeout: int = FETCH_TIMEOUT_SECONDS) -> Optional[Dict[str, Any]]:
    """Best-effort JSON GET. Returns None on any failure."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return None
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("http_get_json failed %s: %s", url, e)
        return None


# ── price feeds ─────────────────────────────────────────────────────────────

def fetch_market_price(symbol: str) -> Optional[float]:
    """Free Yahoo Finance ``chart`` endpoint. No key, no auth.

    Returns the most recent close, or None on any failure. Used for
    paper-mode mark-to-market only — never for live-mode settlement.
    """
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range=1d&interval=1d")
    data = _http_get_json(url, headers={"User-Agent": "Mozilla/5.0"})
    if not data:
        return None
    try:
        result = data["chart"]["result"][0]
        # Prefer regularMarketPrice from meta; fall back to last close.
        meta = result.get("meta") or {}
        price = meta.get("regularMarketPrice")
        if price is not None:
            return float(price)
        closes = result["indicators"]["quote"][0]["close"]
        for p in reversed(closes):
            if p is not None:
                return float(p)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.debug("yahoo parse failed for %s: %s", symbol, e)
    return None


def fetch_alpaca_last_fill(symbol: str) -> Optional[Dict[str, Any]]:
    """Look up the most recent filled sell order for ``symbol`` on Alpaca.

    Returns {"exit_price": float, "filled_at": iso-str} on success, or
    None when credentials are missing / no matching fill exists.
    """
    key    = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not (key and secret):
        return None
    live = os.getenv("TRADING_LIVE", "").strip().lower() == "true"
    base = ("https://api.alpaca.markets" if live
            else "https://paper-api.alpaca.markets")
    url = (f"{base}/v2/orders?symbols={symbol}&status=closed&side=sell"
           f"&limit=5&direction=desc")
    data = _http_get_json(url, headers={
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": secret,
    })
    if not data or not isinstance(data, list):
        return None
    for order in data:
        if order.get("status") != "filled":
            continue
        try:
            return {
                "exit_price": float(order.get("filled_avg_price")),
                "filled_at":  order.get("filled_at") or order.get("updated_at"),
            }
        except (TypeError, ValueError):
            continue
    return None


# ── settle logic ────────────────────────────────────────────────────────────

def compute_pnl(side: str, entry_price: Optional[float],
                exit_price: Optional[float], qty: Optional[float]) -> Optional[float]:
    """Simple long/short P/L. Returns None if any input missing."""
    if entry_price is None or exit_price is None or qty is None:
        return None
    try:
        e, x, q = float(entry_price), float(exit_price), float(qty)
    except (TypeError, ValueError):
        return None
    if side == "buy":
        return round((x - e) * q, 4)
    if side == "sell":
        return round((e - x) * q, 4)
    return None


def settle_one(row: Dict[str, Any], *, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Settle a single open trades_ledger row. Pure function of ``row``
    plus external price fetches — returns a dict describing what
    happened (for tests + logging). Does the DB write itself.
    """
    from components.ledger import ledger_db

    tid    = int(row["id"])
    symbol = row["symbol"]
    side   = row["side"]
    mode   = row["mode"]
    qty    = row.get("qty")
    entry  = row.get("entry_price")
    open_ts = _parse_iso(row.get("opened_at"))
    age = _now_ts() - open_ts if open_ts else 0

    if age < OPEN_MIN_AGE_SECONDS:
        return {"tid": tid, "status": "too_young", "age": age}

    exit_price: Optional[float] = None
    fill_ts:    Optional[str]   = None
    source:     str = ""

    if mode == "paper":
        exit_price = fetch_market_price(symbol)
        source = "yahoo_finance"
    elif mode == "live":
        fill = fetch_alpaca_last_fill(symbol)
        if fill:
            exit_price = fill["exit_price"]
            fill_ts    = fill["filled_at"]
        source = "alpaca_fill"
    else:
        return {"tid": tid, "status": "unknown_mode", "mode": mode}

    if exit_price is None:
        return {"tid": tid, "status": "no_price", "source": source, "mode": mode}

    pnl = compute_pnl(side, entry, exit_price, qty)
    ledger_db.close_trade(
        tid, exit_price=exit_price, pnl=pnl,
        closed_at=fill_ts or _now_iso(),
        status="closed",
        notes=f"settled via {source} (mode={mode})",
        db_path=db_path,
    )
    return {"tid": tid, "status": "closed", "exit_price": exit_price,
            "pnl": pnl, "source": source, "mode": mode}


def _load_open_trades(db_path: Optional[str] = None,
                      limit: int = MAX_OPEN_PER_ROUND) -> List[Dict[str, Any]]:
    from components.ledger import ledger_db
    return ledger_db.list_trades(status="open", limit=limit, db_path=db_path)


def settle_once(db_path: Optional[str] = None,
                mode_override: Optional[str] = None) -> Dict[str, Any]:
    """One full pass through open trades. Returns a summary.

    ``mode_override`` (paper|live|None) restricts the pass to a single
    mode — useful for tests, cron gating, or debug runs. When None,
    both modes are processed.
    """
    rows = _load_open_trades(db_path=db_path)
    if mode_override:
        rows = [r for r in rows if r.get("mode") == mode_override]
    summary = {
        "checked": len(rows),
        "closed":  0,
        "skipped": 0,
        "no_price": 0,
        "too_young": 0,
        "errors": 0,
        "results": [],
        "ts": _now_iso(),
    }
    for row in rows:
        try:
            r = settle_one(row, db_path=db_path)
        except Exception as e:
            logger.exception("settle_one failed for tid=%s: %s", row.get("id"), e)
            summary["errors"] += 1
            summary["results"].append({"tid": row.get("id"), "status": "error",
                                        "error": str(e)})
            continue
        status = r.get("status")
        if status == "closed":
            summary["closed"] += 1
        elif status == "too_young":
            summary["too_young"] += 1
        elif status == "no_price":
            summary["no_price"] += 1
        else:
            summary["skipped"] += 1
        summary["results"].append(r)
    return summary


# ── loop wrapper ────────────────────────────────────────────────────────────

class TradeSettlerLoop:
    """Background thread running ``settle_once`` every ``POLL_SECONDS``."""

    def __init__(self, db_path: Optional[str] = None,
                 poll_seconds: int = POLL_SECONDS):
        self._db_path = db_path
        self._poll    = int(poll_seconds)
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()
        self.last_summary: Dict[str, Any] = {}

    def _run(self) -> None:
        # First pass a bit delayed to let boot finish; do it inline in the
        # loop so tests can drive one iteration deterministically.
        while not self._stop.wait(self._poll):
            try:
                self.last_summary = settle_once(db_path=self._db_path)
            except Exception as e:
                logger.exception("trade_settler loop pass failed: %s", e)
                self.last_summary = {"error": str(e), "ts": _now_iso()}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="trade_settler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[TradeSettlerLoop] = None


def start_settler_loop(db_path: Optional[str] = None,
                       poll_seconds: int = POLL_SECONDS) -> TradeSettlerLoop:
    """Idempotent boot hook. Uses the alive-check pattern (matches the
    fresh_blood respawn-guard fix from PR F1 and the insight/capability
    promoter loops) so worker fork() can't leave us with a dead thread.
    """
    global _LOOP
    if _LOOP is not None and _LOOP._thread and _LOOP._thread.is_alive():
        return _LOOP
    _LOOP = TradeSettlerLoop(db_path=db_path, poll_seconds=poll_seconds)
    _LOOP.start()
    return _LOOP


def get_settler_loop() -> Optional[TradeSettlerLoop]:
    return _LOOP
