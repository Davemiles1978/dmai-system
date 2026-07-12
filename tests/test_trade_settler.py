"""Tests for the trade_settler that closes the outcome loop on every
trade the autonomous_trader opens.

We stub out network calls (yfinance chart endpoint, Alpaca /orders)
via the ``fetcher`` monkeypatches so no live HTTP happens in tests.
Everything runs against an isolated tmp_path ledger DB.
"""

from __future__ import annotations

import os
import time
import types
import pytest

from components.ledger import ledger_db
from components.wealth import trade_settler


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    db = str(tmp_path / "ledger.db")
    monkeypatch.setenv("DMAI_LEDGER_DB", db)
    ledger_db.init_ledger_db(db)
    return db


def _insert_open(db, *, symbol="AAPL", side="buy", qty=1.0,
                 entry_price=100.0, mode="paper", opened_at=None):
    tid = ledger_db.insert_trade(
        symbol=symbol, side=side, qty=qty, mode=mode,
        entry_price=entry_price, db_path=db,
    )
    if opened_at is not None:
        # Backdate for age-gate tests.
        from components.db import safe_open_kdb
        with safe_open_kdb(db) as c:
            c.execute("UPDATE trades_ledger SET opened_at=? WHERE id=?",
                      (opened_at, tid))
            c.commit()
    return tid


def test_settler_closes_paper_long_with_profit(ledger, monkeypatch):
    """Paper long trade with a market price above entry closes as a win."""
    # Backdate so age gate passes.
    from datetime import datetime, timezone, timedelta
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    tid = _insert_open(ledger, symbol="AAPL", side="buy", qty=2.0,
                       entry_price=100.0, mode="paper", opened_at=old_ts)

    monkeypatch.setattr(trade_settler, "fetch_market_price",
                        lambda symbol: 110.0)

    summary = trade_settler.settle_once(db_path=ledger, mode_override="paper")
    assert summary["closed"] == 1
    assert summary["errors"] == 0
    row = ledger_db.get_trade(tid, db_path=ledger)
    assert row["status"] == "closed"
    assert row["exit_price"] == pytest.approx(110.0)
    # long P/L = (exit - entry) * qty = 20.0
    assert row["pnl"] == pytest.approx(20.0)


def test_settler_closes_paper_short_with_profit(ledger, monkeypatch):
    """Short trade with a market price below entry closes as a win."""
    from datetime import datetime, timezone, timedelta
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    tid = _insert_open(ledger, symbol="TSLA", side="sell", qty=1.0,
                       entry_price=200.0, mode="paper", opened_at=old_ts)

    monkeypatch.setattr(trade_settler, "fetch_market_price",
                        lambda symbol: 180.0)

    summary = trade_settler.settle_once(db_path=ledger, mode_override="paper")
    assert summary["closed"] == 1
    row = ledger_db.get_trade(tid, db_path=ledger)
    # short P/L = (entry - exit) * qty = 20.0
    assert row["pnl"] == pytest.approx(20.0)


def test_settler_respects_min_age(ledger, monkeypatch):
    """A trade opened seconds ago is skipped by the age gate."""
    tid = _insert_open(ledger, symbol="AAPL", side="buy",
                       qty=1.0, entry_price=100.0, mode="paper")
    monkeypatch.setattr(trade_settler, "fetch_market_price",
                        lambda symbol: 110.0)
    summary = trade_settler.settle_once(db_path=ledger, mode_override="paper")
    # young trade filtered out
    assert summary["closed"] == 0
    assert summary["too_young"] >= 1
    row = ledger_db.get_trade(tid, db_path=ledger)
    assert row["status"] == "open"


def test_settler_handles_missing_price_gracefully(ledger, monkeypatch):
    """When fetch_market_price returns None the row stays open."""
    from datetime import datetime, timezone, timedelta
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    tid = _insert_open(ledger, symbol="AAPL", side="buy",
                       qty=1.0, entry_price=100.0, mode="paper",
                       opened_at=old_ts)
    monkeypatch.setattr(trade_settler, "fetch_market_price",
                        lambda symbol: None)
    summary = trade_settler.settle_once(db_path=ledger, mode_override="paper")
    assert summary["closed"] == 0
    assert summary["no_price"] >= 1
    row = ledger_db.get_trade(tid, db_path=ledger)
    assert row["status"] == "open"


def test_settler_idempotent(ledger, monkeypatch):
    """Running settle_once twice doesn't double-close or corrupt rows."""
    from datetime import datetime, timezone, timedelta
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    tid = _insert_open(ledger, symbol="AAPL", side="buy",
                       qty=1.0, entry_price=100.0, mode="paper",
                       opened_at=old_ts)
    monkeypatch.setattr(trade_settler, "fetch_market_price",
                        lambda symbol: 105.0)
    s1 = trade_settler.settle_once(db_path=ledger, mode_override="paper")
    s2 = trade_settler.settle_once(db_path=ledger, mode_override="paper")
    assert s1["closed"] == 1
    # Second pass finds no open rows for this symbol.
    assert s2["closed"] == 0
    row = ledger_db.get_trade(tid, db_path=ledger)
    assert row["status"] == "closed"
    assert row["pnl"] == pytest.approx(5.0)


def test_start_settler_loop_alive_check_pattern(monkeypatch):
    """start_settler_loop must not spawn a second thread if the first
    is already alive - matches the fresh_blood respawn-guard pattern
    (PR F1) so gunicorn preload_app+fork() doesn't leave dead threads.
    """
    # Stub out the actual work so the loop doesn't hit the network.
    monkeypatch.setattr(trade_settler, "settle_once",
                        lambda **kw: {"closed": 0})
    trade_settler._LOOP = None
    l1 = trade_settler.start_settler_loop(poll_seconds=60)
    try:
        assert l1 is not None
        assert l1._thread is not None
        assert l1._thread.is_alive()
        l2 = trade_settler.start_settler_loop(poll_seconds=60)
        # Same loop returned when already alive.
        assert l2 is l1
    finally:
        l1.stop()
