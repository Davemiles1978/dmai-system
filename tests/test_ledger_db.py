"""Tests for the isolated performance ledger (PR #166).

Covers the twin-table schema in ``data/dmai_ledger.db``:
  - idempotent init (call twice, no error),
  - trade insert → read-back round-trip,
  - bet insert → outcome update → server-side P&L recompute,
  - index existence via sqlite_master.

The ledger lives in its own SQLite file (NOT dmai_knowledge.db) so the new
trader/tipster writers never re-contend the write mutex PRs #150–#164 fixed.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.ledger import ledger_db


@pytest.fixture
def db_path(tmp_path):
    p = str(tmp_path / "dmai_ledger.db")
    ledger_db.init_ledger_db(p)
    return p


# ── schema ────────────────────────────────────────────────────────────────────
def test_init_is_idempotent(tmp_path):
    p = str(tmp_path / "dmai_ledger.db")
    ledger_db.init_ledger_db(p)
    # Second call must not raise (CREATE TABLE/INDEX IF NOT EXISTS).
    ledger_db.init_ledger_db(p)


def test_indexes_exist(db_path):
    conn = sqlite3.connect(db_path)
    try:
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    finally:
        conn.close()
    for expected in ("ix_trades_opened", "ix_trades_mode_status",
                     "ix_bets_tipped", "ix_bets_outcome"):
        assert expected in names, f"missing index {expected}"


def test_tables_exist(db_path):
    conn = sqlite3.connect(db_path)
    try:
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert {"trades_ledger", "bets_ledger"} <= names


# ── trades ────────────────────────────────────────────────────────────────────
def test_insert_trade_round_trip(db_path):
    tid = ledger_db.insert_trade(
        symbol="NVDA", side="buy", qty=10, entry_price=100.0, stake=1000.0,
        mode="paper", confidence=0.82, db_path=db_path,
    )
    row = ledger_db.get_trade(tid, db_path=db_path)
    assert row is not None
    assert row["symbol"] == "NVDA"
    assert row["side"] == "buy"
    assert row["qty"] == 10
    assert row["entry_price"] == 100.0
    assert row["stake"] == 1000.0
    assert row["mode"] == "paper"
    assert row["status"] == "open"
    assert row["confidence"] == 0.82
    assert row["source"] == "autonomous_trader"
    assert row["opened_at"]


def test_close_open_trade_for_symbol(db_path):
    tid = ledger_db.insert_trade(
        symbol="AAPL", side="buy", qty=5, entry_price=200.0, stake=1000.0,
        mode="live", db_path=db_path,
    )
    closed_id = ledger_db.close_open_trade_for_symbol(
        "AAPL", exit_price=220.0, pnl=100.0, notes="take_profit", db_path=db_path,
    )
    assert closed_id == tid
    row = ledger_db.get_trade(tid, db_path=db_path)
    assert row["status"] == "closed"
    assert row["exit_price"] == 220.0
    assert row["pnl"] == 100.0
    assert row["closed_at"]


def test_list_trades_filters(db_path):
    ledger_db.insert_trade(symbol="A", side="buy", qty=1, mode="paper", db_path=db_path)
    ledger_db.insert_trade(symbol="B", side="buy", qty=1, mode="live", db_path=db_path)
    paper = ledger_db.list_trades(mode="paper", db_path=db_path)
    assert len(paper) == 1 and paper[0]["symbol"] == "A"
    live = ledger_db.list_trades(mode="live", db_path=db_path)
    assert len(live) == 1 and live[0]["symbol"] == "B"


# ── bets: insert → outcome update → P&L recompute ─────────────────────────────
def test_bet_insert_pending(db_path):
    bid = ledger_db.insert_bet(
        event="18:32 Sheffield", selection="Trap 1", odds=3.5,
        ev=0.12, confidence=0.6, db_path=db_path,
    )
    row = ledger_db.get_bet(bid, db_path=db_path)
    assert row["outcome"] == "pending"
    assert row["stake"] is None
    assert row["pnl"] is None
    assert row["placed_at"] is None
    assert row["ev"] == 0.12
    assert row["confidence"] == 0.6


def test_bet_win_pnl(db_path):
    bid = ledger_db.insert_bet(event="E", selection="S", odds=3.0, db_path=db_path)
    row = ledger_db.update_bet(bid, stake=5.0, outcome="win", db_path=db_path)
    # (odds-1)*stake = (3-1)*5 = 10
    assert row["pnl"] == pytest.approx(10.0)
    assert row["outcome"] == "win"
    assert row["stake"] == 5.0


def test_bet_loss_pnl(db_path):
    bid = ledger_db.insert_bet(event="E", selection="S", odds=3.0, db_path=db_path)
    row = ledger_db.update_bet(bid, stake=5.0, outcome="loss", db_path=db_path)
    assert row["pnl"] == pytest.approx(-5.0)


def test_bet_void_pnl(db_path):
    bid = ledger_db.insert_bet(event="E", selection="S", odds=3.0, db_path=db_path)
    row = ledger_db.update_bet(bid, stake=5.0, outcome="void", db_path=db_path)
    assert row["pnl"] == pytest.approx(0.0)


def test_compute_bet_pnl_direct():
    assert ledger_db.compute_bet_pnl(3.0, 5.0, "win") == pytest.approx(10.0)
    assert ledger_db.compute_bet_pnl(3.0, 5.0, "loss") == pytest.approx(-5.0)
    assert ledger_db.compute_bet_pnl(3.0, 5.0, "void") == 0.0
    assert ledger_db.compute_bet_pnl(3.0, None, "win") is None
    assert ledger_db.compute_bet_pnl(3.0, 5.0, "pending") is None


def test_update_missing_bet_returns_none(db_path):
    assert ledger_db.update_bet(9999, stake=1.0, outcome="win", db_path=db_path) is None


# ── summary ────────────────────────────────────────────────────────────────────
def test_summary_aggregates(db_path):
    t1 = ledger_db.insert_trade(symbol="A", side="buy", qty=1, mode="paper", db_path=db_path)
    ledger_db.close_open_trade_for_symbol("A", exit_price=1, pnl=50.0, db_path=db_path)
    ledger_db.insert_trade(symbol="B", side="buy", qty=1, mode="live", db_path=db_path)

    b1 = ledger_db.insert_bet(event="E1", selection="S", odds=2.0, db_path=db_path)
    ledger_db.update_bet(b1, stake=10.0, outcome="win", db_path=db_path)
    b2 = ledger_db.insert_bet(event="E2", selection="S", odds=2.0, db_path=db_path)
    ledger_db.update_bet(b2, stake=10.0, outcome="win", db_path=db_path)
    b3 = ledger_db.insert_bet(event="E3", selection="S", odds=2.0, db_path=db_path)
    ledger_db.update_bet(b3, stake=10.0, outcome="loss", db_path=db_path)

    s = ledger_db.summary(db_path=db_path)
    assert s["trades"]["total"] == 2
    assert s["trades"]["by_mode"].get("paper") == 1
    assert s["trades"]["by_mode"].get("live") == 1
    assert s["trades"]["closed"] == 1
    assert s["trades"]["total_pnl"] == pytest.approx(50.0)
    assert s["bets"]["total"] == 3
    assert s["bets"]["total_pnl"] == pytest.approx(10.0 + 10.0 - 10.0)
    assert s["bets"]["longest_win_streak"] == 2
    assert s["bets"]["avg_odds"] == pytest.approx(2.0)
