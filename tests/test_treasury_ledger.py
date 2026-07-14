"""Tests for components.treasury.treasury_ledger.

Covers:
- init stamps install_ts + default FX
- FX override validation
- sync mirrors realised trades (USD -> GBP) idempotently
- sync mirrors settled bets (GBP) idempotently
- zero-start rule: rows dated before install_ts are ignored
- paper trades and pending bets are ignored
- manual entries adjust the balance
- summary + list_entries shape
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from components.ledger import ledger_db as ldb
from components.treasury import treasury_ledger as tl


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def paths(tmp_path):
    return {
        "treasury": str(tmp_path / "treasury.db"),
        "ledger":   str(tmp_path / "ledger.db"),
    }


def _seed_trade(ledger_path, *, pnl, closed_at, mode="live",
                status="closed", symbol="AAPL"):
    ldb.init_ledger_db(ledger_path)
    trade_id = ldb.insert_trade(
        symbol=symbol, side="buy", qty=1.0, mode=mode,
        entry_price=100.0, status="open",
        db_path=ledger_path,
    )
    if status == "closed":
        ldb.close_trade(
            trade_id, exit_price=101.0, pnl=pnl,
            closed_at=closed_at, db_path=ledger_path,
        )
    return trade_id


def _seed_bet(ledger_path, *, pnl, settled_at, outcome="win",
              stake=5.0, odds=3.0):
    ldb.init_ledger_db(ledger_path)
    bet_id = ldb.insert_bet(
        event="Track A / Race 1", selection="Trap 1",
        odds=odds, stake=stake, outcome="pending",
        db_path=ledger_path,
    )
    # Direct SQL update - update_bet recomputes pnl from odds/stake
    # so we bypass it for tests where we want to force a specific pnl.
    with sqlite3.connect(ledger_path) as c:
        c.execute(
            "UPDATE bets_ledger SET outcome = ?, pnl = ?, settled_at = ? "
            "WHERE id = ?",
            (outcome, pnl, settled_at, bet_id),
        )
        c.commit()
    return bet_id


# ── init + state ─────────────────────────────────────────────────────────

def test_init_stamps_state(paths):
    state = tl.init_treasury_db(paths["treasury"])
    assert state["install_ts"]
    assert state["fx_usd_gbp"] == tl.DEFAULT_USD_TO_GBP


def test_init_is_idempotent(paths):
    s1 = tl.init_treasury_db(paths["treasury"])
    s2 = tl.init_treasury_db(paths["treasury"])
    assert s1 == s2  # install_ts is preserved


def test_fx_override(paths):
    tl.init_treasury_db(paths["treasury"])
    tl.set_fx_usd_gbp(0.81, db_path=paths["treasury"])
    assert tl.get_fx_usd_gbp(paths["treasury"]) == pytest.approx(0.81)


def test_fx_rejects_non_positive(paths):
    tl.init_treasury_db(paths["treasury"])
    with pytest.raises(ValueError):
        tl.set_fx_usd_gbp(0.0, db_path=paths["treasury"])
    with pytest.raises(ValueError):
        tl.set_fx_usd_gbp(-1.0, db_path=paths["treasury"])


# ── sync from source ledgers ─────────────────────────────────────────────

def test_sync_mirrors_live_trade(paths):
    # Install treasury first so install_ts predates the trade close.
    tl.init_treasury_db(paths["treasury"])
    tl.set_fx_usd_gbp(0.80, db_path=paths["treasury"])
    _seed_trade(paths["ledger"], pnl=100.0,
                closed_at=(datetime.now(timezone.utc)
                           + timedelta(seconds=1)).isoformat())
    r = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    assert r.trades_mirrored == 1
    assert r.balance_gbp == pytest.approx(80.0, abs=0.01)


def test_sync_is_idempotent(paths):
    tl.init_treasury_db(paths["treasury"])
    tl.set_fx_usd_gbp(0.80, db_path=paths["treasury"])
    _seed_trade(paths["ledger"], pnl=50.0,
                closed_at=(datetime.now(timezone.utc)
                           + timedelta(seconds=1)).isoformat())
    r1 = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    r2 = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    assert r1.trades_mirrored == 1
    assert r2.trades_mirrored == 0  # already mirrored
    assert r2.balance_gbp == pytest.approx(40.0, abs=0.01)


def test_zero_start_ignores_pre_install_rows(paths):
    # Insert a trade that closed 1 hour before install_ts.
    old_ts = (datetime.now(timezone.utc)
              - timedelta(hours=1)).isoformat()
    _seed_trade(paths["ledger"], pnl=999.0, closed_at=old_ts)
    tl.init_treasury_db(paths["treasury"])  # install_ts = now
    r = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    assert r.trades_mirrored == 0
    assert r.balance_gbp == 0.0


def test_paper_trades_are_ignored(paths):
    tl.init_treasury_db(paths["treasury"])
    _seed_trade(paths["ledger"], pnl=100.0,
                closed_at=(datetime.now(timezone.utc)
                           + timedelta(seconds=1)).isoformat(),
                mode="paper")
    r = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    assert r.trades_mirrored == 0
    assert r.balance_gbp == 0.0


def test_sync_mirrors_settled_bet(paths):
    tl.init_treasury_db(paths["treasury"])
    _seed_bet(paths["ledger"], pnl=10.0,
              settled_at=(datetime.now(timezone.utc)
                          + timedelta(seconds=1)).isoformat(),
              outcome="win")
    r = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    assert r.bets_mirrored == 1
    assert r.balance_gbp == pytest.approx(10.0)


def test_pending_bets_are_ignored(paths):
    tl.init_treasury_db(paths["treasury"])
    _seed_bet(paths["ledger"], pnl=None,
              settled_at=None, outcome="pending")
    r = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    assert r.bets_mirrored == 0


def test_mixed_sync_and_balance(paths):
    tl.init_treasury_db(paths["treasury"])
    tl.set_fx_usd_gbp(0.80, db_path=paths["treasury"])
    now = datetime.now(timezone.utc)
    _seed_trade(paths["ledger"], pnl=100.0,
                closed_at=(now + timedelta(seconds=1)).isoformat())
    _seed_trade(paths["ledger"], pnl=-25.0,
                closed_at=(now + timedelta(seconds=2)).isoformat(),
                symbol="MSFT")
    _seed_bet(paths["ledger"], pnl=12.5,
              settled_at=(now + timedelta(seconds=3)).isoformat(),
              outcome="win")
    r = tl.sync_from_ledger(
        treasury_db_path=paths["treasury"],
        ledger_db_path=paths["ledger"],
    )
    # (100 - 25) * 0.80 + 12.5 = 60 + 12.5 = 72.5
    assert r.balance_gbp == pytest.approx(72.5)


# ── manual entries ───────────────────────────────────────────────────────

def test_manual_infra_spend(paths):
    tl.init_treasury_db(paths["treasury"])
    tl.record_manual(kind="infra_spend", amount_gbp=-18.5,
                     description="Render Jul", db_path=paths["treasury"])
    assert tl.get_balance(paths["treasury"]) == pytest.approx(-18.5)


def test_manual_rejects_bad_kind(paths):
    tl.init_treasury_db(paths["treasury"])
    with pytest.raises(ValueError):
        tl.record_manual(kind="theft", amount_gbp=-1.0,
                         db_path=paths["treasury"])


def test_summary_shape(paths):
    tl.init_treasury_db(paths["treasury"])
    tl.record_manual(kind="manual_credit", amount_gbp=50.0,
                     db_path=paths["treasury"])
    tl.record_manual(kind="infra_spend", amount_gbp=-10.0,
                     db_path=paths["treasury"])
    s = tl.get_summary(paths["treasury"])
    assert s["balance_gbp"] == pytest.approx(40.0)
    assert "install_ts" in s
    assert "by_kind" in s
    assert s["by_kind"]["manual_credit"]["count"] == 1
    assert s["by_kind"]["infra_spend"]["total_gbp"] == pytest.approx(-10.0)


def test_list_entries_orders_and_filters(paths):
    tl.init_treasury_db(paths["treasury"])
    tl.record_manual(kind="manual_credit", amount_gbp=1.0,
                     description="a", db_path=paths["treasury"])
    tl.record_manual(kind="infra_spend", amount_gbp=-2.0,
                     description="b", db_path=paths["treasury"])
    all_rows = tl.list_entries(db_path=paths["treasury"])
    assert len(all_rows) == 2
    # Newest first
    assert all_rows[0]["description"] == "b"
    filtered = tl.list_entries(kind="infra_spend",
                               db_path=paths["treasury"])
    assert len(filtered) == 1
    assert filtered[0]["kind"] == "infra_spend"
