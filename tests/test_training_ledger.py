"""Tests for components/monetisation/training_ledger.py (PR ZZ-1a).

Covers:
  * Schema init is idempotent
  * Never-zero-stake rule (record refuses stake <= 0 / None / missing fields)
  * Per-day dedup on (event_name, market, selection, date)
  * Settlement P/L math for won/lost/void tips
  * Settlement P/L math for buy/long and sell/short trades
  * Aggregate performance stats (win_rate, ROI, turnover, settled_count)
  * ready_for_live gate reads thresholds correctly
"""
from __future__ import annotations
import logging
import os
import tempfile

import pytest


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("DATA_PATH", str(d) + "/")
    return str(d / "dmai_knowledge.db")


@pytest.fixture()
def tl(db_path):
    logging.disable(logging.CRITICAL)
    from components.monetisation import training_ledger as t
    t.init_schema(db_path)
    return t


# ── schema ────────────────────────────────────────────────────────────────────

def test_init_schema_idempotent(tl, db_path):
    tl.init_schema(db_path)
    tl.init_schema(db_path)  # second call must not raise


def test_paper_bankroll_default(monkeypatch, tl):
    monkeypatch.delenv("BETTING_PAPER_BANKROLL", raising=False)
    assert tl.paper_bankroll() == 100.0


def test_paper_bankroll_env_override(monkeypatch, tl):
    monkeypatch.setenv("BETTING_PAPER_BANKROLL", "500")
    assert tl.paper_bankroll() == 500.0


def test_paper_bankroll_rejects_zero(monkeypatch, tl):
    monkeypatch.setenv("BETTING_PAPER_BANKROLL", "0")
    # Never-zero rule: falls back to 100
    assert tl.paper_bankroll() == 100.0


# ── never-zero-stake ──────────────────────────────────────────────────────────

def test_record_tip_rejects_zero_stake(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="Test 19:30", market="trap_winner", selection="Zero",
        decimal_odds=3.0, model_probability=0.3, confidence=0.5,
        expected_value=0.0, kelly_fraction=0.0, passes_ev_gate=False,
        recommended_stake=0.0, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert tid is None


def test_record_tip_rejects_none_stake(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="Test", market="trap_winner", selection="None",
        decimal_odds=3.0, model_probability=0.3, confidence=0.5,
        expected_value=0.0, kelly_fraction=0.0, passes_ev_gate=False,
        recommended_stake=None, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert tid is None


def test_record_tip_rejects_bad_odds(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="Test", market="trap_winner", selection="Even",
        decimal_odds=1.0, model_probability=0.5, confidence=0.5,
        expected_value=0.0, kelly_fraction=0.0, passes_ev_gate=False,
        recommended_stake=1.0, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert tid is None


def test_record_trade_rejects_zero_stake(tl, db_path):
    trd = tl.record_paper_trade(
        symbol="AAPL", side="buy", entry_price=200.0, qty=0.05,
        stake=0.0, paper_bankroll_amt=100.0, confidence=0.6,
        expected_value=0.02, tier="conservative", passed_ev_gate=True,
        db_path=db_path,
    )
    assert trd is None


def test_record_trade_rejects_bad_price(tl, db_path):
    trd = tl.record_paper_trade(
        symbol="AAPL", side="buy", entry_price=0.0, qty=0.05,
        stake=10.0, paper_bankroll_amt=100.0, confidence=0.6,
        expected_value=0.02, tier="conservative", passed_ev_gate=True,
        db_path=db_path,
    )
    assert trd is None


# ── dedup ─────────────────────────────────────────────────────────────────────

def test_record_tip_dedup_same_day(tl, db_path):
    first = tl.record_paper_tip(
        event_name="Monmore 19:30", market="trap_winner", selection="Frankie",
        decimal_odds=4.5, model_probability=0.28, confidence=0.7,
        expected_value=0.06, kelly_fraction=0.015, passes_ev_gate=True,
        recommended_stake=1.50, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert first is not None
    # Same event/market/selection/date → dedup
    second = tl.record_paper_tip(
        event_name="Monmore 19:30", market="trap_winner", selection="Frankie",
        decimal_odds=4.6, model_probability=0.28, confidence=0.7,
        expected_value=0.06, kelly_fraction=0.015, passes_ev_gate=True,
        recommended_stake=1.50, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert second is None


# ── settlement math ───────────────────────────────────────────────────────────

def test_settle_tip_won(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="A", market="trap_winner", selection="Dog1",
        decimal_odds=4.5, model_probability=0.28, confidence=0.7,
        expected_value=0.06, kelly_fraction=0.015, passes_ev_gate=True,
        recommended_stake=2.00, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert tl.settle_paper_tip(tid, "won", db_path=db_path) is True
    rows = tl.list_paper_tips(limit=1, db_path=db_path)
    assert rows[0]["outcome"] == "won"
    # P/L = stake * (odds - 1) = 2.00 * 3.5 = 7.00
    assert rows[0]["profit_loss"] == pytest.approx(7.00)


def test_settle_tip_lost(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="B", market="trap_winner", selection="Dog2",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7,
        expected_value=0.01, kelly_fraction=0.010, passes_ev_gate=True,
        recommended_stake=1.50, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert tl.settle_paper_tip(tid, "lost", db_path=db_path) is True
    rows = tl.list_paper_tips(limit=1, db_path=db_path)
    assert rows[0]["profit_loss"] == pytest.approx(-1.50)


def test_settle_tip_void(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="C", market="trap_winner", selection="Dog3",
        decimal_odds=5.0, model_probability=0.20, confidence=0.5,
        expected_value=0.0, kelly_fraction=0.0, passes_ev_gate=True,
        recommended_stake=1.00, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert tl.settle_paper_tip(tid, "void", db_path=db_path) is True
    rows = tl.list_paper_tips(limit=1, db_path=db_path)
    assert rows[0]["profit_loss"] == 0.0


def test_settle_tip_bad_outcome(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="D", market="trap_winner", selection="Dog4",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7,
        expected_value=0.01, kelly_fraction=0.01, passes_ev_gate=True,
        recommended_stake=1.0, paper_bankroll_amt=100.0, db_path=db_path,
    )
    with pytest.raises(ValueError):
        tl.settle_paper_tip(tid, "cancelled", db_path=db_path)


def test_settle_tip_idempotent(tl, db_path):
    tid = tl.record_paper_tip(
        event_name="E", market="trap_winner", selection="Dog5",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7,
        expected_value=0.01, kelly_fraction=0.01, passes_ev_gate=True,
        recommended_stake=1.0, paper_bankroll_amt=100.0, db_path=db_path,
    )
    assert tl.settle_paper_tip(tid, "won", db_path=db_path) is True
    # Second attempt returns False (already settled) — no double-count
    assert tl.settle_paper_tip(tid, "lost", db_path=db_path) is False


def test_settle_trade_long_profit(tl, db_path):
    trd = tl.record_paper_trade(
        symbol="AAPL", side="buy", entry_price=200.0, qty=0.5,
        stake=100.0, paper_bankroll_amt=100.0, confidence=0.6,
        expected_value=0.02, tier="conservative", passed_ev_gate=True,
        db_path=db_path,
    )
    assert tl.settle_paper_trade(trd, 210.0, db_path=db_path) is True
    rows = tl.list_paper_trades(limit=1, db_path=db_path)
    # P/L = (210 - 200) * 0.5 = 5.0
    assert rows[0]["profit_loss"] == pytest.approx(5.0)
    assert rows[0]["outcome"] == "won"


def test_settle_trade_short_profit(tl, db_path):
    trd = tl.record_paper_trade(
        symbol="AAPL", side="sell", entry_price=200.0, qty=0.5,
        stake=100.0, paper_bankroll_amt=100.0, confidence=0.6,
        expected_value=0.02, tier="conservative", passed_ev_gate=True,
        db_path=db_path,
    )
    assert tl.settle_paper_trade(trd, 190.0, db_path=db_path) is True
    rows = tl.list_paper_trades(limit=1, db_path=db_path)
    # P/L = (200 - 190) * 0.5 = 5.0
    assert rows[0]["profit_loss"] == pytest.approx(5.0)
    assert rows[0]["outcome"] == "won"


def test_settle_trade_flat(tl, db_path):
    trd = tl.record_paper_trade(
        symbol="AAPL", side="buy", entry_price=200.0, qty=0.5,
        stake=100.0, paper_bankroll_amt=100.0, confidence=0.6,
        expected_value=0.02, tier="conservative", passed_ev_gate=True,
        db_path=db_path,
    )
    assert tl.settle_paper_trade(trd, 200.0, db_path=db_path) is True
    rows = tl.list_paper_trades(limit=1, db_path=db_path)
    assert rows[0]["profit_loss"] == 0.0
    assert rows[0]["outcome"] == "void"


def test_settle_trade_bad_price(tl, db_path):
    trd = tl.record_paper_trade(
        symbol="AAPL", side="buy", entry_price=200.0, qty=0.5,
        stake=100.0, paper_bankroll_amt=100.0, confidence=0.6,
        expected_value=0.02, tier="conservative", passed_ev_gate=True,
        db_path=db_path,
    )
    assert tl.settle_paper_trade(trd, 0.0, db_path=db_path) is False


# ── performance aggregate ─────────────────────────────────────────────────────

def test_performance_empty(tl, db_path):
    perf = tl.performance(db_path=db_path)
    assert perf["bets"]["total_count"] == 0
    assert perf["bets"]["win_rate"] is None
    assert perf["trades"]["total_count"] == 0
    assert perf["ready_for_live"]["bets"]["ok"] is False
    assert perf["ready_for_live"]["trades"]["ok"] is False


def test_performance_win_rate_and_roi(tl, db_path):
    # 2 winners @ 3.0 odds, 1 loser — all £1 stake
    t1 = tl.record_paper_tip(event_name="R1", market="trap_winner", selection="W1",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7, expected_value=0.01,
        kelly_fraction=0.01, passes_ev_gate=True, recommended_stake=1.0,
        paper_bankroll_amt=100.0, db_path=db_path)
    t2 = tl.record_paper_tip(event_name="R2", market="trap_winner", selection="W2",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7, expected_value=0.01,
        kelly_fraction=0.01, passes_ev_gate=True, recommended_stake=1.0,
        paper_bankroll_amt=100.0, db_path=db_path)
    t3 = tl.record_paper_tip(event_name="R3", market="trap_winner", selection="L1",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7, expected_value=0.01,
        kelly_fraction=0.01, passes_ev_gate=True, recommended_stake=1.0,
        paper_bankroll_amt=100.0, db_path=db_path)
    tl.settle_paper_tip(t1, "won", db_path=db_path)
    tl.settle_paper_tip(t2, "won", db_path=db_path)
    tl.settle_paper_tip(t3, "lost", db_path=db_path)
    perf = tl.performance(db_path=db_path)
    # 2 wins × £2 profit − 1 loss × £1 = £3 total P/L over £3 turnover = 100% ROI
    assert perf["bets"]["won"] == 2
    assert perf["bets"]["lost"] == 1
    assert perf["bets"]["settled_count"] == 3
    assert perf["bets"]["win_rate"] == pytest.approx(2 / 3)
    assert perf["bets"]["total_pl"] == pytest.approx(3.0)
    assert perf["bets"]["turnover"] == pytest.approx(3.0)
    assert perf["bets"]["roi_pct"] == pytest.approx(100.0)


def test_ready_for_live_thresholds(tl):
    # Feed known perf → expect not-ready (won't hit 50 settled bets)
    perf = {
        "bets": {"settled_count": 60, "win_rate": 0.25, "roi_pct": 10.0},
        "trades": {"settled_count": 40, "win_rate": 0.55, "roi_pct": 3.0},
    }
    r = tl._readiness(perf)
    assert r["bets"]["ok"] is True
    assert r["trades"]["ok"] is True

    # Any threshold miss → not ok
    perf2 = {
        "bets": {"settled_count": 60, "win_rate": 0.15, "roi_pct": 10.0},  # win_rate too low
        "trades": {"settled_count": 40, "win_rate": 0.55, "roi_pct": 3.0},
    }
    r2 = tl._readiness(perf2)
    assert r2["bets"]["ok"] is False


# ── list filters ──────────────────────────────────────────────────────────────

def test_list_paper_tips_outcome_filter(tl, db_path):
    t1 = tl.record_paper_tip(event_name="X1", market="trap_winner", selection="A",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7, expected_value=0.01,
        kelly_fraction=0.01, passes_ev_gate=True, recommended_stake=1.0,
        paper_bankroll_amt=100.0, db_path=db_path)
    t2 = tl.record_paper_tip(event_name="X2", market="trap_winner", selection="B",
        decimal_odds=3.0, model_probability=0.3, confidence=0.7, expected_value=0.01,
        kelly_fraction=0.01, passes_ev_gate=True, recommended_stake=1.0,
        paper_bankroll_amt=100.0, db_path=db_path)
    tl.settle_paper_tip(t1, "won", db_path=db_path)
    won_only = tl.list_paper_tips(outcome="won", db_path=db_path)
    pending_only = tl.list_paper_tips(outcome="pending", db_path=db_path)
    assert len(won_only) == 1 and won_only[0]["outcome"] == "won"
    assert len(pending_only) == 1 and pending_only[0]["outcome"] == "pending"
