"""Tests for the pnl_pct + cumulative pnl_pct math in the records
endpoint helpers. We exercise the pure-Python helpers directly rather
than spinning up Flask.
"""

from __future__ import annotations

import pytest

from dmai_core_complete import _stake_basis, _financial_state, _pnl_to_outcome


def test_stake_basis_prefers_explicit_stake():
    row = {"stake": 50.0, "entry_price": 100.0, "qty": 2.0}
    assert _stake_basis(row) == 50.0


def test_stake_basis_falls_back_to_notional():
    row = {"stake": None, "entry_price": 100.0, "qty": 2.0}
    assert _stake_basis(row) == 200.0


def test_stake_basis_zero_when_nothing_useful():
    assert _stake_basis({}) == 0.0
    assert _stake_basis({"stake": 0}) == 0.0
    assert _stake_basis({"stake": "abc"}) == 0.0


def test_stake_basis_handles_short_side_absolute_value():
    # negative qty representation shouldn't produce negative basis
    row = {"stake": None, "entry_price": 100.0, "qty": -2.0}
    assert _stake_basis(row) == 200.0


def test_pnl_to_outcome_win_loss_scratch():
    assert _pnl_to_outcome(10.0, "closed")  == "win"
    assert _pnl_to_outcome(-5.0, "closed")  == "loss"
    assert _pnl_to_outcome(0.0, "closed")   == "scratch"
    assert _pnl_to_outcome(None, "closed")  == "closed"
    assert _pnl_to_outcome(1.0, "open")     == "open"
    assert _pnl_to_outcome(None, None)      == "open"


def test_financial_state_roi_and_win_rate():
    fs = _financial_state(
        system="trader", mode="training",
        cum_pnl=25.0, cum_stake=100.0,
        wins=3, losses=1, settled=4,
        open_exposure=50.0,
    )
    assert fs["total_pnl"]     == 25.0
    assert fs["total_staked"]  == 100.0
    assert fs["roi_pct"]       == 25.0
    assert fs["win_count"]     == 3
    assert fs["loss_count"]    == 1
    assert fs["win_rate_pct"]  == 75.0
    assert fs["open_exposure"] == 50.0
    # bankroll may be None in tests - that's fine, just verify field exists
    assert "bankroll" in fs
    assert "note" in fs
    assert fs["note"].startswith("training")


def test_financial_state_live_mode_note():
    fs = _financial_state(
        system="betting", mode="live",
        cum_pnl=0, cum_stake=0,
        wins=0, losses=0, settled=0,
        open_exposure=0,
    )
    assert fs["roi_pct"] is None
    assert fs["win_rate_pct"] is None
    assert fs["note"].startswith("live")
