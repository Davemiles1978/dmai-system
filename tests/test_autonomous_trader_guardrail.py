"""Tests for the hard manual-only guardrail on the broker buy path (PR #165).

User policy is absolute: all bets and all trades MANUAL. AutonomousTrader's
``enabled`` flag (prod default: disabled) is a soft gate, so ``execute_buy`` —
the only method that actually places a broker order — grows a hard refusal at
the very top. A trade is permitted only when BOTH the
``AUTONOMOUS_TRADER_MANUAL_TOKEN`` env var is set AND a matching
``manual_approval_token`` kwarg is passed.

The guardrail lives on ``AggressiveTrader.execute_buy`` (components/wealth/
aggressive_trader.py) — that is the concrete method the caller chain
``_tick_inner -> _maybe_execute -> self.trader.execute_buy`` invokes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.wealth.aggressive_trader import AggressiveTrader

_TOKEN_ENV = "AUTONOMOUS_TRADER_MANUAL_TOKEN"


def _trader():
    # paper=True + dummy keys: no network touched during construction.
    return AggressiveTrader(api_key="dummy", secret_key="dummy", paper=True)


def test_execute_buy_blocked_without_env_token(monkeypatch):
    monkeypatch.delenv(_TOKEN_ENV, raising=False)
    trader = _trader()
    with pytest.raises(PermissionError):
        trader.execute_buy("AAPL", 0.9)


def test_execute_buy_blocked_when_token_mismatch(monkeypatch):
    monkeypatch.setenv(_TOKEN_ENV, "expected-token")
    trader = _trader()
    with pytest.raises(PermissionError):
        trader.execute_buy("AAPL", 0.9, manual_approval_token="wrong-token")


def test_execute_buy_proceeds_past_guardrail_with_matching_token(monkeypatch):
    monkeypatch.setenv(_TOKEN_ENV, "secret-123")
    trader = _trader()
    # Mock everything below the guardrail so no real broker call happens; a
    # sentinel return from get_account proves execution moved past the guard.
    called = {"get_account": False}

    def _fake_get_account():
        called["get_account"] = True
        return {"error": "stopped-below-guardrail"}

    monkeypatch.setattr(trader, "get_account", _fake_get_account)

    result = trader.execute_buy("AAPL", 0.9, manual_approval_token="secret-123")

    assert called["get_account"] is True  # passed the guardrail
    assert result == {"error": "stopped-below-guardrail"}
