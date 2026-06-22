"""Adapter wrapping ``components/wealth`` trading modules."""

from __future__ import annotations

import os
import sys
from typing import Any

from dmai.adapters._base import AdapterBase
from dmai.config import settings


class WealthTradingAdapter(AdapterBase):
    """Exposes positions, trade execution, and performance for the trader."""

    component_id = "wealth_trading"
    component_name = "Wealth Trading Engine"
    plane = "agent"
    version = "1.0.0"
    capabilities = ["trading", "positions", "performance"]
    dependencies = []

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.wealth.real_trading_executor import RealTradingExecutor

        return RealTradingExecutor(
            api_key=settings.alpaca_api_key or None,
            secret_key=settings.alpaca_secret_key or None,
        )

    async def get_positions(self) -> dict[str, Any]:
        """Return current open positions."""
        if self._impl is None:
            return {"error": self._init_error or "trading unavailable", "positions": []}
        for name in ("get_positions", "positions", "list_positions"):
            if hasattr(self._impl, name):
                return await self._call(getattr(self._impl, name))
        return {"positions": []}

    async def execute_trade(self, symbol: str, side: str, qty: float) -> dict[str, Any]:
        """Execute a trade — gated by operator approval unless autonomous."""
        if settings.self_funding_mode != "autonomous" and self._bus is not None:
            from dmai.core.bus import Event, EventType

            await self._bus.publish(
                Event(
                    event_type=EventType.APPROVAL_REQUIRED,
                    source=self.component_id,
                    payload={"action": "trade", "symbol": symbol, "side": side, "qty": qty},
                )
            )
            return {"status": "approval_required", "symbol": symbol, "side": side, "qty": qty}
        if self._impl is None:
            return {"error": self._init_error or "trading unavailable"}
        for name in ("execute_trade", "place_order", "submit_order"):
            if hasattr(self._impl, name):
                return await self._call(getattr(self._impl, name), symbol, side, qty)
        return {"error": "no trade method available"}

    async def get_performance(self) -> dict[str, Any]:
        """Return trading performance metrics."""
        if self._impl is None:
            return {"error": self._init_error or "trading unavailable"}
        for name in ("get_performance", "performance", "get_pnl"):
            if hasattr(self._impl, name):
                return await self._call(getattr(self._impl, name))
        return {"performance": {}}
