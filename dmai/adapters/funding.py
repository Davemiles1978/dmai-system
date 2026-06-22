"""Adapter wrapping ``components/funding/SelfFundingOrchestrator``."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dmai.adapters._base import AdapterBase


class FundingAdapter(AdapterBase):
    """Exposes self-funding phase control and strategy execution."""

    component_id = "self_funding"
    component_name = "Self-Funding Orchestrator"
    plane = "agent"
    version = "1.0.0"
    capabilities = ["revenue_discovery", "funding_phases", "strategy"]
    dependencies = ["ai_hub"]

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.funding.SelfFundingOrchestrator import SelfFundingOrchestrator

        return SelfFundingOrchestrator(
            data_path=Path(self._data_path("funding")),
            financial_manager=None,
            knowledge_graph=None,
            ai_hub=None,
        )

    async def get_status(self) -> dict[str, Any]:
        """Return overall funding status."""
        if self._impl is None:
            return {"error": self._init_error or "funding unavailable"}
        return await self._call(self._impl.get_status)

    async def start_phase(self, n: int) -> dict[str, Any]:
        """Start a funding phase (operator-initiated)."""
        if self._impl is None:
            return {"error": self._init_error or "funding unavailable"}
        method = {1: "start_learning", 2: "start_phase_2", 3: "start_phase_3"}.get(n)
        if method is None or not hasattr(self._impl, method):
            return {"error": f"phase {n} not supported"}
        return await self._call(getattr(self._impl, method))

    async def execute_strategy(self, strategy: dict[str, Any]) -> dict[str, Any]:
        """Enable a named strategy on a given avenue."""
        if self._impl is None:
            return {"error": self._init_error or "funding unavailable"}
        avenue = strategy.get("avenue", "")
        strategy_id = strategy.get("strategy_id", "")
        return await self._call(self._impl.enable_strategy, avenue, strategy_id)
