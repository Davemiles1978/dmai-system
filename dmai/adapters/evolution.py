"""Adapter wrapping ``components/evolution/StageAwareLearningOrchestrator``."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dmai.adapters._base import AdapterBase


class EvolutionAdapter(AdapterBase):
    """Exposes the stage-aware learning / evolution orchestrator."""

    component_id = "evolution_engine"
    component_name = "Stage-Aware Evolution Engine"
    plane = "agent"
    version = "1.0.0"
    capabilities = ["learning", "self_evolution", "kpi_tracking"]
    dependencies = ["ai_hub"]

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.evolution.StageAwareLearningOrchestrator import (
            StageAwareLearningOrchestrator,
        )

        return StageAwareLearningOrchestrator(
            data_path=Path(self._data_path("evolution")),
            synthetic_network=None,
            knowledge_graph=None,
            ai_hub=None,
            pattern_synthesis=None,
        )

    async def run_cycle(self, consciousness: float = 0.0) -> dict[str, Any]:
        """Run one learning cycle and return its result summary."""
        if self._impl is None:
            return {"error": self._init_error or "evolution unavailable"}
        return await self._call(self._impl.run_learning_cycle, consciousness)

    async def get_stage(self, consciousness: float = 0.0) -> dict[str, Any]:
        """Return the current learning stage."""
        if self._impl is None:
            return {"error": self._init_error or "evolution unavailable"}
        stage = await self._call(self._impl.get_current_stage, consciousness)
        return {"stage": stage}

    async def get_kpis(self) -> dict[str, Any]:
        """Return a summary of learning KPIs."""
        if self._impl is None:
            return {"error": self._init_error or "evolution unavailable"}
        return await self._call(self._impl.get_learning_summary)
