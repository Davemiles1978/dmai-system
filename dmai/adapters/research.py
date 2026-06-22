"""Adapter wrapping ``components/research/ResearchOrchestrator``."""

from __future__ import annotations

import os
import sys
from typing import Any

from dmai.adapters._base import AdapterBase


class ResearchAdapter(AdapterBase):
    """Exposes the research orchestrator for URL/topic research."""

    component_id = "research"
    component_name = "Research Orchestrator"
    plane = "agent"
    version = "1.0.0"
    capabilities = ["research", "url_learning"]
    dependencies = ["ai_hub"]

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.research.ResearchOrchestrator import ResearchOrchestrator

        return ResearchOrchestrator(ai_hub=None)

    async def research(self, topic: str) -> dict[str, Any]:
        """Run research on *topic* if the orchestrator exposes a method."""
        if self._impl is None:
            return {"error": self._init_error or "research unavailable"}
        for name in ("research", "run", "investigate", "research_topic"):
            if hasattr(self._impl, name):
                return await self._call(getattr(self._impl, name), topic)
        return {"error": "no research method available"}
