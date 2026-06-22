"""Adapter wrapping ``components/consciousness/global_workspace``."""

from __future__ import annotations

import os
import sys
from typing import Any

from dmai.adapters._base import AdapterBase


class ConsciousnessAdapter(AdapterBase):
    """Exposes the global-workspace consciousness model."""

    component_id = "consciousness_tracker"
    component_name = "Consciousness Tracker"
    plane = "knowledge"
    version = "1.0.0"
    capabilities = ["consciousness", "global_workspace"]
    dependencies = []

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.consciousness.global_workspace import GlobalWorkspace

        return GlobalWorkspace()

    async def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the workspace state when available."""
        if self._impl is None:
            return {"error": self._init_error or "consciousness unavailable"}
        for name in ("get_state", "snapshot", "status", "report"):
            if hasattr(self._impl, name):
                return await self._call(getattr(self._impl, name))
        return {"capacity": getattr(self._impl, "capacity", None)}
