"""Adapter wrapping ``components/phase7/P7_MasterControl``."""

from __future__ import annotations

import os
import sys
from typing import Any

from dmai.adapters._base import AdapterBase
from dmai.config import settings


class MasterControlAdapter(AdapterBase):
    """Exposes goal management and kill-switch controls."""

    component_id = "master_control"
    component_name = "Master Control"
    plane = "governance"
    version = "1.0.0"
    capabilities = ["goal_setting", "risk_assessment", "kill_switch"]
    dependencies = []

    def _build_impl(self) -> Any:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from components.phase7.P7_MasterControl import MasterControl

        return MasterControl(master_key=settings.master_key)

    async def set_goal(self, goal: str) -> dict[str, Any]:
        """Register an operator goal."""
        if self._impl is None:
            return {"error": self._init_error or "master control unavailable"}
        return await self._call(self._impl.set_goal, goal)

    async def get_goals(self) -> dict[str, Any]:
        """Return current master-control status (incl. goals)."""
        if self._impl is None:
            return {"error": self._init_error or "master control unavailable"}
        return await self._call(self._impl.get_status)

    async def pause(self) -> dict[str, Any]:
        """Pause the whole system via the orchestrator."""
        from dmai.core.orchestrator import orchestrator

        await orchestrator.pause_all()
        return {"status": "paused"}

    async def resume(self) -> dict[str, Any]:
        """Resume the system via the orchestrator."""
        from dmai.core.orchestrator import orchestrator

        await orchestrator.resume_all()
        return {"status": "resumed"}

    async def kill(self) -> dict[str, Any]:
        """Activate the emergency kill switch."""
        from dmai.core.orchestrator import orchestrator

        await orchestrator.emergency_kill()
        return {"status": "killed"}
