"""The top-level DMAI system coordinator.

The orchestrator owns the shared singletons (registry, bus, OPAR loop, health
monitor), drives startup/shutdown in dependency order, submits tasks to the
OPAR loop, and runs the periodic analytics/evolution/funding background cycles.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from dmai.core.bus import Event, EventBus, EventType, bus as default_bus
from dmai.core.opar import OPARContext, OPARLoop, OPARResult
from dmai.registry.health import HealthMonitor
from dmai.registry.registry import registry as default_registry

logger = logging.getLogger("dmai.orchestrator")

ANALYTICS_INTERVAL = 5 * 60
EVOLUTION_INTERVAL = 60 * 60
FUNDING_INTERVAL = 6 * 60 * 60


class DMAIOrchestrator:
    """Coordinates the entire DMAI runtime."""

    def __init__(self) -> None:
        self.bus: EventBus = default_bus
        self.registry = default_registry
        self.opar = OPARLoop(self.bus)
        self.health = HealthMonitor(self.registry, self.bus)
        self._paused = False
        self._running = False
        self._bg_tasks: list[asyncio.Task] = []
        self.registry.set_bus(self.bus)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Initialise all components in dependency order and start cycles."""
        if self._running:
            return
        logger.info("DMAI orchestrator starting")
        from dmai.db.session import AsyncSessionLocal

        self.bus.attach_persistence(AsyncSessionLocal)
        await self.registry.load_all_from_manifest()
        self.health.start()
        self._running = True
        self._bg_tasks = [
            asyncio.create_task(self._periodic(ANALYTICS_INTERVAL, self._analytics_cycle)),
            asyncio.create_task(self._periodic(EVOLUTION_INTERVAL, self._evolution_cycle)),
            asyncio.create_task(self._periodic(FUNDING_INTERVAL, self._funding_cycle)),
        ]
        await self.bus.publish(
            Event(event_type=EventType.AGENT_STARTED, source="orchestrator", payload={"system": "dmai"})
        )

    async def stop(self) -> None:
        """Gracefully shut down cycles, health monitor, and components."""
        logger.info("DMAI orchestrator stopping")
        self._running = False
        for task in self._bg_tasks:
            task.cancel()
        for task in self._bg_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._bg_tasks.clear()
        await self.health.stop()
        for entry in self.registry.list_all():
            if entry.get("loaded"):
                await self.registry.unload(entry["id"])
        await self.bus.publish(
            Event(event_type=EventType.AGENT_STOPPED, source="orchestrator", payload={"system": "dmai"})
        )

    # ------------------------------------------------------------------ #
    # Task submission
    # ------------------------------------------------------------------ #
    async def run_task(
        self,
        task_type: str,
        input_data: Optional[dict[str, Any]] = None,
        priority: int = 5,
        agent_id: Optional[str] = None,
    ) -> OPARResult:
        """Submit a task to the OPAR loop, routed to a suitable agent."""
        if self._paused:
            raise RuntimeError("system is paused")
        agent = self._resolve_agent(task_type, agent_id)
        if agent is None:
            raise KeyError(f"No agent available for task_type '{task_type}'")
        ctx = OPARContext(
            task_type=task_type,
            input_data=input_data or {},
            agent_id=getattr(agent, "component_id", agent_id or ""),
            metadata={"priority": priority},
        )
        return await self.opar.run(agent, ctx)

    def _resolve_agent(self, task_type: str, agent_id: Optional[str]) -> Any:
        if agent_id:
            return self.registry.get(agent_id)
        # Route by capability match.
        for entry in self.registry.list_all():
            if task_type in entry.get("capabilities", []) and entry.get("loaded"):
                return self.registry.get(entry["id"])
        # Fallback: an agent whose id starts with the task type.
        for entry in self.registry.list_all():
            if entry["id"].startswith(task_type) and entry.get("loaded"):
                return self.registry.get(entry["id"])
        return None

    # ------------------------------------------------------------------ #
    # Operator controls
    # ------------------------------------------------------------------ #
    async def pause_all(self) -> None:
        """Pause acceptance of new tasks."""
        self._paused = True
        logger.warning("DMAI paused by operator")

    async def resume_all(self) -> None:
        """Resume normal operation."""
        self._paused = False
        logger.info("DMAI resumed by operator")

    async def emergency_kill(self) -> None:
        """Hard-stop everything immediately."""
        logger.critical("EMERGENCY KILL activated")
        await self.bus.publish(
            Event(event_type=EventType.KILL_SWITCH_ACTIVATED, source="operator", payload={})
        )
        self._paused = True
        await self.stop()

    # ------------------------------------------------------------------ #
    # Background cycles
    # ------------------------------------------------------------------ #
    async def _periodic(self, interval: int, coro) -> None:
        while self._running:
            await asyncio.sleep(interval)
            if self._paused or not self._running:
                continue
            try:
                await coro()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Background cycle %s failed: %s", coro.__name__, exc)

    async def _analytics_cycle(self) -> None:
        if self.registry.get("analytics_agent"):
            await self.run_task("analytics", {"trigger": "scheduled"}, agent_id="analytics_agent")

    async def _evolution_cycle(self) -> None:
        engine = self.registry.get("evolution_engine")
        if engine and hasattr(engine, "run_cycle"):
            try:
                result = await engine.run_cycle()  # type: ignore[attr-defined]
                await self.bus.publish(
                    Event(
                        event_type=EventType.EVOLUTION_CYCLE_COMPLETE,
                        source="orchestrator",
                        payload={"result": result},
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Evolution cycle failed: %s", exc)

    async def _funding_cycle(self) -> None:
        if self.registry.get("finance_controller_agent"):
            await self.run_task(
                "budgeting", {"trigger": "scheduled"}, agent_id="finance_controller_agent"
            )

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """Return a snapshot of overall system status."""
        return {
            "running": self._running,
            "paused": self._paused,
            "components": self.registry.list_all(),
            "active_runs": self.opar.get_active_runs(),
        }


# Process-wide orchestrator singleton.
orchestrator = DMAIOrchestrator()
