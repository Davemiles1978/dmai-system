"""Background health monitoring for registered components."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import Any, Deque

from dmai.core.bus import Event, EventBus, EventType
from dmai.registry.registry import ComponentRegistry

logger = logging.getLogger("dmai.health")

CHECK_INTERVAL_SECONDS = 60
HISTORY_PER_COMPONENT = 10

_HEALTH_SCORE = {"ok": 100, "healthy": 100, "degraded": 50, "error": 0, "unknown": 60}


class HealthMonitor:
    """Periodically checks component health and tracks short history."""

    def __init__(self, registry: ComponentRegistry, bus: EventBus) -> None:
        self._registry = registry
        self._bus = bus
        self._history: dict[str, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=HISTORY_PER_COMPONENT)
        )
        self._task: asyncio.Task | None = None
        self._last_status: dict[str, str] = {}

    def start(self) -> None:
        """Start the background monitoring loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the background monitoring loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.check_once()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Health check cycle failed: %s", exc)
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    async def check_once(self) -> dict[str, Any]:
        """Run a single health-check sweep across loaded components."""
        summary = await self._registry.health_summary()
        for cid, health in summary.items():
            self._history[cid].append(health)
            prev = self._last_status.get(cid)
            now = health.get("status", "unknown")
            if prev and prev in ("ok", "healthy") and now in ("degraded", "error"):
                await self._bus.publish(
                    Event(
                        event_type=EventType.HEALTH_DEGRADED,
                        source="health_monitor",
                        payload={"component_id": cid, "from": prev, "to": now, "health": health},
                    )
                )
            self._last_status[cid] = now
        return summary

    def get_history(self, component_id: str) -> list[dict[str, Any]]:
        """Return the recent health history for a component."""
        return list(self._history.get(component_id, []))

    async def get_system_health(self) -> dict[str, Any]:
        """Return an aggregate health score (0-100) plus per-component data."""
        summary = await self._registry.health_summary()
        if not summary:
            return {"score": 100, "components": {}, "checked": 0}
        total = sum(_HEALTH_SCORE.get(h.get("status", "unknown"), 60) for h in summary.values())
        score = round(total / len(summary))
        return {"score": score, "components": summary, "checked": len(summary)}
