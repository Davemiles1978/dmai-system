"""Asynchronous in-process event bus for inter-component communication.

Components and agents publish :class:`Event` objects; subscribers register
async handlers keyed by event type (or subscribe to *all* events). Every
published event is appended to a bounded in-memory log and, when a database
session factory is available, persisted to SQLite/Postgres for the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Optional

logger = logging.getLogger("dmai.bus")

EventHandler = Callable[["Event"], Awaitable[None]]


class EventType:
    """Canonical built-in event type names used across DMAI."""

    TASK_CREATED = "TASK_CREATED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_STOPPED = "AGENT_STOPPED"
    REVENUE_RECEIVED = "REVENUE_RECEIVED"
    EVOLUTION_CYCLE_COMPLETE = "EVOLUTION_CYCLE_COMPLETE"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    COMPONENT_STATUS_CHANGED = "COMPONENT_STATUS_CHANGED"

    # Additional domain events emitted by agents.
    MARKET_OPPORTUNITY_FOUND = "MARKET_OPPORTUNITY_FOUND"
    ANALYTICS_INSIGHT = "ANALYTICS_INSIGHT"
    HEALTH_DEGRADED = "HEALTH_DEGRADED"


@dataclass
class Event:
    """A single message routed through the :class:`EventBus`."""

    event_type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the event to a JSON-friendly dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }


class EventBus:
    """Fan-out async event bus with a bounded log and optional persistence."""

    MAX_LOG = 10_000

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._global_subscribers: list[EventHandler] = []
        self._log: Deque[Event] = deque(maxlen=self.MAX_LOG)
        self._lock = asyncio.Lock()
        self._session_factory: Optional[Callable[[], Any]] = None

    def attach_persistence(self, session_factory: Callable[[], Any]) -> None:
        """Attach an async SQLAlchemy session factory for event persistence.

        Persistence is best-effort: failures never block publishing.
        """
        self._session_factory = session_factory

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register *handler* to receive events of *event_type*."""
        self._subscribers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register *handler* to receive every published event."""
        self._global_subscribers.append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler for *event_type*."""
        handlers = self._subscribers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish *event*, fanning it out to all matching subscribers."""
        async with self._lock:
            self._log.append(event)

        await self._persist(event)

        handlers = list(self._subscribers.get(event.event_type, []))
        handlers.extend(self._global_subscribers)
        if not handlers:
            return

        results = await asyncio.gather(
            *(self._safe_dispatch(h, event) for h in handlers),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):  # pragma: no cover - defensive
                logger.warning("Event handler raised: %s", res)

    async def _safe_dispatch(self, handler: EventHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Handler %s failed for %s: %s", handler, event.event_type, exc)

    async def _persist(self, event: Event) -> None:
        if self._session_factory is None:
            return
        try:
            from dmai.db.models import EventModel

            async with self._session_factory() as session:  # type: ignore[misc]
                session.add(
                    EventModel(
                        id=event.event_id,
                        event_type=event.event_type,
                        source=event.source,
                        payload=event.payload,
                        correlation_id=event.correlation_id,
                        created_at=event.timestamp,
                    )
                )
                await session.commit()
        except Exception as exc:  # pragma: no cover - persistence is best-effort
            logger.debug("Event persistence skipped: %s", exc)

    def get_recent_events(
        self, limit: int = 100, event_type: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Return the most recent events (newest first), optionally filtered."""
        events = list(self._log)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        events.reverse()
        return [e.to_dict() for e in events[:limit]]


# Module-level singleton shared by the whole runtime.
bus = EventBus()
