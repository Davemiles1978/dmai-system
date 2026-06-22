"""The :class:`BaseComponent` contract every DMAI component must satisfy."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid runtime import cycle
    from dmai.core.bus import Event, EventBus


class ComponentStatus(str, Enum):
    """Lifecycle states a component can be in."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"


@dataclass
class ComponentHealth:
    """Point-in-time health snapshot for a component."""

    status: str = "unknown"
    message: str = ""
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "status": self.status,
            "message": self.message,
            "last_checked": self.last_checked.isoformat(),
            "metrics": self.metrics,
        }


class BaseComponent(ABC):
    """Abstract base every plug-and-play component inherits from.

    Subclasses declare identity via the class attributes (``component_id`` etc.)
    and implement :meth:`initialize`, :meth:`health_check` and :meth:`shutdown`.
    The registry injects the event bus and config at load time.
    """

    component_id: str = "base_component"
    component_name: str = "Base Component"
    plane: str = "tool"
    version: str = "1.0.0"
    capabilities: list[str] = []
    dependencies: list[str] = []

    def __init__(self) -> None:
        self._bus: Optional["EventBus"] = None
        self._config: dict[str, Any] = {}
        self._status: ComponentStatus = ComponentStatus.UNLOADED
        self._logger = logging.getLogger(f"dmai.component.{self.component_id}")

    # ------------------------------------------------------------------ #
    # Abstract contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def initialize(self, config: dict[str, Any], bus: "EventBus") -> bool:
        """Prepare the component for use. Return True on success."""

    @abstractmethod
    async def health_check(self) -> ComponentHealth:
        """Return a fresh :class:`ComponentHealth` snapshot."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release all resources held by the component."""

    # ------------------------------------------------------------------ #
    # Optional overrides
    # ------------------------------------------------------------------ #
    async def on_event(self, event: "Event") -> None:
        """React to a bus event. Default: no-op."""

    async def on_config_change(self, new_config: dict[str, Any]) -> None:
        """React to a runtime config update. Default: store it."""
        self._config = new_config

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @property
    def status(self) -> ComponentStatus:
        """Current lifecycle status."""
        return self._status

    def _bind(self, config: dict[str, Any], bus: "EventBus") -> None:
        """Attach the injected config and bus (used by the registry)."""
        self._config = config or {}
        self._bus = bus

    def describe(self) -> dict[str, Any]:
        """Return a static description of this component."""
        return {
            "id": self.component_id,
            "name": self.component_name,
            "plane": self.plane,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "status": self._status.value,
        }
