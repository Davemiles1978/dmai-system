"""Shared helpers for component adapters.

Existing DMAI components are synchronous and have heavy, optional dependencies.
:class:`AdapterBase` wraps them so that:

* import / construction failures never crash the registry (the adapter loads in
  a degraded state and reports it via ``health_check``);
* synchronous component methods are run in a thread pool to preserve the async
  contract of :class:`BaseComponent`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Optional

from dmai.core.bus import EventBus
from dmai.registry.component_base import (
    BaseComponent,
    ComponentHealth,
    ComponentStatus,
)


class AdapterBase(BaseComponent):
    """Base class for adapters around existing components."""

    def __init__(self) -> None:
        super().__init__()
        self._impl: Any = None
        self._init_error: Optional[str] = None

    def _build_impl(self) -> Any:
        """Construct the wrapped implementation. Override in subclasses.

        Implementations should raise on failure; the caller handles it.
        """
        raise NotImplementedError

    async def initialize(self, config: dict[str, Any], bus: EventBus) -> bool:
        """Attempt to build the wrapped component; degrade gracefully on error."""
        self._bind(config, bus)
        try:
            self._impl = await asyncio.to_thread(self._build_impl)
            self._status = ComponentStatus.DISABLED
        except Exception as exc:  # adapters must never hard-fail the registry
            self._init_error = str(exc)
            self._impl = None
            self._status = ComponentStatus.ERROR
            self._logger.warning("%s init degraded: %s", self.component_id, exc)
        # Returning True keeps the component registered even when degraded.
        return True

    async def health_check(self) -> ComponentHealth:
        """Report whether the wrapped implementation is available."""
        if self._impl is not None:
            return ComponentHealth(status="ok", message="wrapped component ready")
        return ComponentHealth(
            status="degraded",
            message=self._init_error or "wrapped component not loaded",
        )

    async def shutdown(self) -> None:
        """Drop the reference to the wrapped component."""
        self._impl = None
        self._status = ComponentStatus.UNLOADED

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a (possibly blocking) implementation method in a thread."""
        if self._impl is None:
            return {"error": self._init_error or "component unavailable"}
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.warning("%s call failed: %s", self.component_id, exc)
            return {"error": str(exc)}

    @staticmethod
    def _data_path(name: str) -> str:
        path = os.path.join(os.getcwd(), "data", name)
        os.makedirs(path, exist_ok=True)
        return path
