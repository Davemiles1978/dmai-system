"""The :class:`ComponentRegistry` — live, plug-and-play component management.

The registry is a process-wide singleton. It tracks component manifests and
live instances, drives their lifecycle (load/enable/disable/unload/reload), and
persists manifest state to the database when available (falling back to the
in-memory + JSON manifest otherwise, so it works standalone).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Type

from dmai.core.bus import Event, EventBus, EventType, bus as default_bus
from dmai.registry.component_base import (
    BaseComponent,
    ComponentHealth,
    ComponentStatus,
)
from dmai.registry.loader import ComponentLoader
from dmai.registry.manifest import BUILT_IN_COMPONENTS

logger = logging.getLogger("dmai.registry")


class ComponentRegistry:
    """Singleton registry of component manifests and live instances."""

    _instance: Optional["ComponentRegistry"] = None

    def __new__(cls) -> "ComponentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialised", False):
            return
        self._manifests: dict[str, dict[str, Any]] = {}
        self._classes: dict[str, Type[BaseComponent]] = {}
        self._instances: dict[str, BaseComponent] = {}
        self._lock = asyncio.Lock()
        self._loader = ComponentLoader()
        self._bus: EventBus = default_bus
        self._initialised = True

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #
    def set_bus(self, bus: EventBus) -> None:
        """Attach the event bus instances will be initialised with."""
        self._bus = bus

    def register(self, component_class: Type[BaseComponent], manifest_entry: dict[str, Any]) -> None:
        """Register a component *class* with its manifest entry."""
        cid = manifest_entry["id"]
        self._classes[cid] = component_class
        self._manifests[cid] = dict(manifest_entry)
        self._manifests[cid].setdefault("status", "disabled")

    async def load_all_from_manifest(self) -> None:
        """Register every built-in component and load those marked enabled."""
        for entry in BUILT_IN_COMPONENTS:
            self._manifests[entry["id"]] = dict(entry)
        # Pick up any dynamically-dropped components on disk.
        for entry in self._loader.discover():
            if "id" in entry:
                self._manifests[entry["id"]] = dict(entry)

        for cid, entry in list(self._manifests.items()):
            if entry.get("status") == "enabled":
                try:
                    await self.load(cid, entry.get("config", {}))
                    await self.enable(cid)
                except Exception as exc:
                    logger.warning("Auto-load of %s failed: %s", cid, exc)
                    entry["status"] = "error"
        await self._persist_all()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def load(self, component_id: str, config: Optional[dict[str, Any]] = None) -> BaseComponent:
        """Instantiate and initialise a component."""
        async with self._lock:
            if component_id in self._instances:
                return self._instances[component_id]

            manifest = self._manifests.get(component_id)
            if manifest is None:
                raise KeyError(f"Unknown component '{component_id}'")

            self._set_status(component_id, ComponentStatus.LOADING)
            cls = self._classes.get(component_id)
            if cls is None:
                cls = self._loader.load_from_entry_point(manifest["entry_point"])
                self._classes[component_id] = cls

            instance = cls()
            instance._bind(config or manifest.get("config", {}), self._bus)
            ok = await instance.initialize(instance._config, self._bus)
            if not ok:
                self._set_status(component_id, ComponentStatus.ERROR)
                raise RuntimeError(f"Component '{component_id}' failed to initialize")

            instance._status = ComponentStatus.DISABLED
            self._instances[component_id] = instance
            self._set_status(component_id, ComponentStatus.DISABLED)
            return instance

    async def enable(self, component_id: str) -> None:
        """Enable a loaded component (subscribing it to the bus)."""
        inst = self._require(component_id)
        inst._status = ComponentStatus.ENABLED
        self._bus.subscribe_all(inst.on_event)
        self._set_status(component_id, ComponentStatus.ENABLED)
        await self._emit_status(component_id, "enabled")

    async def disable(self, component_id: str) -> None:
        """Disable a component without unloading it."""
        inst = self._require(component_id)
        inst._status = ComponentStatus.DISABLED
        self._bus.unsubscribe(component_id, inst.on_event)
        self._set_status(component_id, ComponentStatus.DISABLED)
        await self._emit_status(component_id, "disabled")

    async def unload(self, component_id: str) -> None:
        """Fully tear down a component instance."""
        async with self._lock:
            inst = self._instances.pop(component_id, None)
            if inst is not None:
                try:
                    await inst.shutdown()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Shutdown of %s raised: %s", component_id, exc)
            self._set_status(component_id, ComponentStatus.UNLOADED)
        await self._emit_status(component_id, "unloaded")

    async def reload(self, component_id: str) -> BaseComponent:
        """Hot-reload a component: shutdown then re-init and re-enable."""
        self._set_status(component_id, ComponentStatus.UPDATING)
        was_enabled = (
            component_id in self._instances
            and self._instances[component_id].status == ComponentStatus.ENABLED
        )
        config = self._manifests.get(component_id, {}).get("config", {})
        await self.unload(component_id)
        inst = await self.load(component_id, config)
        if was_enabled:
            await self.enable(component_id)
        return inst

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get(self, component_id: str) -> Optional[BaseComponent]:
        """Return the live instance for *component_id*, or ``None``."""
        return self._instances.get(component_id)

    def list_all(self) -> list[dict[str, Any]]:
        """Return every known component with its current status."""
        out: list[dict[str, Any]] = []
        for cid, manifest in self._manifests.items():
            inst = self._instances.get(cid)
            entry = dict(manifest)
            entry["status"] = (
                inst.status.value if inst else manifest.get("status", "unloaded")
            )
            entry["loaded"] = inst is not None
            out.append(entry)
        return out

    async def health_summary(self) -> dict[str, Any]:
        """Run health checks across all loaded components."""
        results: dict[str, Any] = {}
        for cid, inst in self._instances.items():
            try:
                health = await inst.health_check()
                results[cid] = health.to_dict()
            except Exception as exc:  # pragma: no cover - defensive
                results[cid] = ComponentHealth(status="error", message=str(exc)).to_dict()
        return results

    # ------------------------------------------------------------------ #
    # Dynamic install
    # ------------------------------------------------------------------ #
    async def install_from_dict(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Install a brand-new component described by a manifest dict."""
        cid = manifest["id"]
        self._manifests[cid] = dict(manifest)
        self._manifests[cid].setdefault("status", "disabled")
        if manifest.get("status") == "enabled":
            await self.load(cid, manifest.get("config", {}))
            await self.enable(cid)
        await self._persist_one(cid)
        await self._emit_status(cid, "installed")
        return self._manifests[cid]

    async def uninstall(self, component_id: str) -> None:
        """Unload and forget a component entirely."""
        await self.unload(component_id)
        self._manifests.pop(component_id, None)
        self._classes.pop(component_id, None)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _require(self, component_id: str) -> BaseComponent:
        inst = self._instances.get(component_id)
        if inst is None:
            raise KeyError(f"Component '{component_id}' is not loaded")
        return inst

    def _set_status(self, component_id: str, status: ComponentStatus) -> None:
        if component_id in self._manifests:
            self._manifests[component_id]["status"] = status.value

    async def _emit_status(self, component_id: str, status: str) -> None:
        await self._bus.publish(
            Event(
                event_type=EventType.COMPONENT_STATUS_CHANGED,
                source="registry",
                payload={"component_id": component_id, "status": status},
            )
        )

    async def _persist_all(self) -> None:
        for cid in list(self._manifests):
            await self._persist_one(cid)

    async def _persist_one(self, component_id: str) -> None:
        manifest = self._manifests.get(component_id)
        if manifest is None:
            return
        try:
            from sqlalchemy import select

            from dmai.db.models import ComponentModel
            from dmai.db.session import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                existing = await session.get(ComponentModel, component_id)
                if existing is None:
                    session.add(
                        ComponentModel(
                            id=component_id,
                            name=manifest.get("name", component_id),
                            version=manifest.get("version", "1.0.0"),
                            plane=manifest.get("plane", "agent"),
                            status=manifest.get("status", "disabled"),
                            capabilities=manifest.get("capabilities", []),
                            dependencies=manifest.get("dependencies", []),
                            config_schema=manifest.get("config_schema", {}),
                            entry_point=manifest.get("entry_point", ""),
                            manifest=manifest,
                        )
                    )
                else:
                    existing.status = manifest.get("status", existing.status)
                    existing.manifest = manifest
                await session.commit()
                _ = select  # keep import meaningful for future queries
        except Exception as exc:  # pragma: no cover - registry works without DB
            logger.debug("Registry persistence skipped for %s: %s", component_id, exc)


# Process-wide singleton.
registry = ComponentRegistry()
