"""FastAPI router exposing the component registry at ``/registry``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dmai.registry.registry import registry

router = APIRouter(prefix="/registry", tags=["registry"])


class ComponentManifest(BaseModel):
    """Request body for installing a new component."""

    id: str
    name: str
    version: str = "1.0.0"
    plane: str = "agent"
    capabilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] = Field(default_factory=dict)
    entry_point: str
    status: str = "disabled"
    config: dict[str, Any] = Field(default_factory=dict)


@router.get("/components")
async def list_components() -> dict[str, Any]:
    """List all registered components with status."""
    return {"components": registry.list_all()}


@router.get("/components/{component_id}")
async def get_component(component_id: str) -> dict[str, Any]:
    """Return a single component's manifest, status, and health."""
    items = {c["id"]: c for c in registry.list_all()}
    if component_id not in items:
        raise HTTPException(status_code=404, detail="component not found")
    inst = registry.get(component_id)
    health = None
    if inst is not None:
        health = (await inst.health_check()).to_dict()
    return {"component": items[component_id], "health": health}


@router.post("/components/{component_id}/enable")
async def enable_component(component_id: str) -> dict[str, Any]:
    """Enable a component, loading it first if necessary."""
    if registry.get(component_id) is None:
        await registry.load(component_id)
    await registry.enable(component_id)
    return {"status": "enabled", "component_id": component_id}


@router.post("/components/{component_id}/disable")
async def disable_component(component_id: str) -> dict[str, Any]:
    """Disable a loaded component."""
    await registry.disable(component_id)
    return {"status": "disabled", "component_id": component_id}


@router.post("/components/{component_id}/reload")
async def reload_component(component_id: str) -> dict[str, Any]:
    """Hot-reload a component."""
    await registry.reload(component_id)
    return {"status": "reloaded", "component_id": component_id}


@router.post("/install")
async def install_component(manifest: ComponentManifest) -> dict[str, Any]:
    """Install a new component from a manifest."""
    result = await registry.install_from_dict(manifest.model_dump())
    return {"status": "installed", "manifest": result}


@router.delete("/components/{component_id}")
async def uninstall_component(component_id: str) -> dict[str, Any]:
    """Uninstall a component entirely."""
    await registry.uninstall(component_id)
    return {"status": "uninstalled", "component_id": component_id}


@router.get("/health")
async def registry_health() -> dict[str, Any]:
    """Return aggregate health for all loaded components."""
    return await registry.health_summary()
