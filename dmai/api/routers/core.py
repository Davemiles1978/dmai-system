"""Core API routes: health, status, ask, events."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter

from dmai.api.schemas.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    StatusResponse,
)
from dmai.core.bus import bus
from dmai.core.orchestrator import orchestrator

router = APIRouter(tags=["core"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok")


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Return overall system status."""
    return StatusResponse(**orchestrator.status())


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Route a free-form prompt to the AI hub."""
    hub = orchestrator.registry.get("ai_hub")
    if hub is None:
        return AskResponse(error="ai_hub not loaded")
    result = await hub.query(req.prompt, req.model)  # type: ignore[attr-defined]
    return AskResponse(
        text=result.get("text", ""),
        model=result.get("model", "none"),
        error=result.get("error"),
    )


@router.get("/events")
async def events(limit: int = 100, event_type: Optional[str] = None) -> dict[str, Any]:
    """Return recent events from the bus log."""
    return {"events": bus.get_recent_events(limit=limit, event_type=event_type)}
