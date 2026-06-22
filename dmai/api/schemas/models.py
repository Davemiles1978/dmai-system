"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness/health payload."""

    status: str = "ok"
    version: str = "2.0.0"


class StatusResponse(BaseModel):
    """System status payload."""

    running: bool
    paused: bool
    components: list[dict[str, Any]] = Field(default_factory=list)
    active_runs: list[dict[str, Any]] = Field(default_factory=list)


class AskRequest(BaseModel):
    """A free-form question routed to the AI hub."""

    prompt: str
    model: Optional[str] = None


class AskResponse(BaseModel):
    """The AI hub's answer."""

    text: str = ""
    model: str = "none"
    error: Optional[str] = None


class RunAgentRequest(BaseModel):
    """Trigger an agent task."""

    task_type: Optional[str] = None
    input_data: dict[str, Any] = Field(default_factory=dict)
    priority: int = 5


class ApprovalDecision(BaseModel):
    """Operator approval/rejection notes."""

    notes: str = ""
