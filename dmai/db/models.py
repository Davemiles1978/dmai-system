"""SQLAlchemy 2.x ORM models for DMAI persistence.

All primary keys are string UUIDs and JSON columns hold flexible payloads so
the same schema works on both SQLite (development) and Postgres (production).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all DMAI ORM models."""


class ComponentModel(Base):
    """Persisted registry state for a single component."""

    __tablename__ = "components"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    plane: Mapped[str] = mapped_column(String(32), default="agent")
    status: Mapped[str] = mapped_column(String(32), default="disabled")
    capabilities: Mapped[dict] = mapped_column(JSON, default=list)
    dependencies: Mapped[dict] = mapped_column(JSON, default=list)
    config_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    entry_point: Mapped[str] = mapped_column(String(512), default="")
    manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class EventModel(Base):
    """Persisted event-bus message."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class TaskModel(Base):
    """A task submitted to the OPAR loop."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    task_type: Mapped[str] = mapped_column(String(128), index=True)
    agent_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_data: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="created")
    priority: Mapped[int] = mapped_column(default=5)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ApprovalModel(Base):
    """A pending operator approval request."""

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(64), default="action")
    source: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    decision_notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RevenueModel(Base):
    """An income or expense ledger entry."""

    __tablename__ = "revenue"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    direction: Mapped[str] = mapped_column(String(16), default="income")  # income|expense
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    source: Mapped[str] = mapped_column(String(128), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class EvolutionCycleModel(Base):
    """A record of a single evolution / learning cycle."""

    __tablename__ = "evolution_cycles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    stage: Mapped[str] = mapped_column(String(64), default="")
    kpis: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AgentRunModel(Base):
    """A single agent OPAR execution record."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    agent_id: Mapped[str] = mapped_column(String(128), index=True)
    task_type: Mapped[str] = mapped_column(String(128), default="")
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    performance_score: Mapped[float] = mapped_column(Float, default=0.0)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


__all__ = [
    "Base",
    "ComponentModel",
    "EventModel",
    "TaskModel",
    "ApprovalModel",
    "RevenueModel",
    "EvolutionCycleModel",
    "AgentRunModel",
]
