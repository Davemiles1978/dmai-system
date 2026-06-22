"""Operator control routes (pause/resume/kill, approvals)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from dmai.api.middleware.auth import require_operator
from dmai.api.schemas.models import ApprovalDecision
from dmai.core.orchestrator import orchestrator

router = APIRouter(prefix="/operator", tags=["operator"], dependencies=[Depends(require_operator)])


@router.post("/pause")
async def pause() -> dict[str, Any]:
    """Pause all task acceptance."""
    await orchestrator.pause_all()
    return {"status": "paused"}


@router.post("/resume")
async def resume() -> dict[str, Any]:
    """Resume normal operation."""
    await orchestrator.resume_all()
    return {"status": "resumed"}


@router.post("/kill")
async def kill() -> dict[str, Any]:
    """Activate the emergency kill switch."""
    await orchestrator.emergency_kill()
    return {"status": "killed"}


@router.get("/pending")
async def pending_approvals() -> dict[str, Any]:
    """List pending operator approvals."""
    pending: list[dict[str, Any]] = []
    try:
        from sqlalchemy import select

        from dmai.db.models import ApprovalModel
        from dmai.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            rows = await session.scalars(
                select(ApprovalModel).where(ApprovalModel.status == "pending")
            )
            pending = [
                {
                    "id": r.id,
                    "kind": r.kind,
                    "source": r.source,
                    "description": r.description,
                    "payload": r.payload,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception:  # pragma: no cover - works without DB
        pending = []
    return {"pending": pending}


@router.post("/approve/{request_id}")
async def approve(request_id: str, decision: ApprovalDecision | None = None) -> dict[str, Any]:
    """Approve a pending action and release any waiting OPAR run."""
    await _decide(request_id, "approved", decision.notes if decision else "")
    orchestrator.opar.resolve_approval(request_id, True)
    return {"status": "approved", "id": request_id}


@router.post("/reject/{request_id}")
async def reject(request_id: str, decision: ApprovalDecision | None = None) -> dict[str, Any]:
    """Reject a pending action and release any waiting OPAR run."""
    await _decide(request_id, "rejected", decision.notes if decision else "")
    orchestrator.opar.resolve_approval(request_id, False)
    return {"status": "rejected", "id": request_id}


async def _decide(request_id: str, status: str, notes: str) -> None:
    try:
        from dmai.db.models import ApprovalModel
        from dmai.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            row = await session.get(ApprovalModel, request_id)
            if row is None:
                return
            row.status = status
            row.decision_notes = notes
            row.decided_at = datetime.now(timezone.utc)
            await session.commit()
    except Exception:  # pragma: no cover - works without DB
        pass
