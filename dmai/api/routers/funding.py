"""Self-funding routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from dmai.core.orchestrator import orchestrator

router = APIRouter(prefix="/funding", tags=["funding"])


def _funding():
    return orchestrator.registry.get("self_funding")


@router.get("/status")
async def funding_status() -> dict[str, Any]:
    """Return current funding status."""
    impl = _funding()
    if impl is None:
        return {"loaded": False}
    return {"loaded": True, "status": await impl.get_status()}  # type: ignore[attr-defined]


@router.post("/start")
async def funding_start(phase: int = 1) -> dict[str, Any]:
    """Start a funding phase (operator-initiated)."""
    impl = _funding()
    if impl is None:
        return {"error": "self_funding not loaded"}
    return {"result": await impl.start_phase(phase)}  # type: ignore[attr-defined]


@router.get("/history")
async def funding_history(limit: int = 50) -> dict[str, Any]:
    """Return recent revenue ledger entries."""
    history: list[dict[str, Any]] = []
    try:
        from sqlalchemy import select

        from dmai.db.models import RevenueModel
        from dmai.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            rows = await session.scalars(
                select(RevenueModel).order_by(RevenueModel.created_at.desc()).limit(limit)
            )
            history = [
                {
                    "id": r.id,
                    "direction": r.direction,
                    "amount": r.amount,
                    "source": r.source,
                    "category": r.category,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception:  # pragma: no cover - works without DB
        history = []
    return {"history": history}
