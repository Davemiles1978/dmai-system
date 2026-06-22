"""Evolution / learning routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from dmai.core.orchestrator import orchestrator

router = APIRouter(prefix="/evolution", tags=["evolution"])


def _engine():
    return orchestrator.registry.get("evolution_engine")


@router.get("/status")
async def evolution_status() -> dict[str, Any]:
    """Return current evolution stage and KPIs."""
    engine = _engine()
    if engine is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "stage": await engine.get_stage(),  # type: ignore[attr-defined]
        "kpis": await engine.get_kpis(),  # type: ignore[attr-defined]
    }


@router.post("/cycle")
async def evolution_cycle() -> dict[str, Any]:
    """Trigger a single evolution / learning cycle."""
    engine = _engine()
    if engine is None:
        return {"error": "evolution_engine not loaded"}
    return {"result": await engine.run_cycle()}  # type: ignore[attr-defined]


@router.get("/history")
async def evolution_history(limit: int = 20) -> dict[str, Any]:
    """Return recent evolution cycle history from the DB."""
    history: list[dict[str, Any]] = []
    try:
        from sqlalchemy import select

        from dmai.db.models import EvolutionCycleModel
        from dmai.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            rows = await session.scalars(
                select(EvolutionCycleModel)
                .order_by(EvolutionCycleModel.created_at.desc())
                .limit(limit)
            )
            history = [
                {
                    "id": r.id,
                    "stage": r.stage,
                    "kpis": r.kpis,
                    "score": r.score,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception:  # pragma: no cover - history works without DB
        history = []
    return {"history": history}
