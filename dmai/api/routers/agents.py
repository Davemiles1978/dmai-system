"""Agent management routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from dmai.api.schemas.models import RunAgentRequest
from dmai.core.orchestrator import orchestrator

router = APIRouter(prefix="/agents", tags=["agents"])

_AGENT_IDS = {
    "market_research_agent",
    "offer_design_agent",
    "outreach_agent",
    "landing_page_agent",
    "analytics_agent",
    "coding_agent",
    "qa_critic_agent",
    "compliance_agent",
    "finance_controller_agent",
    "upgrade_lab_agent",
}


@router.get("")
async def list_agents() -> dict[str, Any]:
    """List all agent components with status."""
    agents = [c for c in orchestrator.registry.list_all() if c["id"] in _AGENT_IDS]
    return {"agents": agents}


@router.get("/{agent_id}")
async def get_agent(agent_id: str) -> dict[str, Any]:
    """Return a single agent's status and health."""
    inst = orchestrator.registry.get(agent_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="agent not loaded")
    return {"agent": inst.describe(), "health": (await inst.health_check()).to_dict()}


@router.post("/{agent_id}/run")
async def run_agent(agent_id: str, req: RunAgentRequest) -> dict[str, Any]:
    """Trigger an agent task through the OPAR loop."""
    inst = orchestrator.registry.get(agent_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="agent not loaded")
    task_type = req.task_type or (inst.capabilities[0] if inst.capabilities else "execute")
    result = await orchestrator.run_task(
        task_type=task_type,
        input_data=req.input_data,
        priority=req.priority,
        agent_id=agent_id,
    )
    return {"result": result.to_dict()}


@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: str) -> dict[str, Any]:
    """Disable (stop) an agent without unloading it."""
    await orchestrator.registry.disable(agent_id)
    return {"status": "stopped", "agent_id": agent_id}
