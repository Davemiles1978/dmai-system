"""OPAR loop — the Observe / Plan / Act / Reflect execution cycle.

Every agent task flows through this loop. The loop is agent-agnostic: it
receives an agent object exposing async ``observe``, ``plan``, ``act`` and
``reflect`` methods and drives them in order, emitting lifecycle events and
persisting the result. High-risk plans pause for operator approval.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from dmai.core.bus import Event, EventBus, EventType

logger = logging.getLogger("dmai.opar")

RISK_APPROVAL_THRESHOLD = 0.7


@dataclass
class OPARContext:
    """Immutable description of the task entering the loop."""

    task_type: str
    input_data: dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Result of the Observe phase."""

    context: OPARContext
    current_state: dict[str, Any] = field(default_factory=dict)
    available_tools: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priority: int = 5


@dataclass
class PlannedStep:
    """A single planned action within a :class:`Plan`."""

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    fallback: Optional[str] = None


@dataclass
class Plan:
    """Result of the Plan phase."""

    observation: Observation
    steps: list[PlannedStep] = field(default_factory=list)
    estimated_duration: float = 0.0
    risk_score: float = 0.0


@dataclass
class ActionResult:
    """Result of the Act phase."""

    plan: Plan
    steps_executed: int = 0
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class Reflection:
    """Result of the Reflect phase."""

    result: ActionResult
    lessons_learned: list[str] = field(default_factory=list)
    performance_score: float = 0.0
    suggestions: list[str] = field(default_factory=list)


@dataclass
class OPARResult:
    """The complete record of one OPAR run."""

    context: OPARContext
    observation: Optional[Observation] = None
    plan: Optional[Plan] = None
    result: Optional[ActionResult] = None
    reflection: Optional[Reflection] = None
    total_duration_ms: float = 0.0
    success: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly summary."""
        return {
            "task_id": self.context.task_id,
            "task_type": self.context.task_type,
            "agent_id": self.context.agent_id,
            "success": self.success,
            "total_duration_ms": self.total_duration_ms,
            "risk_score": self.plan.risk_score if self.plan else None,
            "performance_score": (
                self.reflection.performance_score if self.reflection else None
            ),
            "outputs": self.result.outputs if self.result else {},
            "errors": self.result.errors if self.result else [],
            "created_at": self.created_at.isoformat(),
        }


class OPARAgent(Protocol):
    """Structural type an agent must satisfy to run through the loop."""

    component_id: str

    async def observe(self, context: OPARContext) -> Observation: ...
    async def plan(self, observation: Observation) -> Plan: ...
    async def act(self, plan: Plan) -> ActionResult: ...
    async def reflect(self, result: ActionResult) -> Reflection: ...


class OPARLoop:
    """Drives the four OPAR phases for a given agent and context."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._active: dict[str, OPARContext] = {}
        self._history: list[OPARResult] = []
        self._approvals: dict[str, asyncio.Event] = {}
        self._approval_decisions: dict[str, bool] = {}

    async def run(self, agent: OPARAgent, context: OPARContext) -> OPARResult:
        """Execute the full Observe→Plan→Act→Reflect cycle for *agent*."""
        context.agent_id = context.agent_id or getattr(agent, "component_id", "")
        started = time.perf_counter()
        self._active[context.task_id] = context
        opar = OPARResult(context=context)

        await self._bus.publish(
            Event(
                event_type=EventType.TASK_CREATED,
                source=context.agent_id,
                payload={"task_id": context.task_id, "task_type": context.task_type},
                correlation_id=context.task_id,
            )
        )

        try:
            opar.observation = await agent.observe(context)
            await self._phase_event("observe", context, {"priority": opar.observation.priority})

            opar.plan = await agent.plan(opar.observation)
            await self._phase_event(
                "plan", context, {"risk_score": opar.plan.risk_score, "steps": len(opar.plan.steps)}
            )

            if opar.plan.risk_score > RISK_APPROVAL_THRESHOLD:
                approved = await self._await_approval(context, opar.plan)
                if not approved:
                    opar.success = False
                    opar.result = ActionResult(
                        plan=opar.plan, success=False, errors=["rejected by operator"]
                    )
                    await self._finish(opar, started, failed=True)
                    return opar

            opar.result = await agent.act(opar.plan)
            await self._phase_event(
                "act", context, {"success": opar.result.success, "steps": opar.result.steps_executed}
            )

            opar.reflection = await agent.reflect(opar.result)
            await self._phase_event(
                "reflect", context, {"score": opar.reflection.performance_score}
            )

            opar.success = opar.result.success
            await self._finish(opar, started, failed=not opar.success)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("OPAR run failed for %s", context.task_id)
            opar.success = False
            if opar.result is None:
                opar.result = ActionResult(
                    plan=opar.plan or Plan(observation=opar.observation or Observation(context)),
                    success=False,
                    errors=[str(exc)],
                )
            await self._finish(opar, started, failed=True)
        finally:
            self._active.pop(context.task_id, None)

        return opar

    async def _finish(self, opar: OPARResult, started: float, failed: bool) -> None:
        opar.total_duration_ms = (time.perf_counter() - started) * 1000.0
        self._history.append(opar)
        if len(self._history) > 500:
            self._history = self._history[-500:]

        await self._bus.publish(
            Event(
                event_type=EventType.TASK_FAILED if failed else EventType.TASK_COMPLETED,
                source=opar.context.agent_id,
                payload=opar.to_dict(),
                correlation_id=opar.context.task_id,
            )
        )
        await self._persist(opar)

    async def _phase_event(self, phase: str, ctx: OPARContext, extra: dict[str, Any]) -> None:
        await self._bus.publish(
            Event(
                event_type=f"OPAR_{phase.upper()}",
                source=ctx.agent_id,
                payload={"task_id": ctx.task_id, "phase": phase, **extra},
                correlation_id=ctx.task_id,
            )
        )

    async def _await_approval(self, ctx: OPARContext, plan: Plan) -> bool:
        """Emit an APPROVAL_REQUIRED event and block until decided."""
        approval_id = ctx.task_id
        gate = asyncio.Event()
        self._approvals[approval_id] = gate
        await self._bus.publish(
            Event(
                event_type=EventType.APPROVAL_REQUIRED,
                source=ctx.agent_id,
                payload={
                    "approval_id": approval_id,
                    "task_type": ctx.task_type,
                    "risk_score": plan.risk_score,
                    "steps": [s.action for s in plan.steps],
                },
                correlation_id=ctx.task_id,
            )
        )
        await self._persist_approval(approval_id, ctx, plan)
        try:
            await asyncio.wait_for(gate.wait(), timeout=3600)
        except asyncio.TimeoutError:
            logger.warning("Approval %s timed out; treating as rejected", approval_id)
            return False
        finally:
            self._approvals.pop(approval_id, None)
        return self._approval_decisions.pop(approval_id, False)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        """Resolve a pending approval; returns True if one was waiting."""
        gate = self._approvals.get(approval_id)
        if gate is None:
            return False
        self._approval_decisions[approval_id] = approved
        gate.set()
        return True

    def get_active_runs(self) -> list[dict[str, Any]]:
        """Return contexts for currently-executing runs."""
        return [
            {"task_id": c.task_id, "task_type": c.task_type, "agent_id": c.agent_id}
            for c in self._active.values()
        ]

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent completed OPAR runs (newest first)."""
        return [r.to_dict() for r in reversed(self._history[-limit:])]

    async def _persist(self, opar: OPARResult) -> None:
        try:
            from dmai.db.models import AgentRunModel
            from dmai.db.session import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                session.add(
                    AgentRunModel(
                        agent_id=opar.context.agent_id or "unknown",
                        task_type=opar.context.task_type,
                        success=opar.success,
                        duration_ms=opar.total_duration_ms,
                        performance_score=(
                            opar.reflection.performance_score if opar.reflection else 0.0
                        ),
                        result=opar.to_dict(),
                    )
                )
                await session.commit()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("OPAR persistence skipped: %s", exc)

    async def _persist_approval(self, approval_id: str, ctx: OPARContext, plan: Plan) -> None:
        try:
            from dmai.db.models import ApprovalModel
            from dmai.db.session import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                session.add(
                    ApprovalModel(
                        id=approval_id,
                        kind="opar_plan",
                        source=ctx.agent_id,
                        description=f"High-risk plan for {ctx.task_type}",
                        payload={"risk_score": plan.risk_score, "task_id": ctx.task_id},
                    )
                )
                await session.commit()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Approval persistence skipped: %s", exc)
