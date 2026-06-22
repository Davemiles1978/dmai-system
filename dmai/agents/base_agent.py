"""Base class shared by all native OPAR agents.

A :class:`BaseAgent` is a :class:`BaseComponent` that implements the four OPAR
phase methods. Concrete agents override :meth:`observe`, :meth:`plan`,
:meth:`act` and :meth:`reflect`. The base provides sensible defaults plus the
``_ai_call`` helper (routed through the AI hub adapter) and result persistence.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from dmai.config import settings
from dmai.core.bus import Event, EventBus, EventType
from dmai.core.opar import (
    ActionResult,
    Observation,
    OPARContext,
    Plan,
    PlannedStep,
    Reflection,
)
from dmai.registry.component_base import (
    BaseComponent,
    ComponentHealth,
    ComponentStatus,
)


class BaseAgent(BaseComponent):
    """Common scaffolding for native DMAI agents."""

    plane = "agent"

    def __init__(self) -> None:
        super().__init__()
        self._runs = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # BaseComponent contract
    # ------------------------------------------------------------------ #
    async def initialize(self, config: dict[str, Any], bus: EventBus) -> bool:
        """Default init: store config and bus, mark ready."""
        self._bind(config, bus)
        self._status = ComponentStatus.DISABLED
        self._logger = logging.getLogger(f"dmai.agent.{self.component_id}")
        return True

    async def health_check(self) -> ComponentHealth:
        """Report status and run counters."""
        status = "ok" if self._last_error is None else "degraded"
        return ComponentHealth(
            status=status,
            message=self._last_error or "operational",
            metrics={"runs": self._runs, "state": self._status.value},
        )

    async def shutdown(self) -> None:
        """Default shutdown: nothing to release."""
        self._status = ComponentStatus.UNLOADED

    # ------------------------------------------------------------------ #
    # OPAR entry point
    # ------------------------------------------------------------------ #
    async def run_opar(self, task_type: str, input_data: dict[str, Any]) -> Any:
        """Submit a task to the shared OPAR loop for this agent."""
        from dmai.core.orchestrator import orchestrator

        ctx = OPARContext(task_type=task_type, input_data=input_data, agent_id=self.component_id)
        return await orchestrator.opar.run(self, ctx)

    # ------------------------------------------------------------------ #
    # Default OPAR phases (overridable)
    # ------------------------------------------------------------------ #
    async def observe(self, context: OPARContext) -> Observation:
        """Default observation: surface the input and standard tools."""
        return Observation(
            context=context,
            current_state={"input": context.input_data},
            available_tools=["ai_hub"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 5)),
        )

    async def plan(self, observation: Observation) -> Plan:
        """Default single-step plan with low risk."""
        return Plan(
            observation=observation,
            steps=[PlannedStep(action="execute", params=observation.context.input_data)],
            estimated_duration=1.0,
            risk_score=0.1,
        )

    async def act(self, plan: Plan) -> ActionResult:
        """Default act: subclasses should override with real work."""
        return ActionResult(plan=plan, steps_executed=len(plan.steps), outputs={}, success=True)

    async def reflect(self, result: ActionResult) -> Reflection:
        """Default reflection: score by success and error count."""
        self._runs += 1
        score = 90.0 if result.success and not result.errors else 40.0
        self._last_error = result.errors[0] if result.errors else None
        await self._store_result(result, score)
        return Reflection(
            result=result,
            lessons_learned=[],
            performance_score=score,
            suggestions=[],
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _default_constraints(self) -> list[str]:
        constraints = []
        if settings.self_funding_mode != "autonomous":
            constraints.append("no external spend without operator approval")
        return constraints

    async def _ai_call(self, prompt: str, model_preference: Optional[str] = None) -> dict[str, Any]:
        """Route an LLM query through the AI hub adapter when available."""
        from dmai.core.orchestrator import orchestrator

        hub = orchestrator.registry.get("ai_hub")
        if hub is not None and hasattr(hub, "query"):
            try:
                return await hub.query(prompt, model_preference)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.warning("AI call failed: %s", exc)
                return {"text": "", "error": str(exc), "model": model_preference or "none"}
        return {"text": "", "error": "ai_hub unavailable", "model": "none"}

    async def _require_approval(self, description: str, payload: dict[str, Any]) -> None:
        """Emit an APPROVAL_REQUIRED event for money/external-API actions."""
        if settings.self_funding_mode == "autonomous":
            return
        if self._bus is not None:
            await self._bus.publish(
                Event(
                    event_type=EventType.APPROVAL_REQUIRED,
                    source=self.component_id,
                    payload={"description": description, **payload},
                )
            )

    async def _store_result(self, result: ActionResult, score: float) -> None:
        try:
            from dmai.db.models import AgentRunModel
            from dmai.db.session import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                session.add(
                    AgentRunModel(
                        agent_id=self.component_id,
                        task_type=result.plan.observation.context.task_type,
                        success=result.success,
                        duration_ms=result.duration_ms,
                        performance_score=score,
                        result={"outputs": result.outputs, "errors": result.errors},
                    )
                )
                await session.commit()
        except Exception as exc:  # pragma: no cover - best effort
            self._logger.debug("Agent result persistence skipped: %s", exc)
