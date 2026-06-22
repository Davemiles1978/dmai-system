"""AnalyticsAgent — monitors revenue, conversion, and component performance."""

from __future__ import annotations

from typing import Any

from dmai.core.bus import Event, EventType
from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent


class AnalyticsAgent(BaseAgent):
    """Aggregates metrics, flags underperformers, and emits insights."""

    component_id = "analytics_agent"
    component_name = "Analytics Agent"
    version = "1.0.0"
    capabilities = ["analytics", "reporting"]
    dependencies = ["ai_hub"]

    async def observe(self, context: OPARContext) -> Observation:
        metrics = await self._gather_metrics()
        return Observation(
            context=context,
            current_state={"metrics": metrics},
            available_tools=["ai_hub", "db"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 4)),
        )

    async def plan(self, observation: Observation) -> Plan:
        return Plan(
            observation=observation,
            steps=[PlannedStep("analyze", observation.current_state["metrics"], "insights")],
            estimated_duration=2.0,
            risk_score=0.05,
        )

    async def act(self, plan: Plan) -> ActionResult:
        metrics = plan.observation.current_state["metrics"]
        insights = self._derive_insights(metrics)
        if self._bus is not None:
            await self._bus.publish(
                Event(
                    event_type=EventType.ANALYTICS_INSIGHT,
                    source=self.component_id,
                    payload={"insights": insights, "metrics": metrics},
                )
            )
        return ActionResult(
            plan=plan,
            steps_executed=1,
            outputs={"insights": insights, "metrics": metrics},
            success=True,
        )

    async def _gather_metrics(self) -> dict[str, Any]:
        metrics = {"total_income": 0.0, "total_expense": 0.0, "runs": 0, "failed_runs": 0}
        try:
            from sqlalchemy import func, select

            from dmai.db.models import AgentRunModel, RevenueModel
            from dmai.db.session import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                income = await session.scalar(
                    select(func.coalesce(func.sum(RevenueModel.amount), 0.0)).where(
                        RevenueModel.direction == "income"
                    )
                )
                expense = await session.scalar(
                    select(func.coalesce(func.sum(RevenueModel.amount), 0.0)).where(
                        RevenueModel.direction == "expense"
                    )
                )
                runs = await session.scalar(select(func.count(AgentRunModel.id)))
                failed = await session.scalar(
                    select(func.count(AgentRunModel.id)).where(AgentRunModel.success.is_(False))
                )
                metrics.update(
                    total_income=float(income or 0.0),
                    total_expense=float(expense or 0.0),
                    runs=int(runs or 0),
                    failed_runs=int(failed or 0),
                )
        except Exception as exc:  # pragma: no cover - analytics works without DB
            self._logger.debug("Metric gather skipped: %s", exc)
        return metrics

    @staticmethod
    def _derive_insights(metrics: dict[str, Any]) -> list[str]:
        insights: list[str] = []
        net = metrics["total_income"] - metrics["total_expense"]
        insights.append(f"Net position: ${net:.2f}")
        runs = metrics.get("runs", 0)
        if runs:
            fail_rate = metrics.get("failed_runs", 0) / runs
            if fail_rate > 0.2:
                insights.append(f"High failure rate ({fail_rate:.0%}) — investigate agents")
        if net < 0:
            insights.append("Spending exceeds income — tighten budget allocation")
        return insights
