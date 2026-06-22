"""FinanceControllerAgent — budget allocation and spend control."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dmai.config import settings
from dmai.core.bus import Event, EventType
from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent

ALLOCATION = {"reserve": 0.40, "reinvestment": 0.40, "scale_up": 0.20}


class FinanceControllerAgent(BaseAgent):
    """Tracks income/expense, enforces spend limits, allocates budget."""

    component_id = "finance_controller_agent"
    component_name = "Finance Controller Agent"
    version = "1.0.0"
    capabilities = ["budgeting", "spend_control", "reporting"]
    dependencies = ["self_funding"]

    async def observe(self, context: OPARContext) -> Observation:
        ledger = await self._ledger_summary()
        return Observation(
            context=context,
            current_state={"ledger": ledger, "action": context.input_data.get("action")},
            available_tools=["self_funding", "db"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 7)),
        )

    async def plan(self, observation: Observation) -> Plan:
        return Plan(
            observation=observation,
            steps=[PlannedStep("allocate_budget", observation.current_state, "allocation")],
            estimated_duration=1.0,
            risk_score=0.1,
        )

    async def act(self, plan: Plan) -> ActionResult:
        ledger = plan.observation.current_state["ledger"]
        action = plan.observation.current_state.get("action")
        net = ledger["income"] - ledger["expense"]
        allocation = {k: round(max(net, 0.0) * pct, 2) for k, pct in ALLOCATION.items()}

        errors: list[str] = []
        if action and action.get("type") == "spend":
            amount = float(action.get("amount", 0))
            if ledger["spent_today"] + amount > settings.spend_limit_daily:
                errors.append("daily spend limit would be exceeded")
                await self._require_approval(
                    "Spend exceeds daily limit", {"amount": amount, "limit": settings.spend_limit_daily}
                )

        report = {
            "net": round(net, 2),
            "allocation": allocation,
            "ledger": ledger,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._maybe_telegram_report(report)
        return ActionResult(
            plan=plan,
            steps_executed=1,
            outputs={"report": report},
            success=not errors,
            errors=errors,
        )

    async def _ledger_summary(self) -> dict[str, Any]:
        summary = {"income": 0.0, "expense": 0.0, "spent_today": 0.0}
        try:
            from sqlalchemy import func, select

            from dmai.db.models import RevenueModel
            from dmai.db.session import AsyncSessionLocal

            today = datetime.now(timezone.utc).date()
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
                rows = await session.scalars(
                    select(RevenueModel).where(RevenueModel.direction == "expense")
                )
                spent_today = sum(
                    r.amount for r in rows if r.created_at and r.created_at.date() == today
                )
                summary.update(
                    income=float(income or 0.0),
                    expense=float(expense or 0.0),
                    spent_today=float(spent_today),
                )
        except Exception as exc:  # pragma: no cover - works without DB
            self._logger.debug("Ledger summary skipped: %s", exc)
        return summary

    async def _maybe_telegram_report(self, report: dict[str, Any]) -> None:
        if self._bus is not None:
            await self._bus.publish(
                Event(
                    event_type=EventType.REVENUE_RECEIVED,
                    source=self.component_id,
                    payload={"financial_report": report},
                )
            )
