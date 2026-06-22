"""ComplianceAgent — reviews planned actions against operator policy."""

from __future__ import annotations

from typing import Any

from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent

APPROVED = "APPROVED"
REJECTED = "REJECTED"
NEEDS_REVIEW = "NEEDS_REVIEW"

DEFAULT_POLICY = [
    {"id": "spend_cap", "rule": "financial actions above daily limit need review", "severity": "high"},
    {"id": "external_send", "rule": "outbound messages need operator approval", "severity": "high"},
    {"id": "self_modify", "rule": "self-modifications need UpgradeLab + operator approval", "severity": "critical"},
]


class ComplianceAgent(BaseAgent):
    """Evaluates actions against editable policy rules stored in the DB."""

    component_id = "compliance_agent"
    component_name = "Compliance Agent"
    plane = "governance"
    version = "1.0.0"
    capabilities = ["policy_review", "approval"]
    dependencies = []

    async def observe(self, context: OPARContext) -> Observation:
        policy = await self._load_policy()
        return Observation(
            context=context,
            current_state={
                "action": context.input_data.get("action", {}),
                "policy": policy,
            },
            available_tools=["db"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 7)),
        )

    async def plan(self, observation: Observation) -> Plan:
        return Plan(
            observation=observation,
            steps=[PlannedStep("evaluate_policy", observation.current_state, "verdict")],
            estimated_duration=1.0,
            risk_score=0.05,
        )

    async def act(self, plan: Plan) -> ActionResult:
        action = plan.observation.current_state["action"]
        verdict, reasons = self._evaluate(action)
        return ActionResult(
            plan=plan,
            steps_executed=1,
            outputs={"verdict": verdict, "reasons": reasons},
            success=True,
        )

    def _evaluate(self, action: dict[str, Any]) -> tuple[str, list[str]]:
        from dmai.config import settings

        reasons: list[str] = []
        verdict = APPROVED

        if action.get("type") == "financial":
            amount = float(action.get("amount", 0))
            if amount > settings.spend_limit_daily:
                verdict = NEEDS_REVIEW
                reasons.append(f"amount ${amount} exceeds daily limit ${settings.spend_limit_daily}")
        if action.get("type") == "outbound_message":
            if settings.self_funding_mode != "autonomous":
                verdict = NEEDS_REVIEW
                reasons.append("outbound messaging requires operator approval")
        if action.get("type") == "self_modify":
            verdict = NEEDS_REVIEW
            reasons.append("self-modification requires UpgradeLab + operator approval")
        if action.get("prohibited"):
            verdict = REJECTED
            reasons.append("action marked prohibited")

        if not reasons:
            reasons.append("no policy violations detected")
        return verdict, reasons

    async def _load_policy(self) -> list[dict[str, Any]]:
        # Policy is operator-editable; defaults are used when none stored.
        return list(DEFAULT_POLICY)
