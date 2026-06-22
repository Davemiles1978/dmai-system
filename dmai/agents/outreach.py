"""OutreachAgent — generates personalised outreach sequences (no auto-send)."""

from __future__ import annotations

from typing import Any

from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent


class OutreachAgent(BaseAgent):
    """Builds multi-channel outreach sequences; sending requires approval."""

    component_id = "outreach_agent"
    component_name = "Outreach Agent"
    version = "1.0.0"
    capabilities = ["outreach", "personalization"]
    dependencies = ["ai_hub"]

    async def observe(self, context: OPARContext) -> Observation:
        return Observation(
            context=context,
            current_state={
                "prospects": context.input_data.get("prospects", []),
                "offer": context.input_data.get("offer", {}),
                "channels": context.input_data.get("channels", ["email"]),
            },
            available_tools=["ai_hub"],
            constraints=self._default_constraints() + ["never send without operator approval"],
            priority=int(context.metadata.get("priority", 5)),
        )

    async def plan(self, observation: Observation) -> Plan:
        return Plan(
            observation=observation,
            steps=[PlannedStep("generate_sequences", observation.current_state, "outreach drafts")],
            estimated_duration=4.0,
            risk_score=0.2,
        )

    async def act(self, plan: Plan) -> ActionResult:
        state = plan.observation.current_state
        prospects = state.get("prospects") or [{"name": "Prospect", "context": "general"}]
        sequences: list[dict[str, Any]] = []
        for prospect in prospects:
            prompt = (
                f"Write a personalised {', '.join(state['channels'])} outreach sequence (3 touches) "
                f"to {prospect.get('name', 'a prospect')} (context: {prospect.get('context', '')}) "
                f"about this offer: {state.get('offer')}. Keep each message short and human."
            )
            ai = await self._ai_call(prompt, model_preference="creative")
            sequences.append(
                {"prospect": prospect, "channels": state["channels"], "draft": ai.get("text", "")}
            )

        # Sending is gated: emit an approval request, never auto-send.
        await self._require_approval(
            "Outreach sending requires operator approval",
            {"sequence_count": len(sequences), "channels": state["channels"]},
        )

        return ActionResult(
            plan=plan,
            steps_executed=len(sequences),
            outputs={"sequences": sequences, "sent": False, "approval_required": True},
            success=True,
        )
