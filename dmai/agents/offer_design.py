"""OfferDesignAgent — turns market research into priced product offers."""

from __future__ import annotations

from typing import Any

from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent


class OfferDesignAgent(BaseAgent):
    """Designs offers with pricing tiers, copy, and value propositions."""

    component_id = "offer_design_agent"
    component_name = "Offer Design Agent"
    version = "1.0.0"
    capabilities = ["offer_design", "pricing"]
    dependencies = ["ai_hub"]

    async def observe(self, context: OPARContext) -> Observation:
        return Observation(
            context=context,
            current_state={
                "market": context.input_data.get("market", {}),
                "topic": context.input_data.get("topic", "digital product"),
            },
            available_tools=["ai_hub"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 5)),
        )

    async def plan(self, observation: Observation) -> Plan:
        return Plan(
            observation=observation,
            steps=[PlannedStep("design_offer", observation.current_state, "OfferDesign object")],
            estimated_duration=4.0,
            risk_score=0.1,
        )

    async def act(self, plan: Plan) -> ActionResult:
        state = plan.observation.current_state
        prompt = (
            f"Design a product/service offer for: {state.get('topic')}. "
            f"Market context: {state.get('market')}. "
            "Provide: a one-line value proposition, three pricing tiers (name, price, what's included), "
            "and a short sales paragraph. Be specific and commercial."
        )
        ai = await self._ai_call(prompt, model_preference="creative")
        offer = self._build_offer(state.get("topic", "offer"), ai.get("text", ""))
        return ActionResult(
            plan=plan,
            steps_executed=1,
            outputs={"offer": offer},
            success=bool(ai.get("text")),
            errors=[ai["error"]] if ai.get("error") else [],
        )

    @staticmethod
    def _build_offer(topic: str, text: str) -> dict[str, Any]:
        return {
            "name": f"{topic.title()} Offer",
            "value_proposition": text.splitlines()[0] if text else "",
            "copy": text,
            "tiers": [
                {"name": "Starter", "price": 29.0, "includes": "core features"},
                {"name": "Pro", "price": 99.0, "includes": "core + advanced + support"},
                {"name": "Scale", "price": 299.0, "includes": "everything + priority + custom"},
            ],
        }
