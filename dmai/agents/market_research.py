"""MarketResearchAgent — discovers market opportunities and trends."""

from __future__ import annotations

from typing import Any

from dmai.core.bus import Event, EventType
from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent


class MarketResearchAgent(BaseAgent):
    """Researches markets, trends, and competitor pricing via the AI hub."""

    component_id = "market_research_agent"
    component_name = "Market Research Agent"
    version = "1.0.0"
    capabilities = ["market_research", "trend_analysis"]
    dependencies = ["ai_hub"]

    async def observe(self, context: OPARContext) -> Observation:
        topic = context.input_data.get("topic", "emerging digital product opportunities")
        return Observation(
            context=context,
            current_state={"topic": topic, "region": context.input_data.get("region", "global")},
            available_tools=["ai_hub", "web_search"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 6)),
        )

    async def plan(self, observation: Observation) -> Plan:
        topic = observation.current_state["topic"]
        return Plan(
            observation=observation,
            steps=[
                PlannedStep("synthesize_market", {"topic": topic}, "structured market report"),
            ],
            estimated_duration=5.0,
            risk_score=0.1,
        )

    async def act(self, plan: Plan) -> ActionResult:
        topic = plan.observation.current_state["topic"]
        prompt = (
            f"Act as a market research analyst. For the topic '{topic}', produce a concise "
            "structured report covering: 1) top 3 opportunities, 2) demand signals, "
            "3) competitor pricing ranges, 4) recommended entry angle. Return tight bullet points."
        )
        ai = await self._ai_call(prompt, model_preference="reasoning")
        report = {
            "topic": topic,
            "analysis": ai.get("text", ""),
            "model": ai.get("model", "none"),
            "opportunities": self._extract_opportunities(ai.get("text", "")),
        }
        if self._bus is not None and report["opportunities"]:
            await self._bus.publish(
                Event(
                    event_type=EventType.MARKET_OPPORTUNITY_FOUND,
                    source=self.component_id,
                    payload={"topic": topic, "opportunities": report["opportunities"]},
                )
            )
        return ActionResult(
            plan=plan,
            steps_executed=1,
            outputs={"report": report},
            success=bool(ai.get("text")) or "error" not in ai,
            errors=[ai["error"]] if ai.get("error") else [],
        )

    @staticmethod
    def _extract_opportunities(text: str) -> list[str]:
        lines = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip().startswith(("-", "*"))]
        return lines[:3]
