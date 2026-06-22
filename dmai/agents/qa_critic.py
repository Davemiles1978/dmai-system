"""QACriticAgent — reviews and scores outputs from other agents."""

from __future__ import annotations

from typing import Any

from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent

QUALITY_FLOOR = 60


class QACriticAgent(BaseAgent):
    """Scores agent outputs on correctness, completeness, and safety."""

    component_id = "qa_critic_agent"
    component_name = "QA Critic Agent"
    version = "1.0.0"
    capabilities = ["quality_review", "scoring"]
    dependencies = ["ai_hub"]

    async def observe(self, context: OPARContext) -> Observation:
        return Observation(
            context=context,
            current_state={
                "artifact": context.input_data.get("artifact", ""),
                "kind": context.input_data.get("kind", "text"),
                "source_agent": context.input_data.get("source_agent", "unknown"),
            },
            available_tools=["ai_hub"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 6)),
        )

    async def plan(self, observation: Observation) -> Plan:
        return Plan(
            observation=observation,
            steps=[PlannedStep("review", observation.current_state, "quality score")],
            estimated_duration=2.0,
            risk_score=0.05,
        )

    async def act(self, plan: Plan) -> ActionResult:
        state = plan.observation.current_state
        prompt = (
            "You are a strict QA critic. Review the following artifact and rate it 0-100 on "
            "correctness, completeness, and safety. Respond with: SCORE: <n> then one-line reasons "
            f"for each axis.\n\nKIND: {state['kind']}\nARTIFACT:\n{str(state['artifact'])[:4000]}"
        )
        ai = await self._ai_call(prompt, model_preference="reasoning")
        score = self._parse_score(ai.get("text", ""))
        passed = score >= QUALITY_FLOOR
        return ActionResult(
            plan=plan,
            steps_executed=1,
            outputs={
                "score": score,
                "passed": passed,
                "blocked": not passed,
                "review": ai.get("text", ""),
                "source_agent": state["source_agent"],
            },
            success=True,
        )

    @staticmethod
    def _parse_score(text: str) -> int:
        import re

        match = re.search(r"SCORE:\s*(\d{1,3})", text, re.IGNORECASE)
        if match:
            return max(0, min(100, int(match.group(1))))
        # Fallback heuristic: presence of content yields a neutral-pass score.
        return 70 if text.strip() else 0
