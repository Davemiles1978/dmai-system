"""UpgradeLabAgent — benchmarks models/prompts and proposes self-upgrades."""

from __future__ import annotations

from typing import Any

from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent

PROMOTION_THRESHOLD = 75.0


class UpgradeLabAgent(BaseAgent):
    """Evaluates new models/prompts and auto-promotes when safe + approved."""

    component_id = "upgrade_lab_agent"
    component_name = "Upgrade Lab Agent"
    version = "1.0.0"
    capabilities = ["benchmarking", "model_evaluation", "self_upgrade"]
    dependencies = ["ai_hub", "evolution_engine"]

    async def observe(self, context: OPARContext) -> Observation:
        return Observation(
            context=context,
            current_state={
                "candidate": context.input_data.get("candidate", {}),
                "baseline": context.input_data.get("baseline", {}),
                "benchmark": context.input_data.get("benchmark", "default"),
            },
            available_tools=["ai_hub", "evolution_engine"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 5)),
        )

    async def plan(self, observation: Observation) -> Plan:
        # Promoting an upgrade is high-risk → routes through approval.
        return Plan(
            observation=observation,
            steps=[
                PlannedStep("benchmark", observation.current_state, "scores"),
                PlannedStep("propose_upgrade", {}, "proposal"),
            ],
            estimated_duration=6.0,
            risk_score=0.75,
        )

    async def act(self, plan: Plan) -> ActionResult:
        state = plan.observation.current_state
        candidate_score = await self._benchmark(state.get("candidate", {}), state["benchmark"])
        baseline_score = await self._benchmark(state.get("baseline", {}), state["benchmark"])

        compliance_ok = await self._compliance_check(state.get("candidate", {}))
        promote = candidate_score > baseline_score and candidate_score >= PROMOTION_THRESHOLD and compliance_ok

        proposal = {
            "candidate_score": candidate_score,
            "baseline_score": baseline_score,
            "delta": round(candidate_score - baseline_score, 2),
            "compliance_ok": compliance_ok,
            "auto_promote": promote,
        }
        if promote:
            await self._require_approval(
                "Upgrade Lab proposes auto-promotion of a new model/prompt",
                {"candidate_score": candidate_score, "delta": proposal["delta"]},
            )
        return ActionResult(
            plan=plan,
            steps_executed=2,
            outputs={"proposal": proposal},
            success=True,
        )

    async def _benchmark(self, config: dict[str, Any], benchmark: str) -> float:
        """Run a lightweight benchmark via the AI hub; returns a 0-100 score."""
        if not config:
            return 50.0
        prompt = (
            f"Run benchmark '{benchmark}'. Rate this configuration 0-100 for expected quality: "
            f"{config}. Respond with only a number."
        )
        ai = await self._ai_call(prompt, model_preference=config.get("model"))
        try:
            return max(0.0, min(100.0, float(ai.get("text", "50").strip().split()[0])))
        except (ValueError, IndexError):
            return 50.0

    async def _compliance_check(self, candidate: dict[str, Any]) -> bool:
        from dmai.core.orchestrator import orchestrator

        agent = orchestrator.registry.get("compliance_agent")
        if agent is None:
            return True
        ctx = OPARContext(
            task_type="policy_review",
            input_data={"action": {"type": "self_modify", "candidate": candidate}},
            agent_id="compliance_agent",
        )
        result = await orchestrator.opar.run(agent, ctx)
        verdict = (result.result.outputs.get("verdict") if result.result else None) or ""
        return verdict != "REJECTED"
