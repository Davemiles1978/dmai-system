"""CodingAgent — writes, reviews, and sandbox-executes code; can self-extend."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent

SANDBOX_TIMEOUT = 10


class CodingAgent(BaseAgent):
    """Generates and safely executes code; self-modifications need approval."""

    component_id = "coding_agent"
    component_name = "Coding Agent"
    version = "1.0.0"
    capabilities = ["code_generation", "self_extension"]
    dependencies = ["ai_hub"]

    async def observe(self, context: OPARContext) -> Observation:
        return Observation(
            context=context,
            current_state={
                "spec": context.input_data.get("spec", ""),
                "language": context.input_data.get("language", "python"),
                "self_modify": bool(context.input_data.get("self_modify", False)),
                "run": bool(context.input_data.get("run", False)),
            },
            available_tools=["ai_hub", "sandbox"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 5)),
        )

    async def plan(self, observation: Observation) -> Plan:
        state = observation.current_state
        # Self-modification is high-risk and must route through approval.
        risk = 0.85 if state["self_modify"] else 0.2
        steps = [PlannedStep("generate_code", {"spec": state["spec"]}, "source code")]
        if state["run"]:
            steps.append(PlannedStep("run_sandbox", {}, "execution result"))
        return Plan(observation=observation, steps=steps, estimated_duration=4.0, risk_score=risk)

    async def act(self, plan: Plan) -> ActionResult:
        state = plan.observation.current_state
        prompt = (
            f"Write {state['language']} code for this specification:\n{state['spec']}\n"
            "Return only the code, no explanation."
        )
        ai = await self._ai_call(prompt, model_preference="coding")
        code = self._strip_fences(ai.get("text", ""))
        outputs: dict[str, Any] = {"code": code, "language": state["language"]}

        if state["self_modify"]:
            await self._require_approval(
                "CodingAgent self-modification proposed — route to UpgradeLab",
                {"spec": state["spec"], "bytes": len(code)},
            )
            outputs["self_modify_pending"] = True

        if state["run"] and state["language"] == "python" and code:
            outputs["execution"] = await self._run_python(code)

        return ActionResult(
            plan=plan,
            steps_executed=len(plan.steps),
            outputs=outputs,
            success=bool(code),
            errors=[ai["error"]] if ai.get("error") else [],
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return text

    async def _run_python(self, code: str) -> dict[str, Any]:
        """Execute code in a subprocess sandbox with a hard timeout."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            path = fh.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SANDBOX_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                return {"timed_out": True, "stdout": "", "stderr": "timeout"}
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode(errors="replace")[:4000],
                "stderr": stderr.decode(errors="replace")[:4000],
            }
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
