"""LandingPageBuilderAgent — generates full HTML landing pages for offers."""

from __future__ import annotations

import html
from typing import Any

from dmai.core.opar import ActionResult, Observation, OPARContext, Plan, PlannedStep
from dmai.agents.base_agent import BaseAgent


class LandingPageBuilderAgent(BaseAgent):
    """Generates complete HTML/CSS/JS landing pages, with A/B variants."""

    component_id = "landing_page_agent"
    component_name = "Landing Page Builder Agent"
    version = "1.0.0"
    capabilities = ["landing_page", "html_generation", "ab_testing"]
    dependencies = ["ai_hub"]

    async def observe(self, context: OPARContext) -> Observation:
        return Observation(
            context=context,
            current_state={
                "offer": context.input_data.get("offer", {}),
                "variants": int(context.input_data.get("variants", 1)),
            },
            available_tools=["ai_hub"],
            constraints=self._default_constraints(),
            priority=int(context.metadata.get("priority", 5)),
        )

    async def plan(self, observation: Observation) -> Plan:
        n = observation.current_state["variants"]
        steps = [
            PlannedStep(f"build_variant_{i}", {"variant": i}, "HTML page")
            for i in range(max(1, n))
        ]
        return Plan(observation=observation, steps=steps, estimated_duration=3.0, risk_score=0.1)

    async def act(self, plan: Plan) -> ActionResult:
        offer = plan.observation.current_state.get("offer", {})
        n = plan.observation.current_state["variants"]
        pages: list[dict[str, Any]] = []
        for i in range(max(1, n)):
            prompt = (
                f"Write the hero headline and subheadline for a landing page (variant {i + 1}) "
                f"selling this offer: {offer}. Return two lines only."
            )
            ai = await self._ai_call(prompt, model_preference="creative")
            headline = (ai.get("text", "") or "Transform Your Results Today").splitlines()
            pages.append(
                {
                    "variant": i + 1,
                    "html": self._render(offer, headline),
                }
            )
        return ActionResult(
            plan=plan,
            steps_executed=len(pages),
            outputs={"pages": pages},
            success=True,
        )

    @staticmethod
    def _render(offer: dict[str, Any], headline_lines: list[str]) -> str:
        headline = html.escape(headline_lines[0] if headline_lines else "Welcome")
        sub = html.escape(headline_lines[1] if len(headline_lines) > 1 else offer.get("value_proposition", ""))
        name = html.escape(str(offer.get("name", "Our Offer")))
        tiers = offer.get("tiers", [])
        tier_html = "".join(
            f'<div class="tier"><h3>{html.escape(str(t.get("name","")))}</h3>'
            f'<p class="price">${t.get("price", 0)}</p>'
            f'<p>{html.escape(str(t.get("includes","")))}</p>'
            f'<button>Choose</button></div>'
            for t in tiers
        )
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}</title>
<style>
  body{{font-family:system-ui,sans-serif;margin:0;background:#0b0e14;color:#e6e6e6}}
  .hero{{padding:80px 20px;text-align:center;background:linear-gradient(135deg,#1a1f2e,#0b0e14)}}
  .hero h1{{font-size:2.5rem;margin:0 0 12px}}
  .tiers{{display:flex;gap:20px;justify-content:center;flex-wrap:wrap;padding:40px}}
  .tier{{background:#161b27;border:1px solid #263041;border-radius:12px;padding:24px;width:220px}}
  .price{{font-size:1.8rem;color:#4fd1c5}}
  button{{background:#4fd1c5;border:0;color:#06121a;padding:10px 18px;border-radius:8px;cursor:pointer;font-weight:600}}
</style></head>
<body>
  <section class="hero"><h1>{headline}</h1><p>{sub}</p></section>
  <section class="tiers">{tier_html}</section>
</body></html>"""
