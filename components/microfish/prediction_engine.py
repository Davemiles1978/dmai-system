"""
PredictionEngine: reverse-engineered Microfish pipeline.

Pipeline:
  1. extract_entities  -> {entities[], relations[]}   (replaces ontology_generator + graph_builder)
  2. generate_personas -> [{id, name, role, beliefs, biases}]  (replaces oasis_profile_generator)
  3. run_simulation    -> [actions]  (replaces simulation_runner + OASIS)
  4. synthesize_verdict-> {verdict, confidence, signals, rationale}  (replaces report_agent)

All stages share one MicrofishLLM client (DMAI's 13-provider waterfall).
Graph + state persisted to SQLite via GraphStore.
"""
from __future__ import annotations
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from .graph_store import GraphStore
from .llm_client import MicrofishLLM

logger = logging.getLogger(__name__)

# ---- prompts (compact; original was ~3k tokens each, distilled to essentials) ----

_ENTITY_SYS = (
    "You are an information extractor. Given a requirement and seed data, "
    "identify the key entities (people, orgs, products, events, factors) and relations between them."
)
_ENTITY_PROMPT = """REQUIREMENT: {requirement}

SEED DATA:
{seed_data}

Extract entities and relations that are relevant to predicting the requirement.
Return JSON:
{{
  "entities": [{{"id": "snake_case_id", "label": "Human Name", "type": "person|org|product|event|factor", "attrs": {{...}}}}],
  "relations": [{{"from": "entity_id", "to": "entity_id", "type": "influences|owns|competes_with|...", "attrs": {{}}}}]
}}
Limit to the {max_entities} most consequential entities."""

_PERSONA_SYS = (
    "You generate realistic stakeholder personas who will reason about a prediction. "
    "Each persona has a viewpoint, biases, and incentives derived from the graph."
)
_PERSONA_PROMPT = """REQUIREMENT: {requirement}

ENTITIES:
{entities}

RELATIONS:
{relations}

Generate {agent_count} distinct personas who would have informed opinions on this requirement.
Mix supportive, skeptical, and neutral viewpoints. Return JSON:
{{
  "agents": [
    {{"id": "agent_1", "name": "...", "role": "...", "stance": "bullish|bearish|neutral",
      "expertise": "...", "key_beliefs": ["...","..."], "biases": ["...","..."]}}
  ]
}}"""

_SIM_ROUND_SYS = (
    "You are simulating one persona's reasoning in a multi-agent deliberation about a prediction. "
    "Stay strictly in character. Be concise (2-3 sentences)."
)
_SIM_ROUND_PROMPT = """REQUIREMENT: {requirement}

YOU ARE: {persona}

PRIOR DISCUSSION (round-by-round):
{history}

It is round {round_num}. Given your role/beliefs/biases, contribute ONE of:
- new_evidence: a fact or data point you bring to bear
- counterpoint: rebut a prior claim
- forecast: your current probability estimate (0-1) and 1-sentence reason

Return JSON: {{"action": "new_evidence|counterpoint|forecast", "content": "...", "probability": <number or null>}}"""

_VERDICT_SYS = (
    "You synthesize a multi-agent deliberation into a final prediction verdict. "
    "Weight evidence by quality, not loudness. Acknowledge uncertainty."
)
_VERDICT_PROMPT = """REQUIREMENT: {requirement}

AGENT POOL:
{agents}

FULL DELIBERATION TIMELINE:
{timeline}

INDIVIDUAL FORECASTS: {forecasts}

Synthesize the final verdict. Return JSON:
{{
  "verdict": "likely|unlikely|uncertain",
  "confidence": <0..1>,
  "probability": <0..1>,
  "signals": [{{"signal": "...", "direction": "supports|opposes", "weight": <0..1>}}],
  "rationale": "<2-4 sentence synthesis>",
  "key_risks": ["...","..."]
}}"""


class PredictionEngine:
    """Public Microfish API. Embedded as a DMAI component."""

    def __init__(self, db_path: str = "data/dmai_knowledge.db", llm: Optional[MicrofishLLM] = None):
        self.store = GraphStore(db_path=db_path)
        self.llm = llm or MicrofishLLM()

    # ---------- public API ----------

    def predict(self, requirement: str, seed_data: str = "",
                max_rounds: int = 2, agent_count: int = 4,
                max_entities: int = 12) -> Dict[str, Any]:
        """Run the full pipeline. Synchronous. Returns verdict dict."""
        t0 = time.time()
        seed_hash = hashlib.sha256((requirement + "|" + seed_data).encode()).hexdigest()[:16]
        pid = self.store.create_prediction(requirement, seed_hash)
        logger.info("PredictionEngine: starting prediction %s", pid)

        try:
            # 1) entity extraction
            graph = self._extract_entities(requirement, seed_data, max_entities)
            self.store.add_entities(pid, graph.get("entities", []))
            self.store.add_relations(pid, graph.get("relations", []))

            # 2) persona generation
            agents = self._generate_personas(requirement, graph, agent_count)
            self.store.add_agents(pid, agents)

            # 3) simulation
            forecasts = self._run_simulation(pid, requirement, agents, max_rounds)

            # 4) verdict
            timeline = self.store.get_timeline(pid)
            verdict = self._synthesize_verdict(requirement, agents, timeline, forecasts)
            verdict["id"] = pid
            verdict["elapsed_seconds"] = round(time.time() - t0, 2)
            verdict["agent_count"] = len(agents)
            verdict["rounds_run"] = max_rounds
            verdict["entity_count"] = len(graph.get("entities", []))

            self.store.finalize_prediction(pid, verdict, status="complete")
            return verdict
        except Exception as e:
            logger.exception("PredictionEngine.predict failed: %s", e)
            fail = {
                "id": pid,
                "verdict": "uncertain",
                "confidence": 0.0,
                "probability": 0.5,
                "error": str(e),
                "elapsed_seconds": round(time.time() - t0, 2),
            }
            self.store.finalize_prediction(pid, fail, status="failed")
            return fail

    def get_prediction(self, pid: str) -> Optional[Dict[str, Any]]:
        return self.store.get_prediction(pid)

    def get_timeline(self, pid: str) -> List[Dict[str, Any]]:
        return self.store.get_timeline(pid)

    # ---------- pipeline stages ----------

    def _extract_entities(self, requirement: str, seed_data: str, max_entities: int) -> Dict[str, Any]:
        seed = (seed_data or "").strip()[:8000] or "(no seed data; reason from requirement alone)"
        result = self.llm.chat_json(
            _ENTITY_PROMPT.format(requirement=requirement, seed_data=seed, max_entities=max_entities),
            system=_ENTITY_SYS,
            default={"entities": [], "relations": []},
        )
        entities = result.get("entities") or []
        relations = result.get("relations") or []
        # dedupe entity ids
        seen, clean_e = set(), []
        for e in entities:
            eid = e.get("id") or e.get("label", "").lower().replace(" ", "_")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            e["id"] = eid
            clean_e.append(e)
        return {"entities": clean_e, "relations": relations}

    def _generate_personas(self, requirement: str, graph: Dict[str, Any], agent_count: int) -> List[Dict[str, Any]]:
        ents_summary = "\n".join(f"- {e['id']} ({e.get('type','?')}): {e.get('label','')}" for e in graph["entities"][:20])
        rels_summary = "\n".join(f"- {r.get('from')} --{r.get('type','rel')}--> {r.get('to')}" for r in graph["relations"][:30])
        result = self.llm.chat_json(
            _PERSONA_PROMPT.format(
                requirement=requirement,
                entities=ents_summary or "(none)",
                relations=rels_summary or "(none)",
                agent_count=agent_count,
            ),
            system=_PERSONA_SYS,
            default={"agents": []},
        )
        agents = result.get("agents") or []
        # ensure unique ids
        for i, a in enumerate(agents):
            if not a.get("id"):
                a["id"] = f"agent_{i+1}"
            a.setdefault("platform", "generic")
        # fallback if LLM returned nothing
        if not agents:
            agents = [
                {"id": f"agent_{i+1}", "name": f"Analyst {i+1}", "role": "generalist",
                 "stance": ["bullish", "bearish", "neutral"][i % 3], "expertise": "general",
                 "key_beliefs": [], "biases": [], "platform": "generic"}
                for i in range(agent_count)
            ]
        return agents[:agent_count]

    def _run_simulation(self, pid: str, requirement: str, agents: List[Dict[str, Any]],
                        max_rounds: int) -> List[float]:
        forecasts: List[float] = []
        history_lines: List[str] = []
        for rnd in range(1, max_rounds + 1):
            for a in agents:
                hist = "\n".join(history_lines[-20:]) or "(start of deliberation)"
                persona = (
                    f"{a.get('name','?')} ({a.get('role','?')}, stance={a.get('stance','?')}, "
                    f"expertise={a.get('expertise','?')}). Beliefs: {a.get('key_beliefs', [])}. "
                    f"Biases: {a.get('biases', [])}."
                )
                turn = self.llm.chat_json(
                    _SIM_ROUND_PROMPT.format(
                        requirement=requirement, persona=persona, history=hist, round_num=rnd
                    ),
                    system=_SIM_ROUND_SYS,
                    default={"action": "forecast", "content": "(no response)", "probability": None},
                )
                act = (turn.get("action") or "forecast").lower()
                content = (turn.get("content") or "").strip()[:1000]
                prob = turn.get("probability")
                self.store.add_action(pid, a["id"], act, content, rnd)
                history_lines.append(f"[R{rnd}] {a.get('name','?')} ({act}): {content}")
                if isinstance(prob, (int, float)) and 0 <= prob <= 1:
                    forecasts.append(float(prob))
        return forecasts

    def _synthesize_verdict(self, requirement: str, agents: List[Dict[str, Any]],
                            timeline: List[Dict[str, Any]], forecasts: List[float]) -> Dict[str, Any]:
        agents_summary = "\n".join(
            f"- {a.get('id')}: {a.get('name','?')} ({a.get('role','?')}, {a.get('stance','?')})"
            for a in agents
        )
        tl_summary = "\n".join(
            f"[R{t['round_num']}] {t['agent_id']} ({t['action_type']}): {(t.get('content') or '')[:200]}"
            for t in timeline[-60:]
        )
        forecasts_str = ", ".join(f"{f:.2f}" for f in forecasts) or "(no numeric forecasts)"
        mean_p = sum(forecasts) / len(forecasts) if forecasts else 0.5

        verdict = self.llm.chat_json(
            _VERDICT_PROMPT.format(
                requirement=requirement,
                agents=agents_summary,
                timeline=tl_summary or "(empty)",
                forecasts=forecasts_str,
            ),
            system=_VERDICT_SYS,
            default={},
        )

        # sanity defaults + clamping
        p = verdict.get("probability")
        if not isinstance(p, (int, float)) or not (0 <= p <= 1):
            p = mean_p
        conf = verdict.get("confidence")
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            # confidence proxy: 1 - dispersion of forecasts
            if len(forecasts) >= 2:
                mean = sum(forecasts) / len(forecasts)
                var = sum((f - mean) ** 2 for f in forecasts) / len(forecasts)
                conf = max(0.0, min(1.0, 1.0 - (var ** 0.5) * 2))
            else:
                conf = 0.5
        v_label = verdict.get("verdict")
        if v_label not in ("likely", "unlikely", "uncertain"):
            v_label = "likely" if p >= 0.6 else ("unlikely" if p <= 0.4 else "uncertain")

        return {
            "verdict": v_label,
            "probability": round(float(p), 3),
            "confidence": round(float(conf), 3),
            "signals": verdict.get("signals") or [],
            "rationale": verdict.get("rationale") or "",
            "key_risks": verdict.get("key_risks") or [],
            "agent_forecasts": forecasts,
        }
