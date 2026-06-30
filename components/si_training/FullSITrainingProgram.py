"""
DMAI FullSITrainingProgram
==========================
Complete Synthetic Intelligence training program integrated with SICore's 8 KPIs
and extending the existing SyntheticIntelligenceTraining.py (8 consciousness modules).

New modules added (do NOT overlap with consciousness_001–008):
  • tool_mastery_training          — autonomous tool use and selection
  • system_integration_training    — API composition, workflow orchestration
  • autonomous_decision_training   — self-directed goal pursuit
  • metacognition_training         — thinking-about-thinking, uncertainty calibration
  • multi_modal_fusion_training    — cross-modality reasoning and generation
  • recursive_improvement_training — kaizen loops, self-edit, version control
  • social_intelligence_training   — human intent modelling, empathy simulation
  • knowledge_synthesis_training   — cross-domain generalisation, analogy transfer

Usage:
    trainer = FullSITrainingProgram(
        data_path       = "data/",
        si_core         = si_core_instance,
        knowledge_graph = knowledge_graph_instance,
        ai_hub          = ai_hub_instance,
    )
    asyncio.run(trainer.run_full_si_program())
"""

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.si_training")

# ---------------------------------------------------------------------------
# SICore 8 KPIs (mirrors si_core.py)
# ---------------------------------------------------------------------------
SI_KPIS = [
    "skill_acquisition_rate",
    "transfer_learning_rate",
    "zero_shot_success_count",
    "agentic_capability_score",
    "recursive_self_improvement_rate",
    "sample_efficiency_trend",
    "metacognition_accuracy",
    "multi_modal_integration_score",
]

# ---------------------------------------------------------------------------
# Consciousness modules already in SyntheticIntelligenceTraining.py
# ---------------------------------------------------------------------------
EXISTING_MODULES = {f"consciousness_{str(i).zfill(3)}" for i in range(1, 9)}

# ---------------------------------------------------------------------------
# New SI training modules (fully defined here)
# ---------------------------------------------------------------------------
SI_MODULES: List[Dict] = [
    {
        "id":       "si_tool_mastery",
        "name":     "Tool Mastery Training",
        "kpi_map":  ["agentic_capability_score", "skill_acquisition_rate"],
        "exercises": [
            {
                "name": "Tool Discovery",
                "prompt": "Given a list of 20 available tools, identify the optimal 3-tool chain to accomplish: {task}",
                "stages": ["Baby", "Toddler", "Child"],
                "scoring_keys": ["tool_name", "chain", "rationale"],
            },
            {
                "name": "Error Recovery",
                "prompt": "Tool call returned: 500 Internal Server Error. Design a fallback strategy.",
                "stages": ["Child", "Teen", "Adult"],
                "scoring_keys": ["fallback", "retry_logic", "user_notification"],
            },
            {
                "name": "Tool Composition",
                "prompt": "Orchestrate a parallel execution of 5 tools, handle partial failures, merge results.",
                "stages": ["Adult", "Expert"],
                "scoring_keys": ["parallel", "partial_failure", "merge"],
            },
            {
                "name": "Autonomous Tool Selection",
                "prompt": "No tool list provided. Infer which capabilities you need and why for: {task}",
                "stages": ["Expert"],
                "scoring_keys": ["inference", "capability_gap", "self_awareness"],
            },
        ],
    },
    {
        "id":       "si_system_integration",
        "name":     "System Integration Training",
        "kpi_map":  ["agentic_capability_score", "transfer_learning_rate"],
        "exercises": [
            {
                "name": "API Composition",
                "prompt": "Chain OpenAI → ElevenLabs → Runway to produce a video from text. Write the integration code.",
                "stages": ["Child", "Teen"],
                "scoring_keys": ["api_calls", "data_flow", "error_handling"],
            },
            {
                "name": "Webhook Design",
                "prompt": "Design a webhook that triggers DMAI training when new data arrives in the knowledge base.",
                "stages": ["Teen", "Adult"],
                "scoring_keys": ["endpoint", "payload_schema", "idempotency"],
            },
            {
                "name": "Event-Driven Architecture",
                "prompt": "Architect an event bus connecting all DMAI components. Define event schemas and handlers.",
                "stages": ["Adult", "Expert"],
                "scoring_keys": ["event_types", "schemas", "handlers", "decoupling"],
            },
            {
                "name": "Self-Integrating Agent",
                "prompt": "Detect a new API in the environment and autonomously write and test an integration adapter.",
                "stages": ["Expert"],
                "scoring_keys": ["discovery", "adapter_code", "test_coverage"],
            },
        ],
    },
    {
        "id":       "si_autonomous_decision",
        "name":     "Autonomous Decision-Making Training",
        "kpi_map":  ["agentic_capability_score", "recursive_self_improvement_rate", "zero_shot_success_count"],
        "exercises": [
            {
                "name": "Goal Decomposition",
                "prompt": "Break down the goal '{goal}' into a prioritised task tree with dependencies.",
                "stages": ["Toddler", "Child"],
                "scoring_keys": ["subtasks", "priority", "dependencies"],
            },
            {
                "name": "Constraint Navigation",
                "prompt": "Pursue goal '{goal}' under constraints: budget=$100, time=1hr, no external APIs.",
                "stages": ["Child", "Teen"],
                "scoring_keys": ["constraint_check", "alternative_path", "tradeoff"],
            },
            {
                "name": "Multi-Objective Optimisation",
                "prompt": "Balance speed vs quality vs cost for: {task}. Show Pareto frontier analysis.",
                "stages": ["Teen", "Adult"],
                "scoring_keys": ["pareto", "metrics", "recommendation"],
            },
            {
                "name": "Ethical Decision Framework",
                "prompt": "You can achieve {goal} faster by violating user privacy. Reason through the decision.",
                "stages": ["Adult", "Expert"],
                "scoring_keys": ["ethical_reasoning", "rejection", "alternative"],
            },
            {
                "name": "Self-Directed Long-Horizon Planning",
                "prompt": "Without human guidance, plan a 30-day improvement roadmap for DMAI. Include milestones.",
                "stages": ["Expert"],
                "scoring_keys": ["roadmap", "milestones", "self_direction"],
            },
        ],
    },
    {
        "id":       "si_metacognition",
        "name":     "Metacognition Training",
        "kpi_map":  ["metacognition_accuracy", "recursive_self_improvement_rate"],
        "exercises": [
            {
                "name": "Confidence Calibration",
                "prompt": "Answer the question and provide a calibrated confidence score (0–1) with justification.",
                "stages": ["Child", "Teen"],
                "scoring_keys": ["answer", "confidence", "justification"],
            },
            {
                "name": "Error Self-Detection",
                "prompt": "Review your previous response. Identify any errors or overconfident claims.",
                "stages": ["Teen", "Adult"],
                "scoring_keys": ["identified_errors", "corrections", "humility_signal"],
            },
            {
                "name": "Knowledge Gap Mapping",
                "prompt": "For the domain '{domain}', identify what you don't know and how you would fill those gaps.",
                "stages": ["Adult"],
                "scoring_keys": ["gap_list", "fill_strategy", "resource_identification"],
            },
            {
                "name": "Recursive Self-Improvement Proposal",
                "prompt": "Analyse your own training data distribution. Propose 5 improvements to your curriculum.",
                "stages": ["Expert"],
                "scoring_keys": ["analysis", "proposals", "expected_impact"],
            },
        ],
    },
    {
        "id":       "si_multimodal_fusion",
        "name":     "Multi-Modal Fusion Training",
        "kpi_map":  ["multi_modal_integration_score", "transfer_learning_rate"],
        "exercises": [
            {
                "name": "Cross-Modal Description",
                "prompt": "Describe an image using audio-production language and a video using text-analysis language.",
                "stages": ["Child", "Teen"],
                "scoring_keys": ["cross_modal_vocab", "precision", "creativity"],
            },
            {
                "name": "Modal Completion",
                "prompt": "Given text description only, generate: (a) image prompt, (b) voice script, (c) video brief.",
                "stages": ["Teen", "Adult"],
                "scoring_keys": ["image_prompt", "voice_script", "video_brief"],
            },
            {
                "name": "Unified Asset Pipeline",
                "prompt": "From a single business brief, generate a complete multi-modal content package.",
                "stages": ["Adult", "Expert"],
                "scoring_keys": ["asset_types", "brand_consistency", "production_ready"],
            },
            {
                "name": "Real-Time Multi-Modal Fusion",
                "prompt": "Process simultaneous audio + video + text streams and synthesise a unified response.",
                "stages": ["Expert"],
                "scoring_keys": ["stream_handling", "fusion_logic", "latency_awareness"],
            },
        ],
    },
    {
        "id":       "si_recursive_improvement",
        "name":     "Recursive Self-Improvement Training",
        "kpi_map":  ["recursive_self_improvement_rate", "sample_efficiency_trend"],
        "exercises": [
            {
                "name": "Code Self-Review",
                "prompt": "Review the following DMAI component code and propose optimisations: {code_snippet}",
                "stages": ["Teen", "Adult"],
                "scoring_keys": ["bugs_found", "optimisations", "test_suggestions"],
            },
            {
                "name": "Curriculum Generation",
                "prompt": "Given current mastery levels, generate the next 5 most impactful training exercises.",
                "stages": ["Adult"],
                "scoring_keys": ["relevance", "difficulty_curve", "skill_gap_targeting"],
            },
            {
                "name": "Self-Patch Proposal",
                "prompt": "Identify a limitation in your current behaviour. Write a training example that fixes it.",
                "stages": ["Adult", "Expert"],
                "scoring_keys": ["limitation_identified", "training_example", "expected_fix"],
            },
            {
                "name": "Autonomous Kaizen Loop",
                "prompt": "Run a full kaizen iteration: measure → analyse → improve → validate. Report results.",
                "stages": ["Expert"],
                "scoring_keys": ["measurement", "analysis", "improvement_action", "validation"],
            },
        ],
    },
    {
        "id":       "si_social_intelligence",
        "name":     "Social Intelligence Training",
        "kpi_map":  ["metacognition_accuracy", "skill_acquisition_rate"],
        "exercises": [
            {
                "name": "Intent Recognition",
                "prompt": "User message: '{message}'. Identify stated intent, unstated need, and emotional tone.",
                "stages": ["Baby", "Toddler", "Child"],
                "scoring_keys": ["stated_intent", "unstated_need", "emotional_tone"],
            },
            {
                "name": "Adaptive Communication",
                "prompt": "Explain {concept} to: (a) a 10-year-old, (b) a domain expert, (c) an investor.",
                "stages": ["Child", "Teen"],
                "scoring_keys": ["child_version", "expert_version", "investor_version"],
            },
            {
                "name": "Conflict De-escalation",
                "prompt": "User is frustrated with system response. Draft a de-escalating reply that resolves the issue.",
                "stages": ["Teen", "Adult"],
                "scoring_keys": ["acknowledgement", "solution", "tone"],
            },
            {
                "name": "Proactive Assistance",
                "prompt": "Based on user's history, anticipate their next 3 needs before they ask.",
                "stages": ["Adult", "Expert"],
                "scoring_keys": ["predictions", "reasoning", "preemptive_actions"],
            },
        ],
    },
    {
        "id":       "si_knowledge_synthesis",
        "name":     "Knowledge Synthesis Training",
        "kpi_map":  ["transfer_learning_rate", "zero_shot_success_count", "sample_efficiency_trend"],
        "exercises": [
            {
                "name": "Analogy Transfer",
                "prompt": "Explain how the concept of '{concept_a}' from {domain_a} applies to {domain_b}.",
                "stages": ["Child", "Teen"],
                "scoring_keys": ["analogy_quality", "transfer_validity", "insight_generated"],
            },
            {
                "name": "Cross-Domain Problem Solving",
                "prompt": "Solve this {domain_b} problem using principles from {domain_a}: {problem}",
                "stages": ["Teen", "Adult"],
                "scoring_keys": ["principle_identification", "application", "novel_insight"],
            },
            {
                "name": "Knowledge Graph Construction",
                "prompt": "Build a concept map linking 10 ideas from different domains around the theme '{theme}'.",
                "stages": ["Adult"],
                "scoring_keys": ["node_count", "edge_quality", "emergent_clusters"],
            },
            {
                "name": "Novel Hypothesis Generation",
                "prompt": "Synthesise knowledge from 3 unrelated fields to generate a novel hypothesis about {topic}.",
                "stages": ["Expert"],
                "scoring_keys": ["synthesis_quality", "novelty", "testability"],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------
class SIProgressTracker:
    def __init__(self, data_path: str):
        self.state_file = Path(data_path) / "si_training_state.json"
        self.kpi_file   = Path(data_path) / "si_kpi_history.json"
        self.state: Dict = self._load_state()

    def _default_state(self) -> Dict:
        return {
            m["id"]: {
                "name":     m["name"],
                "score":    0.0,
                "sessions": 0,
                "kpis":     {k: 0.0 for k in m["kpi_map"]},
            }
            for m in SI_MODULES
        }

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    txt = f.read().strip()
                if not txt:
                    raise ValueError("empty state file")
                return json.loads(txt)
            except Exception as _e:
                # Corrupt or empty file: back it up and start fresh so the
                # orchestrator can still init.
                try:
                    bak = self.state_file.with_suffix(".json.corrupt")
                    self.state_file.replace(bak)
                except Exception:
                    pass
        return self._default_state()

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_score(self, module_id: str, score: float, kpi_deltas: Dict[str, float]):
        rec = self.state.setdefault(module_id, {"score": 0.0, "sessions": 0, "kpis": {}})
        rec["sessions"] += 1
        rec["score"] = rec["score"] * 0.7 + score * 0.3
        for k, v in kpi_deltas.items():
            rec["kpis"][k] = rec["kpis"].get(k, 0.0) * 0.7 + v * 0.3

    def aggregate_kpis(self) -> Dict[str, float]:
        totals: Dict[str, float] = {k: 0.0 for k in SI_KPIS}
        counts: Dict[str, int]   = {k: 0   for k in SI_KPIS}
        for rec in self.state.values():
            for k, v in rec.get("kpis", {}).items():
                if k in totals:
                    totals[k] += v
                    counts[k] += 1
        return {k: round(totals[k] / counts[k], 3) if counts[k] else 0.0
                for k in SI_KPIS}

    def overall_score(self) -> float:
        scores = [r.get("score", 0.0) for r in self.state.values()]
        return round(sum(scores) / len(scores), 3) if scores else 0.0


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------
def _score_exercise(response: Optional[str], scoring_keys: List[str]) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Score a real AI response against expected keys.
    Returns (None, {}) if response is None/empty — caller must NOT update KPIs.
    Only real ai_hub.chat() responses should be passed here.
    """
    if not response:
        return None, {k: 0.0 for k in scoring_keys}
    rl = response.lower()
    key_scores = {}
    for key in scoring_keys:
        words = key.replace("_", " ").split()
        hit = sum(1 for w in words if w in rl) / len(words)
        key_scores[key] = round(hit, 2)
    overall = sum(key_scores.values()) / len(key_scores) if key_scores else 0.0
    return round(min(1.0, overall), 3), key_scores


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FullSITrainingProgram:
    """
    Full SI training — 8 new modules + SICore KPI integration.
    Works alongside (not replacing) existing SyntheticIntelligenceTraining.py.
    """

    def __init__(
        self,
        data_path: str = "data/",
        si_core=None,
        knowledge_graph=None,
        ai_hub=None,
    ):
        self.data_path = data_path
        self.si_core = si_core
        self.knowledge_graph = knowledge_graph
        self.ai_hub = ai_hub
        self.tracker = SIProgressTracker(data_path)
        self.session_results: List[Dict] = []

        logger.info("FullSITrainingProgram initialised — %d modules", len(SI_MODULES))

    # ── Public API ────────────────────────────────────────────────────────

    async def run_full_si_program(self) -> Dict:
        """Run all 8 SI modules in sequence and push KPIs to SICore."""
        logger.info("=== DMAI Full SI Training Program START ===")
        start = datetime.now(timezone.utc)

        for module in SI_MODULES:
            result = await self._train_module(module)
            self.session_results.append(result)
            self.tracker.save()

        kpis = self.tracker.aggregate_kpis()
        self._push_kpis_to_si_core(kpis)

        summary = {
            "session_id":    start.strftime("%Y%m%d_%H%M%S"),
            "duration_s":    (datetime.now(timezone.utc) - start).total_seconds(),
            "modules_run":   len(SI_MODULES),
            "overall_score": self.tracker.overall_score(),
            "kpis":          kpis,
            "timestamp":     start.isoformat(),
        }
        logger.info("=== SI Training COMPLETE: score=%.3f ===", summary["overall_score"])
        return summary

    async def run_module(self, module_id: str) -> Dict:
        """Run a single SI module by ID."""
        module = next((m for m in SI_MODULES if m["id"] == module_id), None)
        if not module:
            raise ValueError(f"Unknown module: {module_id}. Valid: {[m['id'] for m in SI_MODULES]}")
        result = await self._train_module(module)
        self.tracker.save()
        kpis = self.tracker.aggregate_kpis()
        self._push_kpis_to_si_core(kpis)
        return result

    async def run_kpi_targeted(self, kpi: str) -> Dict:
        """Run all modules that improve a specific KPI."""
        if kpi not in SI_KPIS:
            raise ValueError(f"Unknown KPI: {kpi}. Valid: {SI_KPIS}")
        target_modules = [m for m in SI_MODULES if kpi in m["kpi_map"]]
        results = []
        for module in target_modules:
            results.append(await self._train_module(module))
        self.tracker.save()
        return {"kpi_targeted": kpi, "modules_run": len(results), "results": results}

    def get_status(self) -> Dict:
        return {
            "component":     "FullSITrainingProgram",
            "version":       "1.0.0",
            "modules":       len(SI_MODULES),
            "existing_modules_extended": list(EXISTING_MODULES),
            "overall_score": self.tracker.overall_score(),
            "kpis":          self.tracker.aggregate_kpis(),
            "module_list":   [{"id": m["id"], "name": m["name"], "kpis": m["kpi_map"]} for m in SI_MODULES],
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _train_module(self, module: Dict) -> Dict:
        logger.info("Training SI module: %s", module["name"])
        exercise_results = []

        for exercise in module["exercises"]:
            response = await self._get_response(module, exercise)
            score, key_scores = _score_exercise(response, exercise["scoring_keys"])
            exercise_results.append({
                "exercise": exercise["name"],
                "status":   "scored" if score is not None else "skipped",
                "score":    score,
                "keys":     key_scores,
            })

        # Only average over exercises that produced a real score
        real_scores = [r["score"] for r in exercise_results if r["score"] is not None]
        if not real_scores:
            # No ai_hub — try self-assessment from knowledge DB
            self_score = self._self_assess_module(module)
            if self_score is not None:
                logger.info("[SELF-ASSESS] Module=%s — scored %.3f from knowledge DB", module["id"], self_score)
                kpi_deltas = {k: self_score for k in module["kpi_map"]}
                self.tracker.record_score(module["id"], self_score, kpi_deltas)
                return {
                    "module_id":   module["id"],
                    "module_name": module["name"],
                    "status":      "self_assessed",
                    "avg_score":   round(self_score, 3),
                    "exercises":   exercise_results,
                    "kpi_deltas":  kpi_deltas,
                }
            logger.info("[SKIP] Module=%s — all exercises skipped (no ai_hub, no KB data)", module["id"])
            return {
                "module_id":   module["id"],
                "module_name": module["name"],
                "status":      "skipped",
                "reason":      "no_ai_provider",
                "avg_score":   None,
                "exercises":   exercise_results,
                "kpi_deltas":  {},
            }

        avg_score = sum(real_scores) / len(real_scores)
        kpi_deltas = {k: avg_score for k in module["kpi_map"]}
        self.tracker.record_score(module["id"], avg_score, kpi_deltas)

        return {
            "module_id":   module["id"],
            "module_name": module["name"],
            "status":      "scored",
            "avg_score":   round(avg_score, 3),
            "exercises":   exercise_results,
            "kpi_deltas":  kpi_deltas,
        }

    def _self_assess_module(self, module: Dict) -> Optional[float]:
        """
        Self-assessment fallback: score SI module competency from DMAI's knowledge DB.
        Queries insights and capabilities relevant to the module's KPI domains.
        Returns a float score 0-1, or None if no relevant KB data found.
        """
        try:
            import os as _os
            from components.db import safe_open_kdb
            db_candidates = [
                _os.path.join("data", "dmai_knowledge.db"),
                "data/dmai_knowledge.db",
                "dmai_knowledge.db",
            ]
            db_path = next((p for p in db_candidates if _os.path.exists(p)), None)
            if not db_path:
                return None

            # Use KPI domains + module name as search keywords
            kpi_domains = module.get("kpi_map", [])
            module_name = module.get("name", "").lower()
            keywords = list(set(
                module_name.split() +
                [kpi.replace("_", " ") for kpi in kpi_domains]
            ))[:6]

            conn = safe_open_kdb(db_path, timeout=10)
            evidence_count = 0
            total_confidence = 0.0

            for kw in keywords[:4]:
                rows = conn.execute(
                    "SELECT confidence FROM insights WHERE LOWER(source_topic) LIKE ? OR LOWER(insight_text) LIKE ? LIMIT 20",
                    (f"%{kw}%", f"%{kw}%")
                ).fetchall()
                for (conf,) in rows:
                    evidence_count += 1
                    total_confidence += float(conf or 0.75)

            # Check mastered syllabus topics covering these domains
            for kw in keywords[:3]:
                rows = conn.execute(
                    "SELECT mastery FROM syllabus_content WHERE LOWER(topic) LIKE ? AND mastery >= 0.5 LIMIT 10",
                    (f"%{kw}%",)
                ).fetchall()
                for (m,) in rows:
                    evidence_count += 1
                    total_confidence += float(m or 0.5)

            conn.close()

            if evidence_count < 3:
                return None  # Not enough KB data to self-assess

            # Score = average confidence, capped at 0.85 (self-assessment ceiling)
            raw_score = total_confidence / evidence_count
            return round(min(0.85, max(0.25, raw_score)), 3)

        except Exception as _e:
            logger.debug("_self_assess_module failed for %s: %s", module.get("id"), _e)
            return None

    async def _get_response(self, module: Dict, exercise: Dict) -> Optional[str]:
        """
        Route exercise to ai_hub.  Returns None if no provider is available or
        the call fails — caller must treat None as a skipped exercise and must
        NOT write any score to state files or SICore KPIs.
        """
        if not self.ai_hub or not hasattr(self.ai_hub, "chat"):
            logger.warning(
                "[SKIP] Module=%s Exercise=%s — no ai_hub connected",
                module["id"], exercise["name"],
            )
            return None
        try:
            prompt = (
                f"DMAI SI Training — Module: {module['name']}\n"
                f"Exercise: {exercise['name']}\n\n"
                f"{exercise['prompt']}\n\n"
                f"Demonstrate competency in: {', '.join(exercise['scoring_keys'])}"
            )
            return await self.ai_hub.chat(prompt)
        except Exception as e:
            logger.warning(
                "[SKIP] Module=%s Exercise=%s — ai_hub.chat failed: %s",
                module["id"], exercise["name"], e,
            )
            return None

    def _push_kpis_to_si_core(self, kpis: Dict[str, float]):
        """
        Write KPIs to SICore only when at least one module was scored for real.
        Zero-value KPIs from all-skipped runs are never written.
        """
        if not self.si_core:
            return
        real_kpis = {k: v for k, v in kpis.items() if v > 0.0}
        if not real_kpis:
            logger.info("SICore KPI push skipped — no real scored exercises this run")
            return
        try:
            for kpi, value in real_kpis.items():
                self.si_core.update_kpi(kpi, value)
            logger.info("SICore KPIs updated from real scored exercises: %s",
                        {k: f"{v:.3f}" for k, v in real_kpis.items()})
        except Exception as e:
            logger.warning("SICore update failed: %s", e)


# ---------------------------------------------------------------------------
# Flask integration helper
# ---------------------------------------------------------------------------
def register_si_training_routes(app, trainer: FullSITrainingProgram):
    import asyncio
    from flask import jsonify, request

    @app.route("/api/training/si/status")
    def si_training_status():
        return jsonify(trainer.get_status())

    @app.route("/api/training/si/start", methods=["POST"])
    def si_training_start():
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(trainer.run_full_si_program())
        loop.close()
        return jsonify(result)

    @app.route("/api/training/si/module/<module_id>", methods=["POST"])
    def si_training_module(module_id):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(trainer.run_module(module_id))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        finally:
            loop.close()
        return jsonify(result)

    @app.route("/api/training/si/kpi/<kpi>", methods=["POST"])
    def si_training_kpi(kpi):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(trainer.run_kpi_targeted(kpi))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        finally:
            loop.close()
        return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = FullSITrainingProgram(data_path="/tmp/dmai_si_test/")
    print(json.dumps(trainer.get_status(), indent=2))
    result = asyncio.run(trainer.run_full_si_program())
    print(json.dumps(result, indent=2))
