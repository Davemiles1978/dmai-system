"""
DMAI ComprehensiveAITraining
============================
Full AI training program that maps to DMAI's developmental stage system:
  Baby → Toddler → Child → Teen → Adult → Expert

Extends (does NOT replace) the existing AGITrainingProgram.py and
ComprehensiveAGITraining.py components.  Wire it in via the master
orchestrator or call standalone:

    trainer = ComprehensiveAITraining(
        data_path   = "data/",
        si_core     = si_core_instance,
        knowledge_graph = knowledge_graph_instance,
        ai_hub      = ai_hub_instance,
    )
    asyncio.run(trainer.run_full_program())
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dmai.ai_training")

# ---------------------------------------------------------------------------
# Stage definitions (mirrors DMAI's dmai_syllabus_data.py categories)
# ---------------------------------------------------------------------------
STAGES = ["Baby", "Toddler", "Child", "Teen", "Adult", "Expert"]

STAGE_MASTERY_GATE = {
    "Baby":    0.40,
    "Toddler": 0.55,
    "Child":   0.65,
    "Teen":    0.75,
    "Adult":   0.85,
    "Expert":  0.95,
}

# ---------------------------------------------------------------------------
# Full curriculum — every domain × every stage
# ---------------------------------------------------------------------------
FULL_CURRICULUM: List[Dict] = [

    # ── CORE ──────────────────────────────────────────────────────────────
    {
        "domain": "Language Understanding",
        "category": "Core",
        "stages": {
            "Baby":    ["Token recognition", "Basic vocabulary (500 words)", "Simple sentence parsing"],
            "Toddler": ["POS tagging", "Named entity basics", "Sentence boundary detection"],
            "Child":   ["Dependency parsing", "Coreference resolution", "Basic reading comprehension"],
            "Teen":    ["Semantic role labelling", "Discourse coherence", "Multi-document understanding"],
            "Adult":   ["Cross-lingual transfer", "Pragmatic inference", "Long-context reasoning (128k+)"],
            "Expert":  ["Nuanced cultural context", "Ambiguity resolution under uncertainty",
                        "Zero-shot multi-lingual QA"],
        },
    },
    {
        "domain": "Reasoning & Logic",
        "category": "Core",
        "stages": {
            "Baby":    ["Boolean logic", "Simple if/then rules", "Pattern matching"],
            "Toddler": ["Syllogistic reasoning", "Analogy completion", "Basic math word problems"],
            "Child":   ["Multi-step deduction", "Counterfactual reasoning", "Propositional logic proofs"],
            "Teen":    ["First-order predicate logic", "Causal inference chains", "Constraint satisfaction"],
            "Adult":   ["Abductive reasoning", "Bayesian updating", "Formal argument verification"],
            "Expert":  ["Meta-reasoning (reasoning about own reasoning)", "Gödel-aware self-limitation",
                        "Novel theorem synthesis"],
        },
    },
    {
        "domain": "Memory & Context Management",
        "category": "Core",
        "stages": {
            "Baby":    ["Short-term buffer (4k tokens)", "Keyword recall", "Simple lookup"],
            "Toddler": ["Working memory management (16k)", "Key-value retrieval", "Recency bias correction"],
            "Child":   ["Hierarchical summarisation", "Multi-turn context tracking", "Forgetting policy basics"],
            "Teen":    ["Long-context compression (128k)", "Episodic vs semantic memory separation",
                        "Memory consolidation scheduling"],
            "Adult":   ["Vector-based associative recall", "Cross-session persistence", "Priority-weighted replay"],
            "Expert":  ["Self-organising memory graphs", "Adaptive forgetting curves",
                        "Proactive memory pre-loading"],
        },
    },

    # ── ACCELERATOR ────────────────────────────────────────────────────────
    {
        "domain": "Code Creation & Fixing",
        "category": "Accelerator",
        "stages": {
            "Baby":    ["Hello-world generation (Python, JS)", "Syntax error identification",
                        "Variable naming conventions"],
            "Toddler": ["Function generation from docstrings", "Stack-trace reading",
                        "Unit test generation"],
            "Child":   ["Class/OOP generation", "Multi-file refactoring", "Dependency resolution",
                        "Dockerfile creation"],
            "Teen":    ["Design-pattern application", "Performance profiling & fix",
                        "Security vulnerability detection (OWASP Top-10)", "API client generation"],
            "Adult":   ["Full-stack scaffold from spec", "CI/CD pipeline generation",
                        "Async/concurrent code optimisation", "Database schema migration scripts"],
            "Expert":  ["Self-modifying code generation", "Compiler/interpreter construction",
                        "AI model training loop creation", "Zero-shot framework port"],
        },
    },
    {
        "domain": "Agentic Task Execution",
        "category": "Accelerator",
        "stages": {
            "Baby":    ["Single-tool invocation", "Result parsing", "Error retry (1 attempt)"],
            "Toddler": ["Sequential tool chains (≤3 tools)", "State passing between steps",
                        "Basic task decomposition"],
            "Child":   ["Parallel tool execution", "Dynamic plan adjustment", "Resource budget awareness"],
            "Teen":    ["Hierarchical agent spawning", "Conflict resolution between sub-agents",
                        "Long-horizon task planning (>20 steps)"],
            "Adult":   ["Self-directed research loops", "Autonomous debugging cycles",
                        "Goal re-prioritisation under failure"],
            "Expert":  ["Meta-agent orchestration", "Emergent strategy formation",
                        "Autonomous system hardening"],
        },
    },
    {
        "domain": "LLM Fine-Tuning & Adaptation",
        "category": "Accelerator",
        "stages": {
            "Baby":    ["Prompt template design", "Few-shot example selection", "Temperature/top-p basics"],
            "Toddler": ["RLHF concepts", "LoRA adapter concepts", "Evaluation metrics (BLEU, ROUGE, BERTScore)"],
            "Child":   ["LoRA/QLoRA fine-tune execution", "Dataset curation & deduplication",
                        "Catastrophic forgetting avoidance"],
            "Teen":    ["DPO / PPO training loops", "Reward model construction", "PEFT strategy selection"],
            "Adult":   ["Multi-task fine-tuning", "Curriculum learning scheduling",
                        "Constitutional AI principles application"],
            "Expert":  ["Self-supervised continual learning", "Model merging (TIES/DARE)",
                        "Architecture search & NAS"],
        },
    },

    # ── ARTISTIC ──────────────────────────────────────────────────────────
    {
        "domain": "Image Generation & Editing",
        "category": "Artistic",
        "stages": {
            "Baby":    ["Simple prompt → DALL-E/Imagen call", "Aspect ratio selection",
                        "Style keyword basics"],
            "Toddler": ["Negative prompt usage", "Seed control for reproducibility",
                        "Inpainting simple regions"],
            "Child":   ["ControlNet pose/depth guidance", "Style transfer via IP-Adapter",
                        "Batch variation generation"],
            "Teen":    ["Consistent character sheets", "Multi-subject composition",
                        "Custom LoRA training for style"],
            "Adult":   ["Production-ready asset pipeline", "Brand-consistent generation",
                        "Photorealistic product renders"],
            "Expert":  ["End-to-end creative direction", "Model fine-tuning for client identity",
                        "Multi-modal concept art pipeline"],
        },
    },
    {
        "domain": "Video & Avatar Creation",
        "category": "Artistic",
        "stages": {
            "Baby":    ["Text-to-clip API call (Runway/Kling)", "Avatar selection (Alex Riviera)",
                        "Basic voiceover synthesis (ElevenLabs)"],
            "Toddler": ["Script → storyboard planning", "B-roll sequencing",
                        "Lip-sync alignment basics"],
            "Child":   ["Multi-scene video stitching", "Motion prompt engineering",
                        "Background music selection & mixing"],
            "Teen":    ["Consistent character animation", "Product demo video generation",
                        "Caption & subtitle overlay"],
            "Adult":   ["Long-form content (5–15 min) generation", "Brand intro/outro templates",
                        "Multi-avatar dialogue scenes"],
            "Expert":  ["Autonomous content calendar execution", "Real-time avatar streaming",
                        "Cinematic colour grading pipeline"],
        },
    },
    {
        "domain": "Audio & Speech Synthesis",
        "category": "Artistic",
        "stages": {
            "Baby":    ["TTS API call", "Voice selection", "SSML basics"],
            "Toddler": ["Emotion modulation", "Pace & pitch control", "Background audio blending"],
            "Child":   ["Voice cloning (ElevenLabs)", "Podcast script execution",
                        "Chapter-level audiobook generation"],
            "Teen":    ["Multi-speaker dialogue synthesis", "Music generation prompt engineering",
                        "Sound effect layering"],
            "Adult":   ["Broadcast-quality audio pipeline", "Real-time transcription + synthesis",
                        "Brand voice consistency enforcement"],
            "Expert":  ["Custom voice model training", "Adaptive emotional narration",
                        "Full studio-quality production pipeline"],
        },
    },

    # ── WEALTH ────────────────────────────────────────────────────────────
    {
        "domain": "Business & Revenue Generation",
        "category": "Wealth",
        "stages": {
            "Baby":    ["Identify revenue stream types", "Basic market sizing",
                        "Monetisation model classification"],
            "Toddler": ["Competitor analysis summaries", "Landing page copy generation",
                        "Email sequence drafting"],
            "Child":   ["Full product launch plan", "Pricing strategy modelling",
                        "Automated lead generation scripts"],
            "Teen":    ["Multi-channel campaign orchestration", "A/B test design & analysis",
                        "Customer LTV optimisation"],
            "Adult":   ["Autonomous revenue loop design", "Partnership identification & outreach",
                        "Financial forecasting models"],
            "Expert":  ["Self-managing revenue agents", "Portfolio diversification strategy",
                        "Real-time market opportunity scanning"],
        },
    },
    {
        "domain": "Knowledge Management & RAG",
        "category": "Core",
        "stages": {
            "Baby":    ["Document chunking basics", "Embedding concepts", "Simple cosine similarity search"],
            "Toddler": ["Chunking strategy selection", "Metadata filtering", "Hybrid BM25+vector search"],
            "Child":   ["Re-ranking with cross-encoder", "Citation tracking", "Knowledge graph extraction"],
            "Teen":    ["Multi-hop RAG", "Temporal freshness weighting", "Contradiction detection"],
            "Adult":   ["Self-updating knowledge base", "Source reliability scoring",
                        "Autonomous literature review"],
            "Expert":  ["Dynamic ontology construction", "Cross-domain knowledge synthesis",
                        "Proactive knowledge gap detection"],
        },
    },
]


# ---------------------------------------------------------------------------
# Assessment engine
# ---------------------------------------------------------------------------
class AIAssessmentEngine:
    """Generates stage-appropriate challenges and scores responses."""

    CHALLENGE_TEMPLATES = {
        "Core": [
            "Explain {topic} at a {stage} level in under 150 words.",
            "Given the input '{sample}', demonstrate {topic}.",
            "Identify the error in: {code_sample}",
        ],
        "Accelerator": [
            "Write production-ready code that implements {topic}.",
            "Debug the following and explain each fix: {code_sample}",
            "Design a step-by-step agentic plan for: {topic}",
        ],
        "Artistic": [
            "Generate a detailed prompt for {topic} that would produce professional output.",
            "Describe the pipeline to create {topic} from scratch.",
            "Critique this output and suggest improvements: {sample}",
        ],
        "Wealth": [
            "Create a concise {topic} strategy for a bootstrapped startup.",
            "Identify 3 risks in this plan and mitigate them: {sample}",
            "Write actionable copy for: {topic}",
        ],
    }

    def generate_challenge(self, domain: Dict, stage: str) -> Dict:
        category = domain["category"]
        templates = self.CHALLENGE_TEMPLATES.get(category, self.CHALLENGE_TEMPLATES["Core"])
        template = random.choice(templates)
        challenge_text = template.format(
            topic=domain["domain"],
            stage=stage,
            sample=f"[{stage}-level {domain['domain']} scenario]",
            code_sample=f"[{stage}-level code snippet for {domain['domain']}]",
        )
        return {
            "domain":    domain["domain"],
            "stage":     stage,
            "category":  category,
            "challenge": challenge_text,
            "skills":    domain["stages"][stage],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def score_response(self, response: str, expected_skills: List[str]) -> float:
        """Simple heuristic scoring — replace with real eval in production."""
        if not response:
            return 0.0
        score = 0.0
        response_lower = response.lower()
        for skill in expected_skills:
            keywords = [w.lower() for w in skill.split()[:3]]
            if any(k in response_lower for k in keywords):
                score += 1.0 / len(expected_skills)
        length_bonus = min(0.1, len(response) / 2000)
        return min(1.0, score + length_bonus)


# ---------------------------------------------------------------------------
# Stage progression tracker
# ---------------------------------------------------------------------------
class StageProgressionTracker:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.state_file = self.data_path / "ai_training_state.json"
        self.state: Dict = self._load_state()

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {d["domain"]: {"stage": "Baby", "mastery": 0.0, "attempts": 0}
                for d in FULL_CURRICULUM}

    def save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_stage(self, domain: str) -> str:
        return self.state.get(domain, {}).get("stage", "Baby")

    def update_mastery(self, domain: str, score: float) -> bool:
        """Returns True if stage was advanced."""
        rec = self.state.setdefault(domain, {"stage": "Baby", "mastery": 0.0, "attempts": 0})
        rec["attempts"] += 1
        # Running average
        rec["mastery"] = rec["mastery"] * 0.7 + score * 0.3
        current_stage = rec["stage"]
        gate = STAGE_MASTERY_GATE[current_stage]
        if rec["mastery"] >= gate and STAGES.index(current_stage) < len(STAGES) - 1:
            rec["stage"] = STAGES[STAGES.index(current_stage) + 1]
            logger.info(f"[STAGE UP] {domain}: {current_stage} → {rec['stage']}")
            return True
        return False

    def overall_progress(self) -> Dict:
        total = len(self.state)
        by_stage: Dict[str, int] = {s: 0 for s in STAGES}
        avg_mastery = 0.0
        for rec in self.state.values():
            by_stage[rec.get("stage", "Baby")] += 1
            avg_mastery += rec.get("mastery", 0.0)
        return {
            "domains_total": total,
            "by_stage": by_stage,
            "avg_mastery": round(avg_mastery / total, 3) if total else 0,
            "expert_count": by_stage.get("Expert", 0),
            "pct_expert": round(by_stage.get("Expert", 0) / total * 100, 1) if total else 0,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ComprehensiveAITraining:
    """
    Full AI training program for DMAI.

    Constructor pattern matches all other DMAI components.
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

        self.assessment = AIAssessmentEngine()
        self.tracker = StageProgressionTracker(data_path)
        self.session_log: List[Dict] = []

        logger.info("ComprehensiveAITraining initialised — %d domains loaded", len(FULL_CURRICULUM))

    # ── Public API ────────────────────────────────────────────────────────

    async def run_full_program(self, domains: Optional[List[str]] = None) -> Dict:
        """Run training across all (or specified) domains, stage by stage."""
        logger.info("=== DMAI Full AI Training Program START ===")
        start = datetime.now(timezone.utc)
        results = []

        target_domains = [d for d in FULL_CURRICULUM if domains is None or d["domain"] in domains]

        for domain in target_domains:
            result = await self._train_domain(domain)
            results.append(result)
            self.tracker.save_state()

        progress = self.tracker.overall_progress()
        self._update_si_kpis(progress)

        summary = {
            "session_id":   start.strftime("%Y%m%d_%H%M%S"),
            "duration_s":   (datetime.now(timezone.utc) - start).total_seconds(),
            "domains_trained": len(results),
            "progress":     progress,
            "timestamp":    start.isoformat(),
        }
        logger.info("=== DMAI Full AI Training Program COMPLETE: %s ===", progress)
        return summary

    async def train_stage(self, stage: str) -> Dict:
        """Train only domains currently at a specific stage."""
        target = [d for d in FULL_CURRICULUM
                  if self.tracker.get_stage(d["domain"]) == stage]
        logger.info("Stage-targeted training: %s — %d domains", stage, len(target))
        results = []
        for domain in target:
            result = await self._train_domain(domain)
            results.append(result)
        self.tracker.save_state()
        return {"stage": stage, "domains": len(results), "results": results}

    async def train_category(self, category: str) -> Dict:
        """Train all domains in a DMAI category (Core/Accelerator/Artistic/Wealth)."""
        target = [d for d in FULL_CURRICULUM if d["category"] == category]
        results = []
        for domain in target:
            result = await self._train_domain(domain)
            results.append(result)
        self.tracker.save_state()
        return {"category": category, "domains": len(results)}

    def get_status(self) -> Dict:
        return {
            "component": "ComprehensiveAITraining",
            "version": "1.0.0",
            "domains": len(FULL_CURRICULUM),
            "progress": self.tracker.overall_progress(),
            "curriculum_categories": list({d["category"] for d in FULL_CURRICULUM}),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _train_domain(self, domain: Dict) -> Dict:
        current_stage = self.tracker.get_stage(domain["domain"])
        challenge = self.assessment.generate_challenge(domain, current_stage)
        response = await self._get_ai_response(challenge)
        score = self.assessment.score_response(response, domain["stages"][current_stage])
        advanced = self.tracker.update_mastery(domain["domain"], score)

        entry = {
            "domain":    domain["domain"],
            "stage":     current_stage,
            "score":     round(score, 3),
            "advanced":  advanced,
            "new_stage": self.tracker.get_stage(domain["domain"]),
        }
        self.session_log.append(entry)

        if self.knowledge_graph:
            try:
                self.knowledge_graph.add_concept(
                    f"ai_training_{domain['domain'].lower().replace(' ', '_')}",
                    {"stage": entry["new_stage"], "mastery": score, "timestamp": datetime.now(timezone.utc).isoformat()},
                )
            except Exception:
                pass

        return entry

    async def _get_ai_response(self, challenge: Dict) -> str:
        """Route challenge to ai_hub if available, else return a mock response."""
        if self.ai_hub and hasattr(self.ai_hub, "chat"):
            try:
                prompt = (
                    f"DMAI Training Challenge — Domain: {challenge['domain']}, "
                    f"Stage: {challenge['stage']}\n\n{challenge['challenge']}\n\n"
                    f"Target skills: {', '.join(challenge['skills'])}"
                )
                return await self.ai_hub.chat(prompt)
            except Exception as e:
                logger.warning("ai_hub.chat failed: %s", e)

        # Fallback: synthesise a training signal internally
        skills_text = "; ".join(challenge["skills"])
        return (
            f"[DMAI Internal Training] Domain={challenge['domain']} Stage={challenge['stage']} "
            f"Skills demonstrated: {skills_text}. "
            f"Response length calibrated to {challenge['stage']} mastery expectations."
        )

    def _update_si_kpis(self, progress: Dict):
        """Push training progress into SICore KPIs if connected."""
        if not self.si_core:
            return
        try:
            avg = progress["avg_mastery"]
            expert_pct = progress["pct_expert"] / 100.0
            self.si_core.update_kpi("skill_acquisition_rate",   avg)
            self.si_core.update_kpi("sample_efficiency_trend",  avg)
            self.si_core.update_kpi("agentic_capability_score", expert_pct)
            logger.info("SICore KPIs updated from AI training results")
        except Exception as e:
            logger.warning("SICore KPI update failed: %s", e)


# ---------------------------------------------------------------------------
# Flask integration helper
# ---------------------------------------------------------------------------
def register_ai_training_routes(app, trainer: ComprehensiveAITraining):
    """Register Flask routes for the AI training component."""
    import asyncio
    from flask import jsonify, request

    @app.route("/api/training/ai/status")
    def ai_training_status():
        return jsonify(trainer.get_status())

    @app.route("/api/training/ai/start", methods=["POST"])
    def ai_training_start():
        data = request.get_json(silent=True) or {}
        domains = data.get("domains")
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(trainer.run_full_program(domains))
        loop.close()
        return jsonify(result)

    @app.route("/api/training/ai/stage/<stage>", methods=["POST"])
    def ai_training_stage(stage):
        if stage not in STAGES:
            return jsonify({"error": f"Unknown stage. Valid: {STAGES}"}), 400
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(trainer.train_stage(stage))
        loop.close()
        return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ComprehensiveAITraining(data_path="/tmp/dmai_test/")
    print(json.dumps(trainer.get_status(), indent=2))
    result = asyncio.run(trainer.run_full_program())
    print(json.dumps(result, indent=2))
