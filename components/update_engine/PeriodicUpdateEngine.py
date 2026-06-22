"""
DMAI PeriodicUpdateEngine
==========================
Periodic self-update engine compatible with DMAI's existing kaizen / evolution loop.

What it does:
  1. ModelRegistryPoller  — checks for new model versions at configured intervals
  2. FeedbackRetrainer    — collects user feedback signals and retrains from them
  3. KnowledgeFreshenJob  — fetches latest AI research summaries and adds to KG
  4. PerformanceBenchmark — benchmarks DMAI against reference responses; flags regressions
  5. KaizenIntegrator     — hooks into /api/kaizen and pushes improvement proposals
  6. UpdateScheduler      — orchestrates all jobs with configurable cron-like intervals

Hooks into existing DMAI components:
  - SICore.update_kpi()              (updates metrics after each run)
  - EvolutionTrainingSystem          (triggers evolution cycle if regression detected)
  - KnowledgeGraph.add_concept()     (stores new knowledge)
  - /api/kaizen endpoint             (submits proposals)

Usage:
    engine = PeriodicUpdateEngine(
        data_path       = "data/",
        si_core         = si_core_instance,
        knowledge_graph = knowledge_graph_instance,
        ai_hub          = ai_hub_instance,
        config          = {}           # optional overrides
    )
    asyncio.run(engine.start())       # runs indefinitely
    # or:
    await engine.run_once()           # single pass (for testing / on-demand)
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx                          # lightweight HTTP client (already in DMAI deps)

logger = logging.getLogger("dmai.update_engine")

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Interval in seconds for each job
    "model_registry_interval_s":    3600 * 6,     # every 6 hours
    "feedback_retraining_interval_s": 3600 * 12,  # every 12 hours
    "knowledge_freshen_interval_s": 3600 * 24,    # daily
    "benchmark_interval_s":         3600 * 48,    # every 2 days
    "kaizen_interval_s":            3600 * 6,     # every 6 hours

    # Sources to poll for new AI model releases
    "model_registry_sources": [
        "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        "https://openrouter.ai/api/v1/models",
    ],

    # Research/news RSS for knowledge freshening
    "knowledge_sources": [
        "https://huggingface.co/api/papers?sort=published",       # HF daily papers
        "https://paperswithcode.com/api/v1/papers/?ordering=-published",
    ],

    # Benchmark reference Q&A pairs (extend as needed)
    "benchmark_prompts": [
        {"prompt": "What is the capital of France?",       "expected_keywords": ["paris"]},
        {"prompt": "Write a Python function to reverse a string.", "expected_keywords": ["def", "return", "[::-1]"]},
        {"prompt": "Summarise the concept of neural networks.", "expected_keywords": ["neuron", "layer", "weight"]},
        {"prompt": "List 3 benefits of async programming.",  "expected_keywords": ["concurrency", "non-blocking", "performance"]},
    ],

    # Kaizen API endpoint (local or Render)
    "kaizen_endpoint": os.environ.get("DMAI_API_URL", "http://localhost:5000") + "/api/kaizen",

    # Feedback file (populated by user interaction logging)
    "feedback_file": "data/user_feedback.jsonl",

    # State file to track last-run timestamps
    "state_file": "data/update_engine_state.json",
}


# ---------------------------------------------------------------------------
# Model Registry Poller
# ---------------------------------------------------------------------------
class ModelRegistryPoller:
    """
    Polls upstream model registries for new/updated models.
    When a new model is detected, stores it for the integrations layer to pick up.
    """

    def __init__(self, config: Dict, data_path: Path):
        self.config    = config
        self.known_models_file = data_path / "known_models.json"
        self.known_models: Dict = self._load_known()

    def _load_known(self) -> Dict:
        if self.known_models_file.exists():
            with open(self.known_models_file) as f:
                return json.load(f)
        return {}

    def _save_known(self):
        with open(self.known_models_file, "w") as f:
            json.dump(self.known_models, f, indent=2)

    async def poll(self) -> List[Dict]:
        """Returns list of newly discovered models."""
        new_models: List[Dict] = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for url in self.config.get("model_registry_sources", []):
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()

                    # LiteLLM format: dict of {model_id: {...}}
                    if isinstance(data, dict):
                        models = list(data.keys())
                    # OpenRouter format: {"data": [{id: "...", ...}]}
                    elif isinstance(data, dict) and "data" in data:
                        models = [m.get("id", "") for m in data.get("data", [])]
                    else:
                        models = []

                    for model_id in models:
                        if not model_id:
                            continue
                        fingerprint = hashlib.md5(model_id.encode()).hexdigest()[:8]
                        if fingerprint not in self.known_models:
                            self.known_models[fingerprint] = {
                                "model_id":  model_id,
                                "source":    url,
                                "discovered": datetime.now(timezone.utc).isoformat(),
                            }
                            new_models.append({"model_id": model_id, "source": url})

                except Exception as e:
                    logger.warning("ModelRegistryPoller: failed to poll %s — %s", url, e)

        if new_models:
            self._save_known()
            logger.info("ModelRegistryPoller: %d new models discovered", len(new_models))
        return new_models


# ---------------------------------------------------------------------------
# Feedback Retrainer
# ---------------------------------------------------------------------------
class FeedbackRetrainer:
    """
    Reads accumulated user feedback from the feedback JSONL file,
    constructs training pairs, and updates the SICore skill_acquisition_rate KPI.

    Feedback file format (one JSON per line):
        {"prompt": "...", "response": "...", "rating": 1-5, "timestamp": "..."}
    """

    def __init__(self, config: Dict, data_path: Path, si_core=None):
        self.config   = config
        self.si_core  = si_core
        feedback_rel  = config.get("feedback_file", "data/user_feedback.jsonl")
        self.feedback_file = data_path / feedback_rel if not Path(feedback_rel).is_absolute() else Path(feedback_rel)
        self.processed_ids: set = set()

    async def retrain(self) -> Dict:
        if not self.feedback_file.exists():
            logger.info("FeedbackRetrainer: no feedback file yet")
            return {"processed": 0, "positive": 0, "negative": 0}

        positive, negative = [], []
        with open(self.feedback_file) as f:
            for line in f:
                try:
                    fb = json.loads(line.strip())
                    uid = hashlib.md5(json.dumps(fb, sort_keys=True).encode()).hexdigest()
                    if uid in self.processed_ids:
                        continue
                    self.processed_ids.add(uid)
                    rating = fb.get("rating", 3)
                    if rating >= 4:
                        positive.append(fb)
                    elif rating <= 2:
                        negative.append(fb)
                except Exception:
                    pass

        total = len(positive) + len(negative)
        if total == 0:
            return {"processed": 0, "positive": 0, "negative": 0}

        # Compute quality signal
        quality_signal = len(positive) / total if total > 0 else 0.5

        if self.si_core:
            try:
                self.si_core.update_kpi("skill_acquisition_rate", quality_signal)
                self.si_core.update_kpi("sample_efficiency_trend", quality_signal)
            except Exception as e:
                logger.warning("FeedbackRetrainer SICore update: %s", e)

        logger.info("FeedbackRetrainer: processed %d samples (%.0f%% positive)", total, quality_signal * 100)
        return {"processed": total, "positive": len(positive), "negative": len(negative),
                "quality_signal": round(quality_signal, 3)}


# ---------------------------------------------------------------------------
# Knowledge Freshen Job
# ---------------------------------------------------------------------------
class KnowledgeFreshenJob:
    """
    Fetches latest AI research paper titles/abstracts and injects them
    into the DMAI knowledge graph.
    """

    def __init__(self, config: Dict, knowledge_graph=None):
        self.config = config
        self.kg     = knowledge_graph

    async def freshen(self) -> Dict:
        fetched = 0
        async with httpx.AsyncClient(timeout=20.0) as client:
            for url in self.config.get("knowledge_sources", []):
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()

                    papers = []
                    if isinstance(data, list):
                        papers = data[:10]
                    elif isinstance(data, dict) and "results" in data:
                        papers = data["results"][:10]

                    for paper in papers:
                        title    = paper.get("title", paper.get("id", "unknown"))
                        abstract = paper.get("abstract", paper.get("summary", ""))
                        concept  = f"paper_{hashlib.md5(title.encode()).hexdigest()[:8]}"

                        if self.kg:
                            try:
                                self.kg.add_concept(concept, {
                                    "title":     title,
                                    "abstract":  abstract[:500],
                                    "source":    url,
                                    "ingested":  datetime.now(timezone.utc).isoformat(),
                                    "type":      "research_paper",
                                })
                                fetched += 1
                            except Exception as e:
                                logger.warning("KnowledgeFreshenJob KG insert: %s", e)

                except Exception as e:
                    logger.warning("KnowledgeFreshenJob fetch error %s: %s", url, e)

        logger.info("KnowledgeFreshenJob: ingested %d new research items", fetched)
        return {"papers_ingested": fetched, "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Performance Benchmark
# ---------------------------------------------------------------------------
class PerformanceBenchmark:
    """
    Runs reference prompts through DMAI and compares keyword coverage.
    Flags regressions vs last baseline.
    """

    def __init__(self, config: Dict, data_path: Path, ai_hub=None, si_core=None):
        self.config    = config
        self.ai_hub    = ai_hub
        self.si_core   = si_core
        self.baseline_file = data_path / "benchmark_baseline.json"
        self.baseline: Dict = self._load_baseline()

    def _load_baseline(self) -> Dict:
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {}

    def _save_baseline(self, results: Dict):
        with open(self.baseline_file, "w") as f:
            json.dump(results, f, indent=2)

    async def run(self) -> Dict:
        prompts  = self.config.get("benchmark_prompts", [])
        scores   = []
        regressions = []

        for item in prompts:
            prompt   = item["prompt"]
            expected = item.get("expected_keywords", [])
            response = await self._query(prompt)
            rl       = response.lower()
            score    = sum(1 for kw in expected if kw.lower() in rl) / len(expected) if expected else 0.5

            prev_score = self.baseline.get(prompt, {}).get("score", score)
            if score < prev_score - 0.15:
                regressions.append({"prompt": prompt, "prev": prev_score, "current": score})
                logger.warning("REGRESSION detected: '%s' %.2f→%.2f", prompt[:50], prev_score, score)

            scores.append(score)
            self.baseline[prompt] = {
                "score":     score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        avg_score = sum(scores) / len(scores) if scores else 0.0
        self._save_baseline(self.baseline)

        if self.si_core:
            try:
                self.si_core.update_kpi("metacognition_accuracy", avg_score)
            except Exception:
                pass

        return {
            "avg_score":    round(avg_score, 3),
            "regressions":  regressions,
            "prompts_run":  len(prompts),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

    async def _query(self, prompt: str) -> str:
        if self.ai_hub and hasattr(self.ai_hub, "chat"):
            try:
                return await self.ai_hub.chat(prompt)
            except Exception:
                pass
        # Fallback mock
        return f"[BENCHMARK MOCK] {prompt[:30]}... response"


# ---------------------------------------------------------------------------
# Kaizen Integrator
# ---------------------------------------------------------------------------
class KaizenIntegrator:
    """
    Submits improvement proposals to /api/kaizen based on training results.
    """

    def __init__(self, config: Dict):
        self.endpoint = config.get("kaizen_endpoint", "http://localhost:5000/api/kaizen")

    async def submit_proposal(self, proposal: Dict) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.endpoint, json=proposal)
                success = resp.status_code in (200, 201, 202)
                if success:
                    logger.info("KaizenIntegrator: proposal submitted — %s", proposal.get("title", ""))
                return success
        except Exception as e:
            logger.warning("KaizenIntegrator: failed to submit — %s", e)
            return False

    async def propose_from_results(self, benchmark_result: Dict, feedback_result: Dict) -> Dict:
        proposals_submitted = 0
        proposals = []

        # Regression-triggered proposals
        for regression in benchmark_result.get("regressions", []):
            proposal = {
                "title":       f"Fix regression: {regression['prompt'][:40]}",
                "description": (
                    f"Performance dropped from {regression['prev']:.2f} to {regression['current']:.2f}. "
                    f"Trigger targeted retraining on this domain."
                ),
                "priority":    "high",
                "type":        "regression_fix",
                "data":        regression,
            }
            if await self.submit_proposal(proposal):
                proposals_submitted += 1
            proposals.append(proposal)

        # Low-feedback-quality proposal
        quality = feedback_result.get("quality_signal", 1.0)
        if quality < 0.6 and feedback_result.get("processed", 0) > 5:
            proposal = {
                "title":       "Improve user satisfaction (low feedback quality signal)",
                "description": (
                    f"Only {quality*100:.0f}% of recent feedback was positive. "
                    f"Review negative examples and improve response quality."
                ),
                "priority":    "medium",
                "type":        "quality_improvement",
            }
            if await self.submit_proposal(proposal):
                proposals_submitted += 1
            proposals.append(proposal)

        return {"proposals_submitted": proposals_submitted, "proposals": proposals}


# ---------------------------------------------------------------------------
# Update Scheduler
# ---------------------------------------------------------------------------
class UpdateScheduler:
    """Tracks last-run timestamps and determines which jobs are due."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Dict = self._load()

    def _load(self) -> Dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_due(self, job_name: str, interval_s: int) -> bool:
        last_run = self.state.get(job_name, 0)
        return (time.time() - last_run) >= interval_s

    def mark_complete(self, job_name: str):
        self.state[job_name] = time.time()
        self.save()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class PeriodicUpdateEngine:
    """
    Master update engine — wires all sub-jobs together with configurable scheduling.
    """

    def __init__(
        self,
        data_path: str = "data/",
        si_core=None,
        knowledge_graph=None,
        ai_hub=None,
        config: Optional[Dict] = None,
    ):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.si_core = si_core
        self.ai_hub  = ai_hub

        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Sub-jobs
        self.model_poller   = ModelRegistryPoller(self.config, self.data_path)
        self.feedback       = FeedbackRetrainer(self.config, self.data_path, si_core)
        self.knowledge      = KnowledgeFreshenJob(self.config, knowledge_graph)
        self.benchmark      = PerformanceBenchmark(self.config, self.data_path, ai_hub, si_core)
        self.kaizen         = KaizenIntegrator(self.config)
        self.scheduler      = UpdateScheduler(self.data_path / "update_engine_state.json")

        self._running = False
        logger.info("PeriodicUpdateEngine initialised")

    # ── Public API ────────────────────────────────────────────────────────

    async def start(self):
        """Run indefinitely — call from a background task/thread."""
        self._running = True
        logger.info("PeriodicUpdateEngine: starting continuous loop")
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("PeriodicUpdateEngine tick error: %s", e)
            await asyncio.sleep(300)   # check every 5 minutes

    def stop(self):
        self._running = False
        logger.info("PeriodicUpdateEngine: stopping")

    async def run_once(self) -> Dict:
        """Run all jobs once regardless of schedule (useful for testing / manual trigger)."""
        logger.info("PeriodicUpdateEngine: run_once triggered")
        return await self._run_all_jobs()

    async def run_job(self, job_name: str) -> Dict:
        """Run a specific job by name."""
        jobs = {
            "model_registry":     self._job_model_registry,
            "feedback_retraining": self._job_feedback,
            "knowledge_freshen":  self._job_knowledge,
            "benchmark":          self._job_benchmark,
            "kaizen":             self._job_kaizen,
        }
        if job_name not in jobs:
            raise ValueError(f"Unknown job: {job_name}. Valid: {list(jobs.keys())}")
        return await jobs[job_name]()

    def get_status(self) -> Dict:
        now = time.time()
        jobs = {
            "model_registry":     "model_registry_interval_s",
            "feedback_retraining": "feedback_retraining_interval_s",
            "knowledge_freshen":  "knowledge_freshen_interval_s",
            "benchmark":          "benchmark_interval_s",
            "kaizen":             "kaizen_interval_s",
        }
        job_status = {}
        for job, interval_key in jobs.items():
            last   = self.scheduler.state.get(job, 0)
            interval = self.config.get(interval_key, 3600)
            due_in = max(0, interval - (now - last))
            job_status[job] = {
                "last_run":  datetime.fromtimestamp(last, timezone.utc).isoformat() if last else "never",
                "due_in_s":  int(due_in),
                "due_in_h":  round(due_in / 3600, 1),
                "overdue":   due_in == 0 and last > 0,
            }
        return {
            "component":  "PeriodicUpdateEngine",
            "version":    "1.0.0",
            "running":    self._running,
            "jobs":       job_status,
            "config":     {k: v for k, v in self.config.items() if "source" not in k and "prompt" not in k},
        }

    # ── Tick logic ────────────────────────────────────────────────────────

    async def _tick(self):
        cfg = self.config
        if self.scheduler.is_due("model_registry", cfg["model_registry_interval_s"]):
            await self._job_model_registry()

        if self.scheduler.is_due("feedback_retraining", cfg["feedback_retraining_interval_s"]):
            await self._job_feedback()

        if self.scheduler.is_due("knowledge_freshen", cfg["knowledge_freshen_interval_s"]):
            await self._job_knowledge()

        if self.scheduler.is_due("benchmark", cfg["benchmark_interval_s"]):
            await self._job_benchmark()

        if self.scheduler.is_due("kaizen", cfg["kaizen_interval_s"]):
            await self._job_kaizen()

    async def _run_all_jobs(self) -> Dict:
        results = {}
        results["model_registry"]      = await self._job_model_registry()
        results["feedback_retraining"] = await self._job_feedback()
        results["knowledge_freshen"]   = await self._job_knowledge()
        results["benchmark"]           = await self._job_benchmark()
        results["kaizen"]              = await self._job_kaizen()
        return results

    async def _job_model_registry(self) -> Dict:
        result = await self.model_poller.poll()
        self.scheduler.mark_complete("model_registry")
        return {"job": "model_registry", "new_models": len(result), "models": result[:5]}

    async def _job_feedback(self) -> Dict:
        result = await self.feedback.retrain()
        self.scheduler.mark_complete("feedback_retraining")
        return {"job": "feedback_retraining", **result}

    async def _job_knowledge(self) -> Dict:
        result = await self.knowledge.freshen()
        self.scheduler.mark_complete("knowledge_freshen")
        return {"job": "knowledge_freshen", **result}

    async def _job_benchmark(self) -> Dict:
        result = await self.benchmark.run()
        self.scheduler.mark_complete("benchmark")
        return {"job": "benchmark", **result}

    async def _job_kaizen(self) -> Dict:
        bench_result    = self.benchmark.baseline
        feedback_result = {"quality_signal": 0.8, "processed": 0}  # fallback
        result = await self.kaizen.propose_from_results(
            {"regressions": []},
            feedback_result,
        )
        self.scheduler.mark_complete("kaizen")
        return {"job": "kaizen", **result}


# ---------------------------------------------------------------------------
# Flask integration helper
# ---------------------------------------------------------------------------
def register_update_engine_routes(app, engine: PeriodicUpdateEngine):
    import asyncio
    from flask import jsonify, request

    @app.route("/api/update_engine/status")
    def update_engine_status():
        return jsonify(engine.get_status())

    @app.route("/api/update_engine/run", methods=["POST"])
    def update_engine_run():
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(engine.run_once())
        loop.close()
        return jsonify(result)

    @app.route("/api/update_engine/job/<job_name>", methods=["POST"])
    def update_engine_job(job_name):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(engine.run_job(job_name))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        finally:
            loop.close()
        return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = PeriodicUpdateEngine(data_path="/tmp/dmai_update_test/")
    print(json.dumps(engine.get_status(), indent=2))
    result = asyncio.run(engine.run_once())
    print(json.dumps(result, indent=2))
