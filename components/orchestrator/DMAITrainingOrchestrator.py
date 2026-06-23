"""
DMAI Master Training Orchestrator
===================================
Wires all DMAI training programs into a single entry point.

Integrates with existing DMAI components:
  - SICore                       → reads/writes 8 KPIs
  - KnowledgeGraph               → stores training concepts
  - AIIntegrationHub             → routes AI calls
  - EvolutionTrainingSystem      → triggers evolution if regression found
  - KaizenLoop (/api/kaizen)     → submits improvement proposals

New components being wired in:
  - ComprehensiveAITraining      → full AI curriculum (Baby→Expert)
  - FullSITrainingProgram        → 8 new SI modules + KPI integration
  - PeriodicUpdateEngine         → self-update / kaizen integration
  - ExtendedAIIntegrationHub     → 8 new AI providers

Usage (standalone bootstrap):
    python -m components.orchestrator.DMAITrainingOrchestrator

Usage (Flask app integration):
    from components.orchestrator.DMAITrainingOrchestrator import (
        DMAITrainingOrchestrator, register_orchestrator_routes
    )
    orchestrator = DMAITrainingOrchestrator(
        data_path   = app.config["DATA_PATH"],
        si_core     = si_core,
        knowledge_graph = knowledge_graph,
        ai_hub      = ai_hub,
    )
    register_orchestrator_routes(app, orchestrator)
    orchestrator.start_background_updater(app)
"""

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dmai.orchestrator")

# ---------------------------------------------------------------------------
# Lazy imports — components are in the same repo
# ---------------------------------------------------------------------------
def _import_components():
    """Import all training components, handling missing optional deps gracefully."""
    components = {}

    try:
        from components.training.ComprehensiveAITraining import ComprehensiveAITraining
        components["ComprehensiveAITraining"] = ComprehensiveAITraining
    except ImportError as e:
        logger.warning("Could not import ComprehensiveAITraining: %s", e)

    try:
        from components.si_training.FullSITrainingProgram import FullSITrainingProgram
        components["FullSITrainingProgram"] = FullSITrainingProgram
    except ImportError as e:
        logger.warning("Could not import FullSITrainingProgram: %s", e)

    try:
        from components.update_engine.PeriodicUpdateEngine import PeriodicUpdateEngine
        components["PeriodicUpdateEngine"] = PeriodicUpdateEngine
    except ImportError as e:
        logger.warning("Could not import PeriodicUpdateEngine: %s", e)

    try:
        from components.phase11.ExtendedAIIntegrationHub import ExtendedAIIntegrationHub
        components["ExtendedAIIntegrationHub"] = ExtendedAIIntegrationHub
    except ImportError as e:
        logger.warning("Could not import ExtendedAIIntegrationHub: %s", e)

    return components


# ---------------------------------------------------------------------------
# Training run record
# ---------------------------------------------------------------------------
class TrainingRunRecord:
    def __init__(self, data_path: Path):
        self.file = data_path / "training_run_history.jsonl"

    def record(self, run: Dict):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file, "a") as f:
            f.write(json.dumps(run) + "\n")

    def last_runs(self, n: int = 10) -> List[Dict]:
        if not self.file.exists():
            return []
        lines = self.file.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-n:] if l]


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------
class DMAITrainingOrchestrator:
    """
    Single entry point for all DMAI training and self-update operations.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def __init__(
        self,
        data_path: str = "data/",
        si_core=None,
        knowledge_graph=None,
        ai_hub=None,
        evolution_system=None,
        config: Optional[Dict] = None,
    ):
        self.data_path        = Path(data_path)
        self.si_core          = si_core
        self.knowledge_graph  = knowledge_graph
        self.evolution_system = evolution_system
        self.config           = config or {}
        self._update_thread: Optional[threading.Thread] = None

        # Load component classes
        self._classes = _import_components()

        # Instantiate — always use existing ai_hub as base where relevant
        kwargs = dict(
            data_path       = str(self.data_path),
            si_core         = si_core,
            knowledge_graph = knowledge_graph,
            ai_hub          = ai_hub,
        )

        self.ai_trainer    = self._classes["ComprehensiveAITraining"](**kwargs) \
                             if "ComprehensiveAITraining" in self._classes else None

        self.si_trainer    = self._classes["FullSITrainingProgram"](**kwargs) \
                             if "FullSITrainingProgram" in self._classes else None

        self.update_engine = self._classes["PeriodicUpdateEngine"](**kwargs) \
                             if "PeriodicUpdateEngine" in self._classes else None

        self.extended_hub  = self._classes["ExtendedAIIntegrationHub"](
            data_path = str(self.data_path),
            base_hub  = ai_hub,
        ) if "ExtendedAIIntegrationHub" in self._classes else None

        self.run_history = TrainingRunRecord(self.data_path)

        logger.info(
            "DMAITrainingOrchestrator ready — components: AI=%s SI=%s Update=%s ExtHub=%s",
            bool(self.ai_trainer), bool(self.si_trainer),
            bool(self.update_engine), bool(self.extended_hub),
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def run_full_training(self) -> Dict:
        """
        Run the complete DMAI training sequence:
          1. Extended hub initialised (already done in __init__)
          2. AI training program (all domains)
          3. SI training program (all 8 new modules)
          4. Periodic update engine (one-time pass)
          5. Evolution trigger if regressions found
        """
        logger.info("=== DMAI FULL TRAINING SEQUENCE START ===")
        start = datetime.now(timezone.utc)
        results: Dict[str, Any] = {
            "started_at": start.isoformat(),
            "components": {},
        }

        # Step 1: AI Training
        if self.ai_trainer:
            logger.info("Step 1/3: Running AI Training Program...")
            try:
                ai_result = await self.ai_trainer.run_full_program()
                results["components"]["ai_training"] = ai_result
                logger.info("AI Training complete: %s", ai_result.get("progress", {}))
            except Exception as e:
                logger.error("AI Training error: %s", e)
                results["components"]["ai_training"] = {"error": str(e)}
        else:
            results["components"]["ai_training"] = {"skipped": "component not loaded"}

        # Step 2: SI Training
        if self.si_trainer:
            logger.info("Step 2/3: Running SI Training Program...")
            try:
                si_result = await self.si_trainer.run_full_si_program()
                results["components"]["si_training"] = si_result
                logger.info("SI Training complete: score=%.3f", si_result.get("overall_score", 0))
            except Exception as e:
                logger.error("SI Training error: %s", e)
                results["components"]["si_training"] = {"error": str(e)}
        else:
            results["components"]["si_training"] = {"skipped": "component not loaded"}

        # Step 3: Update engine (one-time pass)
        if self.update_engine:
            logger.info("Step 3/3: Running Update Engine...")
            try:
                update_result = await self.update_engine.run_once()
                results["components"]["update_engine"] = update_result
                # Trigger evolution if regressions detected
                bench = update_result.get("benchmark", {})
                if bench.get("regressions") and self.evolution_system:
                    logger.warning("Regressions detected — triggering evolution cycle")
                    await self._trigger_evolution(bench["regressions"])
                    results["evolution_triggered"] = True
            except Exception as e:
                logger.error("Update Engine error: %s", e)
                results["components"]["update_engine"] = {"error": str(e)}
        else:
            results["components"]["update_engine"] = {"skipped": "component not loaded"}

        # Finalise
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        results["duration_s"]   = round(duration, 2)
        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        results["status"]       = "complete"

        self.run_history.record({
            "type":      "full_training",
            "started":   start.isoformat(),
            "duration":  duration,
            "status":    results["status"],
        })

        logger.info("=== DMAI FULL TRAINING SEQUENCE COMPLETE (%.1fs) ===", duration)
        return results

    async def run_quick_training(self, focus: str = "Core") -> Dict:
        """
        Quick training pass — only one category or SI module.
        Useful for scheduled micro-updates without a full run.
        focus: AI category (Core/Accelerator/Artistic/Wealth) or SI module id
        """
        start = datetime.now(timezone.utc)
        results: Dict = {"focus": focus, "started_at": start.isoformat()}

        if focus in ("Core", "Accelerator", "Artistic", "Wealth"):
            if self.ai_trainer:
                results["ai"] = await self.ai_trainer.train_category(focus)
        else:
            # Assume it's an SI module id
            if self.si_trainer:
                try:
                    results["si"] = await self.si_trainer.run_module(focus)
                except ValueError:
                    results["si"] = {"error": f"Unknown module/category: {focus}"}

        results["duration_s"] = round((datetime.now(timezone.utc) - start).total_seconds(), 2)
        return results

    async def run_update_only(self) -> Dict:
        """Run only the periodic update engine (no training)."""
        if self.update_engine:
            return await self.update_engine.run_once()
        return {"error": "PeriodicUpdateEngine not loaded"}

    def start_background_updater(self, flask_app=None):
        """
        Start the PeriodicUpdateEngine in a background thread.
        Call this once from app startup.
        """
        if not self.update_engine:
            logger.warning("Cannot start background updater — PeriodicUpdateEngine not loaded")
            return

        if self._update_thread and self._update_thread.is_alive():
            logger.info("Background updater already running")
            return

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.update_engine.start())
            except Exception as e:
                logger.error("Background updater crashed: %s", e)
            finally:
                loop.close()

        self._update_thread = threading.Thread(target=_run, daemon=True, name="dmai-update-engine")
        self._update_thread.start()
        logger.info("Background update engine started (daemon thread)")

    def stop_background_updater(self):
        if self.update_engine:
            self.update_engine.stop()

    def get_status(self) -> Dict:
        import threading as _th
        tnames = [t.name for t in _th.enumerate()]

        def _up(*kws):
            return any(any(kw.lower() in n.lower() for kw in kws) for n in tnames)

        services = {
            "background_updater":    bool(self._update_thread and self._update_thread.is_alive()),
            "parallel_learner":      _up("parallel", "parallellearner", "web_learn"),
            "autonomous_researcher": _up("research", "autonomousresearch"),
            "stage_learner":         _up("stage", "stagelearner", "learning_loop"),
            "kaizen_repair":         _up("kaizen", "repair"),
            "graph_evolution":       _up("graph", "graphevolution"),
            "kpi_seed":              _up("kpi", "kpiseed"),
            "vocab_ingest":          _up("vocab", "vocabingest"),
        }
        active = sum(1 for v in services.values() if v)

        status: Dict = {
            "component": "DMAITrainingOrchestrator",
            "version":   "1.0.0",
            "background_updater_alive": bool(self._update_thread and self._update_thread.is_alive()),
            # ── Thread / service status for the UI banner ──────────────
            "status":             "healthy" if active >= 3 else "degraded",
            "training_always_on": True,
            "message":            "Training runs 24/7 automatically — no manual start needed",
            "services":           services,
            "active_count":       active,
            "total_threads":      len(tnames),
            "thread_names":       tnames,
            # ── existing fields ────────────────────────────────────────
            "components": {},
            "recent_runs": self.run_history.last_runs(5),
        }

        if self.ai_trainer:
            status["components"]["ai_training"] = self.ai_trainer.get_status()
        if self.si_trainer:
            status["components"]["si_training"] = self.si_trainer.get_status()
        if self.update_engine:
            status["components"]["update_engine"] = self.update_engine.get_status()
        if self.extended_hub:
            status["components"]["extended_hub"] = self.extended_hub.get_status()

        return status

    # ── Private helpers ───────────────────────────────────────────────────

    async def _trigger_evolution(self, regressions: List[Dict]):
        if not self.evolution_system:
            return
        try:
            if hasattr(self.evolution_system, "trigger_evolution"):
                await self.evolution_system.trigger_evolution({
                    "reason":      "benchmark_regression",
                    "regressions": regressions,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                })
            elif hasattr(self.evolution_system, "run_evolution_cycle"):
                self.evolution_system.run_evolution_cycle()
        except Exception as e:
            logger.warning("Evolution trigger failed: %s", e)


# ---------------------------------------------------------------------------
# Flask integration helper
# ---------------------------------------------------------------------------
def register_orchestrator_routes(app, orch: DMAITrainingOrchestrator):
    """
    Register all orchestrator routes + delegate to sub-component routes.
    Call this from dmai_core_complete.py after creating the orchestrator.
    """
    import asyncio
    from flask import jsonify, request

    # Import sub-component route registrars
    try:
        from components.training.ComprehensiveAITraining import register_ai_training_routes
        if orch.ai_trainer:
            register_ai_training_routes(app, orch.ai_trainer)
    except ImportError:
        pass

    try:
        from components.si_training.FullSITrainingProgram import register_si_training_routes
        if orch.si_trainer:
            register_si_training_routes(app, orch.si_trainer)
    except ImportError:
        pass

    try:
        from components.update_engine.PeriodicUpdateEngine import register_update_engine_routes
        if orch.update_engine:
            register_update_engine_routes(app, orch.update_engine)
    except ImportError:
        pass

    try:
        from components.phase11.ExtendedAIIntegrationHub import register_extended_hub_routes
        if orch.extended_hub:
            register_extended_hub_routes(app, orch.extended_hub)
    except ImportError:
        pass

    # Master orchestrator routes
    @app.route("/api/training/status")
    def training_status():
        return jsonify(orch.get_status())

    @app.route("/api/training/full", methods=["POST"])
    def training_full():
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(orch.run_full_training())
        loop.close()
        return jsonify(result)

    @app.route("/api/training/quick", methods=["POST"])
    def training_quick():
        data  = request.get_json(silent=True) or {}
        focus = data.get("focus", "Core")
        loop  = asyncio.new_event_loop()
        result = loop.run_until_complete(orch.run_quick_training(focus))
        loop.close()
        return jsonify(result)

    @app.route("/api/training/update", methods=["POST"])
    def training_update():
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(orch.run_update_only())
        loop.close()
        return jsonify(result)

    @app.route("/api/training/updater/start", methods=["POST"])
    def training_updater_start():
        orch.start_background_updater()
        return jsonify({"status": "started"})

    @app.route("/api/training/updater/stop", methods=["POST"])
    def training_updater_stop():
        orch.stop_background_updater()
        return jsonify({"status": "stopped"})

    logger.info("Orchestrator routes registered")


# ---------------------------------------------------------------------------
# Bootstrap script for drop-in registration in dmai_core_complete.py
# ---------------------------------------------------------------------------
_BOOTSTRAP_SNIPPET = '''
# ─────────────────────────────────────────────────────────────────────────────
# DMAI Training Orchestrator Bootstrap
# Add this block to dmai_core_complete.py after all existing components init
# ─────────────────────────────────────────────────────────────────────────────
from components.orchestrator.DMAITrainingOrchestrator import (
    DMAITrainingOrchestrator,
    register_orchestrator_routes,
)

training_orchestrator = DMAITrainingOrchestrator(
    data_path       = DATA_PATH,          # your existing DATA_PATH variable
    si_core         = si_core,            # your existing SICore instance
    knowledge_graph = knowledge_graph,    # your existing KnowledgeGraph instance
    ai_hub          = ai_hub,             # your existing AIIntegrationHub instance
    evolution_system= evolution_training, # your existing EvolutionTrainingSystem instance
)

register_orchestrator_routes(app, training_orchestrator)
training_orchestrator.start_background_updater(app)
# ─────────────────────────────────────────────────────────────────────────────
'''


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dmai_orchestrator_test/"

    print("\n" + "="*60)
    print("  DMAI Training Orchestrator — Standalone Mode")
    print("="*60 + "\n")

    orch = DMAITrainingOrchestrator(data_path=data_path)
    print("Status:")
    print(json.dumps(orch.get_status(), indent=2))

    print("\nRunning full training sequence...")
    result = asyncio.run(orch.run_full_training())
    print(json.dumps(result, indent=2))

    print("\nBootstrap snippet for dmai_core_complete.py:")
    print(_BOOTSTRAP_SNIPPET)
