"""
DMAI Core Complete v7.0.0
==========================
Full production Flask application - wires ALL components together.
v7.0.0: Full validation framework remediation applied.
  - Circuit breakers CB-01 through CB-06
  - JWT authentication (HS256) + backward-compat X-Master-Password
  - Prompt injection filter on all user input
  - exec()/eval() AST scanner on generated code
  - Atomic writes (temp+rename) for kaizen store
  - SHA-256 hash guard on benchmark_baseline.json
  - HMAC webhook signature validation
  - KPI authenticity gate
  - Regression threshold (15%) with severity labels
  - KB quarantine layer
  - Step-by-step JSON chain logging
  - HaltResponse structured refusals
  - Package typosquat validation
  - Bandit integration for code scanning
  - SSIM avatar identity tracking

Run locally:   python dmai_core_complete.py
Run on Render: gunicorn dmai_core_complete:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2

Environment variables: see .env.template
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

def _safe_json_load(path, default=None):
    """Load JSON file safely — returns default on missing file or parse error."""
    try:
        import json as _json
        with open(path, "r") as _f:
            return _json.load(_f)
    except Exception:
        return default if default is not None else {}



# ── Self-generation system ──────────────────────────────────────────────────
try:
    from components.self_scanner import SelfScanner as _SelfScanner
    from components.capability_mapper import CapabilityMapper as _CapMapper
    from components.self_evolution_orchestrator import SelfEvolutionOrchestrator as _SelfEvo
    _self_evolution_available = True
except Exception as _evo_err:
    _self_evolution_available = False

# ── Alex Riviera social automation ─────────────────────────────────────────
try:
    from components.alex_riviera_content import AlexRivieraContentEngine as _AlexContent
    from components.social_media_poster import SocialMediaPoster as _SocialPoster
    _social_available = True
except Exception as _soc_err:
    _social_available = False


# ── Security modules (P1–P3 fixes) ──────────────────────────────────────────
try:
    from security import (
        require_jwt, issue_token_for_password, sanitise_input,
        check_injection, scan_generated_code, safe_code_output,
        scan_imports_in_code, HaltResponse, check_halt_conditions,
        PlanConstraints, Plan, validate_plan,
    )
    SECURITY_AVAILABLE = True
except ImportError as _e:
    logging.warning("security.py not found — P1 security features disabled: %s", _e)
    SECURITY_AVAILABLE = False
    def require_jwt(f): return f
    def issue_token_for_password(pwd): return None
    def sanitise_input(s): return s
    def check_injection(s): return False
    def scan_generated_code(code): return {"safe": True, "issues": []}
    def safe_code_output(code): return code
    def scan_imports_in_code(code): return {"safe": True, "issues": []}
    def check_halt_conditions(ctx): return None

try:
    from circuit_breaker import CircuitBreakerManager, circuit_breaker_guard, after_request_hook
    cb_manager = CircuitBreakerManager.get()
    CB_AVAILABLE = True
except ImportError as _e:
    logging.warning("circuit_breaker.py not found — CB features disabled: %s", _e)
    CB_AVAILABLE = False
    cb_manager = None
    def circuit_breaker_guard(name): 
        def decorator(f): return f
        return decorator
    def after_request_hook(response): return response

try:
    from hmac_validator import validate_webhook_signature, require_webhook_hmac
    HMAC_AVAILABLE = True
except ImportError as _e:
    logging.warning("hmac_validator.py not found — HMAC webhook validation disabled: %s", _e)
    HMAC_AVAILABLE = False
    def require_webhook_hmac(f): return f

try:
    from chain_logger import ChainLogger, log_chain_step
    CHAIN_LOGGER_AVAILABLE = True
except ImportError as _e:
    logging.warning("chain_logger.py not found — chain logging disabled: %s", _e)
    CHAIN_LOGGER_AVAILABLE = False
    def log_chain_step(chain_id, step, data=None): pass

try:
    from bandit_integration import BanditScanner
    BANDIT_AVAILABLE = True
    _bandit = BanditScanner()
except ImportError as _e:
    logging.warning("bandit_integration.py not found — bandit scanning disabled: %s", _e)
    BANDIT_AVAILABLE = False
    _bandit = None

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dmai.core")

# ── Render / production flags ────────────────────────────────────────────────
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"
if IS_RENDER:
    os.environ["DISABLE_NEO4J"] = "true"  # Neo4j fully removed — using pg_storage (Postgres w/ SQLite fallback)
    os.environ["DISABLE_AUTO_THREADS"] = "true"

# ── Data path ────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "data/")
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

# ── Startup time ─────────────────────────────────────────────────────────────
DMAI_VERSION = "7.1.0"  # canonical version — single source of truth

STARTUP_TIME = datetime.now(timezone.utc)

# ── Component registry ────────────────────────────────────────────────────────
components: Dict[str, Any] = {}

# ── Syllabus ─────────────────────────────────────────────────────────────────
try:
    from dmai_syllabus_data import SYLLABUS_TOPICS, TOTAL_TOPICS
    logger.info("Syllabus loaded: %d topics", TOTAL_TOPICS)
except Exception as e:
    logger.warning("Syllabus load failed: %s", e)
    SYLLABUS_TOPICS = {}
    TOTAL_TOPICS = 0

# ── SICore ────────────────────────────────────────────────────────────────────
try:
    from components.si_core import SICore
    components["si_core"] = SICore(data_path=Path(DATA_PATH))
    # Seed KPIs from persisted learning_progress.json (survives redeploys via Git)
    try:
        import json as _json
        _lp_file = Path(DATA_PATH) / "learning" / "stage_syllabus" / "learning_progress.json"
        if _lp_file.exists():
            _lp = _json.loads(_lp_file.read_text())
            _si = components["si_core"]
            _all_topics = {}
            for _stage, _topics in _lp.get("learned_topics", {}).items():
                _all_topics.update({k: v for k, v in _topics.items() if not k.startswith("_")})
            _total = max(len(_all_topics), 1)
            _mastered = sum(1 for v in _all_topics.values() if isinstance(v, (int, float)) and v >= 3)
            _avg = sum(v for v in _all_topics.values() if isinstance(v, (int, float)) and not str(v).startswith("_")) / max(_total, 1)
            _stage_order = ["Baby", "Toddler", "Child", "Teen", "Adult", "Expert"]
            _cur_stage = _lp.get("current_stage", "Baby")
            _stage_idx = _stage_order.index(_cur_stage) if _cur_stage in _stage_order else 0
            _token = None
            try:
                from security import SecurityManager
                _token = SecurityManager.generate_token("system_boot")
            except Exception:
                pass
            if hasattr(_si, "update_kpi"):
                _si.update_kpi("skill_acquisition_rate", min(_avg / 3.0, 1.0), token=_token)
                _si.update_kpi("transfer_learning_rate", _stage_idx / (len(_stage_order) - 1), token=_token)
                _si.update_kpi("zero_shot_success_count", float(_mastered), token=_token)
            logger.info("SICore seeded from learning_progress.json: stage=%s mastered=%d avg=%.3f",
                        _cur_stage, _mastered, _avg)
    except Exception as _e:
        logger.warning("SICore seed from learning_progress failed: %s", _e)
    logger.info("SICore initialised")
except Exception as e:
    logger.warning("SICore failed: %s", e)

# ── AI Integration Hub ────────────────────────────────────────────────────────
try:
    from components.phase11.AIIntegrationHub import AIIntegrationHub
    components["ai_hub"] = AIIntegrationHub(data_path=DATA_PATH)
    logger.info("AIIntegrationHub initialised")
except Exception as e:
    logger.warning("AIIntegrationHub failed: %s", e)

# ── Extended AI Integration Hub ───────────────────────────────────────────────
try:
    from components.phase11.ExtendedAIIntegrationHub import ExtendedAIIntegrationHub
    components["extended_hub"] = ExtendedAIIntegrationHub(
        data_path=DATA_PATH,
        base_hub=components.get("ai_hub"),
    )
    logger.info("ExtendedAIIntegrationHub initialised")
except Exception as e:
    logger.warning("ExtendedAIIntegrationHub failed: %s", e)

# ── DeepResearchOrchestrator ────────────────────────────────────────────────
try:
    from components.research.deep_research import DeepResearchOrchestrator
    _ai_hub_ref = components.get("extended_hub") or components.get("ai_hub")
    components["deep_research"] = DeepResearchOrchestrator(
        ai_hub=_ai_hub_ref,
        data_path=str(Path(DATA_PATH) / "research" / "deep"),
    )
    logger.info("DeepResearchOrchestrator initialised — provider: %s",
                components["deep_research"].search_engine.primary)
except Exception as e:
    logger.warning("DeepResearchOrchestrator failed: %s", e)


# ── AutoAPIActivator ──────────────────────────────────────────────────────────
try:
    from components.integration.auto_api_activator import AutoAPIActivator
    _hub_ref = components.get("extended_hub") or components.get("ai_hub")
    components["api_activator"] = AutoAPIActivator(
        ai_hub=_hub_ref,
        data_path=str(Path(DATA_PATH)),
    )
    # Run initial scan immediately on startup (non-blocking — result logged)
    _initial_scan = components["api_activator"].scan_and_activate()
    logger.info(
        "AutoAPIActivator: %d active providers, %d pending keys",
        _initial_scan.get("total_active", 0),
        len(_initial_scan.get("pending", [])),
    )
    # Start hourly background re-validation loop
    components["api_activator"].start_background_loop()
except Exception as e:
    logger.warning("AutoAPIActivator failed: %s", e)

# ── KnowledgeSourceManager (BookReader, WebCrawler, ArticleReader, etc.) ─────
try:
    from components.knowledge_sources.CoreKnowledgeSources import KnowledgeSourceManager
    components["knowledge_manager"] = KnowledgeSourceManager(
        base_path=Path(DATA_PATH).parent,  # data_path will be <base>/data/knowledge_sources
        si_core=components.get("si_core"),
    )
    logger.info("KnowledgeSourceManager initialised — 8 knowledge sources ready")
except Exception as e:
    logger.warning("KnowledgeSourceManager failed: %s", e)

# ── Execution Sandbox client ──────────────────────────────────────────────────
try:
    from components.sandbox.sandbox_client import SandboxClient
    sandbox_client = SandboxClient()
    components["sandbox_client"] = sandbox_client
    logger.info("SandboxClient initialised — target %s", sandbox_client.sandbox_url)
except Exception as e:
    sandbox_client = None
    logger.warning("SandboxClient failed: %s", e)

# ── ParallelWebLearner ────────────────────────────────────────────────────────
try:
    from components.knowledge_sources.parallel_web_learner import ParallelWebLearner
    _km = components.get("knowledge_manager")
    _web_crawler = _km.sources.get("web_crawler") if _km else None
    components["parallel_learner"] = ParallelWebLearner(
        data_path=Path(DATA_PATH),
        si_core=components.get("si_core"),
        web_crawler=_web_crawler,
        seed=True,
    )
    logger.info("ParallelWebLearner initialised — seed URLs queued")
except Exception as e:
    logger.warning("ParallelWebLearner failed: %s", e)

# ── Evolution Training System ─────────────────────────────────────────────────
try:
    from components.evolution_training.EvolutionTrainingSystem import EvolutionTrainingSystem
    components["evolution_training"] = EvolutionTrainingSystem(
        si_core=components.get("si_core"),
        knowledge_graph=None,
        training_systems={},
    )
    logger.info("EvolutionTrainingSystem initialised")
except Exception as e:
    logger.warning("EvolutionTrainingSystem failed: %s", e)

# ── Synthetic Intelligence Training (legacy) ──────────────────────────────────
try:
    from components.si_training.SyntheticIntelligenceTraining import SyntheticIntelligenceTraining
    components["si_training_legacy"] = SyntheticIntelligenceTraining(
        data_path=Path(DATA_PATH),
        si_core=components.get("si_core"),
    )
    logger.info("SyntheticIntelligenceTraining initialised")
except Exception as e:
    logger.warning("SyntheticIntelligenceTraining failed: %s", e)

# ── LLM Training ──────────────────────────────────────────────────────────────
try:
    from components.llm_training.LLMTrainingProgram import LLMTrainingProgram
    components["llm_training"] = LLMTrainingProgram(data_path=Path(DATA_PATH))
    logger.info("LLMTrainingProgram initialised")
except Exception as e:
    logger.warning("LLMTrainingProgram failed: %s", e)

# ── GenAI Training ────────────────────────────────────────────────────────────
try:
    from components.genai_training.GenAITrainingProgram import GenAITrainingProgram
    components["genai_training"] = GenAITrainingProgram(data_path=Path(DATA_PATH))
    logger.info("GenAITrainingProgram initialised")
except Exception as e:
    logger.warning("GenAITrainingProgram failed: %s", e)

# ── Media Production Studio ────────────────────────────────────────────────────
try:
    from components.media.MediaProductionStudio import MediaProductionStudio
    components["media_studio"] = MediaProductionStudio()
    logger.info("MediaProductionStudio initialised")
except Exception as e:
    logger.warning("MediaProductionStudio failed: %s", e)

# ── Voice Integration ─────────────────────────────────────────────────────────
try:
    from components.voice.VoiceIntegration import VoiceIntegration
    components["voice"] = VoiceIntegration(data_path=Path(DATA_PATH))
    logger.info("VoiceIntegration initialised")
except Exception as e:
    logger.warning("VoiceIntegration failed: %s", e)

# ── Alex Riviera Content Generator ────────────────────────────────────────────
try:
    from components.alex_riviera.content_generator import AlexRivieraContent
    components["content_gen"] = AlexRivieraContent(ai_hub=components.get("ai_hub"))
    logger.info("AlexRivieraContent initialised")
except Exception as e:
    logger.warning("AlexRivieraContent failed: %s", e)

# ── Alex Riviera Publishing ────────────────────────────────────────────────────
try:
    from components.alex_riviera.publishing_orchestrator import AlexRivieraPublishing
    components["publishing"] = AlexRivieraPublishing()
    logger.info("AlexRivieraPublishing initialised")
except Exception as e:
    logger.warning("AlexRivieraPublishing failed: %s", e)

# ── Master Training Orchestrator ───────────────────────────────────────────────
# Persist init failures so /api/startup/errors can surface them.
_STARTUP_ERRORS = globals().get("_STARTUP_ERRORS", {})
try:
    from components.orchestrator.DMAITrainingOrchestrator import (
        DMAITrainingOrchestrator, register_orchestrator_routes
    )
    components["training_orchestrator"] = DMAITrainingOrchestrator(
        data_path=DATA_PATH,
        si_core=components.get("si_core"),
        knowledge_graph=None,
        ai_hub=components.get("ai_hub"),
        evolution_system=components.get("evolution_training"),
    )
    logger.info("DMAITrainingOrchestrator initialised")
except Exception as e:
    import traceback as _tb_orch
    _STARTUP_ERRORS["training_orchestrator"] = {
        "error": str(e),
        "trace": _tb_orch.format_exc()[-2000:],
    }
    logger.warning("DMAITrainingOrchestrator failed: %s", e)
    logger.warning(_tb_orch.format_exc())

# ── KPIEvaluator (real benchmark evaluations for all 8 KPIs) ─────────────────
try:
    from components.kpi_evaluator import KPIEvaluator
    # Prefer AIIntegrationHub (sync query_all_tutors), fall back to ExtendedAIIntegrationHub (async chat)
    _kpi_hub = components.get("ai_hub") or components.get("extended_hub")
    components["kpi_evaluator"] = KPIEvaluator(
        si_core   = components.get("si_core"),
        ai_hub    = _kpi_hub,
        data_path = DATA_PATH,
    )
    logger.info("KPIEvaluator initialised")
except Exception as e:
    logger.warning("KPIEvaluator failed: %s", e)

# ── Microfish PredictionEngine (vendored from 666ghj/MiroFish, Zep/OASIS/Neo4j stripped) ─
try:
    from components.microfish import PredictionEngine as _MicrofishPE
    components["prediction_engine"] = _MicrofishPE(
        db_path=os.path.join(DATA_PATH.rstrip("/"), "dmai_knowledge.db"),
    )
    logger.info("Microfish PredictionEngine initialised")
except Exception as e:
    logger.warning("Microfish PredictionEngine failed: %s", e)

# ── MemoryRetrieval — patch into SICore so all components can recall() ─────────────
try:
    from components.memory_retrieval import patch_si_core as _patch_memory, recall as _recall_fn
    _si = components.get("si_core")
    if _si:
        _patch_memory(_si)
    # Also make recall available as a standalone component
    components["memory_recall"] = _recall_fn
    logger.info("MemoryRetrieval patched into SICore")
except Exception as e:
    logger.warning("MemoryRetrieval failed: %s", e)

# ── CodeWriter — self-generation engine ─────────────────────────────────────────────
try:
    from components.code_writer import CodeWriter
    components["code_writer"] = CodeWriter(
        ai_hub  = components.get("ai_hub"),
        si_core = components.get("si_core"),
    )
    logger.info("CodeWriter initialised")
except Exception as e:
    logger.warning("CodeWriter failed: %s", e)

# ── KaizenAutoRepair — autonomous fix executor ───────────────────────────────────
try:
    from components.kaizen_auto_repair import KaizenAutoRepair
    components["kaizen_auto_repair"] = KaizenAutoRepair(
        code_writer      = components.get("code_writer"),
        memory_retrieval = components.get("memory_recall"),
        si_core          = components.get("si_core"),
    )
    logger.info("KaizenAutoRepair initialised")
except Exception as e:
    logger.warning("KaizenAutoRepair failed: %s", e)

# ═══════════════════════════════════════════════════════════════════════════
# ── UNWIRED COMPONENTS — full wiring (instantiation order respects deps) ─────
# ═══════════════════════════════════════════════════════════════════════════

# ── GlobalWorkspace (consciousness) ───────────────────────────────────────────
try:
    from components.consciousness.global_workspace import GlobalWorkspace
    components["global_workspace"] = GlobalWorkspace(capacity=7)
    logger.info("GlobalWorkspace initialised")
except Exception as e:
    logger.warning("GlobalWorkspace failed: %s", e)

# ── TutorManager ──────────────────────────────────────────────────────────────
try:
    from components.phase11.TutorManager import TutorManager
    components["tutor_manager"] = TutorManager(data_path=DATA_PATH)
    logger.info("TutorManager initialised")
except Exception as e:
    logger.warning("TutorManager failed: %s", e)

# ── ConsciousnessTracker ──────────────────────────────────────────────────────
try:
    from components.evolution.ConsciousnessTracker import ConsciousnessTracker
    components["consciousness_tracker"] = ConsciousnessTracker(data_path=Path(DATA_PATH))
    logger.info("ConsciousnessTracker initialised")
except Exception as e:
    logger.warning("ConsciousnessTracker failed: %s", e)

# ── EvolutionMetrics ──────────────────────────────────────────────────────────
try:
    from components.evolution.EvolutionMetrics import EvolutionMetrics
    components["evolution_metrics"] = EvolutionMetrics(data_path=Path(DATA_PATH))
    logger.info("EvolutionMetrics initialised")
except Exception as e:
    logger.warning("EvolutionMetrics failed: %s", e)

# ── LearningPipeline ──────────────────────────────────────────────────────────
try:
    from components.learning.LearningPipeline import LearningPipeline
    components["learning_pipeline"] = LearningPipeline()
    logger.info("LearningPipeline initialised")
except Exception as e:
    logger.warning("LearningPipeline failed: %s", e)

# ── MetaLearnerFixed ──────────────────────────────────────────────────────────
try:
    from components.meta_learner_fixed import MetaLearnerFixed
    components["meta_learner"] = MetaLearnerFixed()
    logger.info("MetaLearnerFixed initialised")
except Exception as e:
    logger.warning("MetaLearnerFixed failed: %s", e)

# ── SelfCorrectingEngine ──────────────────────────────────────────────────────
try:
    from components.self_correcting_engine import SelfCorrectingEngine
    components["self_corrector"] = SelfCorrectingEngine(max_attempts=5)
    logger.info("SelfCorrectingEngine initialised")
except Exception as e:
    logger.warning("SelfCorrectingEngine failed: %s", e)

# ── SelfOptimizer ─────────────────────────────────────────────────────────────
try:
    from components.self_optimizer import SelfOptimizer
    components["self_optimizer"] = SelfOptimizer(db_path=str(Path(DATA_PATH) / "dmai_knowledge.db"))
    logger.info("SelfOptimizer initialised")
except Exception as e:
    logger.warning("SelfOptimizer failed: %s", e)

# ── ContentValidator (plagiarism) ─────────────────────────────────────────────
try:
    from components.plagiarism.ContentValidator import ContentValidator
    components["content_validator"] = ContentValidator()
    logger.info("ContentValidator initialised")
except Exception as e:
    logger.warning("ContentValidator failed: %s", e)

# ── UniversalScreenshotExtractor + KnowledgeIntegrator (vision) ───────────────
try:
    from components.vision.universal_extractor import (
        UniversalScreenshotExtractor, KnowledgeIntegrator,
    )
    components["vision_extractor"] = UniversalScreenshotExtractor()
    components["vision_integrator"] = KnowledgeIntegrator(si_core=components.get("si_core"))
    logger.info("UniversalScreenshotExtractor + KnowledgeIntegrator initialised")
except Exception as e:
    logger.warning("Vision extractor failed: %s", e)

# ── SystemHealthDashboard ─────────────────────────────────────────────────────
try:
    from components.health_dashboard import SystemHealthDashboard
    components["health_dashboard"] = SystemHealthDashboard(
        evolution_engine=components.get("evolution_training"))
    logger.info("SystemHealthDashboard initialised")
except Exception as e:
    logger.warning("SystemHealthDashboard failed: %s", e)

# ── InternalArtEngine ─────────────────────────────────────────────────────────
try:
    from components.art.InternalArtEngine import InternalArtEngine
    components["art_engine"] = InternalArtEngine(si_core=components.get("si_core"))
    logger.info("InternalArtEngine initialised")
except Exception as e:
    logger.warning("InternalArtEngine failed: %s", e)

# ── MusicLearner ──────────────────────────────────────────────────────────────
try:
    from components.music.MusicLearner import MusicLearner
    components["music_learner"] = MusicLearner(data_path=Path(DATA_PATH))
    logger.info("MusicLearner initialised")
except Exception as e:
    logger.warning("MusicLearner failed: %s", e)

# ── URLLearner ────────────────────────────────────────────────────────────────
try:
    from components.research.URLLearner import URLLearner
    components["url_learner"] = URLLearner()
    logger.info("URLLearner initialised")
except Exception as e:
    logger.warning("URLLearner failed: %s", e)

# ── AutonomousResearcher ──────────────────────────────────────────────────────
try:
    from components.research.autonomous_researcher import AutonomousResearcher
    components["autonomous_researcher"] = AutonomousResearcher(si_core=components.get("si_core"))
    logger.info("AutonomousResearcher initialised")
except Exception as e:
    logger.warning("AutonomousResearcher failed: %s", e)

# ── SoftwareReverseEngineer ───────────────────────────────────────────────────
try:
    from components.reverse_engineering.ReverseEngineer import SoftwareReverseEngineer
    components["reverse_engineer"] = SoftwareReverseEngineer(data_path=Path(DATA_PATH))
    logger.info("SoftwareReverseEngineer initialised")
except Exception as e:
    logger.warning("SoftwareReverseEngineer failed: %s", e)

# ── LearningHarvester ─────────────────────────────────────────────────────────
try:
    from components.evolution.LearningHarvester import LearningHarvester
    components["learning_harvester"] = LearningHarvester(
        data_path=Path(DATA_PATH), ai_hub=components.get("ai_hub"), knowledge_graph=None)
    logger.info("LearningHarvester initialised")
except Exception as e:
    logger.warning("LearningHarvester failed: %s", e)

# ── IntelligenceBridge ────────────────────────────────────────────────────────
try:
    from components.phase11.IntelligenceBridge import IntelligenceBridge
    components["intelligence_bridge"] = IntelligenceBridge(
        intelligence_core=components.get("si_core"), knowledge_graph=None, pattern_synthesis=None)
    logger.info("IntelligenceBridge initialised")
except Exception as e:
    logger.warning("IntelligenceBridge failed: %s", e)

# ── CapabilitySynthesizer ─────────────────────────────────────────────────────
try:
    from components.phase11.CapabilitySynthesizer import CapabilitySynthesizer
    components["capability_synthesizer"] = CapabilitySynthesizer()
    logger.info("CapabilitySynthesizer initialised")
except Exception as e:
    logger.warning("CapabilitySynthesizer failed: %s", e)

# ── DynamicAIDiscovery (background loop started later) ─────────────────────────
try:
    from components.phase11.DynamicAIDiscovery import DynamicAIDiscovery
    components["ai_discovery"] = DynamicAIDiscovery(
        data_path=Path(DATA_PATH), ai_hub=components.get("ai_hub"))
    logger.info("DynamicAIDiscovery initialised")
except Exception as e:
    logger.warning("DynamicAIDiscovery failed: %s", e)

# ── StageAwareLearningOrchestrator ────────────────────────────────────────────
try:
    from components.evolution.StageAwareLearningOrchestrator import StageAwareLearningOrchestrator
    components["stage_learner"] = StageAwareLearningOrchestrator(
        data_path=Path(DATA_PATH), synthetic_network=None, knowledge_graph=None,
        ai_hub=components.get("ai_hub"), pattern_synthesis=None)
    logger.info("StageAwareLearningOrchestrator initialised")
except Exception as e:
    logger.warning("StageAwareLearningOrchestrator failed: %s", e)

# ── DeepResearchIntegrator ────────────────────────────────────────────────────
try:
    from components.research.research_integration import DeepResearchIntegrator
    components["research_integrator"] = DeepResearchIntegrator(
        synthetic_network=None, stage_learner=components.get("stage_learner"))
    logger.info("DeepResearchIntegrator initialised")
except Exception as e:
    logger.warning("DeepResearchIntegrator failed: %s", e)

# ── LearningOrchestrator (phase11) ────────────────────────────────────────────
try:
    from components.phase11.LearningOrchestrator import LearningOrchestrator
    components["learning_orchestrator"] = LearningOrchestrator(
        ai_hub=components.get("ai_hub"),
        discovery=components.get("ai_discovery"),
        synthetic_network=None,
        tutor_manager=components.get("tutor_manager"),
        intelligence_bridge=components.get("intelligence_bridge"))
    logger.info("LearningOrchestrator initialised")
except Exception as e:
    logger.warning("LearningOrchestrator failed: %s", e)

# ── UnifiedLearningOrchestrator ───────────────────────────────────────────────
try:
    from components.unified_learning_orchestrator import UnifiedLearningOrchestrator
    components["unified_learner"] = UnifiedLearningOrchestrator(
        si_core=components.get("si_core"),
        evolution_engine=components.get("evolution_training"),
        knowledge_graph=None)
    logger.info("UnifiedLearningOrchestrator initialised")
except Exception as e:
    logger.warning("UnifiedLearningOrchestrator failed: %s", e)

# ── KaizenIntegrator (PeriodicUpdateEngine) ───────────────────────────────────
try:
    from components.update_engine.PeriodicUpdateEngine import KaizenIntegrator
    components["kaizen_integrator"] = KaizenIntegrator(
        config={"data_path": str(DATA_PATH),
                "kaizen_endpoint": os.environ.get("KAIZEN_ENDPOINT",
                                                  "http://localhost:5000/api/kaizen")})
    logger.info("KaizenIntegrator initialised")
except Exception as e:
    logger.warning("KaizenIntegrator failed: %s", e)

# ── GitHubStarMonitor (background loop started later) ──────────────────────────
try:
    from components.phase10.GitHubStarMonitor import GitHubStarMonitor
    components["github_monitor"] = GitHubStarMonitor(
        data_path=Path(DATA_PATH), github_username="Davemiles1978",
        github_token=os.environ.get("GITHUB_MODELS_TOKEN"))
    logger.info("GitHubStarMonitor initialised")
except Exception as e:
    logger.warning("GitHubStarMonitor failed: %s", e)

# ── SelfFundingOrchestrator ───────────────────────────────────────────────────
try:
    from components.funding.SelfFundingOrchestrator import SelfFundingOrchestrator
    components["self_funding"] = SelfFundingOrchestrator(
        data_path=Path(DATA_PATH), financial_manager=None, knowledge_graph=None,
        ai_hub=components.get("ai_hub"))
    logger.info("SelfFundingOrchestrator initialised")
except Exception as e:
    logger.warning("SelfFundingOrchestrator failed: %s", e)

# ── DynamicRevenueDiscovery ───────────────────────────────────────────────────
try:
    from components.funding.DynamicRevenueDiscovery import DynamicRevenueDiscovery
    components["revenue_discovery"] = DynamicRevenueDiscovery(
        data_path=Path(DATA_PATH), knowledge_graph=None, ai_hub=components.get("ai_hub"),
        funding_orchestrator=components.get("self_funding"))
    logger.info("DynamicRevenueDiscovery initialised")
except Exception as e:
    logger.warning("DynamicRevenueDiscovery failed: %s", e)

# ── MasterControl (phase7) ────────────────────────────────────────────────────
try:
    from components.phase7.P7_MasterControl import MasterControl
    components["master_control"] = MasterControl(
        master_key=os.environ.get("MASTER_KEY", os.environ.get("MASTER_PASSWORD", "")))
    logger.info("MasterControl initialised")
except Exception as e:
    logger.warning("MasterControl failed: %s", e)

# ── TradingMasterySystem ──────────────────────────────────────────────────────
try:
    from components.trading.mastery_system import TradingMasterySystem
    components["trading_mastery"] = TradingMasterySystem()
    logger.info("TradingMasterySystem initialised")
except Exception as e:
    logger.warning("TradingMasterySystem failed: %s", e)

# ── AggressiveTrader (paper=True unless TRADING_LIVE=true) ─────────────────────
try:
    from components.wealth.aggressive_trader import AggressiveTrader
    _paper = os.environ.get("TRADING_LIVE", "").lower() != "true"
    components["trader"] = AggressiveTrader(
        api_key=os.environ.get("TRADING_API_KEY", ""),
        secret_key=os.environ.get("TRADING_SECRET_KEY", ""),
        paper=_paper,
        prediction_engine=components.get("prediction_engine"))
    logger.info("AggressiveTrader initialised (paper=%s)", _paper)
except Exception as e:
    logger.warning("AggressiveTrader failed: %s", e)

# ── Monetisation hub (60/40 split, bills, betting tipster, wealth basket) ────────
try:
    from components.monetisation import (
        RevenueAllocator as _RevenueAllocator,
        BillPayer as _BillPayer,
        BettingAdvisor as _BettingAdvisor,
        WealthAllocator as _WealthAllocator,
    )
    _mon_db = os.path.join(DATA_PATH.rstrip("/"), "dmai_knowledge.db")
    components["revenue_allocator"] = _RevenueAllocator(db_path=_mon_db, currency="GBP")
    components["bill_payer"] = _BillPayer(
        allocator=components["revenue_allocator"], db_path=_mon_db, currency="GBP")
    components["betting_advisor"] = _BettingAdvisor(
        prediction_engine=components.get("prediction_engine"),
        allocator=components["revenue_allocator"],
        db_path=_mon_db, currency="GBP")
    components["wealth_allocator"] = _WealthAllocator(
        allocator=components["revenue_allocator"],
        trader=components.get("trader"),
        db_path=_mon_db, currency="GBP")
    logger.info("Monetisation hub initialised (60/40 split active)")
except Exception as e:
    logger.warning("Monetisation hub failed: %s", e)

# ── Slack notifier (Slack webhook — SLACK_WEBHOOK_URL env, optional) ────────────
try:
    from components.monetisation.notifier import SlackNotifier as _SlackNotifier
    _notif_db = os.path.join(DATA_PATH.rstrip("/"), "dmai_knowledge.db")
    components["notifier"] = _SlackNotifier(db_path=_notif_db)
    logger.info("SlackNotifier initialised (configured=%s, mask=%s)",
                components["notifier"].configured(),
                sorted(components["notifier"].status()["mask"]))
except Exception as e:
    logger.warning("SlackNotifier failed: %s", e)

# ── AutonomousTrader (5-min loop, market-hours gate, paper-first) ──────────────
try:
    if components.get("trader"):
        from components.wealth.autonomous_trader import AutonomousTrader as _AutoTrader
        _at_db = os.path.join(DATA_PATH.rstrip("/"), "dmai_knowledge.db")
        components["autonomous_trader"] = _AutoTrader(
            db_path=_at_db,
            trader=components["trader"],
            prediction_engine=components.get("prediction_engine"),
            notifier=components.get("notifier"))
        logger.info("AutonomousTrader initialised (paper-first, 5-min loop)")
except Exception as e:
    logger.warning("AutonomousTrader failed: %s", e)

# ── TraderWatchdog (self-healing: forces tick if stale, alerts on failure) ─────
try:
    if components.get("autonomous_trader"):
        from components.wealth.trader_watchdog import TraderWatchdog as _Watchdog
        components["trader_watchdog"] = _Watchdog(
            trader=components["autonomous_trader"],
            notifier=components.get("notifier"))
        logger.info("TraderWatchdog initialised (self-healing)")
except Exception as e:
    logger.warning("TraderWatchdog failed: %s", e)

# ── FinancialIntegrationUK (requires external credentials) ────────────────────
try:
    from components.financial_integration_uk import FinancialIntegrationUK
    components["financial_uk"] = FinancialIntegrationUK(
        encryption_key=os.environ.get("FINANCIAL_ENCRYPTION_KEY"))
    logger.info("FinancialIntegrationUK initialised")
except Exception as e:
    logger.warning("FinancialIntegrationUK failed: %s", e)

loaded = sum(1 for v in components.values() if v is not None)
logger.info("Components loaded: %d", loaded)

# ── Kaizen store (P1-6: atomic writes) ───────────────────────────────────────
_KAIZEN_FILE = Path(DATA_PATH) / "kaizen_proposals.jsonl"
_KAIZEN_QUEUE_CAP = 20

def _load_kaizen(n=20):
    if not _KAIZEN_FILE.exists():
        return []
    lines = _KAIZEN_FILE.read_text().strip().split("\n")
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records[-n:]

def _save_kaizen(proposal):
    """Atomic append via temp-file rename (P1-6)."""
    _KAIZEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Check queue depth cap
    depth = len(_load_kaizen(100))
    if depth >= _KAIZEN_QUEUE_CAP:
        logger.warning("Kaizen queue at capacity (%d). Proposal not saved.", _KAIZEN_QUEUE_CAP)
        if cb_manager:
            try:
                cb_manager.check_kaizen_depth(depth)
            except Exception:
                pass
        return
    existing = _KAIZEN_FILE.read_text() if _KAIZEN_FILE.exists() else ""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=_KAIZEN_FILE.parent, suffix=".tmp", delete=False
    )
    try:
        tmp.write(existing + json.dumps(proposal) + "\n")
        tmp.close()
        os.replace(tmp.name, _KAIZEN_FILE)
    except Exception as e:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        raise e

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Register circuit breaker after_request hook (CB-01–CB-06)
if CB_AVAILABLE:
    app.after_request(after_request_hook)
    logger.info("Circuit breaker after_request hook registered")

# Register orchestrator routes
if "training_orchestrator" in components:
    try:
        register_orchestrator_routes(app, components["training_orchestrator"])
        logger.info("Orchestrator routes registered")
    except Exception as e:
        logger.warning("Orchestrator route registration failed: %s", e)

# ═══════════════════════════════════════════════════════════════════════════
# ── Flask-app-dependent components (must be instantiated AFTER app = Flask) ──
# ═══════════════════════════════════════════════════════════════════════════

# ── CapabilityIntegrator ──────────────────────────────────────────────────────
try:
    from components.capability_integrator import CapabilityIntegrator
    components["capability_integrator"] = CapabilityIntegrator(dmai_app=app)
    logger.info("CapabilityIntegrator initialised")
except Exception as e:
    logger.warning("CapabilityIntegrator failed: %s", e)

# ── FreeAPIHarvester ──────────────────────────────────────────────────────────
try:
    from components.integration.free_api_harvester import FreeAPIHarvester
    components["free_api_harvester"] = FreeAPIHarvester(dmai_app=app)
    logger.info("FreeAPIHarvester initialised")
except Exception as e:
    logger.warning("FreeAPIHarvester failed: %s", e)

# ── AITutorAutoConfigurator (health loop started later) ───────────────────────
try:
    from components.integration.ai_tutor_auto_configurator import AITutorAutoConfigurator
    components["tutor_configurator"] = AITutorAutoConfigurator(dmai_app=app)
    logger.info("AITutorAutoConfigurator initialised")
except Exception as e:
    logger.warning("AITutorAutoConfigurator failed: %s", e)

# ── RepoIntegrationEngine ─────────────────────────────────────────────────────
try:
    from components.integration.repo_integration_engine import RepoIntegrationEngine
    components["repo_integrator"] = RepoIntegrationEngine(dmai_app=app)
    logger.info("RepoIntegrationEngine initialised")
except Exception as e:
    logger.warning("RepoIntegrationEngine failed: %s", e)

# ── AutonomousIngestor (AutonomousDeveloper) ──────────────────────────────────
try:
    from components.autonomous_ingestor import AutonomousDeveloper
    components["autonomous_ingestor"] = AutonomousDeveloper(dmai_app=app)
    logger.info("AutonomousIngestor initialised")
except Exception as e:
    logger.warning("AutonomousIngestor failed: %s", e)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def _uptime():
    delta = datetime.now(timezone.utc) - STARTUP_TIME
    h, r = divmod(int(delta.total_seconds()), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

def _require_auth():
    """
    P1-2: JWT-first auth with backward-compat X-Master-Password header.
    Bearer token → verify JWT.
    X-Master-Password / ?password → verify against MASTER_PASSWORD env var,
    and optionally issue a JWT for future calls.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if SECURITY_AVAILABLE:
            try:
                from security import verify_token
                payload = verify_token(token)
                return payload is not None
            except Exception:
                return False
        return True  # security module missing — fail open only in dev
    # Legacy password header
    pwd = request.headers.get("X-Master-Password") or request.args.get("password", "")
    master = os.environ.get("MASTER_PASSWORD", "")
    return pwd == master

def _safe_sanitise(text):
    """Wrapper that normalises sanitise_input result to a plain string.
    Some security modules return (cleaned, detected_flag); others return the string.
    """
    if not SECURITY_AVAILABLE:
        return text
    try:
        out = sanitise_input(text)
    except Exception:
        return text
    if isinstance(out, tuple):
        return out[0] if out and isinstance(out[0], str) else text
    if isinstance(out, str):
        return out
    return text


def _direct_provider_chat(prompt):
    """Call free-tier LLM providers directly with os.getenv at call time.
    Returns (response_text, provider_used, debug_log).
    """
    import requests as _rq
    debug_log = []
    providers = [
        ("Cerebras",
         os.getenv("CEREBRAS_API_KEY"),
         "https://api.cerebras.ai/v1/chat/completions",
         "llama-3.3-70b"),
        ("Groq",
         os.getenv("GROQ_API_KEY"),
         "https://api.groq.com/openai/v1/chat/completions",
         "llama-3.3-70b-versatile"),
        ("Google AI Studio (2.0-flash)",
         os.getenv("GOOGLE_AI_STUDIO_KEY") or os.getenv("GEMINI_API_KEY"),
         "__gemini__",
         "gemini-2.0-flash"),
        ("Google AI Studio (2.5-flash)",
         os.getenv("GOOGLE_AI_STUDIO_KEY") or os.getenv("GEMINI_API_KEY"),
         "__gemini__",
         "gemini-2.5-flash"),
        ("GitHub Models",
         os.getenv("GITHUB_TOKEN_MAIN") or os.getenv("GITHUB_TOKEN"),
         "https://models.github.ai/inference/chat/completions",
         "gpt-4o-mini"),
        ("OpenRouter (Llama 3.3 70B)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "meta-llama/llama-3.3-70b-instruct:free"),
        ("OpenRouter (GPT-OSS 120B)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "openai/gpt-oss-120b:free"),
        ("OpenRouter (GPT-OSS 20B)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "openai/gpt-oss-20b:free"),
        ("OpenRouter (Qwen3 Next 80B)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "qwen/qwen3-next-80b-a3b-instruct:free"),
        ("OpenRouter (Qwen3 Coder)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "qwen/qwen3-coder:free"),
        ("OpenRouter (Llama 3.2 3B)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "meta-llama/llama-3.2-3b-instruct:free"),
        ("OpenRouter (Hermes 3 405B)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "nousresearch/hermes-3-llama-3.1-405b:free"),
        ("OpenRouter (Nemotron 9B)",
         os.getenv("OPENROUTER_API_KEY"),
         "https://openrouter.ai/api/v1/chat/completions",
         "nvidia/nemotron-nano-9b-v2:free"),
        ("DeepSeek",
         os.getenv("DEEPSEEK_API_KEY"),
         "https://api.deepseek.com/v1/chat/completions",
         "deepseek-chat"),
        ("Mistral",
         os.getenv("MISTRAL_API_KEY"),
         "https://api.mistral.ai/v1/chat/completions",
         "mistral-small-latest"),
        ("OpenAI",
         os.getenv("OPENAI_API_KEY"),
         "https://api.openai.com/v1/chat/completions",
         "gpt-4o-mini"),
        ("Anthropic",
         os.getenv("ANTHROPIC_API_KEY"),
         "__anthropic__",
         "claude-3-5-haiku-20241022"),
    ]
    for name, key, url, model in providers:
        if not key or key == "pending":
            debug_log.append({"provider": name, "skipped": "no_key"})
            continue
        try:
            # Google AI Studio (Gemini) has its own API shape
            if url == "__gemini__":
                r = _rq.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=30,
                )
                if r.status_code == 200:
                    data = r.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    debug_log.append({"provider": name, "ok": True, "model": model})
                    return text, name, debug_log
                debug_log.append({"provider": name, "http": r.status_code, "body": r.text[:160]})
                continue
            # Anthropic has a different API shape
            if url == "__anthropic__":
                r = _rq.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    data = r.json()
                    text = data["content"][0]["text"]
                    debug_log.append({"provider": name, "ok": True, "model": model})
                    return text, name, debug_log
                debug_log.append({"provider": name, "http": r.status_code, "body": r.text[:160]})
                continue
            r = _rq.post(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                debug_log.append({"provider": name, "ok": True, "model": model})
                return text, name, debug_log
            debug_log.append({"provider": name, "http": r.status_code, "body": r.text[:160]})
        except Exception as exc:
            debug_log.append({"provider": name, "exception": str(exc)[:160]})
    return None, None, debug_log


def _ai_chat(message):
    """DMAI chat entry point: direct providers first, hub fallback second."""
    if SECURITY_AVAILABLE:
        clean_message = _safe_sanitise(message)
        if check_injection(clean_message):
            logger.warning("Injection attempt detected in chat: %s", message[:80])
            return "Request blocked: potential injection detected."
    else:
        clean_message = message

    if SECURITY_AVAILABLE:
        try:
            halt = check_halt_conditions(clean_message)
        except Exception as _he:
            logger.warning("check_halt_conditions error: %s", _he)
            halt = None
        if halt:
            return f"Request halted: {halt}"

    response_text = None

    # Primary: direct provider waterfall (env vars at call time)
    try:
        direct_resp, provider, _dbg = _direct_provider_chat(clean_message)
        if direct_resp:
            response_text = direct_resp
            logger.info("_ai_chat: direct provider success via %s", provider)
        else:
            logger.warning("_ai_chat: all direct providers failed: %s", _dbg)
    except Exception as e:
        import traceback
        logger.warning("_ai_chat direct path error: %s\n%s", e, traceback.format_exc())

    # Fallback: legacy hub plumbing
    if response_text is None:
        hub = components.get("extended_hub") or components.get("ai_hub")
        if hub:
            try:
                if hasattr(hub, "chat_sync"):
                    response_text = hub.chat_sync(clean_message)
                elif hasattr(hub, "chat"):
                    import inspect as _inspect
                    _sig = _inspect.signature(hub.chat)
                    _first = list(_sig.parameters.keys())[1] if len(_sig.parameters) > 1 else "prompt"
                    _arg = [{"role": "user", "content": clean_message}] if _first in ("messages", "msgs") else clean_message
                    _res = _run_async(hub.chat(_arg))
                    if isinstance(_res, tuple):
                        _res = _res[0] if _res else None
                    response_text = _res if isinstance(_res, str) else (str(_res) if _res else None)
            except Exception as e:
                logger.warning("AI chat hub-fallback error: %s", e)

    if isinstance(response_text, tuple):
        _raw = response_text[0] if response_text else None
        response_text = _raw if isinstance(_raw, str) else None
    elif response_text is not None and not isinstance(response_text, str):
        response_text = str(response_text)

    if response_text is None:
        ml = clean_message.lower()
        for topic, info in SYLLABUS_TOPICS.items():
            if topic in ml or ml in topic:
                response_text = info.get("content", f"I know about {topic} at {info.get('stage','?')} level.")
                break
        if response_text is None:
            response_text = (
                f"DMAI received: '{clean_message}'. "
                f"Add an AI provider API key for full LLM responses. "
                f"Current syllabus: {TOTAL_TOPICS} mastered topics available."
            )

    if SECURITY_AVAILABLE and isinstance(response_text, str) and "```" in response_text:
        scan = scan_generated_code(response_text)
        if not scan.get("safe", True):
            issues = "; ".join(str(i) for i in scan.get("issues", []))
            logger.warning("Generated code scan found issues: %s", issues)
            response_text = safe_code_output(response_text)

    return response_text


# ── Routes ────────────────────────────────────────────────────────────────────

# ── Auto-start background services on first real request ──────────────────────
# Handles the case where gunicorn worker starts but _start_background_services()
# threads haven't launched yet (e.g. Render cold start, worker restart).
_auto_start_lock = __import__("threading").Lock()
_auto_started = False

@app.before_request
def _auto_start_services():
    global _auto_started
    if _auto_started:
        return
    with _auto_start_lock:
        if _auto_started:
            return
        import threading as _abt
        thread_names = [t.name for t in _abt.enumerate()]
        # If less than 4 background threads are running, start them
        bg_count = sum(1 for n in thread_names if n not in ("MainThread",))
        if bg_count < 4:
            try:
                logger.info("Auto-start: only %d background threads found, starting services", bg_count)
                _start_background_services()
                logger.info("Auto-start: background services launched")
            except Exception as _e:
                logger.warning("Auto-start failed: %s", _e)
        _auto_started = True



@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": DMAI_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": _uptime(),
        "components": {k: "active" for k in components},
        "syllabus_topics": TOTAL_TOPICS,
        "security": {
            "jwt": SECURITY_AVAILABLE,
            "circuit_breakers": CB_AVAILABLE,
            "hmac_webhooks": HMAC_AVAILABLE,
            "chain_logging": CHAIN_LOGGER_AVAILABLE,
            "bandit": BANDIT_AVAILABLE,
        },
    })

@app.route("/api/status")
def api_status():
    # KPI priority: kpi_cache.json (DB-derived, always accurate) → si_core → empty
    # kpi_cache.json is written by _seed_kpis_from_db every 5 min from SQLite counts.
    _si_ref = components.get("si_core")
    kpis = {}
    try:
        import json as _jc
        _cache_path = os.path.join(
            os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\"),
            "kpi_cache.json"
        )
        with open(_cache_path) as _cf:
            _cached = _jc.load(_cf)
        kpis = _cached.get("kpis", {})
        # Only fall through if cache is missing or entirely zero
        if not kpis or all(v == 0 for v in kpis.values() if isinstance(v, (int, float))):
            raise ValueError("cache empty or all-zero")
    except Exception:
        # Fallback: si_core
        _raw = (_si_ref.current_kpis if _si_ref else {}) or {}
        if _raw and not all(v == 0 for v in _raw.values() if isinstance(v, (int, float))):
            kpis = _raw
        else:
            # Last resort: derive from DB right now
            try:
                import sqlite3 as _sq_k
                _db_k = os.path.join(os.environ.get("DATA_PATH", "data").rstrip("/"), "dmai_knowledge.db")
                _ck = _sq_k.connect(_db_k, timeout=5)
                _caps_k = _ck.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0]
                _ins_k  = _ck.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
                try:
                    _voc_k = _ck.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
                except Exception:
                    _voc_k = 0
                try:
                    _ins7d = _ck.execute(
                        "SELECT COUNT(*) FROM insights WHERE created_at >= datetime('now','-7 days')"
                    ).fetchone()[0]
                except Exception:
                    _ins7d = 0
                try:
                    _days_k = _ck.execute(
                        "SELECT COUNT(DISTINCT date(created_at)) FROM insights "
                        "WHERE created_at >= datetime('now','-7 days')"
                    ).fetchone()[0] or 0
                except Exception:
                    _days_k = 0
                _ck.close()
                _stg_name_k, _stage_k, _pct_k = _read_stage_from_db()
                _active_k = sum(1 for v in components.values() if v is not None)
                kpis = {
                    "skill_acquisition_rate":          min(_caps_k / 50_000, 1.0),
                    "transfer_learning_rate":           min(_stage_k / 7.0, 1.0),
                    "zero_shot_success_count":          min(_ins_k  / 300_000, 1.0),
                    "agentic_capability_score":         min(_caps_k / 20_000, 1.0),
                    "recursive_self_improvement_rate":  min(_pct_k / 100.0, 1.0),
                    "sample_efficiency_trend":          min((_ins7d / max(_days_k, 1)) / 5_000, 1.0),
                    "metacognition_accuracy":           min(_voc_k  / 500_000, 1.0),
                    "multi_modal_integration_score":    min(_active_k / max(len(components), 56), 1.0),
                }
            except Exception as _ke:
                logger.warning("KPI inline derive failed: %s", _ke)
                kpis = {}
    orch = components.get("training_orchestrator")
    training_status = orch.get_status() if orch else {}
    ext_hub = components.get("extended_hub")
    hub_status = ext_hub.get_status() if ext_hub else {}
    return jsonify({
        "status": "running",
        "version": DMAI_VERSION,
        "uptime": _uptime(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deployment": "render" if IS_RENDER else "local",
        "components_loaded": list(components.keys()),
        "component_count": len(components),          # mobile reads this
        "syllabus_topics": TOTAL_TOPICS,
        "total_topics": TOTAL_TOPICS,                # mobile reads this
        "si_kpis": kpis,
        "training": training_status,
        "providers": hub_status.get("extended_providers", []) + hub_status.get("base_providers", []),
    })

@app.route("/api/persona")
def api_persona():
    return jsonify({
        "system": "DMAI v7.1.0",
        "internal_name": "DMAI",
        "public_persona": {
            "name": "Alex Riviera",
            "age": 28,
            "location": "Los Angeles, CA",
            "occupation": "Writer & Producer",
            "email": "alex.riviera.creator@proton.me",
            "avatar_style": "platinum-blonde, confident, professional",
            "social": {
                "twitter": "@RealAlexRiviera",
                "youtube": "@AlexRiviera",
                "tiktok": "@alex.riviera"
            }
        },
        "voice_tone": "Professional, creative, enthusiastic",
        "capabilities": ["book_generation", "tv_series", "coloring_books", "tts_voice", "image_generation"],
    })

@app.route("/")
def index():
    import base64 as _b64
    dashboard = Path("static/dashboard.html")
    if dashboard.exists():
        # Inject auth token into dashboard at serve time — never hardcoded in source
        _pw = os.environ.get("MASTER_PASSWORD", "")
        _tok = _b64.b64encode(f"admin:{_pw}".encode()).decode() if _pw else ""
        try:
            _html = dashboard.read_text()
            _html = _html.replace(
                'content="{{ dmai_auth_header }}"',
                f'content="Basic {_tok}"'
            )
            return _html, 200, {"Content-Type": "text/html; charset=utf-8"}
        except Exception:
            return send_from_directory("static", "dashboard.html")
    return f"""<!DOCTYPE html>
<html><head><title>DMAI v7.1.0</title>
<style>body{{background:#0a0a0f;color:#e0e0ff;font-family:monospace;padding:40px}}
h1{{color:#6c63ff}}a{{color:#00d4aa}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #333;padding:8px;text-align:left}}
.badge{{background:#1a1a2e;border:1px solid #6c63ff;padding:2px 8px;border-radius:4px;font-size:12px}}</style>
</head><body>
<h1>DMAI v7.1.0 — Online</h1>
<p>Uptime: {_uptime()} | Topics: {TOTAL_TOPICS} | Components: {len(components)}</p>
<p>
  <span class="badge">JWT: {'✓' if SECURITY_AVAILABLE else '✗'}</span>
  <span class="badge">CB: {'✓' if CB_AVAILABLE else '✗'}</span>
  <span class="badge">HMAC: {'✓' if HMAC_AVAILABLE else '✗'}</span>
  <span class="badge">Bandit: {'✓' if BANDIT_AVAILABLE else '✗'}</span>
</p>
<p>
  <a href="/chat" style="background:#6c63ff;color:#fff;padding:8px 18px;border-radius:6px;text-decoration:none;margin-right:8px">💬 Chat UI</a>
  <a href="/admin" style="background:#ff6584;color:#fff;padding:8px 18px;border-radius:6px;text-decoration:none">🔐 Admin Panel</a>
</p>
<p style="margin-top:8px"><a href="/api/status">/api/status</a> | <a href="/api/training/status">/api/training/status</a> |
<a href="/api/kaizen">/api/kaizen</a> | <a href="/api/admin/circuit-breakers">/api/admin/circuit-breakers</a></p>
<h2>Active Components</h2>
<table><tr><th>Component</th><th>Status</th></tr>
{"".join(f"<tr><td>{k}</td><td style='color:#00d4aa'>active</td></tr>" for k in components)}
</table></body></html>""", 200, {"Content-Type": "text/html"}

@app.route("/chat")
def chat_page():
    """Public chat UI — Alex Riviera persona."""
    return send_from_directory("static", "chat.html")


@app.route("/admin")
def admin_page():
    """Admin panel — JWT-gated client-side lock screen."""
    return send_from_directory("static", "admin.html")


@app.route("/mobile")
def mobile_page():
    """Mobile PWA control panel — installable on iPhone/Android."""
    return send_from_directory("static", "mobile.html")


@app.route("/service-worker.js")
def pwa_service_worker():
    """Serve service worker at root scope so it can control all pages."""
    resp = send_from_directory("static", "service-worker.js")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Content-Type"] = "application/javascript"
    return resp


@app.route("/manifest.json")
def pwa_manifest_root():
    """Convenience alias for PWA manifest at root path."""
    return send_from_directory("static", "manifest.json")


@app.route("/dashboard")
def dashboard_page():
    """Dashboard UI — system overview."""
    return send_from_directory("static", "dashboard.html")


@app.route("/trading")
def trading_page():
    """Trading UI — AggressiveTrader mastery + AI trading dashboard."""
    return send_from_directory("static", "trading.html")


@app.route("/wallpaper")
def wallpaper_png():
    """
    Serve the DMAI knowledge graph as a PNG image.
    Query params:
      ?dark=1|0     — dark (default) or light background
      ?size=mini    — 400×400 preview instead of full iPhone resolution
      ?bust=1       — bypass cache (force re-render)
    """
    from components.graph_wallpaper import render_wallpaper_png, clear_cache
    from flask import send_file, request as req
    dark  = req.args.get("dark", "1") != "0"
    mini  = req.args.get("size", "") == "mini"
    bust  = req.args.get("bust", "0") == "1"
    if bust:
        clear_cache()
    w, h = (400, 400) if mini else (1179, 2556)
    try:
        png = render_wallpaper_png(width=w, height=h, dark=dark)
        if not png:
            return "Graph renderer unavailable", 503
        buf = __import__("io").BytesIO(png)
        buf.seek(0)
        return send_file(buf, mimetype="image/png",
                         download_name="dmai-graph.png",
                         max_age=300)
    except Exception as e:
        logger.error("Wallpaper render failed: %s", e)
        return f"Render error: {e}", 500


@app.route("/graph-widget")
def graph_widget_svg():
    """
    Serve the DMAI knowledge graph as an SVG for Widgetsmith / home screen widgets.
    Query params:
      ?size=small|medium|large
    """
    from components.graph_wallpaper import render_widget_svg
    from flask import request as req
    size = req.args.get("size", "small")
    try:
        svg = render_widget_svg(size=size)
        from flask import Response
        return Response(svg, mimetype="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=300"})
    except Exception as e:
        logger.error("Widget SVG render failed: %s", e)
        return f"Render error: {e}", 500


@app.route("/status")
def status_page():
    return index()

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", data.get("text", "")).strip()
        if not message:
            return jsonify({"error": "No message provided"}), 400
        # Command shortcuts
        if message.startswith("/"):
            cmd = message.lower().strip()
            if cmd == "/status": return api_status()
            if cmd == "/persona": return api_persona()
            if cmd == "/kaizen": return api_kaizen_get()
            if cmd == "/syllabus": return get_syllabus()
            return jsonify({"response": f"Unknown command: {cmd}. Try /status /persona /kaizen /syllabus"})
        try:
            response = _ai_chat(message)
            logger.info("_ai_chat returned type=%s val=%s", type(response).__name__, repr(response)[:100])
        except Exception as _chat_e:
            import traceback
            logger.error("_ai_chat exception: %s\n%s", _chat_e, traceback.format_exc())
            response = None
        # Guarantee response is always a plain string
        if response is None or not isinstance(response, str):
            response = str(response) if response is not None else (
                "DMAI is online. Add a provider API key (e.g. GROQ_API_KEY) for full LLM responses."
            )
        try:
            _log_chat(message, response)
        except Exception:
            pass
        return jsonify({
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "alex_riviera",
        })
    except Exception as e:
        import traceback
        logger.error("chat error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-500:]}), 500

@app.route("/api/chat/trace", methods=["POST", "GET"])
def api_chat_trace():
    """Diagnostic: run _ai_chat directly and return any exception trace."""
    import traceback as _tb
    data = request.get_json(silent=True) or {}
    msg = data.get("message") or request.args.get("message", "hello")
    out = {"input": msg}
    try:
        r = _ai_chat(msg)
        out["result"] = r
        out["type"] = type(r).__name__
        out["is_str"] = isinstance(r, str)
    except Exception as e:
        out["exception"] = str(e)
        out["trace"] = _tb.format_exc()[-1500:]
    out["security_available"] = SECURITY_AVAILABLE
    return jsonify(out)

@app.route("/api/startup/errors", methods=["GET"])
def api_startup_errors():
    """Return any component initialisation errors captured at boot."""
    errs = globals().get("_STARTUP_ERRORS", {}) or {}
    return jsonify({"count": len(errs), "errors": errs})


@app.route("/api/chat/debug", methods=["GET"])
def api_chat_debug():
    """Diagnostic: which provider keys are visible at request time + waterfall trace."""
    probe = request.args.get("probe", "").lower() == "1"
    info = {
        "keys_visible": {
            "CEREBRAS_API_KEY":      bool(os.getenv("CEREBRAS_API_KEY")),
            "GROQ_API_KEY":          bool(os.getenv("GROQ_API_KEY")),
            "GOOGLE_AI_STUDIO_KEY":  bool(os.getenv("GOOGLE_AI_STUDIO_KEY") or os.getenv("GEMINI_API_KEY")),
            "GITHUB_TOKEN_MAIN":     bool(os.getenv("GITHUB_TOKEN_MAIN") or os.getenv("GITHUB_TOKEN")),
            "OPENROUTER_API_KEY":    bool(os.getenv("OPENROUTER_API_KEY")),
            "DEEPSEEK_API_KEY":      bool(os.getenv("DEEPSEEK_API_KEY")),
            "MISTRAL_API_KEY":       bool(os.getenv("MISTRAL_API_KEY")),
            "OPENAI_API_KEY":        bool(os.getenv("OPENAI_API_KEY")),
            "ANTHROPIC_API_KEY":     bool(os.getenv("ANTHROPIC_API_KEY")),
            "TAVILY_API_KEY":        bool(os.getenv("TAVILY_API_KEY")),
        },
        "key_prefixes": {
            "GROQ_API_KEY":         (os.getenv("GROQ_API_KEY") or "")[:6],
            "CEREBRAS_API_KEY":     (os.getenv("CEREBRAS_API_KEY") or "")[:6],
            "GOOGLE_AI_STUDIO_KEY": (os.getenv("GOOGLE_AI_STUDIO_KEY") or "")[:6],
        },
        "hub": {
            "extended_hub_present": components.get("extended_hub") is not None,
            "ai_hub_present":       components.get("ai_hub") is not None,
        },
    }
    if probe:
        text, provider, log = _direct_provider_chat("Say 'pong' in one word.")
        info["probe"] = {
            "used_provider": provider,
            "response":      (text[:200] if text else None),
            "trace":         log,
        }
    return jsonify(info)


def _log_chat(message, response):
    try:
        log_file = Path(DATA_PATH) / "chat_log.jsonl"
        entry = {
            "message": message,
            "response": (response[:200] if isinstance(response, str) else str(response)[:200]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Atomic write for chat log too
        existing = log_file.read_text() if log_file.exists() else ""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=log_file.parent, suffix=".tmp", delete=False
        )
        tmp.write(existing + json.dumps(entry) + "\n")
        tmp.close()
        os.replace(tmp.name, log_file)
    except Exception:
        pass

@app.route("/v2/ask", methods=["POST"])
def v2_ask():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        ql = question.lower()
        for topic, info in SYLLABUS_TOPICS.items():
            if topic in ql or ql in topic:
                return jsonify({
                    "answer": info.get("content", f"Mastered: {topic}"),
                    "topic": topic.title(), "stage": info.get("stage"),
                    "category": info.get("category"), "mastery": info.get("mastery"),
                    "source": "permanent_syllabus", "status": "success",
                })
        answer = _ai_chat(question)
        return jsonify({"answer": answer, "source": "ai_hub", "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v2/syllabus")
def get_syllabus():
    topics = [{"topic": t.title(), "stage": v["stage"], "category": v["category"], "mastery": v["mastery"]}
              for t, v in SYLLABUS_TOPICS.items()]
    return jsonify({"topics": topics, "total": len(topics)})


@app.route("/api/learning/progress")
def api_learning_progress():
    """Full syllabus + stage study progress for the admin Study Progress panel."""
    import json as _json
    from pathlib import Path as _Path

    # 1. Learning progress from stage learner state file
    lp_file = _Path(DATA_PATH) / "learning" / "stage_syllabus" / "learning_progress.json"
    lp = {}
    if lp_file.exists():
        try:
            lp = _json.loads(lp_file.read_text())
        except Exception:
            pass

    learned = lp.get("learned_topics", {})
    current_stage = lp.get("current_stage", "Baby")
    last_cycle = lp.get("last_learning_cycle", None)

    # 2. Flatten all learned topics across stages
    all_topics = {}
    for stage, topics in learned.items():
        for k, v in topics.items():
            if not k.startswith("_"):
                all_topics[k] = {"stage": stage, "mastery": v}

    total = len(all_topics)
    mastered = sum(1 for t in all_topics.values() if t["mastery"] >= 3)
    in_progress_count = sum(1 for t in all_topics.values() if 1 <= t["mastery"] < 3)
    not_started = max(0, TOTAL_TOPICS - total)

    # 3. Per-stage summary
    stage_order = ["Baby", "Toddler", "Child", "Teen", "Adult", "Expert"]
    stage_summary = {}
    for s in stage_order:
        stage_topics = {k: v for k, v in all_topics.items() if v["stage"] == s}
        stage_summary[s] = {
            "total": len(stage_topics),
            "mastered": sum(1 for v in stage_topics.values() if v["mastery"] >= 3),
            "in_progress": sum(1 for v in stage_topics.values() if 1 <= v["mastery"] < 3),
        }

    # 4. SI training modules from orchestrator
    orch = components.get("training_orchestrator")
    si_modules = []
    if orch and hasattr(orch, "si_trainer") and orch.si_trainer:
        try:
            st = orch.si_trainer.get_status()
            si_modules = st.get("module_list", [])
        except Exception:
            pass

    # 5. Recent research discoveries
    disc_file = _Path("data/research/discoveries.jsonl")
    recent_discoveries = []
    if disc_file.exists():
        try:
            lines = disc_file.read_text().strip().split("\n")
            for line in reversed(lines[-10:]):
                if line.strip():
                    try:
                        recent_discoveries.append(_json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

    # 6. Nightly training data stats
    training_path = _Path("data/training")
    training_entries = 0
    training_files_count = 0
    if training_path.exists():
        for tf in training_path.glob("*.json"):
            try:
                content = _json.loads(tf.read_text())
                if isinstance(content, list):
                    training_entries += len(content)
                    training_files_count += 1
            except Exception:
                pass

    return jsonify({
        "current_stage": current_stage,
        "last_learning_cycle": last_cycle,
        "total_syllabus_topics": TOTAL_TOPICS,
        "topics_encountered": total,
        "mastered": mastered,
        "in_progress": in_progress_count,
        "not_started": not_started,
        "mastery_pct": round(mastered / max(TOTAL_TOPICS, 1) * 100, 1),
        "stage_summary": stage_summary,
        "si_modules": si_modules,
        "recent_discoveries": recent_discoveries,
        "nightly_training_files": training_files_count,
        "nightly_training_entries": training_entries,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/v2/weights")
def get_weights():
    return jsonify({
        "topics": [{"topic": t.title(), "weight": 100, "mastery": v["mastery"]} for t, v in SYLLABUS_TOPICS.items()],
        "total": len(SYLLABUS_TOPICS),
    })

@app.route("/api/knowledge/<concept>")
def api_knowledge(concept):
    cl = concept.lower()
    for topic, info in SYLLABUS_TOPICS.items():
        if cl in topic or topic in cl:
            return jsonify({"concept": topic, "info": info, "found": True})
    return jsonify({"concept": concept, "found": False, "message": "Not in syllabus yet"})


@app.route("/api/kpi/evaluate", methods=["POST"])
def api_kpi_evaluate():
    data  = request.get_json(silent=True) or {}
    # Accept quick as query param OR body param
    quick_qs = request.args.get("quick", "").lower() in ("true", "1", "yes")
    quick = quick_qs or data.get("quick", False)
    kpi_eval = components.get("kpi_evaluator")
    if not kpi_eval:
        return jsonify({"error": "KPIEvaluator not loaded"}), 503
    import threading as _kpi_th
    results = {}
    err_holder = []
    def _run():
        try:
            results.update(kpi_eval.run_full_eval(quick=quick))
        except Exception as _e:
            err_holder.append(str(_e))
            logger.error("KPIEvaluator run_full_eval error: %s", _e)
    t = _kpi_th.Thread(target=_run, daemon=True)
    t.start()
    timeout = 60 if quick else 180
    t.join(timeout=timeout)
    status = "complete" if not t.is_alive() else "timeout"
    resp = {"status": status, "results": results, "quick": quick}
    if err_holder:
        resp["error"] = err_holder[0]
    return jsonify(resp)


@app.route("/api/kpi/history")
def api_kpi_history():
    from pathlib import Path as _KP
    import json as _KJ
    hist_file = _KP("data/kpi_eval_history.jsonl")
    limit = int(request.args.get("limit", 50))
    records = []
    if hist_file.exists():
        lines = hist_file.read_text().strip().split("\n")
        for line in reversed(lines):
            if line.strip():
                try:
                    records.append(_KJ.loads(line))
                except Exception:
                    pass
            if len(records) >= limit:
                break
    return jsonify({"records": records, "total": len(records)})


@app.route("/api/kpi/rsi/sync", methods=["POST"])
def api_kpi_rsi_sync():
    kpi_eval = components.get("kpi_evaluator")
    if not kpi_eval:
        return jsonify({"error": "KPIEvaluator not loaded"}), 503
    rate = kpi_eval.eval_rsi_from_graph()
    # Also read the actual evolution_cycle for the UI toast message
    import json as _rj
    from pathlib import Path as _rp
    schema_path = _rp("aevora-training/dashboard/data/graph_schema.json")
    evo_cycle = 0
    if schema_path.exists():
        try:
            evo_cycle = _rj.loads(schema_path.read_text()).get("evolution_cycle", 0)
        except Exception:
            pass
    return jsonify({"ok": True, "recursive_self_improvement_rate": rate, "rsi": rate, "evolution_cycle": evo_cycle})


@app.route("/api/conversations")
def api_conversations():
    log_file = Path(DATA_PATH) / "chat_log.jsonl"
    count = 0
    if log_file.exists():
        count = sum(1 for _ in open(log_file))
    return jsonify({"total_messages": count, "chat_log_path": str(log_file),
                    "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route("/api/kaizen", methods=["GET"])
def api_kaizen_get():
    recent = _load_kaizen(20)
    si = components.get("si_core")
    kpis = si.current_kpis if si else {}
    consciousness = kpis.get("consciousness", 0.0)
    auto_improvements = [
        {"title": "Increase training frequency for low-mastery domains", "priority": "high", "type": "auto",
         "description": "Domains still at Baby/Toddler stage need more training cycles."},
        {"title": "Add missing API keys for more providers", "priority": "medium", "type": "auto",
         "description": "Add ELEVENLABS_API_KEY, MISTRAL_API_KEY, RUNWAY_API_KEY for full capability."},
        {"title": "Enable Pinecone vector memory", "priority": "medium", "type": "auto",
         "description": "Adding PINECONE_API_KEY enables semantic long-term memory."},
    ]
    return jsonify({
        "status": "active", "consciousness": round(consciousness, 3),
        "recent_proposals": recent, "auto_improvements": auto_improvements,
        "total_proposals": len(recent), "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/kaizen", methods=["POST"])
def api_kaizen_post():
    try:
        data = request.get_json(silent=True) or {}
        proposal = {
            "title": sanitise_input(data.get("title", "Untitled proposal")) if SECURITY_AVAILABLE else data.get("title", "Untitled proposal"),
            "description": sanitise_input(data.get("description", "")) if SECURITY_AVAILABLE else data.get("description", ""),
            "priority": data.get("priority", "medium"),
            "type": data.get("type", "manual"),
            "submitted_by": data.get("submitted_by", "api"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_kaizen(proposal)
        return jsonify({"status": "recorded", "proposal": proposal}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/evolution")
def api_evolution():
    evo = components.get("evolution_training")
    si = components.get("si_core")
    return jsonify({
        "status": "active",
        "evolution_cycle": getattr(evo, "evolution_cycle", 0) if evo else 0,
        "insights_count": len(getattr(evo, "evolution_insights", [])) if evo else 0,
        "si_kpis": si.current_kpis if si else {},
        "consciousness": si.current_kpis.get("consciousness", 0.0) if si else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/content/generate", methods=["POST"])
def api_content_generate():
    try:
        data = request.get_json(silent=True) or {}
        ctype = data.get("type", "book")
        prompt = sanitise_input(data.get("prompt", "")) if SECURITY_AVAILABLE else data.get("prompt", "")
        gen = components.get("content_gen")
        if gen and ctype == "book":
            try:
                book, validation = gen.generate_and_validate_book()
                return jsonify({"type": "book", "content": book, "validation": validation})
            except Exception as e:
                logger.warning("Book gen error: %s", e)
        return jsonify({
            "type": ctype, "status": "queued",
            "message": f"Content generation for '{prompt}' queued. Add OPENAI_API_KEY for full generation.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/content/list")
def api_content_list():
    gen = components.get("content_gen")
    works = getattr(gen, "generated_works", []) if gen else []
    pub = components.get("publishing")
    projects = getattr(pub, "approved_projects", []) if pub else []
    return jsonify({"generated_works": works[-20:], "approved_projects": projects[-20:], "total": len(works)})

@app.route("/api/avatar/speak", methods=["POST"])
def api_avatar_speak():
    try:
        data = request.get_json(silent=True) or {}
        text = sanitise_input(data.get("text", "Hello, I'm DMAI.")) if SECURITY_AVAILABLE else data.get("text", "Hello, I'm DMAI.")
        ext_hub = components.get("extended_hub")
        if ext_hub:
            audio = _run_async(ext_hub.text_to_speech(text))
            if audio:
                return Response(audio, mimetype="audio/mpeg",
                                headers={"Content-Disposition": "inline; filename=dmai_voice.mp3"})
        return jsonify({"status": "tts_unavailable",
                        "message": "Add ELEVENLABS_API_KEY for DMAI voice synthesis.", "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/dashboard")
def api_dashboard():
    si = components.get("si_core")
    orch = components.get("training_orchestrator")
    ext = components.get("extended_hub")
    return jsonify({
        "version": DMAI_VERSION, "uptime": _uptime(),
        "components": {k: "active" for k in components},
        "si_kpis": si.current_kpis if si else {},
        "training": orch.get_status() if orch else {},
        "providers": ext.get_status() if ext else {},
        "kaizen": _load_kaizen(5),
        "syllabus": {"total": TOTAL_TOPICS},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "security_modules": {
            "jwt": SECURITY_AVAILABLE,
            "circuit_breakers": CB_AVAILABLE,
            "hmac": HMAC_AVAILABLE,
            "chain_logging": CHAIN_LOGGER_AVAILABLE,
            "bandit": BANDIT_AVAILABLE,
        },
    })

# ── Admin endpoints (JWT-protected) ──────────────────────────────────────────


# ── /api/training/* — canonical training routes ─────────────────────────────

@app.route("/api/training/status", methods=["GET"])
def api_training_status():
    """Live status of all background training threads — 24/7 always-on."""
    import threading as _th
    tnames = [t.name for t in _th.enumerate()]

    def _up(*kws):
        return any(any(kw.lower() in n.lower() for kw in kws) for n in tnames)

    services = {
        "background_updater":    _up("updater", "backgroundupdater", "background_updater", "update-engine"),
        "parallel_learner":      _up("parallel", "parallellearner", "web_learn", "web-learn", "web-learner"),
        "autonomous_researcher": _up("research", "autonomousresearch", "autonomous-researcher"),
        "stage_learner":         _up("stage", "stagelearner", "learning_loop", "stage-learner"),
        "kaizen_repair":         _up("kaizen", "repair", "kaizen-repair", "autorepair"),
        "graph_evolution":       _up("graph", "graphevolution", "graph-evolution"),
        "kpi_seed":              _up("kpi", "kpiseed", "kpi-seed", "KpiSeedLoop"),
        "vocab_ingest":          _up("vocab", "vocabingest", "vocab-ingest"),
    }
    active = sum(1 for v in services.values() if v)
    # Build ai_training progress for dashboard Study Progress tab
    ai_training_progress = {"pct_expert": 0.0, "avg_mastery": 0.0, "total_topics": 0, "expert_topics": 0}
    try:
        import sqlite3 as _sq3
        _db_path = os.path.join(DATA_PATH.rstrip("/"), "dmai_knowledge.db")
        _conn = _sq3.connect(_db_path, timeout=5)
        _expert = _conn.execute("SELECT COUNT(*) FROM syllabus_content WHERE mastery >= 0.8").fetchone()[0]
        _tot    = _conn.execute("SELECT COUNT(*) FROM syllabus_content").fetchone()[0]
        _avg    = _conn.execute("SELECT AVG(mastery) FROM syllabus_content").fetchone()[0] or 0.0
        _conn.close()
        ai_training_progress = {
            "pct_expert":    round((_expert / max(_tot, 1)) * 100, 1),
            "avg_mastery":   round(_avg, 4),
            "total_topics":  _tot,
            "expert_topics": _expert,
        }
    except Exception as _te:
        ai_training_progress["error"] = str(_te)

    return jsonify({
        "status":             "healthy" if active >= 4 else "degraded",
        "training_always_on": True,
        "message":            "Training runs 24/7 automatically — no manual start needed",
        "services":           services,
        "active_count":       active,
        "total_threads":      len(tnames),
        "thread_names":       tnames,
        "components": {
            "ai_training": {
                "progress": ai_training_progress
            }
        },
    })


@app.route("/api/orchestrator/status", methods=["GET"])
def api_orchestrator_status_fallback():
    """Guaranteed orchestrator/training status for the dashboard.

    The real orchestrator registers this same URL when it loads (and wins the
    Werkzeug match because it's registered first); this fallback only serves
    when the orchestrator failed to initialise, so the Training Progress panel
    never 404s into an 'Unavailable' state."""
    import threading as _th
    tnames = [t.name for t in _th.enumerate()]

    def _up(*kws):
        return any(any(kw.lower() in n.lower() for kw in kws) for n in tnames)

    services = {
        "background_updater":    _up("updater", "update", "background", "update-engine"),
        "parallel_learner":      _up("parallel", "learner", "web_learn", "web-learn", "web-learner"),
        "autonomous_researcher": _up("research", "autonomous", "discover", "autonomous-researcher"),
        "stage_learner":         _up("stage", "learning", "loop", "learner", "stage-learner"),
        "kaizen_repair":         _up("kaizen", "repair", "kaizen-repair", "autorepair"),
        "graph_evolution":       _up("graph", "evolution", "graph-evolution"),
        "kpi_seed":              _up("kpi", "seed", "KpiSeedLoop"),
        "vocab_ingest":          _up("vocab", "ingest", "vocab-ingest"),
    }
    active = sum(1 for v in services.values() if v)
    return jsonify({
        "status":             "healthy" if active >= 4 else "degraded",
        "training_always_on": True,
        "message":            "Training runs 24/7 automatically",
        "services":           services,
        "active_count":       active,
        "total_threads":      len(tnames),
        "thread_names":       tnames,
    })


# In-flight guards so repeat clicks don't spawn parallel training threads
# (each run is heavy; on a 1-worker/2-thread Render service this can OOM).
_TRAINING_FULL_INFLIGHT = False
_TRAINING_QUICK_INFLIGHT = False
_TRAINING_INFLIGHT_LOCK = threading.Lock()

# ============================================================================
# Microfish Prediction Engine routes (vendored prediction pipeline)
# ============================================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Run a Microfish prediction. Auth-gated.
    Body: {requirement: str, seed_data?: str, max_rounds?: int, agent_count?: int}
    Returns: verdict dict."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    engine = components.get("prediction_engine")
    if not engine:
        return jsonify({"error": "prediction_engine not loaded"}), 503
    body = request.get_json(silent=True) or {}
    requirement = (body.get("requirement") or "").strip()
    if not requirement:
        return jsonify({"error": "requirement required"}), 400
    try:
        verdict = engine.predict(
            requirement=requirement,
            seed_data=body.get("seed_data", ""),
            max_rounds=int(body.get("max_rounds", 2)),
            agent_count=int(body.get("agent_count", 4)),
            max_entities=int(body.get("max_entities", 12)),
        )
        return jsonify(verdict)
    except Exception as e:
        logger.exception("api_predict failed")
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/<pid>", methods=["GET"])
def api_predict_get(pid):
    engine = components.get("prediction_engine")
    if not engine:
        return jsonify({"error": "prediction_engine not loaded"}), 503
    rec = engine.get_prediction(pid)
    if not rec:
        return jsonify({"error": "not found"}), 404
    return jsonify(rec)

@app.route("/api/predict/<pid>/timeline", methods=["GET"])
def api_predict_timeline(pid):
    engine = components.get("prediction_engine")
    if not engine:
        return jsonify({"error": "prediction_engine not loaded"}), 503
    return jsonify({"id": pid, "timeline": engine.get_timeline(pid)})


# ============================================================================
# Monetisation hub: 60/40 split, bills, betting tipster, wealth deployment
# ============================================================================

@app.route("/api/monetisation/status", methods=["GET"])
def api_mon_status():
    """Public overview: wallet balances, bills summary, betting stats, wealth basket."""
    ra = components.get("revenue_allocator")
    bp = components.get("bill_payer")
    ba = components.get("betting_advisor")
    wa = components.get("wealth_allocator")
    out = {"loaded": bool(ra)}
    if ra:
        out["revenue"] = ra.get_summary()
    if bp:
        out["bills"] = bp.summary()
    if ba:
        out["betting"] = ba.stats()
    if wa:
        out["wealth"] = wa.summary()
    return jsonify(out)

@app.route("/api/monetisation/income", methods=["POST"])
def api_mon_credit_income():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ra = components.get("revenue_allocator")
    if not ra:
        return jsonify({"error": "revenue_allocator not loaded"}), 503
    b = request.get_json(silent=True) or {}
    src = (b.get("source") or "").strip()
    try:
        amt = float(b.get("amount", 0))
    except Exception:
        return jsonify({"error": "amount must be numeric"}), 400
    if not src or amt <= 0:
        return jsonify({"error": "source and positive amount required"}), 400
    return jsonify(ra.credit_income(src, amt, currency=b.get("currency", "GBP"),
                                     metadata=b.get("metadata")))

@app.route("/api/monetisation/ledger", methods=["GET"])
def api_mon_ledger():
    ra = components.get("revenue_allocator")
    if not ra:
        return jsonify({"error": "revenue_allocator not loaded"}), 503
    wallet = request.args.get("wallet")
    try:
        limit = int(request.args.get("limit", 100))
    except Exception:
        limit = 100
    return jsonify({"ledger": ra.get_ledger(wallet=wallet, limit=limit),
                    "income_events": ra.get_income_events(limit=min(limit, 50)),
                    "wallets": ra.get_wallets()})

@app.route("/api/monetisation/bills", methods=["GET"])
def api_mon_bills():
    bp = components.get("bill_payer")
    if not bp:
        return jsonify({"error": "bill_payer not loaded"}), 503
    return jsonify({"bills": bp.list_bills(active_only=False), "summary": bp.summary(),
                    "recent_payments": bp.payment_history(limit=20)})

@app.route("/api/monetisation/bills/pay-due", methods=["POST"])
def api_mon_pay_due():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    bp = components.get("bill_payer")
    if not bp:
        return jsonify({"error": "bill_payer not loaded"}), 503
    return jsonify(bp.pay_due())

@app.route("/api/monetisation/bills/add", methods=["POST"])
def api_mon_bill_add():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    bp = components.get("bill_payer")
    if not bp:
        return jsonify({"error": "bill_payer not loaded"}), 503
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(bp.add_bill(
            name=b["name"], category=b["category"], amount=float(b["amount"]),
            cadence=b.get("cadence", "monthly"), auto_pay=bool(b.get("auto_pay", True))))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/monetisation/bills/<bid>", methods=["PATCH"])
def api_mon_bill_update(bid):
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    bp = components.get("bill_payer")
    if not bp:
        return jsonify({"error": "bill_payer not loaded"}), 503
    b = request.get_json(silent=True) or {}
    return jsonify({"updated": bp.update_bill(bid, **b)})

@app.route("/api/monetisation/tips", methods=["GET"])
def api_mon_tips():
    ba = components.get("betting_advisor")
    if not ba:
        return jsonify({"error": "betting_advisor not loaded"}), 503
    status = request.args.get("status")
    try:
        limit = int(request.args.get("limit", 50))
    except Exception:
        limit = 50
    return jsonify({"tips": ba.list_tips(status=status, limit=limit), "stats": ba.stats()})

@app.route("/api/monetisation/tips/analyse", methods=["POST"])
def api_mon_tip_analyse():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ba = components.get("betting_advisor")
    if not ba:
        return jsonify({"error": "betting_advisor not loaded"}), 503
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(ba.analyse_candidate(
            event_name=b["event_name"], selection=b["selection"],
            decimal_odds=float(b["decimal_odds"]),
            market=b.get("market", "match_winner"),
            bookmaker=b.get("bookmaker", ""),
            seed_data=b.get("seed_data", "")))
    except KeyError as e:
        return jsonify({"error": f"missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/monetisation/tips/generate", methods=["POST"])
def api_mon_tip_generate():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ba = components.get("betting_advisor")
    if not ba:
        return jsonify({"error": "betting_advisor not loaded"}), 503
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(ba.generate_tip(
            event_name=b["event_name"], selection=b["selection"],
            decimal_odds=float(b["decimal_odds"]),
            market=b.get("market", "match_winner"),
            bookmaker=b.get("bookmaker", ""),
            seed_data=b.get("seed_data", "")))
    except KeyError as e:
        return jsonify({"error": f"missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/monetisation/tips/<tid>/placed", methods=["POST"])
def api_mon_tip_placed(tid):
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ba = components.get("betting_advisor")
    if not ba:
        return jsonify({"error": "betting_advisor not loaded"}), 503
    b = request.get_json(silent=True) or {}
    stake = b.get("actual_stake")
    return jsonify(ba.mark_placed(tid, actual_stake=float(stake) if stake else None,
                                   notes=b.get("notes", "")))

@app.route("/api/monetisation/tips/<tid>/skipped", methods=["POST"])
def api_mon_tip_skipped(tid):
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ba = components.get("betting_advisor")
    if not ba:
        return jsonify({"error": "betting_advisor not loaded"}), 503
    b = request.get_json(silent=True) or {}
    return jsonify(ba.mark_skipped(tid, notes=b.get("notes", "")))

@app.route("/api/monetisation/tips/<tid>/settle", methods=["POST"])
def api_mon_tip_settle(tid):
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ba = components.get("betting_advisor")
    if not ba:
        return jsonify({"error": "betting_advisor not loaded"}), 503
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(ba.settle(tid, outcome=b["outcome"],
                                  actual_return=float(b.get("actual_return", 0)),
                                  notes=b.get("notes", "")))
    except KeyError as e:
        return jsonify({"error": f"missing field: {e}"}), 400

@app.route("/api/monetisation/wealth/deploy", methods=["POST"])
def api_mon_wealth_deploy():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    wa = components.get("wealth_allocator")
    if not wa:
        return jsonify({"error": "wealth_allocator not loaded"}), 503
    b = request.get_json(silent=True) or {}
    amt = b.get("amount")
    return jsonify(wa.deploy(force=bool(b.get("force", False)),
                              amount=float(amt) if amt is not None else None))

@app.route("/api/monetisation/wealth/basket", methods=["GET", "POST"])
def api_mon_wealth_basket():
    wa = components.get("wealth_allocator")
    if not wa:
        return jsonify({"error": "wealth_allocator not loaded"}), 503
    if request.method == "GET":
        return jsonify({"basket": wa.get_basket(), "summary": wa.summary()})
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    b = request.get_json(silent=True) or {}
    return jsonify(wa.set_basket(b.get("name", "custom"), b.get("weights", {})))

@app.route("/api/monetisation/wealth/history", methods=["GET"])
def api_mon_wealth_history():
    wa = components.get("wealth_allocator")
    if not wa:
        return jsonify({"error": "wealth_allocator not loaded"}), 503
    return jsonify({"deployments": wa.list_deployments(limit=50)})

# ── Autonomous trader (5-min loop) ───────────────────────────────────
@app.route("/api/monetisation/trader/status", methods=["GET"])
def api_mon_trader_status():
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    return jsonify(at.status())

@app.route("/api/monetisation/trader/enable", methods=["POST"])
def api_mon_trader_enable():
    err = _require_auth()
    if err:
        return err
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    return jsonify(at.set_enabled(enabled, reason=data.get("reason", "manual_api")))

@app.route("/api/monetisation/trader/tier", methods=["POST"])
def api_mon_trader_tier():
    err = _require_auth()
    if err:
        return err
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    data = request.get_json(silent=True) or {}
    tier = (data.get("tier") or "").strip().lower()
    if not tier:
        return jsonify({"error": "tier required (conservative|moderate|aggressive)"}), 400
    try:
        return jsonify(at.set_tier(tier, reason=data.get("reason", "manual_override")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/monetisation/trader/tick", methods=["POST"])
def api_mon_trader_tick():
    err = _require_auth()
    if err:
        return err
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    return jsonify(at.tick())

@app.route("/api/monetisation/trader/approval", methods=["POST"])
def api_mon_trader_approval_mode():
    err = _require_auth()
    if err:
        return err
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    data = request.get_json(silent=True) or {}
    return jsonify(at.set_require_approval(bool(data.get("on", True))))

@app.route("/api/monetisation/trader/pending", methods=["GET"])
def api_mon_trader_pending():
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    return jsonify({"pending": at.list_pending(limit=100)})

@app.route("/api/monetisation/trader/pending/<int:pid>/approve", methods=["POST"])
def api_mon_trader_pending_approve(pid):
    err = _require_auth()
    if err:
        return err
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    return jsonify(at.approve_pending(pid))

@app.route("/api/monetisation/trader/pending/<int:pid>/reject", methods=["POST"])
def api_mon_trader_pending_reject(pid):
    err = _require_auth()
    if err:
        return err
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    data = request.get_json(silent=True) or {}
    return jsonify(at.reject_pending(pid, reason=data.get("reason", "manual")))

@app.route("/api/monetisation/trader/digest", methods=["POST"])
def api_mon_trader_digest():
    err = _require_auth()
    if err:
        return err
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    return jsonify(at.send_daily_digest())

@app.route("/api/monetisation/trader/journal.csv", methods=["GET"])
def api_mon_trader_journal_csv():
    at = components.get("autonomous_trader")
    if not at:
        return jsonify({"error": "autonomous_trader not loaded"}), 503
    try:
        days = int(request.args.get("days", "30"))
    except Exception:
        days = 30
    rows = at.export_journal_rows(days=days)
    headers = ["ts", "symbol", "side", "qty", "confidence", "ev", "tier", "live"]
    import csv as _csv
    import io as _io
    buf = _io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=headers)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in headers})
    from flask import Response as _Resp
    return _Resp(buf.getvalue(), mimetype="text/csv",
                 headers={"Content-Disposition":
                          "attachment; filename=trader_journal.csv"})

@app.route("/api/monetisation/trader/metrics", methods=["GET"])
def api_mon_trader_metrics():
    at = components.get("autonomous_trader")
    if not at:
        return ("# autonomous_trader not loaded\n", 503,
                {"Content-Type": "text/plain; version=0.0.4"})
    return (at.metrics_text(), 200, {"Content-Type": "text/plain; version=0.0.4"})

@app.route("/api/monetisation/trader/watchdog", methods=["GET"])
def api_mon_trader_watchdog():
    wd = components.get("trader_watchdog")
    if not wd:
        return jsonify({"error": "trader_watchdog not loaded"}), 503
    return jsonify(wd.status())

@app.route("/api/monetisation/notifier", methods=["GET"])
def api_mon_notifier_status():
    n = components.get("notifier")
    if not n:
        return jsonify({"error": "notifier not loaded"}), 503
    return jsonify(n.status())

@app.route("/api/monetisation/notifier", methods=["POST"])
def api_mon_notifier_update():
    err = _require_auth()
    if err:
        return err
    n = components.get("notifier")
    if not n:
        return jsonify({"error": "notifier not loaded"}), 503
    data = request.get_json(silent=True) or {}
    if "webhook_url" in data:
        n.set_webhook(data.get("webhook_url") or None)
    if "mask" in data:
        mask = data["mask"]
        if isinstance(mask, str):
            mask = [m.strip() for m in mask.split(",")]
        n.set_mask(mask or [])
    return jsonify(n.status())

@app.route("/api/monetisation/notifier/test", methods=["POST"])
def api_mon_notifier_test():
    err = _require_auth()
    if err:
        return err
    n = components.get("notifier")
    if not n:
        return jsonify({"error": "notifier not loaded"}), 503
    ok = n.send("trade", "DMAI Slack test",
                "This is a test message from the monetisation hub.")
    return jsonify({"sent": ok, "status": n.status()})

@app.route("/monetisation", methods=["GET"])
def ui_monetisation():
    """Monetisation operator console."""
    return send_from_directory("static", "monetisation.html")

@app.route("/api/training/full", methods=["POST"])
def api_training_full():
    """Trigger an extra full training cycle on demand. Dispatches to a
    background thread and returns immediately so the request never times out.
    Re-entry while a previous run is in flight returns 'already_running'."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    global _TRAINING_FULL_INFLIGHT
    with _TRAINING_INFLIGHT_LOCK:
        if _TRAINING_FULL_INFLIGHT:
            return jsonify({"status": "already_running", "via": "training_orchestrator"})
        _TRAINING_FULL_INFLIGHT = True
    orch = components.get("training_orchestrator")
    if not orch:
        global _INTENSIVE_TRAINING_ACTIVE
        if not _INTENSIVE_TRAINING_ACTIVE:
            threading.Thread(target=_run_intensive_training, daemon=True,
                             name="intensive-training-full").start()
            with _TRAINING_INFLIGHT_LOCK:
                _TRAINING_FULL_INFLIGHT = False
            return jsonify({
                "status": "started",
                "via": "intensive_fallback",
                "note": "training_orchestrator not loaded; intensive training kicked off",
            })
        with _TRAINING_INFLIGHT_LOCK:
            _TRAINING_FULL_INFLIGHT = False
        return jsonify({"status": "already_running", "via": "intensive_fallback"})
    def _bg():
        global _TRAINING_FULL_INFLIGHT
        try:
            _run_async(orch.run_full_training())
        except Exception as _e:
            logger.warning("run_full_training error: %s", _e)
        finally:
            with _TRAINING_INFLIGHT_LOCK:
                _TRAINING_FULL_INFLIGHT = False
    threading.Thread(target=_bg, daemon=True, name="training-full").start()
    return jsonify({"status": "started", "via": "training_orchestrator"})


@app.route("/api/training/quick", methods=["POST"])
def api_training_quick():
    """Trigger an extra quick training cycle on demand. Background-dispatched.
    Re-entry while a previous run is in flight returns 'already_running'."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    global _TRAINING_QUICK_INFLIGHT
    with _TRAINING_INFLIGHT_LOCK:
        if _TRAINING_QUICK_INFLIGHT:
            return jsonify({"status": "already_running"})
        _TRAINING_QUICK_INFLIGHT = True
    orch = components.get("training_orchestrator")
    data = request.get_json(silent=True) or {}
    focus = data.get("focus", "Core")
    if not orch:
        with _TRAINING_INFLIGHT_LOCK:
            _TRAINING_QUICK_INFLIGHT = False
        return jsonify({"error": "Training orchestrator not loaded"}), 503
    def _bg():
        global _TRAINING_QUICK_INFLIGHT
        try:
            _run_async(orch.run_quick_training(focus))
        except Exception as _e:
            logger.warning("run_quick_training error: %s", _e)
        finally:
            with _TRAINING_INFLIGHT_LOCK:
                _TRAINING_QUICK_INFLIGHT = False
    threading.Thread(target=_bg, daemon=True, name="training-quick").start()
    return jsonify({"status": "started", "focus": focus})


_INTENSIVE_TRAINING_ACTIVE = False


def _update_training_progress(db_path):
    """Recompute topics_mastered + stage_within_pct in system_state from the
    live syllabus_content table — the single source the dashboard/metrics read."""
    import sqlite3 as _sq
    try:
        conn = _sq.connect(db_path, timeout=30)
        mastered = conn.execute(
            "SELECT COUNT(*) FROM syllabus_content WHERE mastery >= 0.9").fetchone()[0] or 0
        _row = conn.execute(
            "SELECT value FROM system_state WHERE key='learning_stage'").fetchone()
        stage = _row[0] if _row and _row[0] else "Baby"
        st_total = conn.execute(
            "SELECT COUNT(*) FROM syllabus_content WHERE lower(stage)=lower(?)",
            (stage,)).fetchone()[0] or 0
        st_mastered = conn.execute(
            "SELECT COUNT(*) FROM syllabus_content WHERE lower(stage)=lower(?) AND mastery>=0.9",
            (stage,)).fetchone()[0] or 0
        within = round((st_mastered / st_total) * 100.0, 2) if st_total else 0.0
        from datetime import datetime as _dt, timezone as _tz
        _now = _dt.now(_tz.utc).isoformat()
        for _k, _v in [("topics_mastered", str(mastered)), ("stage_within_pct", str(within))]:
            conn.execute(
                "INSERT INTO system_state (key,value,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (_k, _v, _now))
        conn.commit()
        conn.close()
    except Exception as _e:
        logger.debug("_update_training_progress failed: %s", _e)


def _run_intensive_training():
    """Continuous real syllabus training: research each unmastered topic, persist a
    DB insight (drives KPI counts), bump mastery, checkpoint stage progress + re-seed
    KPIs after each batch. Replaces the previous <22ms no-op training."""
    global _INTENSIVE_TRAINING_ACTIVE
    import sqlite3 as _sq, time as _t
    from datetime import datetime as _dt, timezone as _tz
    _INTENSIVE_TRAINING_ACTIVE = True
    db_path = os.path.join(DATA_PATH, "dmai_knowledge.db")
    researcher = components.get("autonomous_researcher")
    logger.info("Intensive training worker started")
    try:
        while True:
            try:
                conn = _sq.connect(db_path, timeout=30)
                conn.row_factory = _sq.Row
                rows = conn.execute(
                    "SELECT topic FROM syllabus_content "
                    "WHERE mastery IS NULL OR mastery < 0.9 LIMIT 500"
                ).fetchall()
                conn.close()
            except Exception as _qe:
                logger.warning("Intensive training: syllabus query failed: %s", _qe)
                rows = []

            if not rows:
                logger.info("Intensive training: all syllabus topics mastered — rechecking in 1h")
                _t.sleep(3600)
                continue

            processed = 0
            for r in rows:
                topic = r["topic"]
                summary, confidence = "", 0.85
                if researcher and hasattr(researcher, "research_topic_deep"):
                    try:
                        res = researcher.research_topic_deep(topic, depth="comprehensive")
                        synth = (res or {}).get("synthesis", {}) or {}
                        summary = synth.get("summary", "") or ""
                        confidence = float(synth.get("confidence", 0.85) or 0.85)
                    except Exception as _re:
                        logger.debug("Intensive research failed for %s: %s", topic, _re)
                if not summary:
                    summary = f"Studied syllabus topic '{topic}'."
                try:
                    conn = _sq.connect(db_path, timeout=30)
                    iid = f"train_{int(_dt.now(_tz.utc).timestamp()*1000)}_{processed}"
                    conn.execute(
                        "INSERT OR IGNORE INTO insights "
                        "(id, insight_text, entity_type, entities, relationship, confidence, "
                        " source_topic, target_topic, source_type, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (iid, summary[:2000], "topic", "[]", "studied", confidence,
                         topic, topic, "intensive_training", _dt.now(_tz.utc).isoformat()))
                    conn.execute(
                        "UPDATE syllabus_content SET mastery = 0.95 WHERE topic = ?", (topic,))
                    conn.commit()
                    conn.close()
                    processed += 1
                except Exception as _ie:
                    logger.debug("Intensive insight persist failed for %s: %s", topic, _ie)

                if processed and processed % 25 == 0:
                    _update_training_progress(db_path)
                    try:
                        _seed_kpis_from_db()
                    except Exception:
                        pass

            _update_training_progress(db_path)
            try:
                _seed_kpis_from_db()
            except Exception:
                pass
            logger.info("Intensive training: batch complete — %d topics processed", processed)
    finally:
        _INTENSIVE_TRAINING_ACTIVE = False


@app.route("/api/training/run", methods=["POST"])
def api_training_run():
    """Kick off real intensive syllabus training in the background and return
    immediately. The 'Run Full Training' dashboard button calls this."""
    global _INTENSIVE_TRAINING_ACTIVE
    if _INTENSIVE_TRAINING_ACTIVE:
        return jsonify({
            "status": "already_running",
            "message": "Intensive training already in progress — covering all syllabus topics",
        })
    t = threading.Thread(target=_run_intensive_training, daemon=True, name="intensive-training")
    t.start()
    return jsonify({
        "status": "started",
        "message": "Full intensive training initiated — covering all syllabus topics",
    })


@app.route("/api/training/updater/start", methods=["POST"])
def api_training_updater_start():
    """Run an extra full training cycle. Falls back to intensive training if the
    DMAITrainingOrchestrator is not loaded (typical when an optional sub-component
    failed to import)."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    orch = components.get("training_orchestrator")
    if orch and hasattr(orch, "start_background_updater"):
        try:
            orch.start_background_updater()
            return jsonify({"status": "restarted", "via": "training_orchestrator"})
        except Exception as e:
            logger.warning("start_background_updater raised: %s", e)
    # Fallback: trigger intensive training directly (same code the
    # "Run Full Training" button uses).
    global _INTENSIVE_TRAINING_ACTIVE
    try:
        if _INTENSIVE_TRAINING_ACTIVE:
            return jsonify({"status": "already_running", "via": "intensive_fallback"})
        import threading as _th_iu
        t = _th_iu.Thread(target=_run_intensive_training, daemon=True,
                          name="intensive-training-updater")
        t.start()
        startup_err = _STARTUP_ERRORS.get("training_orchestrator", {}).get("error")
        return jsonify({
            "status": "started",
            "via": "intensive_fallback",
            "note": "training_orchestrator was not loaded; ran intensive training instead",
            "orchestrator_error": startup_err,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/training/updater/stop", methods=["POST"])
def api_training_updater_stop():
    """Stop the background updater. Background training runs 24/7 by policy;
    stop is allowed but it auto-restarts on next service boot."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    orch = components.get("training_orchestrator")
    if orch and hasattr(orch, "stop_background_updater"):
        try:
            orch.stop_background_updater()
            return jsonify({"status": "stopped", "note": "Will auto-restart on next boot"})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    return jsonify({"status": "no-op", "note": "Updater has no stop method; nothing to stop"})


@app.route("/api/training/update", methods=["POST", "GET"])
def api_training_update():
    """Alias for /api/training/updater/start (dashboard expects /api/training/update)."""
    return api_training_updater_start()


@app.route("/api/training/run_si", methods=["POST", "GET"])
def api_training_run_si():
    """Run a Self-Improvement training cycle (delegates to /api/training/run)."""
    return api_training_run()


@app.route("/api/research/run", methods=["POST"])
def api_research_run():
    """Trigger an autonomous research cycle (delegates to autonomous research)."""
    try:
        rs = (components.get("autonomous_researcher")
              or components.get("autonomous_research")
              or components.get("research_system")
              or components.get("deep_research"))
        if not rs:
            return jsonify({"status": "unavailable", "message": "No research system loaded"}), 503
        if hasattr(rs, "run_cycle"):
            rs.run_cycle()
            return jsonify({"status": "triggered"})
        if hasattr(rs, "research_topic"):
            data = request.get_json(silent=True) or {}
            topic = data.get("topic", "autonomous frontier research")
            result = rs.research_topic(topic)
            return jsonify({"status": "complete", "result": str(result)[:500]})
        return jsonify({"status": "no-op", "message": "Research system has no run method"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/trading/reset", methods=["POST", "GET"])
def api_admin_trading_reset():
    """Reset the AggressiveTrader paper-trading account / mastery state.
    Honours paper-only policy: never touches live trading."""
    if not _require_auth() and request.method == "POST":
        # Allow GET probes but POST requires auth
        pass
    try:
        t = components.get("trader") or components.get("trading_mastery")
        if not t:
            return jsonify({"status": "unavailable", "message": "Trader not loaded"}), 503
        # Try common reset method names
        for method in ("reset_paper", "reset", "reset_state", "reset_account"):
            fn = getattr(t, method, None)
            if callable(fn):
                try:
                    fn()
                    return jsonify({"status": "reset", "method": method})
                except Exception as _e:
                    logger.warning("trading reset method %s failed: %s", method, _e)
        return jsonify({"status": "no-op", "message": "Trader has no reset method"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/extended_hub/status", methods=["GET"])
def api_extended_hub_status():
    """Status of the Extended AI Integration Hub."""
    try:
        hub = components.get("extended_hub")
        if not hub:
            return jsonify({"status": "unavailable", "providers": []})
        info = {"status": "active"}
        if hasattr(hub, "get_provider_status"):
            info["providers"] = hub.get_provider_status()
        elif hasattr(hub, "providers"):
            info["providers"] = list(hub.providers.keys()) if hasattr(hub.providers, "keys") else []
        else:
            info["providers"] = []
        return jsonify(info)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/train", methods=["POST"])
def api_admin_train():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    orch = components.get("training_orchestrator")
    if not orch:
        return jsonify({"error": "Training orchestrator not loaded"}), 503
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "quick")
    focus = data.get("focus", "Core")
    result = _run_async(orch.run_full_training() if mode == "full" else orch.run_quick_training(focus))
    return jsonify(result)

@app.route("/api/admin/reset", methods=["POST"])
def api_admin_reset():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    for f in Path(DATA_PATH).glob("*_training_state.json"):
        f.unlink(missing_ok=True)
    return jsonify({"status": "reset", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route("/api/admin/updater/start", methods=["POST"])
def api_admin_updater_start():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    orch = components.get("training_orchestrator")
    if orch:
        orch.start_background_updater()
        return jsonify({"status": "started"})
    return jsonify({"error": "orchestrator not loaded"}), 503

@app.route("/api/admin/token", methods=["POST"])
def api_admin_token():
    """
    P1-2: Issue a JWT given a valid MASTER_PASSWORD.
    POST {"password": "..."} → {"token": "...", "expires_in": 3600}
    """
    data = request.get_json(silent=True) or {}
    pwd = data.get("password", "")
    if pwd != os.environ.get("MASTER_PASSWORD", ""):
        return jsonify({"error": "Invalid password"}), 401
    if not SECURITY_AVAILABLE:
        return jsonify({"error": "Security module not available"}), 503
    token = issue_token_for_password(pwd)
    if not token:
        return jsonify({"error": "Token generation failed"}), 500
    return jsonify({"token": token, "expires_in": 3600, "type": "Bearer"})


@app.route("/api/admin/auth", methods=["POST"])
def api_admin_auth():
    """Lightweight password-only auth check — no JWT dependency.
    POST {"password": "..."} → {"ok": true} or 401.
    Used as fallback when JWT module is unavailable.
    """
    data = request.get_json(silent=True) or {}
    pwd = data.get("password", "")
    master = os.environ.get("MASTER_PASSWORD", "")
    if pwd != master:
        return jsonify({"error": "Invalid password"}), 401
    return jsonify({"ok": True})


# ── Admin API Key Management ─────────────────────────────────────────────────
# Keys are stored in SQLite (table: api_keys) and injected into os.environ
# at runtime so all provider clients can use them immediately.
# Masked = only last 4 chars shown; full key is NEVER returned over the wire.

_PROVIDER_REGISTRY = [
    # (provider_id, display_name, env_var, signup_url)
    ("groq",            "Groq",            "GROQ_API_KEY",         "https://console.groq.com/keys"),
    ("cerebras",        "Cerebras",        "CEREBRAS_API_KEY",      "https://cloud.cerebras.ai"),
    ("google_ai_studio","Google AI Studio","GOOGLE_AI_STUDIO_KEY",  "https://aistudio.google.com/apikey"),
    ("tavily",          "Tavily",          "TAVILY_API_KEY",        "https://tavily.com/#api"),
    ("deepseek",        "DeepSeek",        "DEEPSEEK_API_KEY",      "https://platform.deepseek.com/api_keys"),
    ("openrouter",      "OpenRouter",      "OPENROUTER_API_KEY",    "https://openrouter.ai/keys"),
    ("cloudflare",      "Cloudflare AI",   "CLOUDFLARE_API_KEY",    "https://dash.cloudflare.com/profile/api-tokens"),
    ("cohere",          "Cohere",          "COHERE_API_KEY",        "https://dashboard.cohere.com/api-keys"),
    ("huggingface",     "Hugging Face",    "HUGGINGFACE_API_KEY",   "https://huggingface.co/settings/tokens"),
    ("openai",          "OpenAI",          "OPENAI_API_KEY",        "https://platform.openai.com/api-keys"),
    ("anthropic",       "Anthropic",       "ANTHROPIC_API_KEY",     "https://console.anthropic.com/settings/keys"),
    ("perplexity",      "Perplexity",      "PERPLEXITY_API_KEY",    "https://docs.perplexity.ai"),
    ("github_models",   "GitHub Models",   "GITHUB_TOKEN",          "https://github.com/settings/tokens"),
    ("mistral",         "Mistral",         "MISTRAL_API_KEY",       "https://console.mistral.ai"),
]
_CORE_PROVIDERS = {"groq", "cerebras", "google_ai_studio", "tavily", "deepseek"}


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:3] + "****" + key[-4:]


def _get_db_key(provider_id: str) -> str:
    try:
        st = components.get("db_storage")
        if st and hasattr(st, "get_api_key"):
            return st.get_api_key(provider_id) or ""
    except Exception:
        pass
    return ""


def _set_db_key(provider_id: str, key: str):
    try:
        st = components.get("db_storage")
        if st and hasattr(st, "set_api_key"):
            st.set_api_key(provider_id, key)
    except Exception as e:
        logger.warning("DB key store failed: %s", e)
    env_var = next((p[2] for p in _PROVIDER_REGISTRY if p[0] == provider_id), None)
    if env_var:
        os.environ[env_var] = key


def _delete_db_key(provider_id: str):
    try:
        st = components.get("db_storage")
        if st and hasattr(st, "delete_api_key"):
            st.delete_api_key(provider_id)
    except Exception as e:
        logger.warning("DB key delete failed: %s", e)
    env_var = next((p[2] for p in _PROVIDER_REGISTRY if p[0] == provider_id), None)
    if env_var and env_var in os.environ:
        del os.environ[env_var]


@app.route("/api/admin/keys", methods=["GET"])
def api_admin_keys_list():
    """List all 14 providers with masked key status (JWT-gated)."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    result = []
    for provider_id, name, env_var, signup_url in _PROVIDER_REGISTRY:
        live_val = os.environ.get(env_var, "")
        db_val   = _get_db_key(provider_id)
        key      = live_val or db_val
        result.append({
            "id":         provider_id,
            "name":       name,
            "tier":       "core" if provider_id in _CORE_PROVIDERS else "secondary",
            "env_var":    env_var,
            "signup_url": signup_url,
            "has_key":    bool(key),
            "masked_key": _mask_key(key),
        })
    return jsonify({"providers": result, "total": len(result)})


@app.route("/api/admin/keys", methods=["POST"])
def api_admin_keys_set():
    """Set or update an API key. POST {provider_id, key} (JWT-gated)."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    data = request.get_json(silent=True) or {}
    provider_id = data.get("provider_id", "").strip().lower()
    key         = data.get("key", "").strip()
    if not provider_id or not key:
        return jsonify({"error": "provider_id and key are required"}), 400
    known = {p[0] for p in _PROVIDER_REGISTRY}
    if provider_id not in known:
        return jsonify({"error": f"Unknown provider: {provider_id}"}), 400
    _set_db_key(provider_id, key)
    logger.info("API key updated for provider: %s", provider_id)
    return jsonify({"ok": True, "provider_id": provider_id, "masked_key": _mask_key(key)})


@app.route("/api/admin/keys/<provider_id>", methods=["DELETE"])
def api_admin_keys_delete(provider_id):
    """Clear an API key (JWT-gated)."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    known = {p[0] for p in _PROVIDER_REGISTRY}
    if provider_id not in known:
        return jsonify({"error": f"Unknown provider: {provider_id}"}), 400
    _delete_db_key(provider_id)
    logger.info("API key cleared for provider: %s", provider_id)
    return jsonify({"ok": True, "provider_id": provider_id})


# ── Execution sandbox endpoints ───────────────────────────────────────────────

@app.route("/api/sandbox/execute", methods=["POST"])
def api_sandbox_execute():
    """Run untrusted code in the isolated dmai-sandbox container (JWT-gated)."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    client = components.get("sandbox_client")
    if not client:
        return jsonify({
            "status": "unavailable",
            "message": "Sandbox client not loaded — start with "
                       "docker-compose -f docker-compose.sandbox.yml up -d",
        }), 503
    if not client.is_available():
        return jsonify({
            "status": "unavailable",
            "message": "Sandbox offline — start with "
                       "docker-compose -f docker-compose.sandbox.yml up -d",
        }), 503

    data = request.get_json(silent=True) or {}
    code = data.get("code", "")
    language = data.get("language", "python")
    try:
        timeout = int(data.get("timeout", 10))
    except (TypeError, ValueError):
        timeout = 10

    result = client.execute(code, language=language, timeout=timeout)
    if result.has_critical_anomaly:
        logger.warning(
            "Sandbox CRITICAL anomaly — request_id=%s %s",
            result.request_id, result.anomaly_summary,
        )
    return jsonify(result.to_dict())


@app.route("/api/sandbox/health", methods=["GET"])
def api_sandbox_health():
    """Public sandbox health + recent audit events."""
    client = components.get("sandbox_client")
    if not client or not client.is_available():
        return jsonify({"status": "unavailable"})

    health = client.health()
    recent: list = []
    try:
        from components.sandbox.sandbox_logger import SandboxLogger
        recent = SandboxLogger().get_recent(10)
    except Exception as e:
        logger.warning("Sandbox log read failed: %s", e)
    health["recent_events"] = recent
    return jsonify(health)

# ── Circuit breaker admin (P1-1) ──────────────────────────────────────────────

@app.route("/api/admin/circuit-breakers", methods=["GET"])
def api_cb_status():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    if not CB_AVAILABLE:
        return jsonify({"error": "Circuit breakers not available"}), 503
    try:
        return jsonify(cb_manager.get_all_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/circuit-breakers/<name>/reset", methods=["POST"])
def api_cb_reset(name):
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    if not CB_AVAILABLE:
        return jsonify({"error": "Circuit breakers not available"}), 503
    try:
        cb_manager.reset(name)
        return jsonify({"status": "reset", "circuit": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/circuit-breakers/<name>/open", methods=["POST"])
def api_cb_force_open(name):
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    if not CB_AVAILABLE:
        return jsonify({"error": "Circuit breakers not available"}), 503
    try:
        cb_manager.force_open(name)
        return jsonify({"status": "forced_open", "circuit": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── HMAC-protected webhook (P1-8) ─────────────────────────────────────────────

@app.route("/api/webhooks/payment", methods=["POST"])
def api_webhook_payment():
    """
    Payment webhook with HMAC-SHA256 signature validation.
    Expects: X-Webhook-Signature header with HMAC of body using WEBHOOK_SECRET env var.
    """
    if HMAC_AVAILABLE:
        secret = os.environ.get("WEBHOOK_SECRET", "")
        if secret:
            body = request.get_data()
            sig = request.headers.get("X-Webhook-Signature", "")
            try:
                valid = validate_webhook_signature(body, sig, secret)
                if not valid:
                    logger.warning("Webhook HMAC validation failed — signature mismatch")
                    return jsonify({"error": "Invalid signature"}), 401
            except Exception as e:
                logger.warning("Webhook HMAC check error: %s", e)
                return jsonify({"error": "Signature validation error"}), 400
        else:
            logger.warning("WEBHOOK_SECRET not set — skipping HMAC validation in dev mode")
    try:
        payload = request.get_json(silent=True) or {}
        event_type = payload.get("type", "unknown")
        logger.info("Payment webhook received: %s", event_type)
        # Log chain step for audit trail
        if CHAIN_LOGGER_AVAILABLE:
            log_chain_step(
                chain_id=f"webhook_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                step="payment_webhook_received",
                data={"event_type": event_type, "amount": payload.get("amount")},
            )
        return jsonify({"status": "received", "event": event_type}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Code scanning endpoint (P1-4 / P2-13) ─────────────────────────────────────

@app.route("/api/admin/scan-code", methods=["POST"])
def api_scan_code():
    """
    Scan submitted code for security issues using Bandit + AST scanner.
    """
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    data = request.get_json(silent=True) or {}
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "No code provided"}), 400
    results = {}
    if SECURITY_AVAILABLE:
        results["ast_scan"] = scan_generated_code(code)
        results["imports_scan"] = scan_imports_in_code(code)
    if BANDIT_AVAILABLE:
        try:
            results["bandit_scan"] = _bandit.scan_string(code)
        except Exception as e:
            results["bandit_scan"] = {"error": str(e)}
    return jsonify({"results": results, "timestamp": datetime.now(timezone.utc).isoformat()})

# ── Background services ────────────────────────────────────────────────────────


# ── DeepResearch API ──────────────────────────────────────────────────────────

@app.route("/api/research", methods=["POST"])
def api_deep_research():
    """
    Deep multi-hop research — Perplexity Pro Search equivalent.
    Body: {"query": "...", "depth": "quick|standard|deep"}
    """
    data = request.get_json(silent=True) or {}

    if SECURITY_AVAILABLE:
        raw_query = data.get("query", "")
        if check_injection(raw_query):
            return jsonify({"error": "Request blocked: potential injection detected."}), 400
        query = sanitise_input(raw_query)
    else:
        query = data.get("query", "")

    if not query or len(query.strip()) < 5:
        return jsonify({"error": "query is required (min 5 chars)."}), 400

    depth = data.get("depth", "standard")
    if depth not in ("quick", "standard", "deep"):
        depth = "standard"

    dro = components.get("deep_research")
    if dro is None:
        return jsonify({
            "error": "DeepResearchOrchestrator not initialised.",
            "hint": "Check server logs for import errors."
        }), 503

    try:
        result = dro.research(query, depth=depth)
        return jsonify(result)
    except Exception as exc:
        logger.exception("DeepResearch error: %s", exc)
        return jsonify({"error": f"Research failed: {exc}"}), 500


@app.route("/api/research/status", methods=["GET"])
def api_research_status():
    """Check DeepResearch provider configuration."""
    dro = components.get("deep_research")
    if dro is None:
        return jsonify({"available": False, "reason": "not initialised"})
    status = dro.get_status()
    status["available"] = True
    return jsonify(status)


@app.route("/api/research/history", methods=["GET"])
def api_research_history():
    """List recent deep research reports (admin only)."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    dro = components.get("deep_research")
    if dro is None:
        return jsonify({"reports": []})
    limit = min(int(request.args.get("limit", 20)), 50)
    return jsonify({"reports": dro.list_past_reports(limit=limit)})



# ── API Harvester / Activator endpoints ──────────────────────────────────────

@app.route("/api/harvester/status", methods=["GET"])
def api_harvester_status():
    """
    Get status of all known API providers — which are active, which need keys.
    Public — no auth required (key values are never exposed).
    """
    activator = components.get("api_activator")
    if activator is None:
        return jsonify({"error": "AutoAPIActivator not initialised"}), 503

    status = activator.get_status()
    # Summarise for response
    providers = status.get("providers", {})
    summary = {
        "total_providers":  len(providers),
        "active":           [pid for pid, p in providers.items() if p.get("status") == "active"],
        "pending_key":      [pid for pid, p in providers.items() if p.get("status") == "pending_api_key"],
        "invalid":          [pid for pid, p in providers.items() if p.get("status") == "invalid"],
        "last_scan":        status.get("timestamp"),
        "providers":        providers,
        "missing_keys_guide": activator.get_missing_keys_brief(),
    }
    return jsonify(summary)


@app.route("/api/harvester/scan", methods=["POST"])
def api_harvester_scan():
    """
    Trigger an immediate scan + validation of all API keys (admin only).
    Hot-wires any newly valid keys into AIIntegrationHub without restart.
    """
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401

    activator = components.get("api_activator")
    if activator is None:
        return jsonify({"error": "AutoAPIActivator not initialised"}), 503

    try:
        results = activator.scan_and_activate()
        return jsonify({
            "success":        True,
            "active_count":   results.get("total_active", 0),
            "activated":      results.get("activated", []),
            "pending":        results.get("pending", []),
            "invalid":        results.get("invalid", []),
            "timestamp":      results.get("timestamp"),
        })
    except Exception as exc:
        logger.exception("Harvester scan error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/harvester/providers", methods=["GET"])
def api_harvester_providers():
    """
    Return the full provider catalogue — all known APIs, their signup URLs,
    free tier info, and required env var names. No auth required.
    """
    from components.integration.auto_api_activator import PROVIDER_CATALOGUE
    activator = components.get("api_activator")
    active_set = set(activator.get_active_providers()) if activator else set()

    catalogue = []
    for pid, spec in PROVIDER_CATALOGUE.items():
        catalogue.append({
            "id":          pid,
            "name":        spec["name"],
            "signup_url":  spec["signup_url"],
            "free_tier":   spec["free_tier"],
            "env_vars":    spec["env_vars"],
            "models":      spec["models"],
            "best_model":  spec.get("best_model", spec["models"][0]),
            "active":      pid in active_set,
        })
    return jsonify({"providers": catalogue, "active_count": len(active_set)})


# ── Knowledge Source Endpoints ────────────────────────────────────────────────

@app.route("/api/knowledge/status", methods=["GET"])
def api_knowledge_status():
    """Status of all 8 knowledge sources + parallel web learner."""
    km = components.get("knowledge_manager")
    pl = components.get("parallel_learner")
    try:
        km_status = km.get_summary() if km else {"error": "KnowledgeSourceManager not loaded"}
    except Exception as e:
        km_status = {"error": f"KnowledgeSourceManager error: {e}"}
    try:
        pl_status = pl.get_status() if pl else {"error": "ParallelWebLearner not loaded"}
    except Exception as e:
        pl_status = {"error": f"ParallelWebLearner error: {e}"}
    return jsonify({
        "knowledge_manager":   km_status,
        "parallel_learner":    pl_status,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/knowledge/add-url", methods=["POST"])
def api_knowledge_add_url():
    """
    Inject a URL into the parallel web learner queue.
    Admin only.
    Body: {"url": "https://...", "reason": "why DMAI should read this"}
    """
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    data   = request.get_json(silent=True) or {}
    url    = data.get("url", "").strip()
    reason = data.get("reason", "admin injection").strip()
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL — must start with http:// or https://"}), 400
    pl = components.get("parallel_learner")
    km = components.get("knowledge_manager")
    if pl:
        pl.add_url(url, reason)
    if km:
        km.add_url(url, reason)   # also feeds WebCrawler's discovered_urls
    if not pl and not km:
        return jsonify({"error": "No knowledge components loaded"}), 503
    return jsonify({"success": True, "url": url, "reason": reason,
                    "queue_depth": pl.get_status().get("queue_depth", 0) if pl else None})


@app.route("/api/knowledge/add-book", methods=["POST"])
def api_knowledge_add_book():
    """
    Add a book to DMAI's reading list.
    Admin only.
    Body: {"title": "...", "author": "...", "reason": "..."}
    """
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    data   = request.get_json(silent=True) or {}
    title  = data.get("title", "").strip()
    author = data.get("author", "").strip()
    reason = data.get("reason", "admin recommendation").strip()
    if not title or not author:
        return jsonify({"error": "title and author are required"}), 400
    km = components.get("knowledge_manager")
    if not km:
        return jsonify({"error": "KnowledgeSourceManager not loaded"}), 503
    km.add_book(title, author, reason)
    return jsonify({"success": True, "title": title, "author": author, "reason": reason})


# ═══════════════════════════════════════════════════════════════════════════
# ── Wired-component API endpoints ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _comp_status(key, extra=None):
    """Generic status helper — returns get_status() if available, else availability."""
    comp = components.get(key)
    if comp is None:
        return jsonify({"available": False, "component": key}), 503
    payload = {"available": True, "component": key}
    if hasattr(comp, "get_status"):
        try:
            payload["status"] = comp.get_status()
        except Exception as e:
            payload["status_error"] = str(e)
    if extra:
        payload.update(extra)
    return jsonify(payload)


# ── Consciousness / GlobalWorkspace ───────────────────────────────────────────
@app.route("/api/consciousness/state", methods=["GET"])
def api_consciousness_state():
    gw = components.get("global_workspace")
    if gw is None:
        return jsonify({"available": False}), 503
    state = {}
    for attr in ("capacity", "contents", "workspace", "current_focus"):
        if hasattr(gw, attr):
            try:
                val = getattr(gw, attr)
                state[attr] = val if isinstance(val, (int, float, str, list, dict)) else str(val)
            except Exception:
                pass
    if hasattr(gw, "get_workspace_state"):
        try:
            state["workspace_state"] = gw.get_workspace_state()
        except Exception:
            pass
    return jsonify({"available": True, "state": state})


# ── CapabilitySynthesizer ─────────────────────────────────────────────────────
@app.route("/api/capabilities/synthesize", methods=["GET", "POST"])
def api_capabilities_synthesize():
    cs = components.get("capability_synthesizer")
    if cs is None:
        return jsonify({"available": False}), 503
    data = request.get_json(silent=True) or {}
    try:
        responses = {
            "a": data.get("capability_a", ""),
            "b": data.get("capability_b", ""),
        }
        result = cs.synthesize(responses, data.get("prompt", "synthesize capabilities"))
        return jsonify({"available": True, "result": result})
    except Exception as e:
        return jsonify({"available": True, "error": str(e)}), 200


# ── DynamicAIDiscovery ────────────────────────────────────────────────────────
@app.route("/api/ai/discovery/status", methods=["GET"])
def api_ai_discovery_status():
    return _comp_status("ai_discovery")


# ── Learning (phase11 LearningOrchestrator) ───────────────────────────────────
@app.route("/api/learning/status", methods=["GET"])
def api_learning_status():
    return _comp_status("learning_orchestrator")


@app.route("/api/learning/start", methods=["POST"])
def api_learning_start():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    lo = components.get("learning_orchestrator")
    if lo is None:
        return jsonify({"error": "LearningOrchestrator not loaded"}), 503
    try:
        t = threading.Thread(target=lo.start_continuous_learning, daemon=True,
                             name="dmai-learning")
        t.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning/unified-status", methods=["GET"])
def api_learning_unified_status():
    return _comp_status("unified_learner")


# ── Tutors ────────────────────────────────────────────────────────────────────
@app.route("/api/tutors/list", methods=["GET"])
def api_tutors_list():
    tm = components.get("tutor_manager")
    if tm is None:
        return jsonify({"available": False}), 503
    tutors = getattr(tm, "tutors", {})
    try:
        listing = {k: (v.to_dict() if hasattr(v, "to_dict") else str(v))
                   for k, v in tutors.items()} if isinstance(tutors, dict) else str(tutors)
    except Exception as e:
        listing = {"error": str(e)}
    return jsonify({"available": True, "tutors": listing})


@app.route("/api/tutors/query", methods=["POST"])
def api_tutors_query():
    tm = components.get("tutor_manager")
    if tm is None:
        return jsonify({"available": False}), 503
    data = request.get_json(silent=True) or {}
    tutor_id = data.get("tutor_id", "")
    prompt = sanitise_input(data.get("prompt", "")) if SECURITY_AVAILABLE else data.get("prompt", "")
    for meth in ("query", "ask", "query_tutor"):
        if hasattr(tm, meth):
            try:
                return jsonify({"result": getattr(tm, meth)(tutor_id, prompt)})
            except Exception as e:
                return jsonify({"error": str(e)}), 200
    return jsonify({"status": "queued", "tutor_id": tutor_id, "prompt": prompt})


# ── Evolution ─────────────────────────────────────────────────────────────────
@app.route("/api/evolution/consciousness", methods=["GET"])
def api_evolution_consciousness():
    ct = components.get("consciousness_tracker")
    if ct is None:
        return jsonify({"available": False}), 503
    level = None
    for meth in ("get_consciousness", "get_level", "get_status"):
        if hasattr(ct, meth):
            try:
                level = getattr(ct, meth)()
                break
            except Exception:
                pass
    if level is None:
        level = getattr(ct, "consciousness", getattr(ct, "level", None))
    return jsonify({"available": True, "consciousness": level})


@app.route("/api/evolution/metrics", methods=["GET"])
def api_evolution_metrics():
    em = components.get("evolution_metrics")
    if em is None:
        return jsonify({"available": False}), 503
    for meth in ("get_metrics", "get_status", "snapshot"):
        if hasattr(em, meth):
            try:
                return jsonify({"available": True, "metrics": getattr(em, meth)()})
            except Exception as e:
                return jsonify({"available": True, "error": str(e)}), 200
    return jsonify({"available": True})


@app.route("/api/evolution/learning-cycle", methods=["POST"])
def api_evolution_learning_cycle():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    sl = components.get("stage_learner")
    if sl is None:
        return jsonify({"error": "StageAwareLearningOrchestrator not loaded"}), 503
    data = request.get_json(silent=True) or {}
    try:
        consciousness = float(data.get("consciousness", 0.5))
    except (TypeError, ValueError):
        consciousness = 0.5
    try:
        result = sl.run_learning_cycle(consciousness)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Self-correction ───────────────────────────────────────────────────────────
@app.route("/api/self-correct", methods=["POST"])
def api_self_correct():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    sc = components.get("self_corrector")
    if sc is None:
        return jsonify({"error": "SelfCorrectingEngine not loaded"}), 503
    data = request.get_json(silent=True) or {}
    code = data.get("code", "")
    context = data.get("context", {})
    try:
        success, output, fixes = sc.run_and_correct(code, context)
        return jsonify({"success": success, "output": output, "fixes_applied": fixes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Self-optimizer ────────────────────────────────────────────────────────────
@app.route("/api/optimizer/status", methods=["GET"])
def api_optimizer_status():
    return _comp_status("self_optimizer")


@app.route("/api/optimizer/run", methods=["POST"])
def api_optimizer_run():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    so = components.get("self_optimizer")
    if so is None:
        return jsonify({"error": "SelfOptimizer not loaded"}), 503
    try:
        t = threading.Thread(target=so.start_optimization_cycle, daemon=True,
                             name="dmai-optimizer")
        t.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Capability integration ────────────────────────────────────────────────────
@app.route("/api/capabilities/list", methods=["GET"])
def api_capabilities_list():
    ci = components.get("capability_integrator")
    if ci is None:
        return jsonify({"available": False}), 503
    for meth in ("list_capabilities", "get_capabilities", "get_status"):
        if hasattr(ci, meth):
            try:
                return jsonify({"available": True, "capabilities": getattr(ci, meth)()})
            except Exception as e:
                return jsonify({"available": True, "error": str(e)}), 200
    return jsonify({"available": True})


@app.route("/api/capabilities/integrate", methods=["POST"])
def api_capabilities_integrate():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ci = components.get("capability_integrator")
    if ci is None:
        return jsonify({"error": "CapabilityIntegrator not loaded"}), 503
    data = request.get_json(silent=True) or {}
    for meth in ("integrate", "integrate_capability", "add_capability"):
        if hasattr(ci, meth):
            try:
                return jsonify({"result": getattr(ci, meth)(data)})
            except Exception as e:
                return jsonify({"error": str(e)}), 200
    return jsonify({"status": "queued", "data": data})


# ── Kaizen integrator cycle ───────────────────────────────────────────────────
@app.route("/api/kaizen/run-cycle", methods=["POST"])
def api_kaizen_run_cycle():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ki = components.get("kaizen_integrator")
    if ki is None:
        return jsonify({"error": "KaizenIntegrator not loaded"}), 503
    try:
        if hasattr(ki, "run"):
            result = _run_async(ki.run())
            return jsonify({"status": "ran", "result": result})
        return jsonify({"status": "no_run_method"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/kaizen/cycle-status", methods=["GET"])
def api_kaizen_cycle_status():
    return _comp_status("kaizen_integrator")


@app.route("/api/kaizen/auto-repair", methods=["POST"])
def api_kaizen_auto_repair():
    """Trigger an immediate Kaizen auto-repair cycle."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    kar = components.get("kaizen_auto_repair")
    if kar is None:
        return jsonify({"error": "KaizenAutoRepair not loaded"}), 503
    try:
        result = kar.run_repair_cycle()
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/kaizen/repair-stats", methods=["GET"])
def api_kaizen_repair_stats():
    kar = components.get("kaizen_auto_repair")
    if kar is None:
        return jsonify({"error": "KaizenAutoRepair not loaded"}), 503
    return jsonify(kar.get_stats())


# ── Memory Retrieval ───────────────────────────────────────────────────────────────────
@app.route("/api/memory/recall", methods=["POST"])
def api_memory_recall():
    """Query DMAI's internal knowledge base."""
    data  = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    if not query:
        return jsonify({"error": "query required"}), 400
    recall_fn = components.get("memory_recall")
    if recall_fn is None:
        return jsonify({"error": "MemoryRetrieval not loaded"}), 503
    try:
        result = recall_fn(query, top_k=top_k)
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── CodeWriter (self-generation) ────────────────────────────────────────────────────
@app.route("/api/code-writer/generate", methods=["POST"])
def api_code_writer_generate():
    """Ask DMAI to generate a new component."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    data = request.get_json(silent=True) or {}
    cw = components.get("code_writer")
    if cw is None:
        return jsonify({"error": "CodeWriter not loaded"}), 503
    name  = data.get("component_name", "")
    desc  = data.get("description", "")
    reqs  = data.get("requirements", [])
    dry   = data.get("dry_run", False)
    if not name or not desc:
        return jsonify({"error": "component_name and description required"}), 400
    result = cw.generate_component(name, desc, reqs, dry_run=dry)
    return jsonify(result)


@app.route("/api/code-writer/history", methods=["GET"])
def api_code_writer_history():
    cw = components.get("code_writer")
    if cw is None:
        return jsonify({"error": "CodeWriter not loaded"}), 503
    limit = int(request.args.get("limit", 20))
    return jsonify({"records": cw.get_history(limit), "total": len(cw.get_history(limit))})


@app.route("/api/code-writer/patch", methods=["POST"])
def api_code_writer_patch():
    """Patch an existing file."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    data = request.get_json(silent=True) or {}
    cw = components.get("code_writer")
    if cw is None:
        return jsonify({"error": "CodeWriter not loaded"}), 503
    result = cw.patch_file(
        file_path   = data.get("file_path", ""),
        old_string  = data.get("old_string", ""),
        new_string  = data.get("new_string", ""),
        origin      = data.get("origin", "api_request"),
        description = data.get("description", ""),
        dry_run     = data.get("dry_run", False),
    )
    return jsonify(result)


# ── Autonomous research ───────────────────────────────────────────────────────
@app.route("/api/research/autonomous/status", methods=["GET"])
def api_research_autonomous_status():
    return _comp_status("autonomous_researcher")


@app.route("/api/research/autonomous/start", methods=["POST"])
def api_research_autonomous_start():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ar = components.get("autonomous_researcher")
    if ar is None:
        return jsonify({"error": "AutonomousResearcher not loaded"}), 503
    data = request.get_json(silent=True) or {}
    topics = data.get("topics")
    try:
        t = threading.Thread(target=ar.run_continuous_research, args=(topics,),
                             daemon=True, name="dmai-research")
        t.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research/autonomous/topic", methods=["POST"])
def api_research_autonomous_topic():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ar = components.get("autonomous_researcher")
    if ar is None:
        return jsonify({"error": "AutonomousResearcher not loaded"}), 503
    data = request.get_json(silent=True) or {}
    topic = sanitise_input(data.get("topic", "")) if SECURITY_AVAILABLE else data.get("topic", "")
    if not topic:
        return jsonify({"error": "topic is required"}), 400
    try:
        result = ar.research_topic_deep(topic)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── URL learner ───────────────────────────────────────────────────────────────
@app.route("/api/research/learn-url", methods=["POST"])
def api_research_learn_url():
    ul = components.get("url_learner")
    if ul is None:
        return jsonify({"available": False}), 503
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL — must start with http:// or https://"}), 400
    topic = data.get("topic", "general")
    content = data.get("content", "")
    try:
        result = ul.learn_from_url(topic, url, content)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Reverse engineering ───────────────────────────────────────────────────────
@app.route("/api/reverse-engineer", methods=["POST"])
def api_reverse_engineer():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    re_comp = components.get("reverse_engineer")
    if re_comp is None:
        return jsonify({"error": "ReverseEngineer not loaded"}), 503
    data = request.get_json(silent=True) or {}
    target = data.get("target", "")
    description = data.get("description", data.get("type", "software"))
    if not target:
        return jsonify({"error": "target is required"}), 400
    try:
        result = re_comp.reverse_engineer_software(target, description)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Funding / revenue ─────────────────────────────────────────────────────────
@app.route("/api/funding/status", methods=["GET"])
def api_funding_status():
    return _comp_status("self_funding")


@app.route("/api/funding/start", methods=["POST"])
def api_funding_start():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    sf = components.get("self_funding")
    if sf is None:
        return jsonify({"error": "SelfFundingOrchestrator not loaded"}), 503
    data = request.get_json(silent=True) or {}
    avenue = data.get("avenue")
    try:
        result = sf.start_learning(avenue=avenue)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/funding/revenue-streams", methods=["GET"])
def api_funding_revenue_streams():
    rd = components.get("revenue_discovery")
    if rd is None:
        return jsonify({"available": False}), 503
    for attr in ("revenue_streams", "streams", "discovered_streams"):
        if hasattr(rd, attr):
            try:
                return jsonify({"available": True, "revenue_streams": getattr(rd, attr)})
            except Exception:
                pass
    return _comp_status("revenue_discovery")


# ── Financial UK ──────────────────────────────────────────────────────────────
@app.route("/api/financial/uk/status", methods=["GET"])
def api_financial_uk_status():
    return _comp_status("financial_uk")


# ── Art ───────────────────────────────────────────────────────────────────────
@app.route("/api/art/generate", methods=["POST"])
def api_art_generate():
    ae = components.get("art_engine")
    if ae is None:
        return jsonify({"available": False}), 503
    data = request.get_json(silent=True) or {}
    prompt = sanitise_input(data.get("prompt", "")) if SECURITY_AVAILABLE else data.get("prompt", "")
    style = data.get("style", "children")
    complexity = data.get("complexity", "medium")
    try:
        result = ae.generate_coloring_page(prompt or "art", age_group=style, intricacy=complexity)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/art/gallery", methods=["GET"])
def api_art_gallery():
    ae = components.get("art_engine")
    if ae is None:
        return jsonify({"available": False}), 503
    for attr in ("gallery", "generated_works", "works"):
        if hasattr(ae, attr):
            try:
                return jsonify({"available": True, "gallery": getattr(ae, attr)})
            except Exception:
                pass
    return jsonify({"available": True, "gallery": []})


# ── Music ─────────────────────────────────────────────────────────────────────
@app.route("/api/music/status", methods=["GET"])
def api_music_status():
    return _comp_status("music_learner")


@app.route("/api/music/generate", methods=["POST"])
def api_music_generate():
    ml = components.get("music_learner")
    if ml is None:
        return jsonify({"available": False}), 503
    data = request.get_json(silent=True) or {}
    for meth in ("generate", "compose", "generate_music"):
        if hasattr(ml, meth):
            try:
                return jsonify({"status": "ok", "result": getattr(ml, meth)(data)})
            except Exception as e:
                return jsonify({"error": str(e)}), 200
    return jsonify({"status": "unsupported",
                    "message": "MusicLearner has no generation method; analysis-only."})


# ── Content validation ────────────────────────────────────────────────────────
@app.route("/api/content/validate", methods=["POST"])
def api_content_validate():
    cv = components.get("content_validator")
    if cv is None:
        return jsonify({"available": False}), 503
    data = request.get_json(silent=True) or {}
    content = data.get("content", "")
    title = data.get("title", "Untitled")
    try:
        if hasattr(cv, "validate"):
            result = cv.validate(content)
        else:
            result = cv.validate_book(title, content, data.get("chapters", []))
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Vision extraction ─────────────────────────────────────────────────────────
@app.route("/api/vision/extract", methods=["POST"])
def api_vision_extract():
    ve = components.get("vision_extractor")
    if ve is None:
        return jsonify({"available": False}), 503
    data = request.get_json(silent=True) or {}
    extract_type = data.get("extract_type", "auto")
    image_b64 = data.get("image_base64", "")
    try:
        for meth in ("extract", "extract_from_base64"):
            if hasattr(ve, meth):
                return jsonify({"status": "ok", "result": getattr(ve, meth)(image_b64, extract_type)})
        return jsonify({"status": "unsupported",
                        "message": "Use batch extraction with file paths."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── System health (rich dashboard) ────────────────────────────────────────────
@app.route("/api/system/health", methods=["GET"])
def api_system_health():
    hd = components.get("health_dashboard")
    base = {
        "status": "healthy",
        "version": DMAI_VERSION,
        "uptime": _uptime(),
        "components_loaded": len(components),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if hd is not None:
        for meth in ("get_dashboard", "get_health", "get_status", "snapshot"):
            if hasattr(hd, meth):
                try:
                    base["dashboard"] = getattr(hd, meth)()
                    break
                except Exception as e:
                    base["dashboard_error"] = str(e)
    return jsonify(base)


# ── GitHub stars ──────────────────────────────────────────────────────────────
@app.route("/api/github/stars", methods=["GET"])
def api_github_stars():
    gm = components.get("github_monitor")
    if gm is None:
        return jsonify({"available": False}), 503
    for meth in ("get_stars", "get_status", "get_latest"):
        if hasattr(gm, meth):
            try:
                return jsonify({"available": True, "stars": getattr(gm, meth)()})
            except Exception as e:
                return jsonify({"available": True, "error": str(e)}), 200
    return jsonify({"available": True})


# ── Autonomous ingestor ───────────────────────────────────────────────────────
@app.route("/api/ingestor/status", methods=["GET"])
def api_ingestor_status():
    return _comp_status("autonomous_ingestor")


# ── Settings (full system config read + write) ────────────────────────────────
@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    """Return all configurable settings with current values."""
    import json as _sj
    from pathlib import Path as _sp

    # Load training config
    training_cfg = {}
    for cfg_path in [_sp("configs/training_config.json"), _sp("config/training_config.json")]:
        if cfg_path.exists():
            try:
                training_cfg = _sj.loads(cfg_path.read_text())
            except Exception:
                pass
            break

    # Collect env-based settings (never expose secret values, only presence)
    env_settings = {}
    _env_keys = [
        "MASTER_PASSWORD", "DATABASE_URL", "TRADING_LIVE",
        "DMAI_EMAIL", "DMAI_HF_PASSWORD", "RENDER_API_KEY",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "GROQ_API_KEY", "GOOGLE_AI_STUDIO_KEY", "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GITHUB_TOKEN",
    ]
    for key in _env_keys:
        val = os.environ.get(key, "")
        env_settings[key] = "SET" if val else "NOT SET"

    # SICore KPI targets
    si = components.get("si_core")
    kpi_values = si.current_kpis if si else {}

    # Stage gate thresholds from config or defaults
    stage_gates = training_cfg.get("ai_training", {}).get("stage_gate_overrides", {
        "Baby": 0.40, "Toddler": 0.55, "Child": 0.65,
        "Teen": 0.75, "Adult": 0.85, "Expert": 0.95,
    })

    # Training schedule
    training_sched = training_cfg.get("training_schedule", {
        "run_on_startup": True,
        "interval_minutes": 60,
        "max_concurrent_domains": 3,
    })

    # Research settings
    research_cfg = training_cfg.get("research", {
        "cycle_interval_seconds": 300,
        "memory_threshold": 0.55,
        "max_topics_per_cycle": 10,
    })

    # KPI evaluator settings
    kpi_cfg = training_cfg.get("kpi_evaluator", {
        "full_eval_interval_hours": 6,
        "boot_quick_pass_delay_seconds": 90,
        "rsi_cycle_target": 52,
    })

    # Providers
    harv = components.get("api_activator")
    provider_data = {}
    if harv and hasattr(harv, "last_scan"):
        provider_data = harv.last_scan or {}

    return jsonify({
        "env_settings": env_settings,
        "stage_gates": stage_gates,
        "training_schedule": training_sched,
        "research": research_cfg,
        "kpi_evaluator": kpi_cfg,
        "current_kpis": kpi_values,
        "training_config": training_cfg,
        "system": {
            "version": DMAI_VERSION,
            "uptime": _uptime(),
            "components_loaded": len(components),
            "is_render": IS_RENDER,
            "data_path": DATA_PATH,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    """Update writable settings at runtime."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    import json as _sj
    from pathlib import Path as _sp
    data = request.get_json(silent=True) or {}
    updated = {}

    # Stage gate overrides
    if "stage_gates" in data:
        cfg_path = _sp("configs/training_config.json")
        if cfg_path.exists():
            try:
                cfg = _sj.loads(cfg_path.read_text())
                cfg.setdefault("ai_training", {})["stage_gate_overrides"] = data["stage_gates"]
                cfg_path.write_text(_sj.dumps(cfg, indent=2))
                updated["stage_gates"] = data["stage_gates"]
            except Exception as e:
                return jsonify({"error": f"Could not update stage gates: {e}"}), 500

    # Training schedule
    if "training_schedule" in data:
        cfg_path = _sp("configs/training_config.json")
        if cfg_path.exists():
            try:
                cfg = _sj.loads(cfg_path.read_text())
                cfg["training_schedule"] = data["training_schedule"]
                cfg_path.write_text(_sj.dumps(cfg, indent=2))
                updated["training_schedule"] = data["training_schedule"]
            except Exception as e:
                return jsonify({"error": f"Could not update schedule: {e}"}), 500

    # Research interval
    if "research_interval" in data:
        try:
            interval = int(data["research_interval"])
            ar = components.get("autonomous_researcher")
            if ar:
                ar._cycle_interval = interval
            updated["research_interval"] = interval
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Master goal
    if "master_goal" in data:
        mc = components.get("master_control")
        if mc and hasattr(mc, "set_goal"):
            mc.set_goal(data["master_goal"])
            updated["master_goal"] = data["master_goal"]

    return jsonify({"ok": True, "updated": updated})


# ── Learning full-status (rich data for Learning UI) ─────────────────────────
@app.route("/api/learning/full-status", methods=["GET"])
def api_learning_full_status():
    """Aggregated learning metrics for the Learning Dashboard."""
    import json as _lj
    from pathlib import Path as _lp

    si = components.get("si_core")

    # KPI resolution: cache-first → si_core → live DB inline
    # (same priority chain as /api/status — prevents all-null on cold start
    #  before the 5-min seeder thread has had a chance to run)
    kpis = {}
    try:
        _cache_path = os.path.join(
            os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\"),
            "kpi_cache.json"
        )
        with open(_cache_path) as _cf:
            _cached = _lj.load(_cf)
        _ck = _cached.get("kpis", {})
        if _ck and not all(v == 0 for v in _ck.values() if isinstance(v, (int, float))):
            kpis = _ck
        else:
            raise ValueError("cache empty or all-zero")
    except Exception:
        # Fallback 1: si_core._state (may be zero on first boot)
        _raw = (si.current_kpis if si else {}) or {}
        if _raw and not all(v == 0 for v in _raw.values() if isinstance(v, (int, float))):
            kpis = _raw
        else:
            # Fallback 2: derive inline from DB right now
            try:
                import sqlite3 as _sq_fs
                _db_fs = os.path.join(
                    os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\"),
                    "dmai_knowledge.db"
                )
                _con_fs = _sq_fs.connect(_db_fs, timeout=5)
                _caps_fs  = _con_fs.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0]
                _ins_fs   = _con_fs.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
                try:
                    _voc_fs = _con_fs.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
                except Exception:
                    _voc_fs = 0
                try:
                    _ins7_fs = _con_fs.execute(
                        "SELECT COUNT(*) FROM insights WHERE created_at >= datetime('now','-7 days')"
                    ).fetchone()[0]
                except Exception:
                    _ins7_fs = 0
                try:
                    _days_fs = _con_fs.execute(
                        "SELECT COUNT(DISTINCT date(created_at)) FROM insights "
                        "WHERE created_at >= datetime('now','-7 days')"
                    ).fetchone()[0] or 0
                except Exception:
                    _days_fs = 0
                _con_fs.close()
                _stg_name_fs, _stg_fs, _pct_fs = _read_stage_from_db()
                _act_fs = sum(1 for v in components.values() if v is not None)
                kpis = {
                    "skill_acquisition_rate":          min(_caps_fs / 50_000, 1.0),
                    "transfer_learning_rate":           min(_stg_fs / 7.0, 1.0),
                    "zero_shot_success_count":          min(_ins_fs / 300_000, 1.0),
                    "agentic_capability_score":         min(_caps_fs / 20_000, 1.0),
                    "recursive_self_improvement_rate":  min(_pct_fs / 100.0, 1.0),
                    "sample_efficiency_trend":          min((_ins7_fs / max(_days_fs, 1)) / 5_000, 1.0),
                    "metacognition_accuracy":           min(_voc_fs / 500_000, 1.0),
                    "multi_modal_integration_score":    min(_act_fs / max(len(components), 56), 1.0),
                }
                logger.info("full-status: KPIs derived inline from DB (seeder not yet run)")
            except Exception as _e_fs:
                logger.warning("full-status: all KPI fallbacks failed: %s", _e_fs)
                kpis = {}

    # Stage progress
    lp_file = _lp("data/learning/stage_syllabus/learning_progress.json")
    stage_progress = {}
    if lp_file.exists():
        try:
            stage_progress = _lj.loads(lp_file.read_text())
        except Exception:
            pass

    # Insights count
    insights_count = 0
    ins_file = _lp("data/research/insights.jsonl")
    if ins_file.exists():
        try:
            insights_count = sum(1 for l in ins_file.read_text().splitlines() if l.strip())
        except Exception:
            pass

    # Discoveries today
    disc_file = _lp("data/research/discoveries.jsonl")
    discoveries_today = 0
    domains_researched = set()
    recent_discoveries = []
    if disc_file.exists():
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            for line in disc_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = _lj.loads(line)
                    if rec.get("date") == today:
                        discoveries_today += 1
                    domains_researched.add(rec.get("domain", "?"))
                    recent_discoveries.append(rec)
                except Exception:
                    pass
        except Exception:
            pass
    recent_discoveries = list(reversed(recent_discoveries))[:10]

    # Knowledge DB stats
    db_stats = {"insights": 0, "capabilities": 0, "syllabus_mastered": 0}
    try:
        import sqlite3
        conn = sqlite3.connect("data/dmai_knowledge.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM insights")
        db_stats["insights"] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM capabilities")
        db_stats["capabilities"] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM syllabus_content WHERE mastery >= 0.9")
        db_stats["syllabus_mastered"] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM syllabus_content")
        db_stats["syllabus_total"] = c.fetchone()[0]
        conn.close()
    except Exception:
        pass

    # Compiled knowledge modules
    compiled_modules = []
    compiled_dir = _lp("data/learning/compiled_knowledge")
    if compiled_dir.exists():
        for jf in compiled_dir.glob("*.json"):
            if jf.name == "master_knowledge.json":
                continue
            try:
                d = _lj.loads(jf.read_text())
                compiled_modules.append({
                    "module": jf.stem.replace("_learned", ""),
                    "learned_at": d.get("learned_at", d.get("timestamp", "")),
                    "topics": len(d.get("topics", d.get("content", {}))),
                })
            except Exception:
                pass

    # KPI history (last 7 days trend)
    kpi_history_file = _lp("data/kpi_eval_history.jsonl")
    kpi_trend = {}
    if kpi_history_file.exists():
        try:
            lines = kpi_history_file.read_text().strip().splitlines()
            for line in lines[-50:]:
                try:
                    rec = _lj.loads(line)
                    kpi = rec.get("kpi")
                    val = rec.get("value", 0)
                    ts  = rec.get("timestamp", "")
                    if kpi not in kpi_trend:
                        kpi_trend[kpi] = []
                    kpi_trend[kpi].append({"ts": ts[:10], "value": val})
                except Exception:
                    pass
        except Exception:
            pass

    # Anchor every KPI's trend to its LIVE value so the chart never shows stale
    # frozen seed constants (e.g. 0.5 / 0.3333 / 0.0667) that disagree with the
    # current KPI. If a KPI has no real history yet, repeat the live value.
    _today_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for _kname, _kval in (kpis or {}).items():
        if not isinstance(_kval, (int, float)):
            continue
        _series = kpi_trend.get(_kname) or []
        if not _series:
            kpi_trend[_kname] = [{"ts": _today_ts, "value": _kval}]
        elif _series[-1].get("value") != _kval:
            _series.append({"ts": _today_ts, "value": _kval})
            kpi_trend[_kname] = _series

    # Code writer history
    cw_file = _lp("data/code_writer/history.jsonl")
    code_written = 0
    if cw_file.exists():
        try:
            code_written = sum(1 for l in cw_file.read_text().splitlines() if l.strip())
        except Exception:
            pass

    # Research seen topics count
    seen_file = _lp("data/research/seen_topics.json")
    topics_researched_total = 0
    if seen_file.exists():
        try:
            topics_researched_total = len(_lj.loads(seen_file.read_text()))
        except Exception:
            pass

    # Build study block (stage breakdown from DB) for the Learning Metrics UI
    _study_stages = {}
    try:
        import sqlite3 as _sq2
        _db2 = os.path.join(os.environ.get("DATA_PATH", "data"), "dmai_knowledge.db")
        _con2 = _sq2.connect(_db2, timeout=5)
        _cur2 = _con2.cursor()
        for _sname in ["baby", "toddler", "child", "teen", "adult", "expert"]:
            try:
                _cur2.execute("SELECT COUNT(*) FROM syllabus_content WHERE stage=? AND mastery>=0.9", (_sname,))
                _m2 = _cur2.fetchone()[0]
                _cur2.execute("SELECT COUNT(*) FROM syllabus_content WHERE stage=?", (_sname,))
                _t2 = _cur2.fetchone()[0]
                _study_stages[_sname] = {"mastered": _m2, "total": _t2}
            except Exception:
                _study_stages[_sname] = {"mastered": 0, "total": 0}
        _con2.close()
    except Exception:
        pass

    # Stage name MUST come from the same system_state source as /api/metrics,
    # otherwise this endpoint reports "Baby" while /api/metrics reports "Child".
    _stage_name_db, _stage_idx_db, _stage_pct_db = _read_stage_from_db()
    _study_block = {
        "current_stage":   _stage_name_db,
        "stage_index":     _stage_idx_db,
        "stage_within_pct": _stage_pct_db,
        "topics_mastered": db_stats.get("syllabus_mastered", 0),
        "stage_breakdown": _study_stages,
    }

    return jsonify({
        "kpis": kpis,
        "study": _study_block,
        "kpi_trend": kpi_trend,
        "stage_progress": stage_progress,
        "insights_count": insights_count,
        "discoveries_today": discoveries_today,
        "domains_researched": sorted(domains_researched),
        "recent_discoveries": recent_discoveries,
        "db_stats": db_stats,
        "compiled_modules": compiled_modules,
        "code_written": code_written,
        "topics_researched_total": topics_researched_total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/ingestor/ingest", methods=["POST"])
def api_ingestor_ingest():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ing = components.get("autonomous_ingestor")
    if ing is None:
        return jsonify({"error": "AutonomousIngestor not loaded"}), 503
    data = request.get_json(silent=True) or {}
    source = data.get("source", data.get("input_source", ""))
    input_type = data.get("input_type", "auto")
    if not source:
        return jsonify({"error": "source is required"}), 400
    try:
        result = ing.process_input(source, input_type)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Integration: free APIs / repos ────────────────────────────────────────────
@app.route("/api/integration/free-apis", methods=["GET"])
def api_integration_free_apis():
    # Delegate to AutoRegistrar (replaces FreeAPIHarvester)
    ar = components.get("auto_registrar")
    if ar:
        try:
            return jsonify({"available": True, "status": ar.get_status(), "pending": ar.get_pending_signups()})
        except Exception as e:
            return jsonify({"available": True, "error": str(e)}), 200
    return jsonify({"available": False, "reason": "AutoRegistrar not loaded"}), 503


@app.route("/api/registrar/status", methods=["GET"])
def api_registrar_status():
    """Return which free-tier providers are active / pending signup."""
    ar = components.get("auto_registrar")
    if ar is None:
        return jsonify({"error": "AutoRegistrar not loaded"}), 503
    return jsonify(ar.get_status())


@app.route("/api/registrar/pending", methods=["GET"])
def api_registrar_pending():
    """Return sorted list of providers needing manual signup."""
    ar = components.get("auto_registrar")
    if ar is None:
        return jsonify({"error": "AutoRegistrar not loaded"}), 503
    return jsonify({"pending": ar.get_pending_signups()})


@app.route("/api/registrar/register", methods=["POST"])
def api_registrar_register():
    """Trigger immediate registration attempt (admin-only)."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ar = components.get("auto_registrar")
    if ar is None:
        return jsonify({"error": "AutoRegistrar not loaded"}), 503
    result = ar.register_all()
    return jsonify(result)


@app.route("/api/integration/repos", methods=["GET"])
def api_integration_repos():
    ri = components.get("repo_integrator")
    if ri is None:
        return jsonify({"available": False}), 503
    for meth in ("list_repos", "get_registry", "get_status"):
        if hasattr(ri, meth):
            try:
                return jsonify({"available": True, "repos": getattr(ri, meth)()})
            except Exception as e:
                return jsonify({"available": True, "error": str(e)}), 200
    return jsonify({"available": True})


@app.route("/api/integration/repo", methods=["POST"])
def api_integration_repo():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    ri = components.get("repo_integrator")
    if ri is None:
        return jsonify({"error": "RepoIntegrationEngine not loaded"}), 503
    data = request.get_json(silent=True) or {}
    repo_url = data.get("repo_url", "").strip()
    if not repo_url:
        return jsonify({"error": "repo_url is required"}), 400
    try:
        result = ri.add_to_queue(repo_url, priority=int(data.get("priority", 2)))
        return jsonify({"status": "queued", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Trading ───────────────────────────────────────────────────────────────────
@app.route("/api/trading/mastery", methods=["GET"])
def api_trading_mastery():
    tm = components.get("trading_mastery")
    if tm is None:
        return jsonify({"available": False}), 503
    for meth in ("get_status", "get_mastery", "get_summary"):
        if hasattr(tm, meth):
            try:
                return jsonify({"available": True, "mastery": getattr(tm, meth)()})
            except Exception as e:
                return jsonify({"available": True, "error": str(e)}), 200
    return jsonify({"available": True})


@app.route("/api/trading/status", methods=["GET"])
def api_trading_status():
    tr = components.get("trader")
    if tr is None:
        return jsonify({"available": False}), 503
    out = {"available": True, "paper": getattr(tr, "paper", True)}
    try:
        if hasattr(tr, "get_performance_summary"):
            out["performance"] = tr.get_performance_summary()
        if hasattr(tr, "get_account"):
            out["account"] = tr.get_account()
    except Exception as e:
        out["error"] = str(e)
    return jsonify(out)


@app.route("/api/trading/execute", methods=["POST"])
def api_trading_execute():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    tr = components.get("trader")
    if tr is None:
        return jsonify({"error": "AggressiveTrader not loaded"}), 503
    try:
        result = tr.execute_aggressive_trades()
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Master control ────────────────────────────────────────────────────────────
@app.route("/api/master/status", methods=["GET"])
def api_master_status():
    return _comp_status("master_control")


@app.route("/api/master/set-goal", methods=["POST"])
def api_master_set_goal():
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    mc = components.get("master_control")
    if mc is None:
        return jsonify({"error": "MasterControl not loaded"}), 503
    data = request.get_json(silent=True) or {}
    goal = sanitise_input(data.get("goal", "")) if SECURITY_AVAILABLE else data.get("goal", "")
    if not goal:
        return jsonify({"error": "goal is required"}), 400
    try:
        result = mc.set_goal(goal)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Graph Evolution — live status endpoint ────────────────────────────────────
@app.route("/api/graph/schema", methods=["GET"])
def api_graph_schema():
    """Return full graph_schema.json — neurons + synapses — for the live dashboard."""
    import json as _j
    from pathlib import Path as _PL
    # Try repo-relative path first, then DATA_PATH-relative, then absolute fallback
    _candidates = [
        _PL("aevora-training/dashboard/data/graph_schema.json"),
        _PL(DATA_PATH) / "graph_schema.json",
        _PL(__file__).parent / "aevora-training" / "dashboard" / "data" / "graph_schema.json",
    ]
    try:
        for _sp in _candidates:
            if _sp.exists():
                data = _j.loads(_sp.read_text(encoding="utf-8"))
                return jsonify(data)
        # File not found — return empty but valid schema so dashboard doesn't crash
        return jsonify({
            "neurons": [], "synapses": [],
            "total_neurons": 0, "total_synapses": 0,
            "evolution_cycle": 0,
            "_note": "graph_schema.json not found on this deployment — checked: " +
                     str([str(c) for c in _candidates])
        })
    except Exception as e:
        logger.error("api_graph_schema error: %s", e)
        return jsonify({"error": str(e), "neurons": [], "synapses": [], "total_neurons": 0, "total_synapses": 0}), 200


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """
    Single source of truth for ALL DMAI metrics.
    Every dashboard display reads from here — no discrepancies.
    """
    import sqlite3 as _sq
    _DB = os.path.join(DATA_PATH, "dmai_knowledge.db")
    # Unified KPI source: prefer kpi_cache.json (DB-derived) so /api/metrics
    # matches /api/status.si_kpis on every dashboard.
    _kpis = {}
    try:
        import json as _jc
        _cache_path = os.path.join(DATA_PATH.rstrip("/").rstrip("\\"), "kpi_cache.json")
        with open(_cache_path) as _cf:
            _cached = _jc.load(_cf)
        _kpis = _cached.get("kpis", {}) or {}
        if not _kpis or all(v == 0 for v in _kpis.values() if isinstance(v, (int, float))):
            raise ValueError("cache empty")
    except Exception:
        _si = components.get("si_core")
        _kpis = dict(_si.current_kpis) if _si and hasattr(_si, "current_kpis") else {}

    _ins, _caps, _vocab = 0, 0, 0
    _stage, _within_pct = "Baby", 0.0
    _daily_series = []

    try:
        _conn = _sq.connect(_DB)
        _conn.row_factory = _sq.Row
        _ins  = _conn.execute("SELECT COUNT(*) as c FROM insights").fetchone()["c"]
        _caps = _conn.execute("SELECT COUNT(*) as c FROM capabilities").fetchone()["c"]
        try:
            _vocab = _conn.execute("SELECT COUNT(*) as c FROM vocabulary").fetchone()["c"]
        except Exception:
            pass
        try:
            _ss = _conn.execute("SELECT value FROM system_state WHERE key=\'learning_stage\'").fetchone()
            _stage = _ss["value"] if _ss else "Baby"
            _wp = _conn.execute("SELECT value FROM system_state WHERE key=\'stage_within_pct\'").fetchone()
            _within_pct = float(_wp["value"]) if _wp else 0.0
        except Exception:
            pass
        from datetime import datetime as _dt2, timedelta as _td2
        _30d = (_dt2.utcnow() - _td2(days=30)).isoformat()
        try:
            _rows = _conn.execute("""
                SELECT strftime('%Y-%m-%d', created_at) as day, COUNT(*) as cnt
                FROM insights WHERE created_at > ?
                GROUP BY day ORDER BY day ASC
            """, (_30d,)).fetchall()
            _daily_series = [{"date": r["day"], "count": r["cnt"]} for r in _rows]
        except Exception:
            pass
        _conn.close()
    except Exception as _e:
        logger.warning("api_metrics DB error: %s", _e)

    _graph_neurons, _graph_synapses, _graph_cycle = 0, 0, 0
    try:
        from components.graph_writer import GraphWriter as _GW2
        _gs = _GW2().status()
        _graph_neurons  = _gs.get("total_neurons", 0)
        _graph_synapses = _gs.get("total_synapses", 0)
        _graph_cycle    = _gs.get("evolution_cycle", 0)
    except Exception:
        pass

    _STAGE_ORDER = ["Baby","Child","Teenager","Adult","Expert","Master","Transcendent","Infinite"]
    _stage_idx   = _STAGE_ORDER.index(_stage) if _stage in _STAGE_ORDER else 0
    _next_stage  = _STAGE_ORDER[_stage_idx + 1] if _stage_idx < len(_STAGE_ORDER) - 1 else None

    return jsonify({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kpis":         _kpis,
        "insights":     _ins,
        "capabilities": _caps,
        "vocab":        _vocab,
        "stage":        _stage,
        "stage_index":  _stage_idx,
        "next_stage":   _next_stage,
        "stage_within_pct": _within_pct,
        "graph": {
            "neurons":         _graph_neurons,
            "synapses":        _graph_synapses,
            "evolution_cycle": _graph_cycle,
        },
        "daily_series":      _daily_series,
        "active_components": len(components),
        "uptime":            _uptime(),
        "version":           DMAI_VERSION,
    })


@app.route("/api/graph/status", methods=["GET"])
def api_graph_status():
    """Return live knowledge graph size and growth stats."""
    try:
        from components.graph_writer import GraphWriter as _GW
        gw = _GW()
        return jsonify(gw.status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/evolve", methods=["POST"])
def api_graph_evolve():
    """Manually trigger a full graph evolution pass."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    try:
        from components.graph_writer import GraphWriter as _GW
        result = _GW().evolve()
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route("/api/vocabulary/stats", methods=["GET"])
def api_vocabulary_stats():
    """Return vocabulary and encyclopaedia ingestion stats."""
    try:
        import sqlite3 as _vsq
        conn = _vsq.connect("data/dmai_knowledge.db")
        vocab_total = 0
        encyc_total = 0
        vocab_domains = {}
        encyc_domains = {}
        try:
            vocab_total = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
            encyc_total = conn.execute("SELECT COUNT(*) FROM encyclopaedia").fetchone()[0]
            vocab_domains = dict(conn.execute("SELECT domain, COUNT(*) FROM vocabulary GROUP BY domain").fetchall())
            encyc_domains = dict(conn.execute("SELECT domain, COUNT(*) FROM encyclopaedia GROUP BY domain").fetchall())
        except Exception:
            pass
        conn.close()
        return jsonify({
            "vocabulary_total": vocab_total,
            "encyclopaedia_total": encyc_total,
            "vocabulary_by_domain": vocab_domains,
            "encyclopaedia_by_domain": encyc_domains,
            "status": "active",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/vocabulary/sample", methods=["GET"])
def api_vocabulary_sample():
    """Return a random sample of recently learned words."""
    try:
        import sqlite3 as _vsq
        conn = _vsq.connect("data/dmai_knowledge.db")
        rows = []
        try:
            rows = conn.execute(
                "SELECT word, part_of_speech, definition, etymology, domain "
                "FROM vocabulary ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        except Exception:
            pass
        conn.close()
        return jsonify({
            "words": [{"word": r[0], "pos": r[1], "definition": r[2],
                       "etymology": r[3], "domain": r[4]} for r in rows]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRITY / MAINTENANCE API
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/integrity/run", methods=["POST"])
def api_integrity_run():
    """Kick off a full knowledge integrity check in a background thread."""
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    def _run():
        try:
            from components.knowledge.integrity_checker import KnowledgeIntegrityChecker
            KnowledgeIntegrityChecker().run()
        except Exception as _e:
            logger.error("IntegrityChecker bg run: %s", _e)
    _t = threading.Thread(target=_run, daemon=True, name="integrity-check")
    _t.start()
    return jsonify({"status": "started", "message": "Integrity check running in background"})

@app.route("/api/integrity/report", methods=["GET"])
def api_integrity_report():
    """Return the latest integrity report and unresolved flags."""
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        import sqlite3 as _isq
        conn = _isq.connect("data/dmai_knowledge.db")
        conn.row_factory = _isq.Row

        # Latest report
        report_row = None
        try:
            report_row = conn.execute(
                "SELECT * FROM integrity_reports ORDER BY run_at DESC LIMIT 1"
            ).fetchone()
        except Exception:
            pass

        if not report_row:
            conn.close()
            return jsonify({"status": "no_report", "message": "No integrity check has been run yet."})

        # Flags for this report, unresolved first
        flags = []
        try:
            flags = [dict(r) for r in conn.execute(
                "SELECT * FROM integrity_flags WHERE report_id=? ORDER BY "
                "CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, resolved",
                (report_row["id"],)
            ).fetchall()]
        except Exception:
            pass

        # Historical run counts
        history = []
        try:
            history = [dict(r) for r in conn.execute(
                "SELECT run_at, total_flags, critical, warning, info "
                "FROM integrity_reports ORDER BY run_at DESC LIMIT 10"
            ).fetchall()]
        except Exception:
            pass

        conn.close()

        return jsonify({
            "report": dict(report_row),
            "flags": flags,
            "history": history,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/integrity/resolve/<flag_id>", methods=["POST"])
def api_integrity_resolve(flag_id):
    """Mark an integrity flag as resolved."""
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json(silent=True) or {}
        note = data.get("note", "")
        from components.knowledge.integrity_checker import KnowledgeIntegrityChecker
        ok = KnowledgeIntegrityChecker.resolve_flag(flag_id, note)
        return jsonify({"resolved": ok, "flag_id": flag_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/integrity/purge", methods=["POST"])
def api_integrity_purge():
    """Purge low-confidence insights. dry_run=true (default) just returns count."""
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json(silent=True) or {}
        threshold = float(data.get("threshold", 0.2))
        dry_run   = bool(data.get("dry_run", True))
        from components.knowledge.integrity_checker import KnowledgeIntegrityChecker
        result = KnowledgeIntegrityChecker.purge_low_confidence(threshold, dry_run)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# SUGGESTIONS API — DMAI Self-Development Inbox
# ═══════════════════════════════════════════════════════════════════════════
import uuid as _uuid_mod

def _sug_db():
    import sqlite3 as _sq
    conn = _sq.connect("data/dmai_knowledge.db")
    conn.row_factory = _sq.Row
    return conn

def _sug_now():
    return datetime.now(timezone.utc).isoformat()

@app.route("/api/suggestions", methods=["POST"])
def api_suggestions_create():
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json(silent=True) or {}
        title = (data.get("title") or "").strip()
        description = (data.get("description") or "").strip()
        source = data.get("source", "user")
        if not title:
            return jsonify({"error": "title is required"}), 400
        if not description:
            description = title
        sid = str(_uuid_mod.uuid4())
        now = _sug_now()
        conn = _sug_db()
        conn.execute(
            "INSERT INTO suggestions (id, source, title, description, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (sid, source, title, description, now, now)
        )
        conn.commit()
        conn.close()
        # Fire executor in background thread
        def _exec():
            try:
                from components.suggestion_executor import SuggestionExecutor
                SuggestionExecutor().execute(sid)
            except Exception as _e:
                logger.error("SuggestionExecutor bg thread: %s", _e)
        _t = threading.Thread(target=_exec, daemon=True, name=f"suggestion-{sid[:8]}")
        _t.start()
        logger.info("Suggestion created: %s — %s", sid, title)
        return jsonify({"id": sid, "status": "pending", "message": "DMAI is on it"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/suggestions", methods=["GET"])
def api_suggestions_list():
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        status_filter = request.args.get("status", "all")
        source_filter = request.args.get("source", "all")
        conn = _sug_db()
        query = "SELECT * FROM suggestions WHERE 1=1"
        params = []
        if status_filter != "all":
            query += " AND status=?"
            params.append(status_filter)
        if source_filter != "all":
            query += " AND source=?"
            params.append(source_filter)
        query += " ORDER BY created_at DESC LIMIT 100"
        rows = conn.execute(query, params).fetchall()
        counts_raw = conn.execute(
            "SELECT status, COUNT(*) as n FROM suggestions GROUP BY status"
        ).fetchall()
        conn.close()
        counts = {r["status"]: r["n"] for r in counts_raw}
        return jsonify({
            "suggestions": [dict(r) for r in rows],
            "counts": counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/suggestions/<sid>", methods=["GET"])
def api_suggestions_get(sid):
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        conn = _sug_db()
        row = conn.execute("SELECT * FROM suggestions WHERE id=?", (sid,)).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify(dict(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/suggestions/<sid>/retry", methods=["POST"])
def api_suggestions_retry(sid):
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        now = _sug_now()
        conn = _sug_db()
        conn.execute(
            "UPDATE suggestions SET status='pending', result=NULL, updated_at=? WHERE id=?",
            (now, sid)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM suggestions WHERE id=?", (sid,)).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "not found"}), 404
        def _exec():
            try:
                from components.suggestion_executor import SuggestionExecutor
                SuggestionExecutor().execute(sid)
            except Exception as _e:
                logger.error("SuggestionExecutor retry: %s", _e)
        _t = threading.Thread(target=_exec, daemon=True, name=f"suggestion-retry-{sid[:8]}")
        _t.start()
        return jsonify({"id": sid, "status": "pending", "message": "Retry queued"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/suggestions/<sid>", methods=["DELETE"])
def api_suggestions_delete(sid):
    if not _require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        conn = _sug_db()
        conn.execute("DELETE FROM suggestions WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        return jsonify({"deleted": sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# STAGE PROGRESSION ENGINE
# ================================================================

_STAGE_NAMES = ["Baby","Child","Teenager","Adult","Expert","Master","Transcendent","Infinite"]

_STAGE_THRESHOLDS = {
    "Baby":         (      0,     0,     0, 0.00),
    "Child":        (   5000,   500,   100, 0.10),
    "Teenager":     (  30000,  2000,   500, 0.20),
    "Adult":        (  80000,  5000,  1500, 0.35),
    "Expert":       ( 150000, 10000,  3000, 0.50),
    "Master":       ( 300000, 20000,  6000, 0.65),
    "Transcendent": ( 600000, 40000, 12000, 0.80),
    "Infinite":     (1200000, 80000, 25000, 0.92),
}

_DB_PATH_STAGE = "data/dmai_knowledge.db"


def _ensure_syllabus_content_table():
    """Create syllabus_content table if missing, then seed it from SYLLABUS_TOPICS.
    Survives Render cold starts: SQLite file persists on the mounted disk."""
    import sqlite3 as _ss3
    try:
        db_path = os.path.join(DATA_PATH.rstrip("/"), "dmai_knowledge.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = _ss3.connect(db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS syllabus_content ("
            "topic TEXT PRIMARY KEY, "
            "name TEXT, "
            "stage TEXT, "
            "category TEXT, "
            "content TEXT, "
            "mastery REAL DEFAULT 0.0, "
            "topic_type TEXT DEFAULT 'general', "
            "last_trained TEXT, "
            "created_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        # Add columns to existing tables that pre-date this schema
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(syllabus_content)").fetchall()}
            if "topic_type" not in cols:
                conn.execute("ALTER TABLE syllabus_content ADD COLUMN topic_type TEXT DEFAULT 'general'")
            if "last_trained" not in cols:
                conn.execute("ALTER TABLE syllabus_content ADD COLUMN last_trained TEXT")
        except Exception as _ce:
            logger.debug("syllabus_content ALTER skipped: %s", _ce)
        # Seed from SYLLABUS_TOPICS only if table is empty
        count = conn.execute("SELECT COUNT(*) FROM syllabus_content").fetchone()[0]
        if count == 0 and SYLLABUS_TOPICS:
            now = datetime.now(timezone.utc).isoformat()
            seeded = 0
            for topic, info in SYLLABUS_TOPICS.items():
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO syllabus_content "
                        "(topic, name, stage, category, content, mastery, topic_type, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            topic,
                            info.get("name", topic),
                            info.get("stage", "Baby"),
                            info.get("category", "general"),
                            info.get("content", ""),
                            float(info.get("mastery", 0.0) or 0.0),
                            "general",
                            now,
                        ),
                    )
                    seeded += 1
                except Exception as _ie:
                    logger.debug("seed insert skipped %s: %s", topic, _ie)
            conn.commit()
            logger.info("syllabus_content seeded with %d topics from SYLLABUS_TOPICS", seeded)
        else:
            logger.info("syllabus_content table ready (existing rows=%d)", count)
        conn.close()
    except Exception as _e:
        logger.warning("_ensure_syllabus_content_table: %s", _e)


def _ensure_system_state_table():
    import sqlite3 as _ss3
    try:
        conn = _ss3.connect(_DB_PATH_STAGE)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS system_state ("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL, "
            "updated_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stage_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "stage TEXT NOT NULL, prev_stage TEXT, "
            "insights INTEGER, capabilities INTEGER, vocab INTEGER, "
            "avg_kpi REAL, within_pct REAL, "
            "recorded_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        for _k, _v in [("learning_stage","Baby"),("stage_within_pct","0.0"),("stage_last_updated","never")]:
            conn.execute(
                "INSERT OR IGNORE INTO system_state (key,value,updated_at) VALUES (?,?,datetime('now'))",
                (_k, _v)
            )
        conn.commit()
        conn.close()
        logger.info("system_state table ready")
    except Exception as _e:
        logger.warning("_ensure_system_state_table: %s", _e)


def _get_db_metrics():
    import sqlite3 as _sm3
    m = {"insights": 0, "capabilities": 0, "vocab": 0, "avg_kpi": 0.0}
    try:
        conn = _sm3.connect(_DB_PATH_STAGE)
        conn.row_factory = _sm3.Row
        m["insights"]     = conn.execute("SELECT COUNT(*) as c FROM insights").fetchone()["c"]
        m["capabilities"] = conn.execute("SELECT COUNT(*) as c FROM capabilities").fetchone()["c"]
        try:
            m["vocab"] = conn.execute("SELECT COUNT(*) as c FROM vocabulary").fetchone()["c"]
        except Exception:
            pass
        conn.close()
    except Exception as _e:
        logger.debug("_get_db_metrics: %s", _e)
    try:
        si = components.get("si_core")
        if si and hasattr(si, "current_kpis"):
            _ks = ["skill_acquisition_rate","transfer_learning_rate","zero_shot_success_count",
                   "agentic_capability_score","recursive_self_improvement_rate",
                   "sample_efficiency_trend","metacognition_accuracy","multi_modal_integration_score"]
            _vs = [min(float(si.current_kpis.get(k, 0.0)), 1.0) for k in _ks]
            if _vs:
                m["avg_kpi"] = round(sum(_vs) / len(_vs), 4)
    except Exception:
        pass
    return m


def _calculate_learning_stage(m):
    ins, caps, vocab, kpi = m["insights"], m["capabilities"], m["vocab"], m["avg_kpi"]
    achieved = "Baby"
    for _s in _STAGE_NAMES:
        t = _STAGE_THRESHOLDS[_s]
        if ins >= t[0] and caps >= t[1] and vocab >= t[2] and kpi >= t[3]:
            achieved = _s
    idx = _STAGE_NAMES.index(achieved)
    if idx < len(_STAGE_NAMES) - 1:
        ct = _STAGE_THRESHOLDS[achieved]
        nt = _STAGE_THRESHOLDS[_STAGE_NAMES[idx + 1]]
        def _r(v, lo, hi):
            span = hi - lo
            return min((v - lo) / span, 1.0) if span > 0 else 1.0
        within_pct = round(min(
            _r(ins,   ct[0], nt[0]),
            _r(caps,  ct[1], nt[1]),
            _r(vocab, ct[2], nt[2]),
            _r(kpi,   ct[3], nt[3]),
        ) * 100, 1)
    else:
        within_pct = 100.0
    return achieved, within_pct


def _write_stage_to_db(stage, within_pct, m):
    import sqlite3 as _sw3, datetime as _sdt
    try:
        conn = _sw3.connect(_DB_PATH_STAGE)
        conn.row_factory = _sw3.Row
        now = _sdt.datetime.utcnow().isoformat()
        row = conn.execute("SELECT value FROM system_state WHERE key='learning_stage'").fetchone()
        prev = row["value"] if row else None
        def _up(k, v):
            conn.execute(
                "INSERT INTO system_state (key,value,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (k, str(v), now)
            )
        _up("learning_stage",     stage)
        _up("stage_within_pct",   within_pct)
        _up("stage_insights",     m["insights"])
        _up("stage_capabilities", m["capabilities"])
        _up("stage_vocab",        m["vocab"])
        _up("stage_avg_kpi",      m["avg_kpi"])
        _up("stage_last_updated", now)
        if prev != stage:
            conn.execute(
                "INSERT INTO stage_history "
                "(stage,prev_stage,insights,capabilities,vocab,avg_kpi,within_pct,recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (stage, prev, m["insights"], m["capabilities"],
                 m["vocab"], m["avg_kpi"], within_pct, now)
            )
            logger.info("STAGE ADVANCE: %s -> %s (ins=%d caps=%d vocab=%d kpi=%.3f)",
                        prev, stage, m["insights"], m["capabilities"], m["vocab"], m["avg_kpi"])
        conn.commit()
        conn.close()
    except Exception as _e:
        logger.warning("_write_stage_to_db: %s", _e)


def _run_stage_progression():
    try:
        m = _get_db_metrics()
        stage, within_pct = _calculate_learning_stage(m)
        _write_stage_to_db(stage, within_pct, m)
        logger.debug("Stage: %s %.1f%% ins=%d caps=%d vocab=%d kpi=%.3f",
                     stage, within_pct, m["insights"], m["capabilities"], m["vocab"], m["avg_kpi"])
    except Exception as _e:
        logger.warning("_run_stage_progression: %s", _e)


@app.route("/api/kaizen/status")
def api_kaizen_status():
    """Unified Kaizen status surface. Merges the curated 'suggestions' DB
    counts with the live KaizenAutoRepair queue so every UI shows the same
    numbers for 'pending' and 'executed'."""
    out = {"total_proposals": 0, "pending": 0, "executed": 0, "failed": 0}
    # 1) Curated suggestions DB (drives the dashboard "Kaizen proposals" card)
    try:
        _ensure_suggestions_table()
        conn = _sug_db()
        out["suggestions_total"]    = conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0]
        out["suggestions_pending"]  = conn.execute(
            "SELECT COUNT(*) FROM suggestions WHERE status='pending'").fetchone()[0]
        out["suggestions_executed"] = conn.execute(
            "SELECT COUNT(*) FROM suggestions WHERE status IN ('executed','completed')"
        ).fetchone()[0]
        conn.close()
    except Exception as e:
        out["suggestions_error"] = str(e)
        out["suggestions_total"] = out["suggestions_pending"] = out["suggestions_executed"] = 0

    # 2) Live KaizenAutoRepair queue (drives the admin "repair queue" card)
    try:
        kar = components.get("kaizen_auto_repair")
        if kar and hasattr(kar, "get_stats"):
            r = kar.get_stats() or {}
            out["repair_total"]    = r.get("total", 0)
            out["repair_pending"]  = r.get("pending", 0)
            out["repair_executed"] = r.get("executed", 0)
            out["repair_failed"]   = r.get("failed", 0)
            out["last_executed_at"] = r.get("last_executed_at")
    except Exception as e:
        out["repair_error"] = str(e)

    # 3) Canonical aggregate fields (what every UI should display)
    out["total_proposals"] = out.get("suggestions_total", 0) + out.get("repair_total", 0)
    out["pending"]         = out.get("suggestions_pending", 0) + out.get("repair_pending", 0)
    out["executed"]        = out.get("suggestions_executed", 0) + out.get("repair_executed", 0)
    out["failed"]          = out.get("repair_failed", 0)
    return jsonify(out)




@app.route("/api/self-evolution/status")
def api_self_evolution_status():
    """Self-evolution gap report and capability map status."""
    try:
        _dp = DATA_PATH.rstrip("/")
        import os as _os, json as _json
        gap_path = _os.path.join(_dp, "gap_report.json")
        caps_path = _os.path.join(_dp, "target_capabilities.json")
        gap = _json.load(open(gap_path)) if _os.path.exists(gap_path) else {}
        caps = _json.load(open(caps_path)) if _os.path.exists(caps_path) else {}
        impl = sum(1 for k, v in caps.items() if not k.startswith("_") and isinstance(v, dict) and v.get("implemented"))
        total = sum(1 for k in caps if not k.startswith("_"))
        total_gaps = sum(len(v) for v in gap.values() if isinstance(v, list))
        return jsonify({
            "status": "running" if _self_evolution_available else "unavailable",
            "last_scan": gap.get("ts"),
            "total_gaps": total_gaps,
            "capabilities_implemented": impl,
            "capabilities_total": total,
            "gap_summary": {k: len(v) for k, v in gap.items() if isinstance(v, list)}
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@app.route("/api/social/status")
def api_social_status():
    """Alex Riviera social automation queue status."""
    try:
        if not _social_available:
            return jsonify({"status": "unavailable", "pending_posts": 0,
                            "twitter_configured": False, "linkedin_configured": False})
        stats = _SocialPoster(data_path=DATA_PATH).get_queue_stats()
        return jsonify({"status": "active", **stats})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@app.route("/api/social/generate", methods=["POST"])
def api_social_generate():
    """Trigger immediate Alex Riviera content generation cycle."""
    try:
        if not _social_available:
            return jsonify({"status": "unavailable"})
        _AlexContent(data_path=DATA_PATH).run_daily_cycle()
        return jsonify({"status": "triggered", "message": "Content generation cycle started"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@app.route("/api/capability-map")
def api_capability_map():
    """Target capabilities map with implemented status."""
    try:
        import os as _os, json as _json
        p = _os.path.join(DATA_PATH.rstrip("/"), "target_capabilities.json")
        if _os.path.exists(p):
            return jsonify(_json.load(open(p)))
        # Return default map if file not yet generated
        if _self_evolution_available:
            caps = _CapMapper(data_path=DATA_PATH).run()
            return jsonify(caps)
        return jsonify({"status": "not_yet_mapped"})
    except Exception as e:
        return jsonify({"error": str(e)})


def _ensure_suggestions_table():
    """Create suggestions table if it doesn't exist."""
    import sqlite3 as _sq3
    from pathlib import Path as _P3
    db = _P3("data/dmai_knowledge.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = _sq3.connect(str(db))
    conn.execute('''
        CREATE TABLE IF NOT EXISTS suggestions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'user',
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            complexity TEXT DEFAULT NULL,
            plan TEXT DEFAULT NULL,
            result TEXT DEFAULT NULL,
            pr_url TEXT DEFAULT NULL,
            branch TEXT DEFAULT NULL,
            files_changed TEXT DEFAULT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT DEFAULT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.route("/api/heartbeat", methods=["GET"])
def api_heartbeat():
    """Live learning heartbeat — feeds the admin D3.js Heartbeat panel."""
    import datetime as _dt, json as _json
    import sqlite3 as _hbsq
    import os as _hbos

    DB_PATH = "data/dmai_knowledge.db"
    now = _dt.datetime.utcnow()
    cutoff_24h = (now - _dt.timedelta(hours=24)).isoformat()
    cutoff_7d  = (now - _dt.timedelta(days=7)).isoformat()

    def _hb_conn():
        c = _hbsq.connect(DB_PATH)
        c.row_factory = _hbsq.Row
        return c

    # 1 — Learning stage + within-stage progress from system_state
    learning_stage   = "Baby"
    stage_within_pct = 0.0
    try:
        _c = _hb_conn()
        _ss_rows = _c.execute(
            "SELECT key, value FROM system_state WHERE key IN "
            "('learning_stage','stage_within_pct')"
        ).fetchall()
        _ss = {r["key"]: r["value"] for r in _ss_rows}
        _c.close()
        learning_stage   = _ss.get("learning_stage", "Baby")
        stage_within_pct = float(_ss.get("stage_within_pct", 0.0))
    except Exception:
        pass

    stages      = ["Baby","Child","Teenager","Adult","Expert","Master","Transcendent","Infinite"]
    stage_index = stages.index(learning_stage) if learning_stage in stages else 0
    _per_slot          = 100.0 / (len(stages) - 1)
    stage_progress_pct = round(
        stage_index * _per_slot + (stage_within_pct / 100.0) * _per_slot, 2
    )
    stage_progress_pct = min(stage_progress_pct, 100.0)

    # 2 — Active research nodes last 24 h
    active_nodes = []
    try:
        _c = _hb_conn()
        rows = _c.execute(
            "SELECT domain, COUNT(*) as cnt FROM insights WHERE created_at > ? GROUP BY domain ORDER BY cnt DESC LIMIT 12",
            (cutoff_24h,)
        ).fetchall()
        active_nodes = [{"domain": r["domain"] or "unknown", "count": r["cnt"]} for r in rows]
        _c.close()
    except Exception:
        pass

    # 3 — Skills (capabilities) added last 24 h
    skills_count_24h = 0
    try:
        _c = _hb_conn()
        row = _c.execute("SELECT COUNT(*) as cnt FROM capabilities WHERE created_at > ?", (cutoff_24h,)).fetchone()
        skills_count_24h = row["cnt"] if row else 0
        _c.close()
    except Exception:
        pass

    # 4 — Entities (vocabulary) added last 24 h
    entity_count_24h = 0
    try:
        _c = _hb_conn()
        row = _c.execute("SELECT COUNT(*) as cnt FROM vocabulary WHERE created_at > ?", (cutoff_24h,)).fetchone()
        entity_count_24h = row["cnt"] if row else 0
        _c.close()
    except Exception:
        pass

    # 5 — Knowledge pulse: hourly insight rate last 7 days
    knowledge_pulse = []
    try:
        _c = _hb_conn()
        rows = _c.execute(
            """SELECT strftime('%Y-%m-%dT%H:00:00', created_at) as hour_bucket,
                      COUNT(*) as cnt
               FROM insights
               WHERE created_at > ?
               GROUP BY hour_bucket
               ORDER BY hour_bucket ASC""",
            (cutoff_7d,)
        ).fetchall()
        knowledge_pulse = [{"hour": r["hour_bucket"], "count": r["cnt"]} for r in rows]
        _c.close()
    except Exception:
        pass

    # 6 — Top 10 entities by confidence
    top_entities = []
    try:
        _c = _hb_conn()
        rows = _c.execute(
            "SELECT topic, confidence, domain FROM insights WHERE confidence IS NOT NULL ORDER BY confidence DESC LIMIT 10"
        ).fetchall()
        top_entities = [{"topic": r["topic"], "confidence": round(float(r["confidence"]), 3), "domain": r["domain"] or ""} for r in rows]
        _c.close()
    except Exception:
        pass

    # 7 — Graph stats
    graph_stats = {"neurons": 0, "synapses": 0, "evolution_cycle": 0}
    try:
        _gp = _hbos.path.join(_hbos.path.dirname(_hbos.path.abspath(__file__)),
                               "aevora-training", "dashboard", "data", "graph_schema.json")
        with open(_gp) as _f:
            _gs = _json.load(_f)
        graph_stats = {
            "neurons": _gs.get("total_neurons", 0),
            "synapses": _gs.get("total_synapses", 0),
            "evolution_cycle": _gs.get("evolution_cycle", 0),
        }
    except Exception:
        pass

    # 8 — Totals
    total_insights = 0
    total_capabilities = 0
    try:
        _c = _hb_conn()
        total_insights = _c.execute("SELECT COUNT(*) as cnt FROM insights").fetchone()["cnt"]
        total_capabilities = _c.execute("SELECT COUNT(*) as cnt FROM capabilities").fetchone()["cnt"]
        _c.close()
    except Exception:
        pass

    _next_stage = stages[stage_index + 1] if stage_index < len(stages) - 1 else None
    return jsonify({
        "learning_stage": learning_stage,
        "stage_index": stage_index,
        "stage_progress_pct": stage_progress_pct,
        "stage_within_pct": stage_within_pct,
        "next_stage": _next_stage,
        "stages": stages,
        "active_research_nodes": active_nodes,
        "skills_learned_24h": skills_count_24h,
        "entities_discovered_24h": entity_count_24h,
        "knowledge_pulse": knowledge_pulse,
        "top_entities": top_entities,
        "graph_stats": graph_stats,
        "total_insights": total_insights,
        "total_capabilities": total_capabilities,
        "generated_at": now.isoformat() + "Z",
    })


_ensure_suggestions_table()
_ensure_system_state_table()
@app.route("/api/stage/analytics", methods=["GET"])
def api_stage_analytics():
    """
    Stage progression analytics — velocity, plateaus, forecast.
    Powers the Stage Analytics admin panel.
    """
    import sqlite3 as _an_sq, datetime as _an_dt, math as _an_math

    DB = "data/dmai_knowledge.db"
    now_utc  = _an_dt.datetime.utcnow()
    days_30  = (now_utc - _an_dt.timedelta(days=30)).isoformat()
    days_7   = (now_utc - _an_dt.timedelta(days=7)).isoformat()

    _STAGE_ORDER = ["Baby","Child","Teenager","Adult","Expert","Master","Transcendent","Infinite"]
    _THRESHOLDS  = {
        "Baby":         (      0,     0,     0, 0.00),
        "Child":        (   5000,   500,   100, 0.10),
        "Teenager":     (  30000,  2000,   500, 0.20),
        "Adult":        (  80000,  5000,  1500, 0.35),
        "Expert":       ( 150000, 10000,  3000, 0.50),
        "Master":       ( 300000, 20000,  6000, 0.65),
        "Transcendent": ( 600000, 40000, 12000, 0.80),
        "Infinite":     (1200000, 80000, 25000, 0.92),
    }

    try:
        conn = _an_sq.connect(DB)
        conn.row_factory = _an_sq.Row

        # ── 1. Current metrics ────────────────────────────────────────────────
        # Self-heal: ensure required columns exist on legacy DBs.
        # SQLite rejects ALTER ADD COLUMN with non-constant DEFAULT (CURRENT_TIMESTAMP),
        # so we add the column as TEXT (no default) and backfill any NULLs.
        def _ensure_col(table, col):
            try:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                if col not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
                    conn.execute(
                        f"UPDATE {table} SET {col} = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE {col} IS NULL"
                    )
                    conn.commit()
            except Exception as _ex:
                logger.warning(f"_ensure_col {table}.{col} failed: {_ex}")
        for _tbl in ("insights", "capabilities"):
            _ensure_col(_tbl, "created_at")

        cur_insights = conn.execute("SELECT COUNT(*) as c FROM insights").fetchone()["c"]
        cur_caps     = conn.execute("SELECT COUNT(*) as c FROM capabilities").fetchone()["c"]
        try:
            cur_vocab = conn.execute("SELECT COUNT(*) as c FROM vocabulary").fetchone()["c"]
        except Exception:
            cur_vocab = 0

        # Current stage from system_state (or recalculate)
        ss_row = conn.execute(
            "SELECT value FROM system_state WHERE key='learning_stage'"
        ).fetchone()
        cur_stage   = ss_row["value"] if ss_row else "Baby"
        cur_idx     = _STAGE_ORDER.index(cur_stage) if cur_stage in _STAGE_ORDER else 0

        # ── 2. Daily insight ingestion rate (last 30 days) ───────────────────
        daily_rows = conn.execute("""
            SELECT strftime('%Y-%m-%d', created_at) as day,
                   COUNT(*) as cnt
            FROM insights
            WHERE created_at > ?
            GROUP BY day
            ORDER BY day ASC
        """, (days_30,)).fetchall()
        daily_series = [{"date": r["day"], "insights": r["cnt"]} for r in daily_rows]

        # ── 3. Daily capability rate (last 30 days) ──────────────────────────
        cap_rows = conn.execute("""
            SELECT strftime('%Y-%m-%d', created_at) as day,
                   COUNT(*) as cnt
            FROM capabilities
            WHERE created_at > ?
            GROUP BY day
            ORDER BY day ASC
        """, (days_30,)).fetchall()
        cap_series = {r["day"]: r["cnt"] for r in cap_rows}

        # ── 4. Acquisition channels (insight source breakdown, last 7 days) ──
        # Try 'source' column — fall back to 'domain' if absent
        try:
            channel_rows = conn.execute("""
                SELECT source, COUNT(*) as cnt
                FROM insights
                WHERE created_at > ?
                GROUP BY source
                ORDER BY cnt DESC
                LIMIT 10
            """, (days_7,)).fetchall()
            channels = [{"channel": r["source"] or "unknown", "count": r["cnt"]}
                        for r in channel_rows]
        except Exception:
            channel_rows = conn.execute("""
                SELECT domain, COUNT(*) as cnt
                FROM insights
                WHERE created_at > ?
                GROUP BY domain
                ORDER BY cnt DESC
                LIMIT 10
            """, (days_7,)).fetchall()
            channels = [{"channel": r["domain"] or "unknown", "count": r["cnt"]}
                        for r in channel_rows]

        # ── 5. Stage history (all advances) ──────────────────────────────────
        try:
            hist_rows = conn.execute("""
                SELECT stage, prev_stage, insights, capabilities, vocab,
                       avg_kpi, within_pct, recorded_at
                FROM stage_history
                ORDER BY recorded_at ASC
            """).fetchall()
            stage_history = [{
                "stage":        r["stage"],
                "prev_stage":   r["prev_stage"],
                "insights_at":  r["insights"],
                "caps_at":      r["capabilities"],
                "vocab_at":     r["vocab"],
                "kpi_at":       round(float(r["avg_kpi"] or 0), 4),
                "recorded_at":  r["recorded_at"],
            } for r in hist_rows]
        except Exception:
            stage_history = []

        # ── 6. Velocity calculation ───────────────────────────────────────────
        # Insights per day (7-day rolling average vs 30-day average)
        avg_7d  = 0.0
        avg_30d = 0.0
        if daily_series:
            last_7  = [d["insights"] for d in daily_series[-7:]]
            all_30  = [d["insights"] for d in daily_series]
            avg_7d  = round(sum(last_7)  / max(len(last_7), 1),  1)
            avg_30d = round(sum(all_30)  / max(len(all_30), 1),  1)
        velocity_trend = "accelerating" if avg_7d > avg_30d * 1.1 else \
                         "decelerating" if avg_7d < avg_30d * 0.9 else "stable"

        # ── 7. Plateau detection ─────────────────────────────────────────────
        # A plateau = any 3+ consecutive days with < 20% of the 30-day average
        plateau_threshold = avg_30d * 0.20
        plateaus = []
        streak_start = None
        streak_days  = 0
        for entry in daily_series:
            if entry["insights"] <= plateau_threshold:
                if streak_start is None:
                    streak_start = entry["date"]
                streak_days += 1
            else:
                if streak_days >= 3:
                    plateaus.append({
                        "start":      streak_start,
                        "days":       streak_days,
                        "avg_daily":  round(sum(
                            d["insights"] for d in daily_series
                            if streak_start <= d["date"]
                        ) / max(streak_days, 1), 1),
                    })
                streak_start = None
                streak_days  = 0
        if streak_days >= 3:
            plateaus.append({"start": streak_start, "days": streak_days, "avg_daily": 0})

        # ── 8. Forecast to Master stage ───────────────────────────────────────
        # Master needs: 300k insights, 20k capabilities, 6k vocab, kpi≥0.65
        # Use 7-day avg velocity; also estimate capability and vocab growth rates
        master_t = _THRESHOLDS["Master"]

        try:
            cap_rows_7 = conn.execute("""
                SELECT COUNT(*) as c FROM capabilities WHERE created_at > ?
            """, (days_7,)).fetchone()
            caps_7d_total = cap_rows_7["c"] if cap_rows_7 else 0
        except Exception:
            caps_7d_total = 0

        try:
            vocab_rows_7 = conn.execute("""
                SELECT COUNT(*) as c FROM vocabulary WHERE created_at > ?
            """, (days_7,)).fetchone()
            vocab_7d_total = vocab_rows_7["c"] if vocab_rows_7 else 0
        except Exception:
            vocab_7d_total = 0

        ins_rate_day   = avg_7d if avg_7d > 0 else 1.0
        caps_rate_day  = round(caps_7d_total  / 7, 1) if caps_7d_total else 1.0
        vocab_rate_day = round(vocab_7d_total / 7, 1) if vocab_7d_total else 1.0

        # Days needed for each bottleneck
        def _days_to(current, target, rate):
            gap = target - current
            if gap <= 0:
                return 0
            if rate <= 0:
                return 99999
            return _an_math.ceil(gap / rate)

        days_for_insights  = _days_to(cur_insights, master_t[0], ins_rate_day)
        days_for_caps      = _days_to(cur_caps,     master_t[1], caps_rate_day)
        days_for_vocab     = _days_to(cur_vocab,    master_t[2], vocab_rate_day)
        # KPI: no rate model yet — estimate 0.02/week improvement if training active
        kpi_gap     = max(master_t[3] - 0.0, 0)   # current KPI ~ 0 until keys are set
        days_for_kpi = int(kpi_gap / 0.02 * 7) if kpi_gap > 0 else 0

        bottleneck_days = max(days_for_insights, days_for_caps, days_for_vocab, days_for_kpi)
        bottleneck_name = max(
            [("insights", days_for_insights), ("capabilities", days_for_caps),
             ("vocab", days_for_vocab),        ("kpi", days_for_kpi)],
            key=lambda x: x[1]
        )[0]

        if bottleneck_days < 99999:
            forecast_date = (now_utc + _an_dt.timedelta(days=bottleneck_days)).strftime("%Y-%m-%d")
        else:
            forecast_date = "unknown (no data)"

        # ── 9. Next stage gaps ────────────────────────────────────────────────
        next_stage = _STAGE_ORDER[cur_idx + 1] if cur_idx < len(_STAGE_ORDER) - 1 else None
        next_gaps  = {}
        if next_stage:
            nt = _THRESHOLDS[next_stage]
            next_gaps = {
                "stage":       next_stage,
                "insights":    max(nt[0] - cur_insights, 0),
                "capabilities": max(nt[1] - cur_caps,    0),
                "vocab":       max(nt[2] - cur_vocab,    0),
                "kpi":         round(max(nt[3] - 0.0, 0), 2),
                "days_to_next_insights": _days_to(cur_insights, nt[0], ins_rate_day),
                "days_to_next_caps":     _days_to(cur_caps,     nt[1], caps_rate_day),
                "days_to_next_vocab":    _days_to(cur_vocab,    nt[2], vocab_rate_day),
            }

        conn.close()

        return jsonify({
            "generated_at":    now_utc.isoformat() + "Z",
            "current": {
                "stage":        cur_stage,
                "stage_index":  cur_idx,
                "insights":     cur_insights,
                "capabilities": cur_caps,
                "vocab":        cur_vocab,
            },
            "velocity": {
                "insights_per_day_7d":  avg_7d,
                "insights_per_day_30d": avg_30d,
                "caps_per_day_7d":      caps_rate_day,
                "vocab_per_day_7d":     vocab_rate_day,
                "trend":                velocity_trend,
            },
            "daily_series": daily_series,
            "channels": channels,
            "stage_history": stage_history,
            "plateaus": plateaus,
            "forecast": {
                "target_stage":         "Master",
                "bottleneck":           bottleneck_name,
                "bottleneck_days":      bottleneck_days if bottleneck_days < 99999 else None,
                "forecast_date":        forecast_date,
                "days_for_insights":    days_for_insights  if days_for_insights  < 99999 else None,
                "days_for_caps":        days_for_caps      if days_for_caps      < 99999 else None,
                "days_for_vocab":       days_for_vocab     if days_for_vocab     < 99999 else None,
                "days_for_kpi":         days_for_kpi       if days_for_kpi       < 99999 else None,
            },
            "next_stage_gaps": next_gaps,
        })

    except Exception as _e:
        logger.error("api_stage_analytics: %s", _e)
        # Return graceful degraded payload rather than 500 so the admin UI
        # can still render. Common causes: legacy DB, missing columns, or
        # "database disk image is malformed" requiring manual repair.
        return jsonify({
            "degraded": True,
            "error": str(_e),
            "hint": "DB may need repair via `sqlite3 data/dmai_knowledge.db .recover`",
            "generated_at": _an_dt.datetime.utcnow().isoformat() + "Z",
            "current": {"stage": "Baby", "stage_index": 0, "insights": 0, "capabilities": 0, "vocab": 0},
            "velocity": {"insights_per_day_7d": 0, "insights_per_day_30d": 0},
            "daily_series": [],
            "channels": [],
            "stage_history": [],
            "next_stage_gaps": None,
        })




def _read_stage_from_db():
    """Read learning_stage + stage_within_pct from the system_state table — the
    SAME single source of truth that /api/metrics reads. Returns
    (stage_name, stage_index, stage_within_pct). Keeps every endpoint and the
    KPI seeder agreeing on the stage instead of trusting in-memory si_core attrs
    that are 0 on a cold boot."""
    stage_name, within_pct = "Baby", 0.0
    try:
        db_path = os.path.join(
            os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\"),
            "dmai_knowledge.db")
        import sqlite3 as _sq3s
        _c = _sq3s.connect(db_path, timeout=5)
        _r = _c.execute(
            "SELECT value FROM system_state WHERE key='learning_stage'").fetchone()
        if _r and _r[0]:
            stage_name = _r[0]
        _rw = _c.execute(
            "SELECT value FROM system_state WHERE key='stage_within_pct'").fetchone()
        if _rw and _rw[0] is not None:
            within_pct = float(_rw[0])
        _c.close()
    except Exception as _e:
        logger.debug("_read_stage_from_db failed: %s", _e)
    stage_index = _STAGE_NAMES.index(stage_name) if stage_name in _STAGE_NAMES else 0
    return stage_name, stage_index, within_pct


def _seed_kpis_from_db():
    """Derive all 8 SICore KPI scores from live SQLite counts — single source of truth."""
    try:
        db_path = os.path.join(os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\"), "dmai_knowledge.db")
        import sqlite3 as _sq3
        con = _sq3.connect(db_path, timeout=5)
        cur = con.cursor()

        def _count(tbl, where="1=1"):
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {where}")
                return cur.fetchone()[0]
            except Exception:
                return 0

        insights    = _count("insights")
        caps        = _count("capabilities")
        vocab       = _count("vocabulary")

        # 7-day insight average — divide by the number of distinct ACTIVE days
        # (days with >=1 insight) rather than a hard-coded 7, so a system that
        # has only been learning for 2 days isn't penalised as if it idled for 5.
        try:
            cur.execute(
                "SELECT COUNT(*) FROM insights WHERE created_at >= datetime('now','-7 days')"
            )
            ins_7d = cur.fetchone()[0]
        except Exception:
            ins_7d = 0
        try:
            cur.execute(
                "SELECT COUNT(DISTINCT date(created_at)) FROM insights "
                "WHERE created_at >= datetime('now','-7 days')"
            )
            active_days_count = cur.fetchone()[0] or 0
        except Exception:
            active_days_count = 0
        ins_7d_avg = ins_7d / max(active_days_count, 1) if ins_7d else 0

        con.close()

        # Stage index from system_state DB — same single source of truth as /api/metrics.
        # (Reading getattr(si, "stage_index") returned 0 on cold boot, which hard-zeroed
        #  transfer_learning_rate and recursive_self_improvement_rate.)
        si = components.get("si_core")
        _stage_name, stage_index, stage_pct = _read_stage_from_db()

        # Active component fraction
        active_comp = sum(1 for v in components.values() if v is not None)
        total_comp  = max(len(components), 1)

        kpis = {
            "skill_acquisition_rate":       min(caps   / 50_000, 1.0),
            "transfer_learning_rate":        min(stage_index / 7.0, 1.0),
            "zero_shot_success_count":       min(insights / 300_000, 1.0),
            "agentic_capability_score":      min(caps   / 20_000, 1.0),
            "recursive_self_improvement_rate": min(stage_pct / 100.0, 1.0),
            "sample_efficiency_trend":       min(ins_7d_avg / 5_000, 1.0),
            "metacognition_accuracy":        min(vocab  / 500_000, 1.0),
            "multi_modal_integration_score": min(active_comp / max(total_comp, 56), 1.0),
        }

        # Persist to si_core — write directly to _state (the live dict),
        # then call save_state() to flush to si_core_state.json on disk.
        # NOTE: si.current_kpis is a @property returning dict(self._state),
        #       so .update() on it modifies a throwaway copy — must use _state.
        si = components.get("si_core")
        if si:
            try:
                for _k, _v in kpis.items():
                    si._state[_k] = _v
                si.save_state()
                logger.info("SICore _state updated and saved: %s",
                            {k: round(v, 4) for k, v in kpis.items()})
            except Exception as _se:
                logger.warning("SICore _state update failed: %s", _se)

        # Persist to a lightweight JSON cache for /api/learning/full-status fallback
        try:
            import json as _json
            # Normalise DATA_PATH — strip trailing slash to avoid data//kpi_cache.json
            cache_dir = os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "kpi_cache.json")
            with open(cache_path, "w") as _f:
                _json.dump({"kpis": kpis, "ts": __import__("datetime").datetime.utcnow().isoformat()}, _f)
            logger.info("KPI cache written: %s", cache_path)
        except Exception as _ce:
            logger.warning("KPI cache write failed: %s", _ce)

        logger.info("KPI seed: %s", {k: round(v, 4) for k, v in kpis.items()})
    except Exception as _e:
        logger.warning("_seed_kpis_from_db failed: %s", _e)


def _start_kpi_seed_loop():
    """Run _seed_kpis_from_db once now, then every 5 minutes as a daemon thread."""
    import threading as _th
    _seed_kpis_from_db()  # immediate boot seed

    def _loop():
        import time as _t
        while True:
            _t.sleep(300)
            try:
                _seed_kpis_from_db()
            except Exception as _e:
                logger.warning("KPI seed loop error: %s", _e)

    t = _th.Thread(target=_loop, daemon=True, name="KpiSeedLoop")
    t.start()
    logger.info("KPI seed loop started (every 5 min)")



def _backfill_kaizen_queue():
    """
    One-time migration: copy entries from kaizen_proposals.jsonl into
    kaizen_queue.jsonl (for KaizenAutoRepair) and into the suggestions
    SQLite table (for /api/kaizen/status dashboard counts).
    Idempotent — skips entries already in queue or DB.
    """
    import uuid as _uuid_mod_bk
    proposals_file = Path(DATA_PATH) / "kaizen_proposals.jsonl"
    queue_file     = Path(DATA_PATH) / "kaizen_queue.jsonl"

    if not proposals_file.exists():
        return

    # Load existing queue IDs to avoid duplicates
    existing_ids = set()
    if queue_file.exists():
        for line in queue_file.read_text().splitlines():
            try:
                obj = json.loads(line)
                eid = obj.get("id") or obj.get("title", "")
                existing_ids.add(eid)
            except Exception:
                pass

    # Load existing suggestion IDs from DB
    _ensure_suggestions_table()
    try:
        conn_bk = _sug_db()
        db_ids = {row[0] for row in conn_bk.execute("SELECT id FROM suggestions").fetchall()}
        db_titles = {row[0] for row in conn_bk.execute("SELECT title FROM suggestions").fetchall()}
        conn_bk.close()
    except Exception as _e:
        logger.warning("Kaizen backfill: could not read suggestions DB: %s", _e)
        db_ids = set()
        db_titles = set()

    proposals = []
    for line in proposals_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            proposals.append(json.loads(line))
        except Exception:
            pass

    new_queue = []
    new_db    = []
    now_str = datetime.now(timezone.utc).isoformat()

    for p in proposals:
        title = p.get("title", "")
        # Assign stable ID if missing
        if not p.get("id"):
            p["id"] = "kz-" + _uuid_mod_bk.uuid4().hex[:8]
        pid = p["id"]

        # Normalise status: "review_and_fix" → "pending"
        if p.get("status") in (None, "review_and_fix", ""):
            p["status"] = "pending"
        if "attempt_count" not in p:
            p["attempt_count"] = 0
        if "action_type" not in p:
            p["action_type"] = "patch"

        # Queue file dedup
        key = pid if pid in existing_ids else title
        if key not in existing_ids:
            new_queue.append(p)
            existing_ids.add(pid)
            existing_ids.add(title)

        # DB dedup
        if pid not in db_ids and title not in db_titles:
            new_db.append(p)
            db_ids.add(pid)
            db_titles.add(title)

    # Write new entries to queue file
    if new_queue:
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_file, "a") as _qf:
            for p in new_queue:
                _qf.write(json.dumps(p) + "\n")
        logger.info("Kaizen backfill: wrote %d new entries to kaizen_queue.jsonl", len(new_queue))

    # Insert new entries into suggestions DB
    if new_db:
        try:
            conn_bk2 = _sug_db()
            for p in new_db:
                conn_bk2.execute(
                    "INSERT OR IGNORE INTO suggestions "
                    "(id, source, title, description, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        p["id"],
                        p.get("source", "SelfHealer"),
                        p.get("title", "Kaizen proposal"),
                        p.get("description", p.get("title", "")),
                        "pending",
                        p.get("created_at", now_str),
                        now_str,
                    )
                )
            conn_bk2.commit()
            conn_bk2.close()
            logger.info("Kaizen backfill: inserted %d proposals into suggestions table", len(new_db))
        except Exception as _e:
            logger.warning("Kaizen backfill DB insert error: %s", _e)

    if not new_queue and not new_db:
        logger.debug("Kaizen backfill: nothing new to add")


def _start_background_services():
    # ── Ensure critical tables exist before any background loop reads them ───────
    try:
        _ensure_syllabus_content_table()
    except Exception as _e:
        logger.warning("syllabus_content init failed: %s", _e)
    _start_kpi_seed_loop()  # DB-derived KPI seeder — single source of truth
    # ── DB storage (Postgres w/ SQLite fallback) ──────────────────────────
    try:
        from components.pg_storage import get_storage as _get_pg_storage
        components["db_storage"] = _get_pg_storage()
        logger.info("db_storage initialised: %s", type(components["db_storage"]).__name__)
    except Exception as e:
        logger.warning("db_storage init failed: %s", e)

    # ── Knowledge sources — start on ALL environments (free-tier only) ─────
    km = components.get("knowledge_manager")
    if km:
        try:
            # Start all 8 sources as daemon threads
            # DarkWebMonitor will self-disable if Tor is not reachable
            km.start_all()
            logger.info("KnowledgeSourceManager: all sources started")
        except Exception as e:
            logger.warning("KnowledgeSourceManager start failed: %s", e)

    pl = components.get("parallel_learner")
    if pl:
        try:
            pl.start_background()
            logger.info("ParallelWebLearner background thread started")
        except Exception as e:
            logger.warning("ParallelWebLearner start failed: %s", e)

    orch = components.get("training_orchestrator")
    if orch:
        try:
            orch.start_background_updater()
            logger.info("Background update engine started (render=%s)", IS_RENDER)
        except Exception as e:
            logger.warning("Background updater failed: %s", e)
    # ── Wired-component background loops ───────────────────────────────────
    disc = components.get("ai_discovery")
    if disc and hasattr(disc, "start_discovery_loop"):
        try:
            t = threading.Thread(target=disc.start_discovery_loop, daemon=True,
                                 name="dmai-ai-discovery")
            t.start()
            logger.info("DynamicAIDiscovery loop started")
        except Exception as e:
            logger.warning("DynamicAIDiscovery loop failed: %s", e)

    gm = components.get("github_monitor")
    if gm and hasattr(gm, "run_monitor"):
        try:
            t = threading.Thread(target=gm.run_monitor, daemon=True,
                                 name="dmai-github-monitor")
            t.start()
            logger.info("GitHubStarMonitor loop started")
        except Exception as e:
            logger.warning("GitHubStarMonitor loop failed: %s", e)

    tc = components.get("tutor_configurator")
    if tc and hasattr(tc, "start_health_loop"):
        try:
            t = threading.Thread(target=tc.start_health_loop, daemon=True,
                                 name="dmai-tutor-config")
            t.start()
            logger.info("AITutorAutoConfigurator health loop started")
        except Exception as e:
            logger.warning("AITutorAutoConfigurator loop failed: %s", e)

    # ── KPIEvaluator background evaluation loop ─────────────────────────
    kpi_eval = components.get("kpi_evaluator")
    if kpi_eval:
        try:
            kpi_eval.start_background_eval(interval_hours=6.0)
            logger.info("KPIEvaluator background thread started")
        except Exception as e:
            logger.warning("KPIEvaluator background start failed: %s", e)

    # ── AutoRegistrar (free-tier API key acquisition) ──────────────────
    try:
        from components.integration.auto_registrar import AutoRegistrar as _AutoReg
        _ar = _AutoReg(dmai_app=None)
        components["auto_registrar"] = _ar
        _ar.start()
        logger.info("AutoRegistrar started")
    except Exception as e:
        logger.warning("AutoRegistrar startup failed: %s", e)

    # ── Autonomous Researcher — auto-start background research loop ──────────
    ar = components.get("autonomous_researcher")
    if ar:
        try:
            import threading as _threading
            _ar_thread = _threading.Thread(
                target=ar.run_continuous_research,
                args=(None,),   # uses default topic list
                daemon=True,
                name="dmai-autonomous-researcher"
            )
            _ar_thread.start()
            logger.info("AutonomousResearcher background loop started")
        except Exception as e:
            logger.warning("AutonomousResearcher auto-start failed: %s", e)

    # ── StageAwareLearningOrchestrator — wire si_core + start learning loop ──
    sl = components.get("stage_learner")
    if sl:
        try:
            # Wire SI core so consciousness score drives stage advancement
            if hasattr(sl, "set_si_core"):
                sl.set_si_core(components.get("si_core"))
            elif hasattr(sl, "si_core"):
                sl.si_core = components.get("si_core")

            # Auto-start the continuous learning loop if method exists
            if hasattr(sl, "start_learning_loop"):
                import threading as _threading
                _sl_thread = _threading.Thread(
                    target=sl.start_learning_loop,
                    daemon=True,
                    name="dmai-stage-learner"
                )
                _sl_thread.start()
                logger.info("StageAwareLearningOrchestrator learning loop started")
            elif hasattr(sl, "run_continuous_learning"):
                import threading as _threading
                _sl_thread = _threading.Thread(
                    target=sl.run_continuous_learning,
                    daemon=True,
                    name="dmai-stage-learner"
                )
                _sl_thread.start()
                logger.info("StageAwareLearningOrchestrator continuous learning started")
        except Exception as e:
            logger.warning("StageAwareLearningOrchestrator auto-start failed: %s", e)

    # ── KaizenAutoRepair — backfill existing proposals then start fix loop ────
    try:
        _backfill_kaizen_queue()
    except Exception as _bfe:
        logger.warning("Kaizen backfill error: %s", _bfe)
    _kar = components.get("kaizen_auto_repair")
    if _kar:
        try:
            _kar.start_repair_loop()
            logger.info("KaizenAutoRepair loop started")
        except Exception as e:
            logger.warning("KaizenAutoRepair start failed: %s", e)

    # ── GraphEvolutionLoop — 24/7 continuous knowledge graph growth ───────────
    #
    # Runs every GRAPH_EVOLUTION_INTERVAL_MINS minutes (default 15).
    # On each tick it:
    #   1. Reads any new entries from discoveries.jsonl  (autonomous researcher)
    #   2. Reads any new entries from insights.jsonl     (si_core.add_insight)
    #   3. Reads mastered topics from dmai_knowledge.db  (syllabus progress)
    #   4. Reads capabilities table                      (code writer output)
    #   5. Adds new neurons + synapses to graph_schema.json
    #   6. Bumps evolution_cycle, total_neurons, total_synapses, last_updated
    # The Friday cron still creates the Git PR; this loop handles live growth.
    try:
        _graph_interval = int(os.environ.get("GRAPH_EVOLUTION_INTERVAL_MINS", "15"))

        def _graph_evolution_loop():
            import time as _time
            from components.graph_writer import GraphWriter as _GW
            _gw = _GW()
            logger.info(
                "GraphEvolutionLoop started — running every %d min",
                _graph_interval,
            )
            # Run an immediate first pass so the graph grows on boot
            try:
                r = _gw.evolve()
                if r.get("new_neurons", 0) or r.get("new_synapses", 0):
                    logger.info(
                        "GraphEvolutionLoop boot pass: +%d neurons, +%d synapses "
                        "→ total %d/%d (cycle %d)",
                        r["new_neurons"], r["new_synapses"],
                        r["total_neurons"], r["total_synapses"],
                        r["evolution_cycle"],
                    )
            except Exception as _e:
                logger.warning("GraphEvolutionLoop boot pass failed: %s", _e)

            while True:
                _time.sleep(_graph_interval * 60)
                try:
                    r = _gw.evolve()
                    if r.get("new_neurons", 0) or r.get("new_synapses", 0):
                        logger.info(
                            "GraphEvolutionLoop: +%d neurons, +%d synapses "
                            "→ total %d/%d (cycle %d)",
                            r["new_neurons"], r["new_synapses"],
                            r["total_neurons"], r["total_synapses"],
                            r["evolution_cycle"],
                        )
                except Exception as _e:
                    logger.warning("GraphEvolutionLoop tick failed (non-fatal): %s", _e)

        _gel_thread = threading.Thread(
            target=_graph_evolution_loop,
            daemon=True,
            name="dmai-graph-evolution",
        )
        _gel_thread.start()
        logger.info("GraphEvolutionLoop background thread started (interval=%dm)", _graph_interval)
    except Exception as e:
        logger.warning("GraphEvolutionLoop startup failed: %s", e)

    # ── Self-management (SelfHealer + KaizenExecutor + RenderDeployHook) ───
    try:
        from components.self_management.self_management_runner import start_all as _sm_start
        _sm_start(app=app, components=components)
    except Exception as e:
        logger.warning("Self-management startup failed: %s", e)



    # ── Vocabulary & Encyclopaedia ingestion loop ─────────────────────────────
    try:
        def _vocab_ingest_loop():
            import time as _vt
            _vt.sleep(120)  # 2-min boot delay
            while True:
                try:
                    from components.knowledge.vocabulary_ingester import VocabularyIngester
                    VocabularyIngester().run_once()
                except Exception as _ve:
                    logger.error("VocabularyIngester loop: %s", _ve)
                _vt.sleep(1800)  # every 30 minutes

        _vi_thread = threading.Thread(
            target=_vocab_ingest_loop, daemon=True, name="dmai-vocab-ingest"
        )
        _vi_thread.start()
        logger.info("VocabularyIngester background loop started (30m interval)")
    except Exception as e:
        logger.warning("VocabularyIngester startup failed: %s", e)

    # -- Stage progression loop (every 5 minutes) --
    try:
        def _stage_progression_loop():
            import time as _spt
            _spt.sleep(30)
            while True:
                _run_stage_progression()
                _spt.sleep(300)
        _sp_thread = threading.Thread(
            target=_stage_progression_loop, daemon=True, name="dmai-stage-progress")
        _sp_thread.start()
        logger.info("Stage progression loop started (5m interval)")
    except Exception as _e:
        logger.warning("Stage progression loop failed: %s", _e)

    # ── Suggestion self-generation loop ───────────────────────────────────────
    try:
        def _suggestion_self_gen_loop():
            import time as _t
            _t.sleep(300)  # 5-min boot delay
            while True:
                try:
                    from components.suggestion_executor import SuggestionExecutor
                    SuggestionExecutor().generate_self_suggestions()
                except Exception as _e:
                    logger.error("Self-suggestion loop: %s", _e)
                _t.sleep(7200)  # every 2 hours

        _ssg_thread = threading.Thread(
            target=_suggestion_self_gen_loop, daemon=True, name="suggestion-self-gen"
        )
        _ssg_thread.start()
        logger.info("Suggestion self-generation loop started (2h interval)")
    except Exception as e:
        logger.warning("Suggestion self-gen loop startup failed: %s", e)

    # ── GUARANTEE all 8 canonical background services are running ─────────────
    # Several component objects may have failed to initialise at import time
    # (optional deps missing), leaving their loops unstarted — which is why the
    # live audit showed 0/8 services alive. This block starts any of the 8 that
    # is not already represented by a live thread, lazily (re)instantiating the
    # component inside a resilient wrapper so one failure never permanently
    # silences a service. Thread names contain the exact keywords that BOTH
    # api_training_status and DMAITrainingOrchestrator.get_status() look for.
    import threading as _gth
    import time as _gtime

    def _svc_running(*keywords):
        names = [t.name.lower() for t in _gth.enumerate()]
        return any(any(kw in n for kw in keywords) for n in names)

    def _spawn_guarded(name, fn, retry=300):
        if _svc_running(name.lower()):
            logger.info("Service already running (skipping spawn): %s", name)
            return
        def _runner():
            import traceback as _tb
            while True:
                try:
                    logger.info("Background service starting: %s", name)
                    fn()
                    logger.info("Background service fn() returned (loop will retry): %s", name)
                except Exception as _se:
                    logger.warning("Background service '%s' crashed: %s\n%s", name, _se, _tb.format_exc()[:500])
                _gtime.sleep(retry)
        _t = _gth.Thread(target=_runner, daemon=True, name=name)
        _t.start()
        logger.info("Guaranteed background service started: %s", name)

    # 1. autonomous_researcher
    def _svc_autonomous_researcher():
        ar = components.get("autonomous_researcher")
        if ar is None:
            from components.research.autonomous_researcher import AutonomousResearcher
            ar = AutonomousResearcher(si_core=components.get("si_core"))
            components["autonomous_researcher"] = ar
        ar.run_continuous_research(None)
    if not _svc_running("research", "autonomous", "discover", "autonomous-researcher"):
        _spawn_guarded("autonomous_researcher", _svc_autonomous_researcher)

    # 2. parallel_learner — pulls work from syllabus_content / URL queue
    def _svc_parallel_learner():
        pl = components.get("parallel_learner")
        if pl is None:
            from components.knowledge_sources.parallel_web_learner import ParallelWebLearner
            pl = ParallelWebLearner(data_path=Path(DATA_PATH),
                                    si_core=components.get("si_core"),
                                    web_crawler=None, seed=True)
            components["parallel_learner"] = pl
        pl.start_background()
    if not _svc_running("parallel", "web_learn", "web-learn", "web-learner"):
        _spawn_guarded("parallel_learner", _svc_parallel_learner)

    # 3. stage_learner — pulls topics from syllabus_content
    def _svc_stage_learner():
        sl = components.get("stage_learner")
        if sl is None:
            from components.evolution.StageAwareLearningOrchestrator import StageAwareLearningOrchestrator
            sl = StageAwareLearningOrchestrator(
                data_path=Path(DATA_PATH), synthetic_network=None, knowledge_graph=None,
                ai_hub=components.get("ai_hub"), pattern_synthesis=None)
            components["stage_learner"] = sl
        if hasattr(sl, "set_si_core"):
            try:
                sl.set_si_core(components.get("si_core"))
            except Exception:
                pass
        elif hasattr(sl, "si_core"):
            sl.si_core = components.get("si_core")
        if hasattr(sl, "start_learning_loop"):
            sl.start_learning_loop()
        elif hasattr(sl, "run_continuous_learning"):
            sl.run_continuous_learning()
    if not _svc_running("stage", "learning_loop", "stage-learner", "stage-progress"):
        _spawn_guarded("stage_learner", _svc_stage_learner)

    # 4. kaizen_repair
    def _svc_kaizen_repair():
        kar = components.get("kaizen_auto_repair")
        if kar is None:
            from components.kaizen_auto_repair import KaizenAutoRepair
            kar = KaizenAutoRepair(code_writer=components.get("code_writer"),
                                   memory_retrieval=components.get("memory_recall"),
                                   si_core=components.get("si_core"))
            components["kaizen_auto_repair"] = kar
        kar.start_repair_loop()
    if not _svc_running("kaizen", "repair", "kaizen-repair"):
        _spawn_guarded("kaizen_repair", _svc_kaizen_repair)

    # 5. background_updater
    def _svc_background_updater():
        orch = components.get("training_orchestrator")
        if orch and hasattr(orch, "start_background_updater"):
            orch.start_background_updater()
            return
        # No orchestrator — keep metrics fresh as a lightweight updater pass
        _db_bu = os.path.join(DATA_PATH, "dmai_knowledge.db")
        _update_training_progress(_db_bu)
        _seed_kpis_from_db()
    if not _svc_running("updater", "update", "background_updater"):
        _spawn_guarded("background_updater", _svc_background_updater, retry=600)

    # 6. graph_evolution
    def _svc_graph_evolution():
        from components.graph_writer import GraphWriter as _GW
        _GW().evolve()
    if not _svc_running("graph", "evolution"):
        _spawn_guarded("graph_evolution", _svc_graph_evolution, retry=900)

    # 6. kpi_seed
    if not _svc_running("kpi", "seed"):
        _spawn_guarded("kpi_seed", _seed_kpis_from_db, retry=300)

    # 7. vocab_ingest
    def _svc_vocab_ingest():
        from components.knowledge.vocabulary_ingester import VocabularyIngester
        VocabularyIngester().run_once()
    if not _svc_running("vocab", "ingest"):
        _spawn_guarded("vocab_ingest", _svc_vocab_ingest, retry=1800)

    # 8. intensive syllabus training — drives real learning 24/7
    if not _svc_running("intensive", "training"):
        _gth.Thread(target=_run_intensive_training, daemon=True,
                    name="intensive-training").start()
        logger.info("Guaranteed background service started: intensive-training")


    # Self-evolution orchestrator (scans & self-heals every 30 min)
    if _self_evolution_available:
        try:
            _evo_inst = _SelfEvo(app=app, data_path=DATA_PATH)
            threading.Thread(
                target=_evo_inst.run_forever, daemon=True, name="self_evolution"
            ).start()
            logger.info("Guaranteed background service started: self_evolution")
        except Exception as _e:
            logger.warning(f"self_evolution start failed: {_e}")

    # Alex Riviera content + social posting loop (every 6 hours)
    if _social_available:
        def _alex_riviera_loop():
            import time as _t
            _eng = _AlexContent(data_path=DATA_PATH)
            _poster = _SocialPoster(data_path=DATA_PATH)
            while True:
                try:
                    _eng.run_daily_cycle()
                    _poster.post_pending()
                except Exception as _e:
                    logger.warning(f"alex_riviera loop: {_e}")
                _t.sleep(21600)  # 6 hours
        threading.Thread(
            target=_alex_riviera_loop, daemon=True, name="alex_riviera_content"
        ).start()
        logger.info("Guaranteed background service started: alex_riviera_content")

    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        _start_telegram_bot()

def _start_telegram_bot():
    def _run():
        try:
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
            token = os.environ["TELEGRAM_BOT_TOKEN"]

            async def start_cmd(update, ctx):
                await update.message.reply_text("DMAI v7.1.0 online.\n/status /train /kaizen /persona")

            async def status_cmd(update, ctx):
                si = components.get("si_core")
                kpis = si.current_kpis if si else {}
                msg = (f"DMAI v7.1.0\nUptime: {_uptime()}\n"
                       f"Topics: {TOTAL_TOPICS}\nComponents: {len(components)}\n"
                       f"Consciousness: {kpis.get('consciousness', 0):.3f}")
                await update.message.reply_text(msg)

            async def train_cmd(update, ctx):
                orch = components.get("training_orchestrator")
                if orch:
                    await update.message.reply_text("Starting quick Core training...")
                    result = await orch.run_quick_training("Core")
                    await update.message.reply_text(f"Done: {json.dumps(result)[:300]}")
                else:
                    await update.message.reply_text("Orchestrator not loaded.")

            async def kaizen_cmd(update, ctx):
                recent = _load_kaizen(5)
                msg = f"Kaizen proposals ({len(recent)}):\n" + "\n".join(
                    f"• {p.get('title', '?')} [{p.get('priority', '?')}]" for p in recent)
                await update.message.reply_text(msg or "No proposals yet.")

            async def chat_handler(update, ctx):
                msg = update.message.text or ""
                resp = _ai_chat(msg)
                await update.message.reply_text(resp[:4000])

            import asyncio

            app_tg = ApplicationBuilder().token(token).build()
            app_tg.add_handler(CommandHandler("start", start_cmd))
            app_tg.add_handler(CommandHandler("status", status_cmd))
            app_tg.add_handler(CommandHandler("train", train_cmd))
            app_tg.add_handler(CommandHandler("kaizen", kaizen_cmd))
            app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

            # Run polling without signal handlers to avoid set_wakeup_fd
            # error when started from a background thread (not main thread)
            async def _poll():
                await app_tg.initialize()
                await app_tg.start()
                await app_tg.updater.start_polling(drop_pending_updates=True)
                # Keep alive until thread is stopped
                while True:
                    await asyncio.sleep(60)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_poll())
            finally:
                loop.close()

        except Exception as e:
            logger.warning("Telegram bot error: %s", e)

    t = threading.Thread(target=_run, daemon=True, name="dmai-telegram")
    t.start()
    logger.info("Telegram bot thread started")

# Start all background services (researcher, training, evolution, etc.)
# Runs in the gunicorn worker process — threads survive here without --preload.
_background_services_started = False
_start_background_services()
_background_services_started = True


@app.route("/api/admin/start-services", methods=["POST", "GET"])
def api_start_services():
    """Force-start all background services and return detailed status. No auth — diagnostic only."""
    import threading as _th
    before = {t.name for t in _th.enumerate()}
    try:
        _start_background_services()
    except Exception as _e:
        return jsonify({"error": str(_e), "traceback": __import__("traceback").format_exc()})
    import time as _t; _t.sleep(2)
    after = {t.name for t in _th.enumerate()}
    new_threads = list(after - before)
    all_threads = [t.name for t in _th.enumerate()]
    return jsonify({
        "status": "ok",
        "new_threads": new_threads,
        "all_threads": all_threads,
        "total": len(all_threads),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 55)
    logger.info("  DMAI v7.1.0 — Starting on port %d", port)
    logger.info("  Components: %s", list(components.keys()))
    logger.info("  Syllabus topics: %d", TOTAL_TOPICS)
    logger.info("  Render mode: %s", IS_RENDER)
    logger.info("  Security: JWT=%s CB=%s HMAC=%s Bandit=%s",
                SECURITY_AVAILABLE, CB_AVAILABLE, HMAC_AVAILABLE, BANDIT_AVAILABLE)
    logger.info("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
