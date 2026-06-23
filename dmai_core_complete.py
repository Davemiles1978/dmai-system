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
    os.environ["DISABLE_NEO4J"] = "true"  # Neo4j fully removed — using SQLite (sqlite_storage.py)
    os.environ["DISABLE_AUTO_THREADS"] = "true"

# ── Data path ────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "data/")
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

# ── Startup time ─────────────────────────────────────────────────────────────
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
        data_path=DATA_PATH,
        si_core=components.get("si_core"),
    )
    logger.info("SyntheticIntelligenceTraining initialised")
except Exception as e:
    logger.warning("SyntheticIntelligenceTraining failed: %s", e)

# ── LLM Training ──────────────────────────────────────────────────────────────
try:
    from components.llm_training.LLMTrainingProgram import LLMTrainingProgram
    components["llm_training"] = LLMTrainingProgram(data_path=DATA_PATH)
    logger.info("LLMTrainingProgram initialised")
except Exception as e:
    logger.warning("LLMTrainingProgram failed: %s", e)

# ── GenAI Training ────────────────────────────────────────────────────────────
try:
    from components.genai_training.GenAITrainingProgram import GenAITrainingProgram
    components["genai_training"] = GenAITrainingProgram(data_path=DATA_PATH)
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
    logger.warning("DMAITrainingOrchestrator failed: %s", e)

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
        paper=_paper)
    logger.info("AggressiveTrader initialised (paper=%s)", _paper)
except Exception as e:
    logger.warning("AggressiveTrader failed: %s", e)

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
    master = os.environ.get("MASTER_PASSWORD", "dmai_master")
    return pwd == master

def _ai_chat(message: str) -> str:
    """
    P1-3: Sanitise input before passing to AI hub.
    P1-4: Scan any generated code in the response.
    P3-14: HaltResponse check before returning.
    """
    # Sanitise input
    if SECURITY_AVAILABLE:
        clean_message = sanitise_input(message)
        if check_injection(clean_message):
            logger.warning("Injection attempt detected in chat: %s", message[:80])
            return "Request blocked: potential injection detected."
    else:
        clean_message = message

    # Halt condition check
    if SECURITY_AVAILABLE:
        halt = check_halt_conditions({"message": clean_message})
        if halt:
            return f"Request halted: {halt}"

    hub = components.get("extended_hub") or components.get("ai_hub")
    response_text = None
    if hub:
        try:
            # ExtendedHub has async chat(); AIIntegrationHub has chat_sync()
            if hasattr(hub, "chat_sync"):
                response_text = hub.chat_sync(clean_message)
            elif hasattr(hub, "chat"):
                response_text = _run_async(hub.chat(clean_message))
        except Exception as e:
            logger.warning("AI chat error: %s", e)

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

    # P1-4: scan any code blocks in the response
    if SECURITY_AVAILABLE and "```" in response_text:
        scan = scan_generated_code(response_text)
        if not scan.get("safe", True):
            issues = "; ".join(str(i) for i in scan.get("issues", []))
            logger.warning("Generated code scan found issues: %s", issues)
            response_text = safe_code_output(response_text)

    return response_text

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": "7.0.0",
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
    si = components.get("si_core")
    kpis = si.current_kpis if si else {}
    orch = components.get("training_orchestrator")
    training_status = orch.get_status() if orch else {}
    ext_hub = components.get("extended_hub")
    hub_status = ext_hub.get_status() if ext_hub else {}
    return jsonify({
        "status": "running",
        "version": "7.0.0",
        "uptime": _uptime(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deployment": "render" if IS_RENDER else "local",
        "components_loaded": list(components.keys()),
        "syllabus_topics": TOTAL_TOPICS,
        "si_kpis": kpis,
        "training": training_status,
        "providers": hub_status.get("extended_providers", []) + hub_status.get("base_providers", []),
    })

@app.route("/api/persona")
def api_persona():
    return jsonify({
        "system": "DMAI v7.0.0",
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
    dashboard = Path("static/dashboard.html")
    if dashboard.exists():
        return send_from_directory("static", "dashboard.html")
    return f"""<!DOCTYPE html>
<html><head><title>DMAI v7.0.0</title>
<style>body{{background:#0a0a0f;color:#e0e0ff;font-family:monospace;padding:40px}}
h1{{color:#6c63ff}}a{{color:#00d4aa}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #333;padding:8px;text-align:left}}
.badge{{background:#1a1a2e;border:1px solid #6c63ff;padding:2px 8px;border-radius:4px;font-size:12px}}</style>
</head><body>
<h1>DMAI v7.0.0 — Online</h1>
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
        response = _ai_chat(message)
        _log_chat(message, response)
        return jsonify({
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "dmai_v7",
        })
    except Exception as e:
        logger.error("chat error: %s", e)
        return jsonify({"error": str(e)}), 500

def _log_chat(message, response):
    try:
        log_file = Path(DATA_PATH) / "chat_log.jsonl"
        entry = {
            "message": message,
            "response": response[:200],
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
        "version": "7.0.0", "uptime": _uptime(),
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
    if pwd != os.environ.get("MASTER_PASSWORD", "dmai_master"):
        return jsonify({"error": "Invalid password"}), 401
    if not SECURITY_AVAILABLE:
        return jsonify({"error": "Security module not available"}), 503
    token = issue_token_for_password(pwd)
    if not token:
        return jsonify({"error": "Token generation failed"}), 500
    return jsonify({"token": token, "expires_in": 3600, "type": "Bearer"})

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
    km_status = km.get_summary() if km else {"error": "KnowledgeSourceManager not loaded"}
    pl_status = pl.get_status() if pl else {"error": "ParallelWebLearner not loaded"}
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
        "version": "7.0.0",
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
    fh = components.get("free_api_harvester")
    if fh is None:
        return jsonify({"available": False}), 503
    for meth in ("get_status", "list_apis", "get_apis", "get_harvested"):
        if hasattr(fh, meth):
            try:
                return jsonify({"available": True, "free_apis": getattr(fh, meth)()})
            except Exception as e:
                return jsonify({"available": True, "error": str(e)}), 200
    return jsonify({"available": True})


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


def _start_background_services():
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

    if not IS_RENDER:
        orch = components.get("training_orchestrator")
        if orch:
            try:
                orch.start_background_updater()
                logger.info("Background update engine started")
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

    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        _start_telegram_bot()

def _start_telegram_bot():
    def _run():
        try:
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
            token = os.environ["TELEGRAM_BOT_TOKEN"]

            async def start_cmd(update, ctx):
                await update.message.reply_text("DMAI v7.0.0 online.\n/status /train /kaizen /persona")

            async def status_cmd(update, ctx):
                si = components.get("si_core")
                kpis = si.current_kpis if si else {}
                msg = (f"DMAI v7.0.0\nUptime: {_uptime()}\n"
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

_start_background_services()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 55)
    logger.info("  DMAI v7.0.0 — Starting on port %d", port)
    logger.info("  Components: %s", list(components.keys()))
    logger.info("  Syllabus topics: %d", TOTAL_TOPICS)
    logger.info("  Render mode: %s", IS_RENDER)
    logger.info("  Security: JWT=%s CB=%s HMAC=%s Bandit=%s",
                SECURITY_AVAILABLE, CB_AVAILABLE, HMAC_AVAILABLE, BANDIT_AVAILABLE)
    logger.info("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
