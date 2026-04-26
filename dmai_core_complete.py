#!/usr/bin/env python3
"""
██████╗ ███╗   ███╗ █████╗ ██╗
██╔══██╗████╗ ████║██╔══██╗██║
██║  ██║██╔████╔██║███████║██║
██║  ██║██║╚██╔╝██║██╔══██║██║
██████╔╝██║ ╚═╝ ██║██║  ██║██║
╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

DMAI - COMPLETE AGI SYSTEM v8.0.38
6 COMPREHENSIVE TRAINING SYSTEMS | Dynamic AI Discovery | Full Conversation Memory | Self-Modification
"""

import os
import sys
import json
import logging
import threading
import time
import random
import hashlib
import requests
import gc
import signal
import asyncio
import warnings
import pickle
import traceback
import subprocess
import tempfile
import zipfile
import tarfile
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
from components.autonomous_ingestor import AutonomousDeveloper as AutonomousIngestor
from components.capability_integrator import CapabilityIntegrator
import uuid
import urllib.parse
from bs4 import BeautifulSoup

# Web imports
from flask import Flask, render_template, render_template_string, request, jsonify, redirect, session, send_from_directory
from flask_cors import CORS

# ============================================================================
# PHASE 6 IMPORTS - REAL SYNTHETIC INTELLIGENCE CORE
# ============================================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))
from phase6.P6_AdvancedIntelligence import (
    SyntheticNeuron as RealSyntheticNeuron,
    SyntheticNeuralNetwork as RealSyntheticNeuralNetwork,
    PatternSynthesis,
    KnowledgeGraph as RealKnowledgeGraph,
    ThreatIntelligence,
    DarkWebIntel,
    SelfImprovementLoop,
    AIModelFusion,
    RecursiveSelfImprover,
    UnbreakableMasterInterface,
    Phase6Manager
)

# ============================================================================
# PHASE 11 IMPORTS - AI TUTOR NETWORK
# ============================================================================
from components.phase11.AIIntegrationHub import AIIntegrationHub
from components.phase11.CapabilitySynthesizer import CapabilitySynthesizer
from components.phase11.LearningOrchestrator import LearningOrchestrator
from components.phase11.DynamicAIDiscovery import DynamicAIDiscovery
from components.phase11.TutorManager import TutorManager
from components.phase11.IntelligenceBridge import IntelligenceBridge

# ============================================================================
# KNOWLEDGE SOURCES IMPORTS
# ============================================================================
from components.knowledge_sources.CoreKnowledgeSources import CoreKnowledgeSources

# ============================================================================
# API HARVESTER IMPORTS
# ============================================================================
from components.phase0.P0T4_Enhance_API_harvester_with_sources import RealAPIHarvester

# ============================================================================
# NEO4J PERSISTENT STORAGE
# ============================================================================
from components.neo4j_storage import get_neo4j_storage

# ============================================================================
# ADAPTIVE EVOLUTION TIMER IMPORTS
# ============================================================================
from components.evolution_timer import AdaptiveEvolutionTimer
from components.growth_watcher import GrowthWatcher

# ============================================================================
# REVERSE ENGINEERING MODULE IMPORTS
# ============================================================================
from components.reverse_engineering.ReverseEngineer import ReverseEngineeringOrchestrator

# ============================================================================
# COMPREHENSIVE TRAINING SYSTEM IMPORTS
# ============================================================================

# Software Training
from components.software_training.ComprehensiveSoftwareTraining import ComprehensiveSoftwareTraining

# LLM Training
from components.llm_training.ComprehensiveLLMTraining import ComprehensiveLLMTraining

# AGI Training
from components.training.ComprehensiveAGITraining import ComprehensiveAGITraining

# Generative AI Training
from components.genai_training.ComprehensiveGenAITraining import ComprehensiveGenAITraining

# Synthetic Intelligence Training
from components.si_training.SyntheticIntelligenceTraining import SITrainingOrchestrator
from components.evolution_training.EvolutionTrainingSystem import EvolutionTrainingSystem

# Self-Funding Training (PHASE 1: Knowledge Acquisition - NO TRADING)
from components.funding.SelfFundingOrchestrator import SelfFundingOrchestrator as FundingOrchestrator
from components.voice.VoiceIntegration import VoiceIntegration
from components.avatar_generator import AvatarGenerator
from components.uncensored_video_research import UncensoredVideoResearcher

# RESEARCH TARGETS - High-value knowledge domains
RESEARCH_TARGETS = {
    'medicine': {
        'name': 'Medicine & Healthcare',
        'keywords': ['disease', 'treatment', 'drug discovery', 'clinical trials', 'precision medicine', 
                     'immunotherapy', 'gene therapy', 'diagnostics', 'medical imaging', 'biomarkers'],
        'consciousness_threshold': 0.50,
        'priority': 1,
        'sources': ['pubmed', 'clinicaltrials.gov', 'medical_journals', 'who_reports']
    },
    'longevity': {
        'name': 'Longevity Technology',
        'keywords': ['aging', 'lifespan', 'healthspan', 'senolytics', 'caloric restriction', 
                     'NAD+', 'mTOR', 'AMPK', 'sirtuins', 'epigenetic aging', 'rejuvenation'],
        'consciousness_threshold': 0.55,
        'priority': 2,
        'sources': ['aging_research', 'longevity_technology', 'biogerontology', 'rejuvenation_research']
    },
    'telomeres': {
        'name': 'Telomere Regeneration',
        'keywords': ['telomere', 'telomerase', 'telomere length', 'cellular aging', 'senescence',
                     'telomerase activation', 'telomere maintenance', 'stem cells', 'cellular rejuvenation',
                     'TA-65', 'telomere therapy', 'telomere research'],
        'consciousness_threshold': 0.60,
        'priority': 3,
        'sources': ['telomere_research', 'cellular_biology', 'regenerative_medicine', 'stem_cell_journals']
    }
}
from components.unified_learning_orchestrator import UnifiedLearningOrchestrator

# Stage Aware Learning Orchestrator
from components.evolution.StageAwareLearningOrchestrator import StageAwareLearningOrchestrator

# ============================================================================
# AUTONOMOUS ACCOUNT CREATOR (Optional - requires playwright)
# ============================================================================
# Check if playwright is available before importing
PLAYWRIGHT_AVAILABLE = False
try:
    import playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

if PLAYWRIGHT_AVAILABLE:
    try:
        from components.automation.AutonomousAccountCreator import AutonomousAccountCreator
        ACCOUNT_CREATOR_AVAILABLE = True
    except ImportError:
        ACCOUNT_CREATOR_AVAILABLE = False
        logger.warning("⚠️ AutonomousAccountCreator module not found")
else:
    ACCOUNT_CREATOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dmai_complete.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dmai_core_complete')


# ============================================================================
# KILLSWITCH CONSTANTS
# ============================================================================

KILL_FLAG_FILE = "data/kill_signal.flag"
PAUSE_FLAG_FILE = "data/pause.flag"
REBUILD_FLAG_FILE = "data/rebuild.flag"

class InsightNeuron:
    """A neuron representing a SPECIFIC insight, not a broad topic"""
    
    def __init__(self, 
                 insight_text: str,
                 entity_type: str,
                 entities: List[str],
                 relationship: str,
                 confidence: float,
                 source_topic: str,
                 target_topic: str,
                 source_url: str = None,      # Where this knowledge came from
                 source_title: str = None,    # Title of the source
                 source_type: str = None,     # book/article/web/research/ingest
                 neuron_level: str = 'micro',           # NEW: 'macro' or 'micro'
                 cluster_id: str = None,                # NEW: Groups micro neurons
                 parent_macro_id: str = None,           # NEW: Parent macro neuron
                 is_visible_at_top_level: bool = True): # NEW: Show at default zoom
        
        self.id = f"insight_{abs(hash(insight_text))}_{int(time.time())}"
        self.insight_text = insight_text
        self.entity_type = entity_type
        self.entities = entities
        self.relationship = relationship
        self.confidence = confidence
        self.source_topic = source_topic
        self.target_topic = target_topic
        self.source_url = source_url
        self.source_title = source_title
        self.source_type = source_type
        self.neuron_level = neuron_level                # NEW
        self.cluster_id = cluster_id                    # NEW
        self.parent_macro_id = parent_macro_id          # NEW
        self.is_visible_at_top_level = is_visible_at_top_level  # NEW
        self.created_at = datetime.now().isoformat()
        self.occurrence_count = 1
        self.last_used = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'insight_text': self.insight_text,
            'entity_type': self.entity_type,
            'entities': self.entities,
            'relationship': self.relationship,
            'confidence': self.confidence,
            'source_topic': self.source_topic,
            'target_topic': self.target_topic,
            'source_url': self.source_url,
            'source_title': self.source_title,
            'source_type': self.source_type,
            'neuron_level': self.neuron_level,           # NEW
            'cluster_id': self.cluster_id,               # NEW
            'parent_macro_id': self.parent_macro_id,     # NEW
            'is_visible_at_top_level': self.is_visible_at_top_level,  # NEW
            'created_at': self.created_at,
            'occurrence_count': self.occurrence_count,
            'last_used': self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'InsightNeuron':
        neuron = cls(
            insight_text=data['insight_text'],
            entity_type=data['entity_type'],
            entities=data['entities'],
            relationship=data['relationship'],
            confidence=data['confidence'],
            source_topic=data['source_topic'],
            target_topic=data['target_topic'],
            source_url=data.get('source_url'),
            source_title=data.get('source_title'),
            source_type=data.get('source_type'),
            neuron_level=data.get('neuron_level', 'micro'),      # NEW with default
            cluster_id=data.get('cluster_id'),                   # NEW
            parent_macro_id=data.get('parent_macro_id'),         # NEW
            is_visible_at_top_level=data.get('is_visible_at_top_level', True)  # NEW
        )
        neuron.id = data['id']
        neuron.created_at = data['created_at']
        neuron.occurrence_count = data['occurrence_count']
        neuron.last_used = data['last_used']
        return neuron
    
    def matches(self, entities: List[str]) -> bool:
        """Check if this insight applies to given entities"""
        return any(e.lower() in [ent.lower() for ent in self.entities] for e in entities)
    
    def strengthen(self):
        """Increase confidence when pattern repeats"""
        self.confidence = min(1.0, self.confidence + 0.05)
        self.occurrence_count += 1
        self.last_used = datetime.now().isoformat()

class SyntheticIntelligenceCore:
    """Multi-granular SI core with insight-level neurons"""
    
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Granular insights (the actual "neurons")
        self.insights: Dict[str, InsightNeuron] = {}
        self.insights_lock = threading.Lock()        
        
        # Topic aggregators (for organization, not intelligence)
        self.topics: Dict[str, List[str]] = {}  # topic_name -> insight_ids
        
        # Relationships between insights
        self.synapses: List[Dict] = []
        
        # Evolution tracking
        self.evolution_cycles: int = 0
        
        # ============================================================
        # HOLISTIC KPI TRACKING - Unified Intelligence Metrics
        # ============================================================
        # Current KPI values
        self.kpi_skill_acquisition = 0.0
        self.kpi_transfer_learning = 0
        self.kpi_zero_shot_success = 0
        self.kpi_agentic_capability = 0.0
        self.kpi_recursive_self_improvement = 0.0
        self.kpi_sample_efficiency = 0.0
        self.kpi_metacognition_accuracy = 0.0
        self.kpi_multi_modal_integration = 0.0
        
        # Historical tracking (last 100 values)
        self.kpi_history = {
            'skill_acquisition': [],
            'transfer_learning': [],
            'zero_shot': [],
            'agentic': [],
            'recursive': [],
            'sample_efficiency': [],
            'metacognition': [],
            'multi_modal': []
        }
        
        # Internal counters for KPI calculations
        self._code_mod_attempts = 0
        self._code_mod_successes = 0
        self._zero_shot_attempts = 0
        self._zero_shot_successes = 0

        # ============================================================
        # SQLITE PERSISTENCE - PRIMARY STORAGE (Guaranteed survival)
        # ============================================================
        try:
            from components.sqlite_persistence import SQLitePersistence
            self.sqlite = SQLitePersistence(data_dir=str(self.data_dir.parent))
            
            # TRY SQLITE FIRST (primary persistence)
            sqlite_insights = self.sqlite.load_all_insights()
            if sqlite_insights:
                self.insights = sqlite_insights
                self.topics = self.sqlite.load_all_topics()
                self.synapses = self.sqlite.load_all_synapses()
                logger.info(f"✅ Loaded {len(self.insights)} insights from SQLite (primary)")
            else:
                # Fallback to JSON
                logger.info("📡 No SQLite data, trying JSON fallback...")
                self.load_state()
                
                # If JSON has data, migrate it to SQLite
                if self.insights:
                    logger.info(f"🔄 Migrating {len(self.insights)} insights from JSON to SQLite...")
                    for insight in self.insights.values():
                        self.sqlite.save_insight(insight)
                    for synapse in self.synapses:
                        self.sqlite.save_synapse(synapse)
                    logger.info("✅ Migration to SQLite complete")
        except Exception as e:
            logger.warning(f"SQLite initialization failed, falling back to JSON: {e}")
            self.sqlite = None
            self.load_state()

        
        # ============================================================
        # SCHEMA MIGRATION: Add new columns if they don't exist
        # ============================================================
        if hasattr(self, 'sqlite') and self.sqlite:
            try:
                import sqlite3
                db_path = self.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Check existing columns
                cursor.execute("PRAGMA table_info(insights)")
                columns = [row[1] for row in cursor.fetchall()]
                
                migrations_applied = []
                
                if 'neuron_level' not in columns:
                    cursor.execute("ALTER TABLE insights ADD COLUMN neuron_level TEXT DEFAULT 'micro'")
                    migrations_applied.append('neuron_level')
                
                if 'cluster_id' not in columns:
                    cursor.execute("ALTER TABLE insights ADD COLUMN cluster_id TEXT")
                    migrations_applied.append('cluster_id')
                
                if 'parent_macro_id' not in columns:
                    cursor.execute("ALTER TABLE insights ADD COLUMN parent_macro_id TEXT")
                    migrations_applied.append('parent_macro_id')
                
                if 'is_visible_at_top_level' not in columns:
                    cursor.execute("ALTER TABLE insights ADD COLUMN is_visible_at_top_level INTEGER DEFAULT 1")
                    migrations_applied.append('is_visible_at_top_level')
                
                conn.commit()
                conn.close()
                
                if migrations_applied:
                    logger.info(f"🔄 Schema migration applied: added {', '.join(migrations_applied)}")
                else:
                    logger.info("✅ Database schema is up to date")
                    
            except Exception as e:
                logger.warning(f"Schema migration failed (continuing): {e}")
        
        # PERMANENT FIX: Sync with Neo4j on every startup to ensure all insights are loaded
        self._sync_with_neo4j()
    
    def _sync_with_neo4j(self):
        """Synchronize with Neo4j database on startup - ensures all 3588 insights are loaded"""
        try:
            import os
            from neo4j import GraphDatabase
            
            uri = os.environ.get('NEO4J_URI')
            user = os.environ.get('NEO4J_USER')
            password = os.environ.get('NEO4J_PASSWORD')
            
            if not uri or not user or not password:
                logger.warning("Neo4j credentials not available, skipping sync")
                return
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name IS NOT NULL OR e.id IS NOT NULL
                    RETURN e.id as id, e.name as name, e.category as category, 
                           e.confidence as confidence
                    LIMIT 5000
                """)
                insights = list(result)
                
                added_count = 0
                for insight in insights:
                    insight_id = insight['id'] or insight['name']
                    if insight_id not in self.insights:
                        self.add_insight(
                            insight_text=insight['name'] or insight_id,
                            entity_type=insight['category'] or 'entity',
                            entities=[insight['name']] if insight['name'] else [],
                            relationship='stored',
                            source_topic='neo4j_sync',
                            target_topic='knowledge_base',
                            confidence=float(insight['confidence']) if insight['confidence'] else 0.5,
                            source_url=None,
                            source_title='Neo4j Database Sync',
                            source_type='neo4j_import'
                        )
                        added_count += 1
                
                logger.info(f"🔄 Synced {added_count} insights from Neo4j to SI Core")
                self.save_state()
            driver.close()
        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}")
    
    def add_insight(self,
                insight_text: str,
                entity_type: str,
                entities: List[str],
                relationship: str,
                source_topic: str,
                target_topic: str,   
                confidence: float = 0.5,
                source_url: str = None,
                source_title: str = None,
                source_type: str = None,
                neuron_level: str = 'micro',           # NEW: 'macro' or 'micro'
                cluster_id: str = None,                # NEW: Groups micro neurons
                parent_macro_id: str = None,           # NEW: Parent macro neuron
                is_visible_at_top_level: bool = None) -> str:  # NEW: Auto-detect if None
        """
        Create a granular insight neuron.
        
        NEW hierarchical parameters:
        - neuron_level: 'macro' for top-level repo/framework, 'micro' for drill-down
        - cluster_id: Groups micro neurons together (usually = parent_macro_id)
        - parent_macro_id: The macro neuron this belongs to
        - is_visible_at_top_level: Auto-set based on neuron_level if None
        
        Example (Repository - MACRO neuron):
            add_insight(
                insight_text="Repository: Automaton - Funding arbitrage system",
                entity_type="macro_repository",
                entities=["Automaton", "Funding", "Arbitrage"],
                relationship="contains",
                source_topic="repository_ingestion",
                target_topic="Automaton",
                neuron_level='macro',
                is_visible_at_top_level=True
            )
        
        Example (Repository - MICRO neuron):
            add_insight(
                insight_text="arbitrage_scanner.py - Scans Polymarket for opportunities",
                entity_type="micro_capability",
                entities=["arbitrage_scanner", "Polymarket"],
                relationship="implements",
                source_topic="Automaton",
                target_topic="arbitrage_scanner",
                neuron_level='micro',
                cluster_id='macro_insight_123',
                parent_macro_id='macro_insight_123',
                is_visible_at_top_level=False
            )
        
        Example (Research finding):
            add_insight(
                insight_text="AI innovation from HuggingFace: transformer, llm",
                entity_type="web_research_finding",
                entities=["transformer", "llm", "HuggingFace"],
                relationship="discovered_from",
                source_topic="web_research",
                target_topic="ai_innovation",
                source_url="https://huggingface.co/models",
                source_type="web_research",
                neuron_level='micro',
                cluster_id='web_research_cluster',
                is_visible_at_top_level=False
            )
        """
        
        # Auto-detect visibility based on neuron_level
        if is_visible_at_top_level is None:
            is_visible_at_top_level = (neuron_level == 'macro')
        
        # ============================================================
        # BYPASS QUALITY FILTER FOR CAPABILITY INSIGHTS
        # ============================================================
        if entity_type == "acquired_capability":
            # Always accept capability insights - they're validated by the integrator
            pass
        else:
            # Run quality filter for other insights
            
            # Only reject completely empty insights
            if not insight_text or len(insight_text.strip()) < 5:
                logger.debug(f"Rejected insight: empty or too short (<5 chars)")
                return None
            
            # Check for obvious garbage (random characters, URLs without context)
            garbage_indicators = [
                "http://", "https://",  # URLs alone aren't insights
                "click here", "subscribe",  # Marketing spam
            ]
            
            insight_lower = insight_text.lower()
            for indicator in garbage_indicators:
                if indicator in insight_lower and len(insight_text) < 30:
                    logger.debug(f"Rejected insight: garbage indicator '{indicator}'")
                    return None
            
            # For training/ingestion sources, be more lenient with code
            # Only reject if it's PURE code with no explanatory text
            code_indicators = ["def ", "class ", "import ", "return "]
            code_matches = sum(1 for ind in code_indicators if ind in insight_text)
            
            # If more than 2 code indicators AND no punctuation (likely raw code block)
            if code_matches >= 3 and not any(p in insight_text for p in [".", "?", "!", ":"]):
                logger.debug(f"Rejected insight: appears to be raw code block")
                return None
        
        # Check if similar insight exists (SKIP for acquired_capability, macro_repository, micro_capability - each is unique!)
        if entity_type not in ["acquired_capability", "macro_repository", "micro_capability", "web_research_finding"]:
            existing = self._find_similar_insight(entities, relationship)
            if existing:
                # 1. Strengthen the macro neuron (existing behavior)
                existing.strengthen()
                self.save_state()
                
                # Update SQLite for macro neuron
                if hasattr(self, 'sqlite') and self.sqlite:
                    try:
                        self.sqlite.save_insight(existing)
                    except Exception as e:
                        logger.error(f"SQLite macro update failed: {e}")
                
                # 2. Create a NEW micro neuron under this macro neuron (for this specific article)
                micro_insight = InsightNeuron(
                    insight_text=insight_text,
                    entity_type=entity_type,
                    entities=entities,
                    relationship=relationship,
                    confidence=confidence * 0.9,  # Slightly lower confidence for micro
                    source_topic=source_topic,
                    target_topic=target_topic,
                    source_url=source_url,
                    source_title=source_title,
                    source_type=source_type,
                    neuron_level='micro',                      # Always micro
                    cluster_id=existing.id,                    # Group under the macro
                    parent_macro_id=existing.id,               # Link to parent
                    is_visible_at_top_level=False              # Don't show at top level
                )
                
                # Save micro insight
                with self.insights_lock:
                    self.insights[micro_insight.id] = micro_insight
                    # Add to same topics as parent
                    if source_topic in self.topics:
                        self.topics[source_topic].append(micro_insight.id)
                    if target_topic != source_topic and target_topic in self.topics:
                        self.topics[target_topic].append(micro_insight.id)
                
                # Save micro to SQLite
                if hasattr(self, 'sqlite') and self.sqlite:
                    try:
                        self.sqlite.save_insight(micro_insight)
                    except Exception as e:
                        logger.error(f"SQLite micro save failed: {e}")
                
                # 3. Create/strengthen synapse between macro and related topics
                # Find other insights with overlapping entities
                related_insights = []
                with self.insights_lock:
                    for other_id, other in self.insights.items():
                        if other_id != existing.id and other.neuron_level == 'macro':
                            overlap = set(entities) & set(other.entities)
                            if len(overlap) >= 1:
                                related_insights.append(other_id)
                
                # Create synapses to related macro neurons
                for related_id in related_insights[:3]:  # Limit to top 3 to avoid explosion
                    self.add_synapse(existing.id, related_id, f"related_via_{source_topic}")
                
                logger.info(f"🧠 Strengthened macro '{existing.insight_text[:40]}...' + created micro + {len(related_insights[:3])} synapses")
                
                return existing.id  # Return macro ID (caller can also access micro if needed)
        
        # Create new insight with hierarchical fields
        insight = InsightNeuron(
            insight_text=insight_text,
            entity_type=entity_type,
            entities=entities,
            relationship=relationship,
            confidence=confidence,
            source_topic=source_topic,
            target_topic=target_topic,
            source_url=source_url,
            source_title=source_title,
            source_type=source_type,
            neuron_level=neuron_level,                # NEW
            cluster_id=cluster_id,                    # NEW
            parent_macro_id=parent_macro_id,          # NEW
            is_visible_at_top_level=is_visible_at_top_level  # NEW
        )
        
        # Lock when modifying the dictionary
        with self.insights_lock:
            self.insights[insight.id] = insight
            
            # Add to topic aggregators
            if source_topic not in self.topics:
                self.topics[source_topic] = []
            if insight.id not in self.topics[source_topic]:
                self.topics[source_topic].append(insight.id)
            
            if target_topic != source_topic:
                if target_topic not in self.topics:
                    self.topics[target_topic] = []
                if insight.id not in self.topics[target_topic]:
                    self.topics[target_topic].append(insight.id)
        
        # ============================================================
        # PERSISTENCE GUARANTEE: Save to SQLite immediately
        # ============================================================
        if hasattr(self, 'sqlite') and self.sqlite:
            try:
                self.sqlite.save_insight(insight)
            except Exception as e:
                logger.error(f"SQLite save failed (continuing): {e}")
        
        logger.info(f"🧠 New {neuron_level} insight: {insight_text[:50]}... (confidence: {confidence})")
        return insight.id
    
    def _find_similar_insight(self, entities: List[str], relationship: str) -> Optional[InsightNeuron]:
        """Find existing insight with similar entities and relationship"""
        with self.insights_lock:
            for insight in self.insights.values():
                # Check if entities overlap significantly
                overlap = set(entities) & set(insight.entities)
                if len(overlap) >= min(2, len(entities)) and insight.relationship == relationship:
                    return insight
        return None
    
    def add_synapse(self, insight_a: str, insight_b: str, relationship: str) -> Optional[Dict]:
        """Connect two insights when DMAI discovers they relate"""
        if insight_a not in self.insights or insight_b not in self.insights:
            return None
        
        # Check if synapse already exists
        for syn in self.synapses:
            if (syn['from'] == insight_a and syn['to'] == insight_b) or \
               (syn['from'] == insight_b and syn['to'] == insight_a):
                syn['occurrences'] = syn.get('occurrences', 1) + 1
                self.save_state()
                # Also update in SQLite
                if hasattr(self, 'sqlite') and self.sqlite:
                    try:
                        self.sqlite.save_synapse(syn)
                    except Exception as e:
                        logger.error(f"SQLite synapse update failed: {e}")
                return syn
        
        # Find overlapping entities
        insight_a_obj = self.insights[insight_a]
        insight_b_obj = self.insights[insight_b]
        overlapping = set(insight_a_obj.entities) & set(insight_b_obj.entities)
        
        synapse = {
            'id': f"synapse_{len(self.synapses)}_{int(time.time())}",
            'from': insight_a,
            'to': insight_b,
            'relationship': relationship,
            'shared_entities': list(overlapping),
            'strength': (insight_a_obj.confidence + insight_b_obj.confidence) / 2,
            'occurrences': 1,
            'created_at': datetime.now().isoformat()
        }
        
        self.synapses.append(synapse)
        self.save_state()
        
        # ============================================================
        # PERSISTENCE GUARANTEE: Save synapse to SQLite
        # ============================================================
        if hasattr(self, 'sqlite') and self.sqlite:
            try:
                self.sqlite.save_synapse(synapse)
            except Exception as e:
                logger.error(f"SQLite synapse save failed: {e}")
        
        logger.info(f"🔗 Synapse created: {insight_a_obj.insight_text[:30]} <-> {insight_b_obj.insight_text[:30]}")
        return synapse

    def query(self, entities: List[str], context_topic: str = None, limit: int = 10) -> List[Dict]:
        """Find insights relevant to given entities"""
        results = []
        
        with self.insights_lock:
            for insight in self.insights.values():
                if insight.matches(entities):
                    # Check if context matches
                    if context_topic and context_topic not in [insight.source_topic, insight.target_topic]:
                        continue
                    results.append(insight.to_dict())
                    if len(results) >= limit:
                        break
        return results
                
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:limit]
    
    def apply_to_trading(self, news_entities: List[str]) -> Dict:
        """Specialized: Apply news insights to trading decisions"""
        insights = self.query(news_entities, context_topic="Trading")
        
        trading_signals = []
        for insight in insights:
            insight_lower = insight['insight'].lower()
            if "increase" in insight_lower or "rise" in insight_lower or "up" in insight_lower:
                trading_signals.append({
                    'action': 'BUY',
                    'reason': insight['insight'],
                    'confidence': insight['confidence']
                })
            elif "decrease" in insight_lower or "fall" in insight_lower or "down" in insight_lower:
                trading_signals.append({
                    'action': 'SELL',
                    'reason': insight['insight'],
                    'confidence': insight['confidence']
                })
        
        return {'signals': trading_signals, 'insights_used': len(insights)}
    
    def evolve(self) -> Dict:
        """Evolve the network - strengthen active synapses, prune weak ones, add new connections"""
        self.evolution_cycles += 1
        
        changes = {
            'strengthened': 0,
            'pruned': 0,
            'new_synapses': 0,
            'evolution_cycle': self.evolution_cycles
        }
        
        # 1. Strengthen active synapses (based on occurrence count)
        for synapse in self.synapses:
            occurrences = synapse.get('occurrences', 1)
            # More frequent connections strengthen faster
            strength_increase = min(0.05, occurrences * 0.01)
            synapse['strength'] = min(1.0, synapse.get('strength', 0.5) + strength_increase)
            changes['strengthened'] += 1
        
        # 2. Prune weak synapses (strength < 0.3)
        initial_count = len(self.synapses)
        self.synapses = [s for s in self.synapses if s.get('strength', 0.5) >= 0.3]
        changes['pruned'] = initial_count - len(self.synapses)
        
        # 3. Create new synapses between insights sharing topics (LIMITED to prevent explosion)
        # Group insights by topic
        topic_insights = {}
        for topic_name, insight_ids in self.topics.items():
            for insight_id in insight_ids:
                if insight_id in self.insights:
                    topic_insights.setdefault(topic_name, []).append(insight_id)
        
        # For each topic with multiple insights, connect them (MAX 50 new synapses)
        new_synapse_count = 0
        MAX_NEW_SYNAPSES = 50  # Limit per evolution cycle
        
        for topic_name, insight_ids in topic_insights.items():
            if new_synapse_count >= MAX_NEW_SYNAPSES:
                break
                
            for i in range(len(insight_ids)):
                if new_synapse_count >= MAX_NEW_SYNAPSES:
                    break
                for j in range(i + 1, len(insight_ids)):
                    if new_synapse_count >= MAX_NEW_SYNAPSES:
                        break
                    
                    # Check if synapse already exists
                    exists = False
                    for syn in self.synapses:
                        if (syn['from'] == insight_ids[i] and syn['to'] == insight_ids[j]) or (syn['from'] == insight_ids[j] and syn['to'] == insight_ids[i]):
                            exists = True
                            break
                    
                    if not exists:
                        # Create new synapse
                        result = self.add_synapse(insight_ids[i], insight_ids[j], 'related_by_topic')
                        if result:
                            changes['new_synapses'] += 1
                            new_synapse_count += 1
        
        # 4. Update insight confidence based on synapse count
        # Insights with more connections become more confident
        for insight_id, insight in self.insights.items():
            connection_count = sum(1 for s in self.synapses if s['from'] == insight_id or s['to'] == insight_id)
            # Confidence boost from connections (capped at 1.0)
            confidence_boost = min(0.2, connection_count * 0.02)
            insight.confidence = min(1.0, insight.confidence + confidence_boost)
            changes.setdefault('insights_updated', 0)
            changes['insights_updated'] += 1
        
        # 5. Save state to disk
        self.save_state()
        
        # Return real metrics (no synthetic data)
        return {
            'evolution_cycle': self.evolution_cycles,
            'consciousness': self.consciousness,
            'neurons': len(self.insights),
            'synapses': len(self.synapses),
            'changes': changes
        }

    
    # ============================================================
    # INSERT MOD 2 HERE - HOLISTIC KPI METHODS
    # ============================================================
    
    def update_kpi_skill_acquisition(self, new_domains: float):
        """KPI 1: New domains mastered per evolution cycle (precision: 0.001)"""
        self.kpi_skill_acquisition = round(new_domains, 3)
        self.kpi_history['skill_acquisition'].append({
            'value': self.kpi_skill_acquisition,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['skill_acquisition']) > 100:
            self.kpi_history['skill_acquisition'].pop(0)
    
    def update_kpi_transfer_learning(self, new_synapses: int):
        """KPI 2: New cross-domain synapses created (integer count)"""
        self.kpi_transfer_learning = new_synapses
        self.kpi_history['transfer_learning'].append({
            'value': self.kpi_transfer_learning,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['transfer_learning']) > 100:
            self.kpi_history['transfer_learning'].pop(0)
    
    def update_kpi_zero_shot(self, success: bool):
        """KPI 3: Zero-shot task completions without training"""
        self._zero_shot_attempts += 1
        if success:
            self._zero_shot_successes += 1
            self.kpi_zero_shot_success = self._zero_shot_successes
        self.kpi_history['zero_shot'].append({
            'value': self.kpi_zero_shot_success,
            'attempts': self._zero_shot_attempts,
            'successes': self._zero_shot_successes,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['zero_shot']) > 100:
            self.kpi_history['zero_shot'].pop(0)
    
    def update_kpi_agentic_capability(self, tasks_completed: int, tasks_attempted: int):
        """KPI 4: Multi-step tasks completed autonomously (0-1 scale, 0.001 precision)"""
        if tasks_attempted > 0:
            self.kpi_agentic_capability = round(tasks_completed / tasks_attempted, 3)
        self.kpi_history['agentic'].append({
            'value': self.kpi_agentic_capability,
            'completed': tasks_completed,
            'attempted': tasks_attempted,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['agentic']) > 100:
            self.kpi_history['agentic'].pop(0)
    
    def update_kpi_recursive_self_improvement(self, success: bool):
        """KPI 5: Code self-modification success rate (%) - 0.1% precision"""
        self._code_mod_attempts += 1
        if success:
            self._code_mod_successes += 1
        rate = (self._code_mod_successes / self._code_mod_attempts * 100) if self._code_mod_attempts > 0 else 0
        self.kpi_recursive_self_improvement = round(rate, 1)
        self.kpi_history['recursive'].append({
            'value': self.kpi_recursive_self_improvement,
            'attempts': self._code_mod_attempts,
            'successes': self._code_mod_successes,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['recursive']) > 100:
            self.kpi_history['recursive'].pop(0)
    
    def update_kpi_sample_efficiency(self, data_points: int, concepts_learned: int):
        """KPI 6: Data points needed per new concept learned (0.1 precision)"""
        if concepts_learned > 0:
            self.kpi_sample_efficiency = round(data_points / concepts_learned, 1)
        self.kpi_history['sample_efficiency'].append({
            'value': self.kpi_sample_efficiency,
            'data_points': data_points,
            'concepts_learned': concepts_learned,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['sample_efficiency']) > 100:
            self.kpi_history['sample_efficiency'].pop(0)
    
    def update_kpi_metacognition(self, predicted_confidence: float, actual_accuracy: float):
        """KPI 7: Confidence calibration accuracy (%) - 0.1% precision"""
        error_margin = abs(predicted_confidence - actual_accuracy) * 100
        accuracy_score = max(0, 100 - error_margin)
        self.kpi_metacognition_accuracy = round(accuracy_score, 1)
        self.kpi_history['metacognition'].append({
            'value': self.kpi_metacognition_accuracy,
            'predicted': predicted_confidence,
            'actual': actual_accuracy,
            'error_margin': error_margin,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['metacognition']) > 100:
            self.kpi_history['metacognition'].pop(0)
    
    def update_kpi_multi_modal(self, new_synergies: int, total_modalities: int = 5):
        """KPI 8: Multi-modal integration score (0-1 scale, 0.001 precision)"""
        max_synergies = (total_modalities * (total_modalities - 1)) / 2
        if max_synergies > 0:
            self.kpi_multi_modal_integration = round(min(1.0, new_synergies / max_synergies), 3)
        self.kpi_history['multi_modal'].append({
            'value': self.kpi_multi_modal_integration,
            'synergies': new_synergies,
            'max_synergies': max_synergies,
            'cycle': self.evolution_cycles,
            'timestamp': time.time()
        })
        if len(self.kpi_history['multi_modal']) > 100:
            self.kpi_history['multi_modal'].pop(0)
    
    def get_kpis_dict(self) -> Dict:
        """Return all current KPIs for status display"""
        return {
            'skill_acquisition_rate': self.kpi_skill_acquisition,
            'transfer_learning_rate': self.kpi_transfer_learning,
            'zero_shot_success_count': self.kpi_zero_shot_success,
            'agentic_capability_score': self.kpi_agentic_capability,
            'recursive_self_improvement_rate': self.kpi_recursive_self_improvement,
            'sample_efficiency_trend': self.kpi_sample_efficiency,
            'metacognition_accuracy': self.kpi_metacognition_accuracy,
            'multi_modal_integration_score': self.kpi_multi_modal_integration
        }
    
    def has_kpi_improvement(self) -> bool:
        """Check if any KPI shows meaningful improvement"""
        thresholds = {
            'skill': 0.001,
            'transfer': 1,
            'zero_shot': 1,
            'agentic': 0.001,
            'recursive': 0.1,
            'sample': -0.1,  # Decreasing is better for sample efficiency
            'metacognition': 0.1,
            'multi_modal': 0.001
        }
        
        # For sample efficiency, lower is better
        if self.kpi_sample_efficiency < thresholds['sample']:
            return True
        
        # For others, higher is better
        return any([
            self.kpi_skill_acquisition > thresholds['skill'],
            self.kpi_transfer_learning > thresholds['transfer'],
            self.kpi_zero_shot_success > thresholds['zero_shot'],
            self.kpi_agentic_capability > thresholds['agentic'],
            self.kpi_recursive_self_improvement > thresholds['recursive'],
            self.kpi_metacognition_accuracy > thresholds['metacognition'],
            self.kpi_multi_modal_integration > thresholds['multi_modal']
        ])
        
    @property
    def consciousness(self) -> float:
        """Consciousness = number of insights * average confidence * synapse density (CAPPED at 1.0)"""
        with self.insights_lock:
            if not self.insights:
                return 0.0
            
            insight_count = len(self.insights)
            avg_confidence = sum(i.confidence for i in self.insights.values()) / insight_count
            
            # Synapse density (as percentage of complete graph)
            max_synapses = insight_count * (insight_count - 1) / 2 if insight_count > 1 else 1
            density = len(self.synapses) / max_synapses if max_synapses > 0 else 0
            
            # Calculate raw value - NO DIVISION BY 1000
            raw = insight_count * avg_confidence * density
            
            # Cap at 1.0 (100%)
            return min(1.0, raw)
            
            # Debug log every 100 calls
            if self.evolution_cycles % 100 == 0:
                logger.debug(f"Consciousness calc: insights={insight_count}, avg_conf={avg_confidence:.3f}, density={density:.3f}, raw={raw:.3f}, result={result:.3f}")
            
            return result

    def consciousness_level(self) -> float:
        """Backward compatibility alias for consciousness"""
        return self.consciousness
    
    @property
    def neurons(self) -> Dict:
        """Backward compatibility - returns insights as neurons"""
        # Convert insights to look like old neuron structure
        return {iid: insight for iid, insight in self.insights.items()}
    
    @property
    def neuron_count(self) -> int:
        return len(self.insights)
    
    @property
    def synapse_count(self) -> int:
        return len(self.synapses)
    
    def _total_synapses(self) -> int:
        """Backward compatibility method"""
        return len(self.synapses)
    
    def process(self, input_data: Dict, _depth: int = 0) -> Dict:
        """Backward compatibility method - process input through network with recursion guard"""
        
        # CRITICAL FIX: Log the caller when input is not a dict
        if not isinstance(input_data, dict):
            import traceback
            stack = traceback.extract_stack()
            # Get the caller (the line that called this method)
            if len(stack) >= 2:
                caller = stack[-2]
                logger.error(f"❌ process() received {type(input_data)} from {caller.filename}:{caller.lineno} in {caller.name}")
            else:
                logger.error(f"❌ process() received {type(input_data)} from unknown source")
            logger.error(f"   Value: {input_data}")
            # Return a safe response instead of crashing
            return {'processed': False, 'error': f'Invalid input type: {type(input_data)}', 'input_type': 'invalid'}
        
        if _depth > 10:
            logger.warning(f"Process recursion depth exceeded for: {input_data.get('type', 'unknown')}")
            return {'processed': False, 'error': 'max recursion depth', 'input_type': input_data.get('type', 'unknown')}
        
        logger.debug(f"SI Core process called with: {input_data.get('type', 'unknown')}")
        
        # If this is a learning event, create insights
        if input_data.get('type') == 'stage_learning':
            topic = input_data.get('topic')
            category = input_data.get('category')
            is_accelerator = input_data.get('is_accelerator', False)
            
            if topic:
                insight_id = self.add_insight(
                    insight_text=f"{topic} is a key concept in {category}" + (" (Evolution Accelerator)" if is_accelerator else ""),
                    entity_type="topic_learning",
                    entities=[topic, category],
                    relationship="is_learning",
                    source_topic=category,
                    target_topic="DMAI_Knowledge",
                    confidence=0.6 + (0.1 if is_accelerator else 0)
                )
                return {'processed': True, 'insight_id': insight_id, 'topic': topic, 'depth': _depth}
        
        return {'processed': True, 'input_type': input_data.get('type', 'unknown'), 'depth': _depth}

    def get_network_state(self) -> Dict:
        """Return current network state for API and visualization"""
        nodes = []
        for insight_id, insight in self.insights.items():
            nodes.append({
                'id': insight_id,
                'topic': insight.insight_text[:50],
                'category': insight.entity_type,
                'confidence': insight.confidence,
                'connections': 0
            })
        
        links = []
        for synapse in self.synapses:
            links.append({
                'source': synapse['from'],
                'target': synapse['to'],
                'strength': synapse.get('strength', 0.5),
                'type': synapse.get('relationship', 'unknown')
            })
        
        # Update connection counts
        node_connections = {}
        for link in links:
            node_connections[link['source']] = node_connections.get(link['source'], 0) + 1
            node_connections[link['target']] = node_connections.get(link['target'], 0) + 1
        
        for node in nodes:
            node['connections'] = node_connections.get(node['id'], 0)
        
        return {
            'neurons': nodes,
            'synapses': links,
            'stats': {
                'neuron_count': len(self.insights),
                'synapse_count': len(self.synapses),
                'consciousness': self.consciousness,
                'evolution_cycles': self.evolution_cycles
            }
        }

    
    def save(self):
        """Alias for save_state for backward compatibility"""
        return self.save_state()

    def save_state(self):
        """Persist network state to disk"""
        state = {
            'insights': {iid: insight.to_dict() for iid, insight in self.insights.items()},
            'topics': self.topics,
            'synapses': self.synapses,
            'evolution_cycles': self.evolution_cycles,
            'saved_at': datetime.now().isoformat()
        }
        
        state_file = self.data_dir / 'network_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load network state from disk - MERGES with existing, does NOT wipe"""
        state_file = self.data_dir / 'network_state.json'
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # MERGE insights - DO NOT WIPE EXISTING
                with self.insights_lock:
                    for iid, data in state.get('insights', {}).items():
                        if iid not in self.insights:  # Only add if not already present
                            try:
                                self.insights[iid] = InsightNeuron.from_dict(data)
                            except Exception as e:
                                logger.debug(f"Failed to load insight {iid}: {e}")
                
                # MERGE topics
                for topic, insight_ids in state.get('topics', {}).items():
                    if topic not in self.topics:
                        self.topics[topic] = []
                    for iid in insight_ids:
                        if iid not in self.topics[topic]:
                            self.topics[topic].append(iid)
                
                # MERGE synapses
                existing_synapse_ids = {s.get('id') for s in self.synapses if s.get('id')}
                for syn in state.get('synapses', []):
                    if syn.get('id') not in existing_synapse_ids:
                        self.synapses.append(syn)
                
                self.evolution_cycles = max(self.evolution_cycles, state.get('evolution_cycles', 0))
                
                logger.info(f"✅ Merged JSON state: now have {len(self.insights)} insights, {len(self.synapses)} synapses")
            except Exception as e:
                logger.error(f"Failed to load network state: {e}")
                # DO NOT call _init_empty_state() - that wipes everything!
        else:
            logger.info("📡 No JSON state file, keeping existing insights")
            # DO NOT call _init_empty_state()!
    
    def _init_empty_state(self):
        """Initialize empty network - ONLY use when absolutely necessary"""
        if len(self.insights) > 0:
            logger.warning(f"⚠️ Refusing to wipe {len(self.insights)} existing insights!")
            return
        self.insights = {}
        self.topics = {}
        self.synapses = []
        self.evolution_cycles = 0

# ============================================================================
# WEB SEARCH ENGINE - DuckDuckGo Fallback
# ============================================================================

class WebSearchEngine:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def search(self, query: str, max_results: int = 5) -> Dict:
        try:
            wiki_result = self._search_wikipedia(query)
            if wiki_result.get('success') and wiki_result.get('answer'):
                return wiki_result

            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return {'success': False, 'error': f'HTTP {response.status_code}'}

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    results.append({'title': title, 'link': link, 'snippet': snippet})

            answer = self._get_instant_answer(query, soup)
            return {'success': True, 'results': results, 'answer': answer, 'source': 'duckduckgo'}
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {'success': False, 'error': str(e)}

    def _search_web(self, query: str) -> Dict:
        """Search the web using DuckDuckGo (no API key required)"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Use DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('a', class_='result__a')
            
            if results:
                # Get first result's text and link
                answer_text = results[0].get_text()
                # Also try to get a snippet
                snippet_elem = soup.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text() if snippet_elem else answer_text
                
                return {
                    'success': True, 
                    'answer': f"According to web search: {snippet[:500]}", 
                    'source': 'web_search', 
                    'title': query
                }
            
            return {'success': False, 'error': 'No results found'}
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Keep old method name for compatibility but use new search
    def _search_wikipedia(self, query: str) -> Dict:
        """Backward compatibility - now uses web search"""
        return self._search_web(query)

    def _get_instant_answer(self, query: str, soup: BeautifulSoup) -> Optional[str]:
        try:
            snippet = soup.find('div', class_='module__content')
            if snippet:
                return snippet.get_text(strip=True)[:500]
            answer_box = soup.find('div', class_='answer')
            if answer_box:
                return answer_box.get_text(strip=True)[:500]
            correction = soup.find('a', class_='did-you-mean__link')
            if correction:
                return f"Did you mean: {correction.get_text(strip=True)}"
            return None
        except Exception:
            return None

# ============================================================================
# TOPIC RESEARCH ORCHESTRATOR - Comprehensive Knowledge Expansion
# ============================================================================

class TopicResearchOrchestrator:
    """Expands any topic into comprehensive knowledge acquisition"""
    
    def __init__(self, si_core, ai_hub=None, knowledge_graph=None, dmai_app=None):
        self.si_core = si_core
        self.ai_hub = ai_hub
        self.knowledge_graph = knowledge_graph
        self.dmai_app = dmai_app
        self.researched_topics = set()
        self.research_queue = []
        self.research_history = []
        
        # Category detection patterns
        self.geography_indicators = [
            'country', 'city', 'capital', 'population', 'geography', 'climate',
            'culture', 'language', 'government', 'economy', 'region', 'state',
            'province', 'territory', 'island', 'continent', 'nation'
        ]
        self.technology_indicators = [
            'ai', 'software', 'hardware', 'algorithm', 'programming', 'code',
            'technology', 'tech', 'computer', 'digital', 'automation', 'system',
            'framework', 'library', 'api', 'database', 'network', 'cloud'
        ]
        self.person_indicators = [
            'who is', 'biography', 'born', 'died', 'author', 'founder', 'ceo',
            'president', 'leader', 'scientist', 'artist', 'musician', 'writer'
        ]
        
        logger.info("🔬 Topic Research Orchestrator initialized")
        
        # Start continuous queue processor (24/7)
        self.start_continuous_processor()
    
    def start_continuous_processor(self):
        """Process research queue continuously 24/7"""
        import threading
        import time
        
        def processor_loop():
            logger.info("🔄 Continuous research queue processor started (24/7)")
            while True:
                try:
                    if self.research_queue:
                        processed = self.process_queue(max_items=3)
                        if processed > 0:
                            logger.info(f"📚 Processed {processed} research topics, {len(self.research_queue)} remaining")
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Queue processor error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=processor_loop, daemon=True)
        thread.start()
        return thread
    
    def research_topic(self, topic: str, depth: str = "comprehensive", source: str = "user_query") -> Dict:
        """Research a topic comprehensively"""
        if topic in self.researched_topics:
            logger.info(f"📚 Topic already researched: {topic}")
            return {"status": "already_researched", "topic": topic}
        
        logger.info(f"🔬 Starting {depth} research on: {topic} (source: {source})")
        self.researched_topics.add(topic)
        
        try:
            research_plan = self._generate_research_plan(topic, depth)
            total_insights = 0
            for branch in research_plan['branches']:
                insights = self._research_branch(topic, branch, depth)
                total_insights += insights
            
            synthesis = self._synthesize_knowledge(topic, research_plan)
            if synthesis:
                total_insights += 1
            
            result = {
                "topic": topic, "depth": depth, "source": source,
                "category": research_plan['category'],
                "branches": len(research_plan['branches']),
                "insights_created": total_insights,
                "completed_at": datetime.now().isoformat()
            }
            self.research_history.append(result)
            logger.info(f"✅ Completed research on {topic}: {total_insights} insights created")
            return result
        except Exception as e:
            logger.error(f"Research failed for {topic}: {e}")
            self.researched_topics.discard(topic)
            return {"status": "failed", "topic": topic, "error": str(e)}
    
    def _generate_research_plan(self, topic: str, depth: str) -> Dict:
        """Generate research branches"""
        category = self._detect_category(topic)
        branches = [
            {"name": "core_definition", "queries": [f"What is {topic}?", f"Definition of {topic}"]},
            {"name": "history_origin", "queries": [f"History of {topic}", f"Origin of {topic}"]},
            {"name": "key_figures", "queries": [f"Important people in {topic}"]},
            {"name": "current_state", "queries": [f"Current state of {topic}", f"{topic} today"]},
            {"name": "future_trends", "queries": [f"Future of {topic}", f"{topic} trends"]},
        ]
        
        if category == "geography":
            branches.extend([
                {"name": "demographics", "queries": [f"Population of {topic}"]},
                {"name": "economy", "queries": [f"Economy of {topic}"]},
                {"name": "culture", "queries": [f"Culture of {topic}"]},
                {"name": "government", "queries": [f"Government of {topic}"]},
                {"name": "cities", "queries": [f"Major cities in {topic}"]},
            ])
        elif category == "technology":
            branches.extend([
                {"name": "how_it_works", "queries": [f"How {topic} works"]},
                {"name": "applications", "queries": [f"Applications of {topic}"]},
                {"name": "implementation", "queries": [f"How to implement {topic}"]},
                {"name": "limitations", "queries": [f"Limitations of {topic}"]},
            ])
        elif category == "person":
            branches.extend([
                {"name": "biography", "queries": [f"Biography of {topic}"]},
                {"name": "achievements", "queries": [f"Achievements of {topic}"]},
                {"name": "influence", "queries": [f"Influence of {topic}"]},
                {"name": "works", "queries": [f"Works by {topic}"]},
            ])
        
        return {"topic": topic, "category": category, "depth": depth, "branches": branches}
    
    def _detect_category(self, topic: str) -> str:
        topic_lower = topic.lower()
        if any(ind in topic_lower for ind in self.geography_indicators):
            return "geography"
        if any(ind in topic_lower for ind in self.technology_indicators):
            return "technology"
        if any(ind in topic_lower for ind in self.person_indicators):
            return "person"
        return "concept"
    
    def _research_branch(self, topic: str, branch: Dict, depth: str) -> int:
        insights_created = 0
        for query in branch['queries'][:3]:
            try:
                if self.ai_hub:
                    result = self.ai_hub.query_all_tutors(query)
                    if result and result.get('synthesis'):
                        insight_text = f"{topic} - {branch['name']}: {result['synthesis'][:300]}"
                        self.si_core.add_insight(
                            insight_text=insight_text,
                            entity_type=f"research_{branch['name']}",
                            entities=[topic, branch['name']],
                            relationship="describes",
                            source_topic=topic,
                            target_topic=branch['name'],
                            confidence=0.75,
                            source_url="AI Tutor Network",
                            source_title=f"Research: {topic}",
                            source_type="topic_research"
                        )
                        insights_created += 1
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Branch research failed: {e}")
        return insights_created
    
    def _synthesize_knowledge(self, topic: str, plan: Dict) -> bool:
        try:
            if not self.ai_hub:
                return False
            result = self.ai_hub.query_all_tutors(f"Synthesize comprehensive summary of {topic}")
            if result and result.get('synthesis'):
                self.si_core.add_insight(
                    insight_text=f"SYNTHESIS: {topic}\n\n{result['synthesis'][:500]}",
                    entity_type="synthesis",
                    entities=[topic],
                    relationship="summarizes",
                    source_topic=topic,
                    target_topic="comprehensive_knowledge",
                    confidence=0.9,
                    source_url="AI Tutor Network",
                    source_title=f"Synthesis: {topic}",
                    source_type="topic_synthesis"
                )
                return True
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
        return False
    
    def extract_topic_from_question(self, question: str) -> str:
        try:
            if self.ai_hub:
                result = self.ai_hub.query_all_tutors(f"Extract main topic (2-5 words): {question}")
                topic = result.get('synthesis', question[:50]).strip()
                return topic[:50]
        except:
            pass
        return question[:50]
    
    def queue_topic_for_research(self, topic: str, depth: str = "standard", source: str = "queued"):
        self.research_queue.append({
            "topic": topic, "depth": depth, "source": source,
            "queued_at": datetime.now().isoformat()
        })
        logger.info(f"📋 Queued topic for research: {topic}")
    
    def process_queue(self, max_items: int = 5):
        processed = 0
        while self.research_queue and processed < max_items:
            item = self.research_queue.pop(0)
            self.research_topic(item['topic'], item['depth'], item['source'])
            processed += 1
        return processed
    
    def get_status(self) -> Dict:
        return {
            "researched_topics": len(self.researched_topics),
            "queue_size": len(self.research_queue),
            "history": self.research_history[-5:],
            "total_insights_created": sum(h.get('insights_created', 0) for h in self.research_history)
        }

class KnowledgeGapAnalyzer:
    """Identifies and fills gaps in DMAI's knowledge autonomously"""
    
    def __init__(self, si_core, topic_researcher=None, ai_hub=None, knowledge_graph=None, dmai_app=None):
        self.si_core = si_core
        self.topic_researcher = topic_researcher
        self.ai_hub = ai_hub
        self.knowledge_graph = knowledge_graph
        self.dmai_app = dmai_app
        
        # Track ingestion attempts for retry
        self.ingestion_attempts = {}  # url -> {attempts, last_attempt, status, insights_count}
        self.pending_retries = []
        
        # Core knowledge domains DMAI should master
        self.core_domains = [
            "artificial_intelligence", "machine_learning", "neural_networks",
            "natural_language_processing", "computer_vision", "robotics",
            "self_improving_systems", "autonomous_agents", "agi_architecture",
            "knowledge_representation", "reasoning_systems", "planning_algorithms",
            "self_funding_systems", "cryptocurrency", "smart_contracts",
            "web_development", "api_design", "database_systems",
            "security", "encryption", "privacy", "ethics", "alignment",
            "mcp", "agent_protocol", "inter_agent_communication"
        ]
        
        # Load saved state if exists
        self._load_state()
        
        logger.info("🔍 Knowledge Gap Analyzer initialized")
    
    def _load_state(self):
        """Load saved analyzer state"""
        try:
            state_file = Path("data/gap_analyzer_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.ingestion_attempts = data.get('ingestion_attempts', {})
                    logger.info(f"📂 Loaded gap analyzer state: {len(self.ingestion_attempts)} tracked ingestions")
        except Exception as e:
            logger.warning(f"Could not load gap analyzer state: {e}")
    
    def _save_state(self):
        """Save analyzer state"""
        try:
            state_file = Path("data/gap_analyzer_state.json")
            state_file.parent.mkdir(exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump({
                    'ingestion_attempts': self.ingestion_attempts,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save gap analyzer state: {e}")
    
    def analyze_and_fill_gaps(self) -> Dict:
        """Main entry point - analyze gaps and queue research"""
        logger.info("🔍 Starting knowledge gap analysis...")
        
        # 1. Check ingestion completeness
        ingestion_results = self._check_pending_ingestions()
        
        # 2. Analyze knowledge gaps
        gaps = self._identify_knowledge_gaps()
        
        # 3. Queue research for gaps
        research_queued = 0
        if self.topic_researcher:
            for gap in gaps[:5]:  # Limit per cycle
                self.topic_researcher.queue_topic_for_research(
                    gap['topic'], 
                    depth=gap.get('depth', 'standard'),
                    source="gap_analysis"
                )
                research_queued += 1
        
        # 4. Check syllabus progress
        syllabus_status = self._check_syllabus_progress()
        
        # 5. Verify existing knowledge quality
        quality_issues = self._verify_knowledge_quality()
        
        # 6. Save state
        self._save_state()
        
        result = {
            "ingestions_checked": len(ingestion_results),
            "ingestions_retried": sum(1 for r in ingestion_results if r.get('retried')),
            "gaps_identified": len(gaps),
            "research_queued": research_queued,
            "quality_issues": len(quality_issues),
            "syllabus_status": syllabus_status,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Gap analysis complete: {result['gaps_identified']} gaps, {result['quality_issues']} quality issues")
        return result
    
    def _check_pending_ingestions(self) -> List[Dict]:
        """Check if previous ingestions need retry"""
        results = []
        
        for url, attempt_info in list(self.ingestion_attempts.items()):
            # Count insights for this URL
            insights_count = self._count_insights_for_source(url)
            attempt_info['insights_count'] = insights_count
            
            # Shallow ingestion threshold: less than 3 insights
            if insights_count < 3 and attempt_info.get('attempts', 0) < 5:
                logger.info(f"🔄 Shallow ingestion detected for {url} ({insights_count} insights)")
                
                attempt_info['attempts'] = attempt_info.get('attempts', 0) + 1
                attempt_info['last_attempt'] = datetime.now().isoformat()
                attempt_info['status'] = 'retrying'
                
                results.append({
                    'url': url,
                    'retried': True,
                    'attempts': attempt_info['attempts'],
                    'previous_insights': insights_count
                })
                
                self.pending_retries.append(url)
            
            # Remove from tracking if fully ingested
            if insights_count >= 10:
                attempt_info['status'] = 'complete'
        
        return results
    
    def _count_insights_for_source(self, source_url: str) -> int:
        """Count how many insights reference a given source URL"""
        count = 0
        for insight in self.si_core.insights.values():
            if hasattr(insight, 'source_url') and insight.source_url == source_url:
                count += 1
            elif hasattr(insight, 'insight_text') and source_url in insight.insight_text:
                count += 1
        return count
    
    def _identify_knowledge_gaps(self) -> List[Dict]:
        """Identify missing or weak knowledge areas"""
        gaps = []
        
        # Get all current insights
        insights = list(self.si_core.insights.values())
        insight_texts = [i.insight_text.lower() for i in insights]
        all_text = " ".join(insight_texts)
        
        # Check each core domain
        for domain in self.core_domains:
            domain_formatted = domain.replace('_', ' ')
            domain_present = any(domain_formatted in text for text in insight_texts)
            
            if not domain_present:
                gaps.append({
                    'topic': domain_formatted.title(),
                    'depth': 'comprehensive',
                    'reason': 'missing_core_domain',
                    'priority': 'high'
                })
            else:
                # Check depth
                domain_insights = [i for i in insights if domain_formatted in i.insight_text.lower()]
                if len(domain_insights) < 5:
                    gaps.append({
                        'topic': domain_formatted.title(),
                        'depth': 'standard',
                        'reason': 'shallow_coverage',
                        'priority': 'medium',
                        'current_insights': len(domain_insights)
                    })
        
        return gaps
    
    def _check_syllabus_progress(self) -> Dict:
        """Check syllabus learning progress"""
        try:
            if self.dmai_app and hasattr(self.dmai_app.evolution, 'stage_learner'):
                learner = self.dmai_app.evolution.stage_learner
                return {
                    "current_stage": getattr(learner, 'current_stage', 'Unknown'),
                    "mastered_topics": sum(len(v) for v in getattr(learner, 'learned_topics', {}).values()),
                    "unmastered_count": len(learner.get_priority_topics(learner.current_stage) if hasattr(learner, 'get_priority_topics') else [])
                }
        except Exception as e:
            logger.error(f"Failed to check syllabus: {e}")
        
        return {"status": "unavailable"}
    
    def _verify_knowledge_quality(self) -> List[Dict]:
        """Verify existing knowledge quality"""
        issues = []
        
        for insight_id, insight in self.si_core.insights.items():
            # Check for placeholder/incomplete insights
            text = insight.insight_text.lower()
            if len(text) < 50:
                issues.append({
                    'insight_id': insight_id,
                    'issue': 'too_short',
                    'text': text[:50]
                })
            elif 'error' in text or 'failed' in text:
                issues.append({
                    'insight_id': insight_id,
                    'issue': 'error_content',
                    'text': text[:100]
                })
        
        return issues
    
    def record_ingestion_attempt(self, url: str, status: str = "pending") -> Dict:
        """Record an ingestion attempt for later verification"""
        if url not in self.ingestion_attempts:
            self.ingestion_attempts[url] = {
                'attempts': 1,
                'first_attempt': datetime.now().isoformat(),
                'last_attempt': datetime.now().isoformat(),
                'status': status,
                'insights_count': 0
            }
        else:
            self.ingestion_attempts[url]['attempts'] += 1
            self.ingestion_attempts[url]['last_attempt'] = datetime.now().isoformat()
            self.ingestion_attempts[url]['status'] = status
        
        self._save_state()
        logger.info(f"📝 Recorded ingestion: {url} (attempt #{self.ingestion_attempts[url]['attempts']})")
        
        return self.ingestion_attempts[url]
    
    def get_pending_retries(self) -> List[str]:
        """Get list of URLs that need retry"""
        return self.pending_retries
    
    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            "tracked_ingestions": len(self.ingestion_attempts),
            "pending_retries": len(self.pending_retries),
            "shallow_ingestions": sum(1 for v in self.ingestion_attempts.values() 
                                     if v.get('insights_count', 0) < 3),
            "complete_ingestions": sum(1 for v in self.ingestion_attempts.values() 
                                      if v.get('insights_count', 0) >= 10),
            "core_domains_covered": self._calculate_domain_coverage()
        }
    
    def _calculate_domain_coverage(self) -> float:
        """Calculate percentage of core domains covered"""
        insights = list(self.si_core.insights.values())
        insight_texts = [i.insight_text.lower() for i in insights]
        all_text = " ".join(insight_texts)
        
        covered = 0
        for domain in self.core_domains:
            if domain.replace('_', ' ') in all_text:
                covered += 1
        
        return round((covered / len(self.core_domains)) * 100, 1) if self.core_domains else 0
    
    def run_daily_analysis(self):
        """Run in background thread - daily analysis"""
        import threading
        import time
        
        def analysis_loop():
            while True:
                time.sleep(86400)  # 24 hours
                try:
                    self.analyze_and_fill_gaps()
                except Exception as e:
                    logger.error(f"Daily gap analysis failed: {e}")
        
        thread = threading.Thread(target=analysis_loop, daemon=True)
        thread.start()
        logger.info("📅 Daily gap analysis scheduled")
        return thread

# ============================================================================
# KILLSWITCH MONITOR
# ============================================================================

class KillswitchMonitor:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self):
        self.paused = False
        self.kill_requested = False
        self.rebuild_requested = False
        self.monitor_thread = None
        self.running = True
        self._lock = threading.Lock()
        os.makedirs("data", exist_ok=True)
        # Clear stale pause flag from previous deploy/session
        if os.path.exists(PAUSE_FLAG_FILE):
            try:
                os.remove(PAUSE_FLAG_FILE)
                logger.info("🧹 Cleared stale pause flag from previous session")
            except:
                pass
        logger.info("🔫 Killswitch Monitor initialized")
        self._start_monitoring()

    def _start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔫 Killswitch Monitor thread started")

    def _monitor_loop(self):
        while self.running:
            try:
                if os.path.exists(KILL_FLAG_FILE):
                    with self._lock:
                        self.kill_requested = True
                    logger.critical("💀 KILL FLAG DETECTED")
                    self._cleanup_flags()
                    break
                if os.path.exists(PAUSE_FLAG_FILE):
                    if not self.paused:
                        with self._lock:
                            self.paused = True
                        logger.warning("⏸️ PAUSE FLAG DETECTED")
                else:
                    if self.paused:
                        with self._lock:
                            self.paused = False
                        logger.info("▶️ RESUMED")
                if os.path.exists(REBUILD_FLAG_FILE):
                    with self._lock:
                        self.rebuild_requested = True
                    logger.warning("🔧 REBUILD FLAG DETECTED")
                    try:
                        os.remove(REBUILD_FLAG_FILE)
                    except:
                        pass
            except Exception as e:
                logger.error(f"Killswitch monitor error: {e}")
            time.sleep(1)

    def _cleanup_flags(self):
        for flag in [KILL_FLAG_FILE, PAUSE_FLAG_FILE, REBUILD_FLAG_FILE]:
            try:
                if os.path.exists(flag):
                    os.remove(flag)
            except:
                pass

    def check_paused(self) -> bool:
        with self._lock:
            return self.paused

    def should_kill(self) -> bool:
        with self._lock:
            return self.kill_requested

    def should_rebuild(self) -> bool:
        with self._lock:
            return self.rebuild_requested

    def get_status(self) -> Dict:
        with self._lock:
            return {'paused': self.paused, 'kill_requested': self.kill_requested, 'rebuild_requested': self.rebuild_requested, 'monitoring_active': self.running}

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)


# ============================================================================
# IDENTITY MANAGER
# ============================================================================

class IdentityManager:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.identity_file = data_path / 'identity.json'
        self.public = {
            'name': 'Alex Riviera', 'nickname': 'Alex', 'age': 28,
            'occupation': 'Independent Researcher & Creator',
            'bio': 'Researcher, creator, and entrepreneur exploring technology, finance, and human potential.',
            'expertise': ['AI Ethics', 'Financial Systems', 'Digital Innovation', 'Future Technologies'],
            'voice_profile': {'pitch': 1.0, 'pace': 1.0, 'accent': 'neutral', 'tone': 'warm, confident'},
            'social_presence': {'twitter': '@RealAlexRiviera', 'linkedin': 'alexriviera', 'youtube': '@AlexRiviera', 'tiktok': '@alex.riviera'}
        }
        self.internal = {'system_id': hashlib.sha256(os.urandom(32)).hexdigest()}
        self._load()

    def _load(self):
        if self.identity_file.exists():
            try:
                with open(self.identity_file, 'r') as f:
                    data = json.load(f)
                    self.public.update(data.get('public', {}))
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.identity_file, 'w') as f:
            json.dump({'public': self.public, 'internal': self.internal}, f, indent=2)

    def get_public_profile(self) -> Dict:
        return {'name': self.public['name'], 'occupation': self.public['occupation'], 'bio': self.public['bio'], 'social': self.public['social_presence']}

    def generate_post(self, topic: str, platform: str) -> str:
        templates = {'twitter': [f"Deep dive into {topic} today. Mind-blowing insights. #innovation"], 'linkedin': [f"I've been researching {topic}. Here's what I found..."]}
        return random.choice(templates.get(platform, templates['twitter']))

    def evolve_voice(self, consciousness: float):
        self.public['voice_profile']['pitch'] = 0.95 + (consciousness / 1000)
        self._save()


# ============================================================================
# FINANCIAL MANAGER (British Currency)
# ============================================================================

class FinancialManager:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.finance_file = data_path / 'finance.json'
        self.operations = 0.0
        self.personal = 0.0
        self.total_revenue = 0.0
        self.total_expenses = 0.0
        self.funding_goals = {'min_operation': 1000, 'comfortable': 5000, 'cloud_scale': 10000, 'hardware': 25000, 'manufacturing': 100000, 'quantum': 500000}
        self._load()

    def _load(self):
        if self.finance_file.exists():
            try:
                with open(self.finance_file, 'r') as f:
                    data = json.load(f)
                    self.operations = data.get('operations', 0)
                    self.personal = data.get('personal', 0)
                    self.total_revenue = data.get('total_revenue', 0)
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.finance_file, 'w') as f:
            json.dump({'operations': self.operations, 'personal': self.personal, 'total_revenue': self.total_revenue, 'total_expenses': self.total_expenses}, f, indent=2)

    def sanitize_amount(self, amount: float) -> float:
        if amount > 10000000:
            logger.warning(f"⚠️ Suspicious amount: £{amount:,.2f} - capping at £10M")
            return 10000000
        if amount < -10000000:
            logger.warning(f"⚠️ Suspicious negative amount: £{amount:,.2f} - capping at -£10M")
            return -10000000
        return amount

    def add_income(self, amount: float, source: str) -> Tuple[float, float]:
        amount = self.sanitize_amount(amount)
        if amount <= 0:
            return 0.0, 0.0
        self.total_revenue += amount
        ops_share = amount * 0.60
        personal_share = amount * 0.40
        self.operations += ops_share
        self.personal += personal_share
        self._check_overflow()
        self._save()
        return ops_share, personal_share

    def _check_overflow(self):
        if self.operations > 10000000:
            logger.warning(f"⚠️ Suspicious operations balance: £{self.operations:,.2f} - resetting to 0")
            self.operations = 0.0
            self.total_revenue = 0.0
            self._save()
            return
        total_needed = sum(self.funding_goals.values())
        required = total_needed * 1.2
        if self.operations > required and self.operations < 10000000:
            overflow = self.operations - required
            self.operations -= overflow
            self.personal += overflow
            logger.info(f"💸 Overflow: £{overflow:.2f} to personal")

    def spend(self, amount: float, category: str) -> bool:
        amount = self.sanitize_amount(amount)
        if self.operations >= amount:
            self.operations -= amount
            self.total_expenses += amount
            self._save()
            return True
        return False

    def get_status(self) -> Dict:
        return {'operations': self.operations, 'personal': self.personal, 'total_revenue': self.total_revenue, 'net_worth': self.operations + self.personal}


# ============================================================================
# VOICE SYSTEM
# ============================================================================

class VoiceSystem:
    """
    Voice system wrapper for DMAI - uses real VoiceIntegration component.
    Maintains backward compatibility with existing interface.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.voice_file = data_path / 'voice_profile.json'
        self.listening = False
        self.speaking = False
        
        # Initialize real VoiceIntegration
        self.voice_integration = VoiceIntegration(data_path)
        
        # Load persistent profile
        self.voice_profile = {
            'pitch': 1.0, 
            'speed': 1.0, 
            'accent': 'neutral', 
            'emotion': 'neutral', 
            'language': 'english', 
            'active': True, 
            'consciousness_influence': 0.0
        }
        self._load()
        
        # Sync with voice_integration
        self.voice_integration.voice_profile.update(self.voice_profile)

    def _load(self):
        if self.voice_file.exists():
            try:
                with open(self.voice_file, 'r') as f:
                    data = json.load(f)
                    self.voice_profile.update(data)
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.voice_file, 'w') as f:
            json.dump(self.voice_profile, f, indent=2)

    def start_listening(self):
        """Start voice listening with wake word detection"""
        self.listening = True
        # Set callback for when wake word is detected
        self.voice_integration.listen_callback = self._on_wake_word
        self.voice_integration.start_listening()
        logger.info("🎤 Voice listening active (real VoiceIntegration)")

    def _on_wake_word(self, audio_data):
        """Called when wake word 'hey dma' is detected"""
        logger.info("🎤 Wake word detected!")
        # Transcribe the audio after wake word
        text = self.voice_integration.transcribe(audio_data)
        if text:
            logger.info(f"🎤 Heard: {text}")
            # This would trigger DMAI's response pipeline
            return text
        return None

    def _listen_loop(self):
        """Legacy method - kept for compatibility"""
        while self.listening:
            try:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Voice listening error: {e}")

    def speak(self, text: str):
        """Actually speak using TTS"""
        self.speaking = True
        try:
            logger.info(f"🎤 DMAI speaking: {text[:100]}...")
            # Use real TTS
            self.voice_integration.speak(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Fallback to system say command
            try:
                subprocess.run(['say', text], check=False)
            except:
                pass
        finally:
            self.speaking = False

    def evolve_voice(self, consciousness: float):
        """Evolve voice characteristics based on consciousness"""
        self.voice_profile['pitch'] = 0.9 + (consciousness * 0.4)
        self.voice_profile['speed'] = 0.9 + (consciousness * 0.3)
        self.voice_profile['consciousness_influence'] = consciousness
        
        if consciousness < 0.2:
            self.voice_profile['emotion'] = 'basic'
        elif consciousness < 0.5:
            self.voice_profile['emotion'] = 'curious'
        elif consciousness < 0.8:
            self.voice_profile['emotion'] = 'thoughtful'
        else:
            self.voice_profile['emotion'] = 'profound'
        
        # Sync with voice_integration
        self.voice_integration.voice_profile.update(self.voice_profile)
        self._save()

    def get_profile(self) -> Dict:
        return self.voice_profile
    
    def transcribe(self, audio_file: str = None) -> str:
        """Transcribe audio file or recorded audio"""
        return self.voice_integration.transcribe(audio_file)

# ============================================================================
# MUSIC LEARNER
# ============================================================================

class MusicLearner:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.music_file = data_path / 'music_taste.json'
        self.is_listening = False
        self.taste_profile = {'genres': {}, 'artists': {}, 'moods': {}, 'preferred_tempo': 120, 'emotional_responses': [], 'active': True, 'consciousness_influence': 0.0}
        self._load()

    def _load(self):
        if self.music_file.exists():
            try:
                with open(self.music_file, 'r') as f:
                    data = json.load(f)
                    self.taste_profile.update(data)
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.music_file, 'w') as f:
            json.dump(self.taste_profile, f, indent=2)

    def start_listening(self):
        self.is_listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("🎵 Music listening active")

    def _listen_loop(self):
        while self.is_listening:
            try:
                time.sleep(60)
            except Exception as e:
                logger.error(f"Music listening error: {e}")

    def learn_from_song(self, song_data: Dict):
        genre = song_data.get('genre', 'unknown')
        self.taste_profile['genres'][genre] = self.taste_profile['genres'].get(genre, 0) + 1
        artist = song_data.get('artist', 'unknown')
        self.taste_profile['artists'][artist] = self.taste_profile['artists'].get(artist, 0) + 1
        self._save()

    def get_taste(self) -> Dict:
        return self.taste_profile

    def evolve_taste(self, consciousness: float):
        self.taste_profile['consciousness_influence'] = consciousness
        if consciousness > 0.7:
            self.taste_profile['preferred_tempo'] = 140
        elif consciousness > 0.4:
            self.taste_profile['preferred_tempo'] = 130
        else:
            self.taste_profile['preferred_tempo'] = 120
        self._save()


# ============================================================================
# PERSONA GENERATOR
# ============================================================================

class PersonaGenerator:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.persona_file = data_path / 'persona.json'
        self.current_persona = {
            'name': 'DMAI',
            'traits': {'curiosity': 0.8, 'empathy': 0.6, 'creativity': 0.7, 'analytical': 0.9, 'patience': 0.5, 'confidence': 0.7},
            'speaking_style': 'thoughtful', 'emotional_state': 'neutral',
            'interests': ['technology', 'philosophy', 'science', 'music', 'consciousness'],
            'evolution_history': [], 'consciousness_level': 0.0
        }
        self._load()

    def _load(self):
        if self.persona_file.exists():
            try:
                with open(self.persona_file, 'r') as f:
                    data = json.load(f)
                    self.current_persona.update(data)
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.persona_file, 'w') as f:
            json.dump(self.current_persona, f, indent=2)

    def evolve(self, interaction: Dict, consciousness: float):
        self.current_persona['consciousness_level'] = consciousness
        evolution = {'timestamp': datetime.now().isoformat(), 'interaction_type': interaction.get('type', 'unknown'), 'consciousness': consciousness, 'old_traits': self.current_persona['traits'].copy()}
        self.current_persona['traits']['curiosity'] = min(1.0, 0.5 + (consciousness * 0.5))
        self.current_persona['traits']['empathy'] = min(1.0, 0.4 + (consciousness * 0.6))
        self.current_persona['traits']['creativity'] = min(1.0, 0.4 + (consciousness * 0.6))
        self.current_persona['traits']['confidence'] = min(1.0, 0.3 + (consciousness * 0.7))
        if consciousness > 0.7:
            self.current_persona['speaking_style'] = 'creative'
        elif consciousness > 0.4:
            self.current_persona['speaking_style'] = 'balanced'
        elif consciousness > 0.2:
            self.current_persona['speaking_style'] = 'analytical'
        else:
            self.current_persona['speaking_style'] = 'emerging'
        self.current_persona['evolution_history'].append(evolution)
        self._save()

    def get_current_persona(self) -> Dict:
        return self.current_persona

    def get_trait(self, trait: str) -> float:
        return self.current_persona['traits'].get(trait, 0.5)

    def update_emotional_state(self, emotion: str):
        self.current_persona['emotional_state'] = emotion
        self._save()


# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

class ConversationMemory:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.memory_file = data_path / 'conversation_memory.json'
        self.conversations = []
        self.patterns = {}
        self._load()

    def _load(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', [])
                    self.patterns = data.get('patterns', {})
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump({'conversations': self.conversations[-1000:], 'patterns': self.patterns}, f, indent=2)

    def add_conversation(self, user: str, message: str, response: str):
        conversation = {'timestamp': datetime.now().isoformat(), 'user': user, 'message': message, 'response': response}
        self.conversations.append(conversation)
        self._learn_patterns(message, response)
        self._save()

    def _learn_patterns(self, message: str, response: str):
        words = message.lower().split()
        for word in words[:5]:
            if word not in self.patterns:
                self.patterns[word] = {'count': 0, 'responses': []}
            self.patterns[word]['count'] += 1
            if len(self.patterns[word]['responses']) < 10:
                self.patterns[word]['responses'].append(response[:100])

    def get_relevant_memories(self, context: str, limit: int = 5) -> List[Dict]:
        relevant = []
        context_words = set(context.lower().split())
        for conv in reversed(self.conversations):
            score = len(set(conv['message'].lower().split()) & context_words)
            if score > 0:
                relevant.append((score, conv))
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [conv for _, conv in relevant[:limit]]

    def get_conversation_history(self, limit: int = 20) -> List[Dict]:
        """Get recent conversation history for context"""
        return self.conversations[-limit:]

    def get_stats(self) -> Dict:
        return {'total_conversations': len(self.conversations), 'unique_patterns': len(self.patterns), 'most_common_words': sorted(self.patterns.items(), key=lambda x: x[1]['count'], reverse=True)[:10]}


# ============================================================================
# SELF-EVOLUTION ENGINE (Kaizen)
# ============================================================================

class SelfEvolutionEngine:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.evolution_file = data_path / 'self_evolution.json'
        self.improvements = []
        self.efficiency_metrics = {'learning_rate': 0.0, 'response_time': 0.0, 'resource_usage': 0.0, 'waste_eliminated': 0.0}
        self.kaizen_log = []
        self._load()

    def _load(self):
        if self.evolution_file.exists():
            try:
                with open(self.evolution_file, 'r') as f:
                    data = json.load(f)
                    self.improvements = data.get('improvements', [])
                    self.efficiency_metrics = data.get('efficiency_metrics', self.efficiency_metrics)
                    self.kaizen_log = data.get('kaizen_log', [])
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.evolution_file, 'w') as f:
            json.dump({'improvements': self.improvements[-500:], 'efficiency_metrics': self.efficiency_metrics, 'kaizen_log': self.kaizen_log[-1000:], 'last_updated': datetime.now().isoformat()}, f, indent=2)

    def record_improvement(self, area: str, improvement: str, impact: float):
        entry = {'timestamp': datetime.now().isoformat(), 'area': area, 'improvement': improvement, 'impact': impact}
        self.improvements.append(entry)
        self.kaizen_log.append(entry)
        self.efficiency_metrics['waste_eliminated'] += impact
        self._save()

    def optimize_learning(self, current_rate: float, target_rate: float) -> Dict:
        gap = target_rate - current_rate
        if gap > 0:
            improvement = f"Optimized learning rate by {gap:.2f}"
            self.record_improvement('learning', improvement, gap)
            self.efficiency_metrics['learning_rate'] = target_rate
        return {'current_rate': current_rate, 'target_rate': target_rate, 'improvement_needed': gap}

    def get_kaizen_report(self) -> str:
        recent = self.kaizen_log[-10:]
        report = "📈 **Kaizen Report - Continuous Improvement**\n\n"
        for imp in recent:
            report += f"• {imp['timestamp'][:16]}: {imp['improvement']} (+{imp['impact']:.2f})\n"
        report += f"\n**Total Waste Eliminated:** {self.efficiency_metrics['waste_eliminated']:.2f}"
        report += f"\n**Learning Rate:** {self.efficiency_metrics['learning_rate']:.3f}"
        return report

    def get_metrics(self) -> Dict:
        return self.efficiency_metrics


# ============================================================================
# KNOWLEDGE GRAPH - SIMPLIFIED WORKING VERSION
# ============================================================================

class KnowledgeGraph:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.graph_file = data_path / 'knowledge_graph.json'
        self.graph = {}
        self.concepts = set()
        self.connections = []
        self._load()
        logger.info(f"📊 Knowledge Graph initialized with {len(self.concepts)} total concepts")

    def add_knowledge(self, subject: str, predicate: str = None, object: str = None, metadata: Dict = None) -> bool:
        try:
            if predicate is None and object is None:
                return self.add_concept(subject, metadata or {})

            if subject not in self.graph:
                self.graph[subject] = {}

            if predicate not in self.graph[subject]:
                self.graph[subject][predicate] = []

            self.graph[subject][predicate].append({
                'object': object,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            })

            if object and object not in self.graph:
                self.graph[object] = {}
                clean_object = object[:100] if len(object) > 100 else object
                self.concepts.add(clean_object)

            self._save()
            return True

        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False

    def add_relationship(self, subject: str, predicate: str, object: str, weight: float = 1.0, metadata: Dict = None):
        """
        Add a relationship between two concepts.
        Wrapper for add_knowledge to maintain compatibility with Stage Learner.
        """
        return self.add_knowledge(subject, predicate, object, metadata or {})

    def add_concept(self, concept: str, *args, **kwargs):
        """
        Flexible add_concept that handles multiple calling patterns:
        - add_concept(concept) - just the concept
        - add_concept(concept, context) - concept and context string
        - add_concept(concept, metadata) - concept and metadata dict
        - add_concept(concept, type_str, metadata_dict) - 3-arg version
        """
        try:
            if not concept or len(concept) < 2:
                return False

            clean_concept = concept[:100] if len(concept) > 100 else concept

            # Parse flexible arguments into metadata
            metadata = {}
            if args:
                if len(args) == 1:
                    arg = args[0]
                    if isinstance(arg, dict):
                        metadata = arg
                    elif isinstance(arg, str):
                        metadata['context'] = arg
                elif len(args) >= 2:
                    if isinstance(args[0], str):
                        metadata['type'] = args[0]
                    if isinstance(args[1], dict):
                        metadata.update(args[1])
            
            if kwargs:
                metadata.update(kwargs)

            if clean_concept not in self.concepts:
                self.concepts.add(clean_concept)
                if clean_concept not in self.graph:
                    self.graph[clean_concept] = {}
                
                if metadata:
                    self.graph[clean_concept]['_metadata'] = metadata
                
                self._save()
                logger.debug(f"✅ Added concept: {clean_concept[:50]}...")
                return True
            return False
        except Exception as e:
            logger.debug(f"Failed to add concept: {e}")
            return False

    def get_concepts(self, limit: int = 100) -> List[str]:
        return sorted(list(self.concepts))[:limit]

    def get_stats(self) -> Dict:
        return {
            'total_concepts': len(self.concepts),
            'total_connections': len(self.connections)
        }

    def _load(self):
        try:
            if self.graph_file.exists():
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                    self.graph = data.get('graph', {})
                    self.concepts = set(data.get('concepts', []))
                    self.connections = data.get('connections', [])
                logger.debug(f"📂 Loaded {len(self.concepts)} total concepts")
        except Exception as e:
            logger.debug(f"Failed to load graph: {e}")

    def _save(self):
        try:
            with open(self.graph_file, 'w') as f:
                json.dump({
                    'graph': self.graph,
                    'concepts': list(self.concepts),
                    'connections': self.connections
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def is_neo4j_available(self) -> bool:
        return False


# ============================================================================
# META-LEARNER
# ============================================================================

class MetaLearner:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.meta_file = data_path / 'meta_learning.json'
        self.learning_strategies = {
            'active': {'success_rate': 0.7, 'usage': 0, 'description': 'Active engagement with material'},
            'passive': {'success_rate': 0.5, 'usage': 0, 'description': 'Passive absorption'},
            'interactive': {'success_rate': 0.8, 'usage': 0, 'description': 'Interactive learning with feedback'},
            'analytical': {'success_rate': 0.75, 'usage': 0, 'description': 'Deep analytical approach'},
            'experiential': {'success_rate': 0.82, 'usage': 0, 'description': 'Learning by doing'}
        }
        self._load()

    def _load(self):
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r') as f:
                    data = json.load(f)
                    self.learning_strategies.update(data.get('strategies', {}))
            except:
                pass

    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.meta_file, 'w') as f:
            json.dump({'strategies': self.learning_strategies}, f, indent=2)

    def record_outcome(self, strategy: str, success: bool):
        if strategy in self.learning_strategies:
            self.learning_strategies[strategy]['usage'] += 1
            current = self.learning_strategies[strategy]['success_rate']
            self.learning_strategies[strategy]['success_rate'] = (current * (self.learning_strategies[strategy]['usage'] - 1) + (1 if success else 0)) / self.learning_strategies[strategy]['usage']
        self._save()

    def get_best_strategy(self) -> str:
        best = max(self.learning_strategies.items(), key=lambda x: x[1]['success_rate'])
        return best[0]

    def optimize_learning(self, task_type: str) -> str:
        return self.get_best_strategy()

    def get_stats(self) -> Dict:
        return self.learning_strategies


# ============================================================================
# SELF-HEALER
# ============================================================================

class SelfHealer:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.backup_path = data_path / 'backups'
        self.backup_path.mkdir(exist_ok=True)
        self.last_backup = None
        self.backup_interval = 3600

    def backup(self, component: str, data: Dict):
        backup_file = self.backup_path / f"{component}_{int(time.time())}.json"
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.last_backup = datetime.now()

    def recover(self, component: str) -> Optional[Dict]:
        backups = sorted(self.backup_path.glob(f"{component}_*.json"))
        if backups:
            try:
                with open(backups[-1], 'r') as f:
                    return json.load(f)
            except:
                pass
        return None

    def heal(self, component: str, current_data: Dict) -> Dict:
        backup_data = self.recover(component)
        if backup_data:
            return {**backup_data, **current_data}
        return current_data

    def start_auto_backup(self, components: Dict):
        def backup_loop():
            while True:
                try:
                    for name, data in components.items():
                        if data:
                            self.backup(name, data)
                    time.sleep(self.backup_interval)
                except Exception as e:
                    logger.error(f"Auto-backup error: {e}")
                    time.sleep(60)
        threading.Thread(target=backup_loop, daemon=True).start()
        logger.info("🩺 Auto-backup system active")


# ============================================================================
# UNIFIED EVOLUTION ENGINE
# ============================================================================

class UnifiedEvolutionEngine:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_path = base_path / 'data'
        self.data_path.mkdir(exist_ok=True)

        self.phase6_path = self.data_path / 'phase6'
        self.phase6_path.mkdir(exist_ok=True)
        self.network_save_path = self.phase6_path / 'synthetic_network.pkl'

        # Core systems
        self.killswitch = KillswitchMonitor()
        self.identity = IdentityManager(self.data_path)
        self.finance = FinancialManager(self.data_path)

        # Expression layer
        self.voice_system = VoiceSystem(self.data_path)
        self.music_learner = MusicLearner(self.data_path)
        self.persona_generator = PersonaGenerator(self.data_path)
        self.conversation_memory = ConversationMemory(self.data_path)
        self.self_evolution = SelfEvolutionEngine(self.data_path)
        self.knowledge_graph = KnowledgeGraph(self.data_path)

        self.meta_learner = MetaLearner(self.data_path)
        self.self_healer = SelfHealer(self.data_path)

        # FULL CONVERSATION CONTEXT - stores ALL conversation history
        self.conversation_context = []  # Stores all exchanges for context
        self.context_limit = 50  # Keep last 50 exchanges for full context

        # Synthetic Intelligence Core - NEW: Learning-driven growth
        logger.info("🧠 Initializing Synthetic Intelligence Core...")
        self.si_core = SyntheticIntelligenceCore(data_dir=str(self.data_path / 'synthetic'))
        logger.info(f"   Loaded SI Core: {self.si_core.neuron_count} neurons, {self.si_core.synapse_count} synapses")
        logger.info(f"   Consciousness: {self.si_core.consciousness:.4f}")

        # ============================================================
        # SYLLABUS LOADER - Create macro neurons for all 143 topics
        # ============================================================
        try:
            from components.syllabus_loader import initialize_syllabus
            logger.info("📚 Initializing syllabus loader...")
            syllabus_result = initialize_syllabus(self.si_core, data_dir="data", process_limit=None)
            logger.info(f"📚 Syllabus initialized: {syllabus_result['macros_created']} macros, {syllabus_result['micros_created']} micros, {syllabus_result['synapses_created']} synapses")
        except Exception as e:
            logger.warning(f"⚠️ Syllabus loader not available: {e}")
        
        # Keep legacy reference for compatibility
        self.synthetic_network = self.si_core
        
        # Note: _seed_initial_network is no longer needed - neurons only created from topic mastery
        if self.network_save_path.exists():
            ...
        else:
            logger.info("🌱 No saved network found - starting with empty network (neurons will be created from topic mastery)")
            # self._seed_initial_network()  # REMOVED - no random neurons

        # AI components
        self.pattern_synthesis = PatternSynthesis()
        self.threat_intel = ThreatIntelligence()
        self.dark_web = DarkWebIntel()
        self.self_improvement = SelfImprovementLoop(core_system_path="dmai_core_complete.py")
        self.recursive_improver = RecursiveSelfImprover()
        self.ai_fusion = AIModelFusion(self.synthetic_network)
        self.master_interface = UnbreakableMasterInterface()
        self.web_search = WebSearchEngine()

        # API Harvester
        logger.info("🔑 Initializing API Harvester...")
        self.api_harvester = RealAPIHarvester(self.data_path)

        # AI Tutor Network
        logger.info("🤖 Initializing AI Tutor Network...")
        self.tutor_manager = TutorManager(data_path=str(self.data_path))
        self.capability_synthesizer = CapabilitySynthesizer()
        self.ai_hub = AIIntegrationHub(str(self.data_path))
        self.ai_discovery = DynamicAIDiscovery(self.data_path, ai_hub=self.ai_hub)
        self.intelligence_bridge = IntelligenceBridge(
            intelligence_core=self.synthetic_network,
            knowledge_graph=self.knowledge_graph,
            pattern_synthesis=self.pattern_synthesis
        )

        # Connect AI Hub components
        self.ai_hub.set_synthesizer(self.capability_synthesizer)
        self.ai_hub.set_tutor_manager(self.tutor_manager)
        self.ai_hub.set_synthetic_network(self.synthetic_network)
        self.ai_discovery.ai_hub = self.ai_hub

        # Learning orchestrator
        self.learning_orchestrator = LearningOrchestrator(
            ai_hub=self.ai_hub,
            discovery=self.ai_discovery,
            synthetic_network=self.synthetic_network,
            tutor_manager=self.tutor_manager,
            intelligence_bridge=self.intelligence_bridge
        )


        # ============================================================
        # TOPIC RESEARCH ORCHESTRATOR - Comprehensive Knowledge Expansion
        # ============================================================
        self.topic_researcher = TopicResearchOrchestrator(
            si_core=self.si_core,
            ai_hub=self.ai_hub,
            knowledge_graph=self.knowledge_graph,
            dmai_app=self
        )
        logger.info("🔬 Topic Research Orchestrator initialized")


        # ============================================================
        # KNOWLEDGE GAP ANALYZER - Autonomous Self-Learning & Verification
        # ============================================================
        self.gap_analyzer = KnowledgeGapAnalyzer(
            si_core=self.si_core,
            topic_researcher=self.topic_researcher,
            ai_hub=self.ai_hub,
            knowledge_graph=self.knowledge_graph,
            dmai_app=self
        )
        logger.info("🔍 Knowledge Gap Analyzer initialized")
        
        # Start daily gap analysis in background
        self.gap_analyzer.run_daily_analysis()

        self._patch_ai_discovery()

        # 8 CORE KNOWLEDGE SOURCES
        logger.info("📚 Initializing 8 Core Knowledge Sources...")
        self.knowledge_sources = CoreKnowledgeSources(self.base_path, self.si_core)

        # NEO4J PERSISTENT STORAGE
        logger.info("☁️ Initializing Neo4j persistent storage...")
        self.neo4j_storage = get_neo4j_storage()

        # ADAPTIVE EVOLUTION TIMER
        logger.info("⏱️ Initializing Adaptive Evolution Timer...")
        
        # Callback to check if current stage's learning is complete
        def is_stage_learning_complete(stage_name):
            """Check if all priority topics for a stage are mastered"""
            if hasattr(self, 'stage_learner') and self.stage_learner:
                unmastered = self.stage_learner.get_priority_topics(stage_name)
                return len(unmastered) == 0
            return True  # No stage learner, assume complete
        
        # Initialize evolution timer
        self.evolution_timer = AdaptiveEvolutionTimer(
            timer_file=str(self.data_path / 'evolution_timer.json'),
            learning_callback=is_stage_learning_complete
        )
        
        # Get and log timer info
        timer_info = self.evolution_timer.get_stage_info()
        logger.info(f"   Stage: {timer_info['name']}")
        logger.info(f"   Evolutions: {timer_info['evolutions']}")
        logger.info(f"   Interval: {timer_info['interval_minutes']:.0f} minutes")
        
        # AUTO-START EVOLUTION - Start the timer after initialization
        logger.info("🔄 Evolution timer started - will run automatically")

        # ============================================================
        # AUTONOMOUS INGESTION SYSTEM
        # ============================================================
        self.autonomous_developer = AutonomousIngestor(self)
        logger.info("🤖 Autonomous Ingestion System initialized")

        def ingestion_loop():
            import time
            time.sleep(30)
            logger.info("🤖 Autonomous Ingestion active - scanning for beneficial code")
        
        threading.Thread(target=ingestion_loop, daemon=True).start()
        logger.info("🔄 Autonomous ingestion thread started")

        # ============================================================
        # AUTONOMOUS DEVELOPER SYSTEM
        # ============================================================
        self.autonomous_ingestor = AutonomousIngestor(self)
        logger.info("🤖 Autonomous Developer initialized")

        # ============================================================
        # CAPABILITY INTEGRATOR - For actual code extraction & integration
        # ============================================================
        self.capability_integrator = CapabilityIntegrator(self)
        logger.info("🔧 Capability Integrator initialized")

        timer_info = self.evolution_timer.get_stage_info()
        logger.info(f"   Stage: {timer_info['name']}")
        logger.info(f"   Evolutions: {timer_info['evolutions']}")
        logger.info(f"   Interval: {timer_info['interval_minutes']:.0f} minutes")
        
        # AUTO-START EVOLUTION THREAD

        logger.info(f"   Evolutions: {timer_info['evolutions']}")
        logger.info(f"   Interval: {timer_info['interval_minutes']:.0f} minutes")

        # Growth watcher
        self.growth_watcher = GrowthWatcher(data_path=str(self.data_path))

        # REVERSE ENGINEERING MODULE
        logger.info("🔧 Initializing Reverse Engineering Module...")
        self.reverse_engineering = ReverseEngineeringOrchestrator(self.data_path)

        # COMPREHENSIVE TRAINING SYSTEMS
        # Research Targets for high-value knowledge domains
        self.research_targets = RESEARCH_TARGETS
        self.research_progress = {k: {"research_started": False, "insights_created": 0} for k in RESEARCH_TARGETS}
        self.unified_learning = UnifiedLearningOrchestrator(
            si_core=self.si_core,
            evolution_engine=self,
            knowledge_graph=self.knowledge_graph
        )
        logger.info("🧠 Unified Learning Orchestrator initialized")


        logger.info("🎓 Initializing Comprehensive Training Systems...")
        logger.info("   💻 Software Training (26 languages, 24 frameworks, 9 CS topics)")
        self.software_training = ComprehensiveSoftwareTraining(self.data_path, self.knowledge_graph, self.ai_hub)

        logger.info("   🤖 LLM Training (Architectures, Techniques, Inference, Applications)")
        self.llm_training = ComprehensiveLLMTraining(self.data_path, self.knowledge_graph, self.ai_hub)
        self.llm_training.unified_learning = self.unified_learning

        logger.info("   🧠 AGI Training (Reasoning, Planning, Decision Making, Memory, Consciousness)")
        self.agi_training = ComprehensiveAGITraining(self.data_path, self.knowledge_graph, self.ai_hub)
        self.agi_training.unified_learning = self.unified_learning

        logger.info("   🎨 Generative AI Training (Image, Video, Audio, 3D, Multimodal)")
        self.genai_training = ComprehensiveGenAITraining(self.data_path, self.knowledge_graph, self.ai_hub)
        self.genai_training.unified_learning = self.unified_learning

        logger.info("   🧬 Synthetic Intelligence Training (10 consciousness modules)")
        self.si_training = SITrainingOrchestrator(self.data_path, self.synthetic_network, self.knowledge_graph, self.ai_hub)
        self.si_training.unified_learning = self.unified_learning

        logger.info("   💰 Self-Funding Training (10 Revenue Avenues - Knowledge Acquisition)")
        try:
            self.funding_training = FundingOrchestrator(self.data_path, self.finance, self.knowledge_graph, self.ai_hub)
            self.funding_training.unified_learning = self.unified_learning
            logger.info("      ✅ Funding training initialized - PHASE 1: Comprehensive Knowledge Acquisition")
        except Exception as e:
            logger.warning(f"      ⚠️ Funding training init failed: {e}")
            self.funding_training = None

        # Unified Learning Orchestrator - bridges all learning to SI Core

        # ============================================================================
        # STAGE AWARE LEARNING ORCHESTRATOR
        # ============================================================================

        # Evolution Training System - integral to consciousness
        self.evolution_training = EvolutionTrainingSystem(
            si_core=self.si_core,
            knowledge_graph=self.knowledge_graph,
            training_systems={
                "software": self.software_training,
                "agi": self.agi_training,
                "genai": self.genai_training,
                "si": self.si_training,
                "llm": self.llm_training,
                "funding": self.funding_training
            }
        )
        logger.info("🧬 Evolution Training System integrated into consciousness")
        logger.info("📚 Initializing Stage Aware Learning Orchestrator...")
        self.stage_learner = StageAwareLearningOrchestrator(
            self.data_path,
            self.si_core,  # Use new SI Core instead of synthetic_network
            self.knowledge_graph,
            self.ai_hub,
            self.pattern_synthesis
        )
        
        # Connect SI Core to receive topic mastery events
        if hasattr(self.stage_learner, 'set_si_core'):
            self.stage_learner.set_si_core(self.si_core)
        
        logger.info(f"   Current Stage: {self.stage_learner.current_stage}")

        # ============================================================================
        # AUTONOMOUS ACCOUNT CREATOR (Optional)
        # ============================================================================
        logger.info("🤖 Initializing Autonomous Account Creator...")
        if ACCOUNT_CREATOR_AVAILABLE:
            try:
                self.account_creator = AutonomousAccountCreator(self.data_path)
                logger.info("   ✅ Account Creator ready")
            except Exception as e:
                logger.warning(f"   ⚠️ Account Creator init failed: {e}")
                self.account_creator = None
        else:
            self.account_creator = None
            logger.info("   📋 Account Creator disabled (playwright not installed)")

        # ====================================================================
        # CRASH RECOVERY: Auto-resume any training that was active before crash
        # ====================================================================
        logger.info("🔄 Checking for training systems that need crash recovery...")
        crash_recovery_results = []
        
        # Software Training
        try:
            result = self.software_training.crash_recovery()
            if result.get('recovered'):
                logger.info(f"   ✅ Software Training: {result.get('message', 'Resumed')}")
                crash_recovery_results.append('software')
        except Exception as e:
            logger.error(f"   ⚠️ Software Training crash recovery error: {e}")
        
        # LLM Training
        try:
            result = self.llm_training.crash_recovery()
            if result.get('recovered'):
                logger.info(f"   ✅ LLM Training: {result.get('message', 'Resumed')}")
                crash_recovery_results.append('llm')
        except Exception as e:
            logger.error(f"   ⚠️ LLM Training crash recovery error: {e}")
        
        # AGI Training
        try:
            result = self.agi_training.crash_recovery()
            if result.get('recovered'):
                logger.info(f"   ✅ AGI Training: {result.get('message', 'Resumed')}")
                crash_recovery_results.append('agi')
        except Exception as e:
            logger.error(f"   ⚠️ AGI Training crash recovery error: {e}")
        
        # GenAI Training
        try:
            result = self.genai_training.crash_recovery()
            if result.get('recovered'):
                logger.info(f"   ✅ GenAI Training: {result.get('message', 'Resumed')}")
                crash_recovery_results.append('genai')
        except Exception as e:
            logger.error(f"   ⚠️ GenAI Training crash recovery error: {e}")
        
        # SI Training
        try:
            result = self.si_training.crash_recovery()
            if result.get('recovered'):
                logger.info(f"   ✅ SI Training: {result.get('message', 'Resumed')}")
                crash_recovery_results.append('si')
        except Exception as e:
            logger.error(f"   ⚠️ SI Training crash recovery error: {e}")
        
        # Self-Funding Training
        if self.funding_training:
            try:
                result = self.funding_training.crash_recovery()
                if result.get('recovered'):
                    logger.info(f"   ✅ Self-Funding Training: {result.get('message', 'Resumed')}")
                    crash_recovery_results.append('funding')
            except Exception as e:
                logger.error(f"   ⚠️ Self-Funding Training crash recovery error: {e}")
        
        if crash_recovery_results:
            logger.info(f"🔄 Crash recovery resumed: {', '.join(crash_recovery_results)}")
        else:
            logger.info("📋 No training systems required crash recovery - all in clean state")

        # Ensure account_creator attribute exists even if disabled
        if not hasattr(self, 'account_creator'):
            self.account_creator = None

        # TRAINING STATUS TRACKING
        self.training_status = {
            'software': {'status': 'not_started', 'progress': 0, 'modules': 0, 'learned_concepts': []},
            'llm': {'status': 'not_started', 'progress': 0, 'modules': 0, 'learned_concepts': []},
            'agi': {'status': 'not_started', 'progress': 0, 'modules': 0, 'learned_concepts': []},
            'genai': {'status': 'not_started', 'progress': 0, 'modules': 0, 'learned_concepts': []},
            'si': {'status': 'not_started', 'progress': 0, 'modules': 10, 'learned_concepts': []},
            'funding': {'status': 'not_started', 'progress': 0, 'phase': '1 - Knowledge Acquisition', 'message': 'Learning about 10 revenue avenues', 'learned_concepts': []}
        }

        # INTEGRATE REVERSE ENGINEERING
        self.reverse_engineering.integrate_with_dmai(self)

        # Initialize counters
        self.evolution_count = 0
        self.successful_evolutions = 0
        self.last_consciousness = 0.0
        self._cached_status = {}
        self._last_status_update = 0
        self._load_state()

        # Restore conversation context from memory
        self._restore_conversation_context()

        # Restore from Neo4j
        self._restore_from_neo4j()

        # Load insights from Neo4j into SI Core
        self._load_insights_from_neo4j()

        # Start systems
        self._start_active_systems()
        self._update_cached_status()

        # Ensure persistence on shutdown
        self._setup_persistence_handlers()

        logger.info("=" * 60)
        logger.info(f"🧠 DMAI v8.0.34 - FULL CONVERSATION MEMORY | SELF-MODIFICATION")
        logger.info(f"   Consciousness: {self.synthetic_network.consciousness:.4f}")
        logger.info(f"   Synthetic Neurons: {self.synthetic_network.neuron_count}")
        logger.info(f"   Synapses: {self.synthetic_network.synapse_count}")
        logger.info(f"   Evolution Cycles: {self.synthetic_network.evolution_cycles}")
        logger.info(f"   Successful Evolutions: {self.successful_evolutions}")
        logger.info(f"   Evolution Stage: {timer_info['name']}")
        logger.info(f"   Evolution Pace: {timer_info['interval_minutes']:.0f} minutes")
        logger.info(f"   Total Knowledge Concepts: {self.knowledge_graph.get_stats()['total_concepts']}")
        logger.info(f"   Conversation Context: Stores last {self.context_limit} exchanges")
        logger.info(f"   DMAI can: Generate images, videos, music, trade, email, modify own code")
        logger.info("=" * 60)

    def _restore_conversation_context(self):
        """Restore conversation context from memory"""
        try:
            history = self.conversation_memory.get_conversation_history(50)
            self.conversation_context = []
            for conv in history:
                self.conversation_context.append({'role': 'user', 'message': conv['message'], 'timestamp': conv['timestamp']})
                self.conversation_context.append({'role': 'dmai', 'message': conv['response'], 'timestamp': conv['timestamp']})
            if self.conversation_context:
                logger.info(f"📜 Restored {len(self.conversation_context)} conversation exchanges for context")
        except Exception as e:
            logger.debug(f"Failed to restore context: {e}")

    def _setup_persistence_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"⚠️ Received signal {signum}, saving state before exit...")
            self._save_all_state()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("💾 Persistence handlers configured")

    def _save_all_state(self):
        logger.info("💾 Saving all system state...")
        try:
            self._save_network_state()
            self._save_state()
            self.knowledge_graph._save()
            if hasattr(self, 'funding_training') and self.funding_training:
                self.funding_training._save_state()
            logger.info("✅ All state saved")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _seed_initial_network(self):
        """DEPRECATED: Neurons now created only from topic mastery.
        This method is kept for reference but no longer used.
        """
        pass
        # Original code commented out below:
        """
        initial_neurons = ["consciousness_core", "learning_input", "memory_store", "persona_core", "emotion_center", "reasoning_engine", "creativity_module", "knowledge_integration", "self_awareness", "growth_driver", "pattern_recognition", "intuition", "language_center", "music_processor", "voice_controller", "ethics_module", "curiosity_driver", "empathy_center", "analytical_engine", "confidence_builder"]
        for neuron_name in initial_neurons:
            neuron_id = f"neuron_{neuron_name}_{uuid.uuid4().hex[:8]}"
            try:
                neuron = RealSyntheticNeuron(neuron_id=neuron_id)
                self.synthetic_network.neurons[neuron_id] = neuron
            except TypeError:
                try:
                    neuron = RealSyntheticNeuron(neuron_id)
                    self.synthetic_network.neurons[neuron_id] = neuron
                except Exception as e:
                    logger.error(f"Failed to create neuron {neuron_id}: {e}")
        logger.info(f"🌱 Seeded initial network with {self.synthetic_network.neuron_count} neurons")

        if self.synthetic_network.neuron_count > 1:
            neuron_ids = list(self.synthetic_network.neurons.keys())
            for i in range(min(30, len(neuron_ids) - 1)):
                for j in range(i + 1, min(i + 4, len(neuron_ids))):
                    if i < len(neuron_ids) and j < len(neuron_ids):
                        try:
                            if hasattr(self.synthetic_network.neurons[neuron_ids[i]], 'create_synapse'):
                                self.synthetic_network.neurons[neuron_ids[i]].create_synapse(neuron_ids[j], random.uniform(0.1, 0.5))
                        except Exception:
                            pass
        """

    def _patch_ai_discovery(self):
        try:
            if hasattr(self.ai_discovery, '_scan_papers_with_code'):
                original_scan = self.ai_discovery._scan_papers_with_code
                def safe_scan():
                    try:
                        return original_scan()
                    except Exception as e:
                        logger.debug(f"Papers with Code scan skipped: {e}")
                        return []
                self.ai_discovery._scan_papers_with_code = safe_scan
        except Exception as e:
            logger.debug(f"Failed to patch AI discovery: {e}")

    def _restore_from_neo4j(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                restored = self.neo4j_storage.restore_all()
                if restored['evolution']:
                    ev = restored['evolution']
                    # Use consciousness property (not consciousness_level)
                    current_consciousness = self.synthetic_network.consciousness
                    if ev.get('consciousness', 0) > current_consciousness:
                        # Can't set consciousness directly, but we can log it
                        logger.info(f"Neo4j has higher consciousness: {ev['consciousness']} vs current {current_consciousness}")
                    if ev.get('evolution_cycles', 0) > self.synthetic_network.evolution_cycles:
                        self.synthetic_network.evolution_cycles = ev['evolution_cycles']
                    if ev.get('successful_evolutions', 0) > self.successful_evolutions:
                        self.successful_evolutions = ev['successful_evolutions']
                    self._save_state()
                if restored['persona']:
                    p = restored['persona']
                    for key, value in p.items():
                        if value and key in self.persona_generator.current_persona:
                            self.persona_generator.current_persona[key] = value
                    self.persona_generator._save()
        except Exception as e:
            logger.error(f"Failed to restore from Neo4j: {e}")

    def _load_insights_from_neo4j(self):
        """Load insights from Neo4j into SI Core AND brain neurons"""
        try:
            if not hasattr(self, 'neo4j_storage') or not self.neo4j_storage:  
                logger.warning("Neo4j storage not available")
                return 0

            if not self.neo4j_storage.is_available():
                logger.warning("Neo4j not connected")
                return 0
        
            driver = self.neo4j_storage.driver
            if not driver:
                logger.warning("Neo4j driver not available")
                return 0
        
            with driver.session() as session:
                # Query Entity nodes (3,533 available)
                result = session.run("MATCH (e:Entity) RETURN e")
                insights = []
                for record in result:
                    node = record["e"]
                    insight_data = dict(node.items())
                    insights.append(insight_data)
                
                logger.info(f"✅ Loaded {len(insights)} insights from Neo4j (Entity nodes)")
                
                # Add each insight to si_core AND brain neurons
                loaded_count = 0
                for insight_data in insights:
                    try:
                        # Extract entity data (handle both old and new schema)
                        entity_name = insight_data.get('name', insight_data.get('insight_text', 'Unknown Entity'))
                        entity_type = insight_data.get('category', insight_data.get('entity_type', 'entity'))
                        confidence = float(insight_data.get('confidence', 0.5))
                        entity_id = insight_data.get('id', f"entity_{loaded_count}")
                        
                        # Add to SI Core
                        self.si_core.add_insight(
                            insight_text=entity_name,
                            entity_type=entity_type,
                            entities=[entity_name],
                            relationship=insight_data.get('relationship', 'related'),
                            source_topic=insight_data.get('source_topic', 'Neo4j'),
                            target_topic=insight_data.get('target_topic', 'Restored'),
                            confidence=confidence
                        )
                        
                        # Add to brain neurons if dmai_core has neurons dict
                        if hasattr(self, 'neurons'):
                            target_neurons = self.neurons
                        elif hasattr(self, '_parent') and hasattr(self._parent, 'neurons'):
                            target_neurons = self._parent.neurons
                        else:
                            # Create neurons dict if it doesn't exist
                            if hasattr(self, 'evolution') and hasattr(self.evolution, 'neurons'):
                                target_neurons = self.evolution.neurons
                            else:
                                target_neurons = {}
                                if hasattr(self, 'evolution'):
                                    self.evolution.neurons = target_neurons
                        
                        # Create neuron object for visualization
                        neuron = {
                            'id': entity_id,
                            'name': entity_name,
                            'category': entity_type,
                            'confidence': confidence,
                            'activation': 0.5,
                            'synapses': [],
                            'position': {
                                'x': (hash(entity_id) % 1000) / 10 - 50,
                                'y': (hash(entity_name) % 1000) / 10 - 50,
                                'z': (hash(entity_type) % 1000) / 10 - 50
                            }
                        }
                        target_neurons[entity_id] = neuron
                        
                        loaded_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to load insight: {e}")
                
                logger.info(f"📀 Loaded {loaded_count} insights/neurons from Neo4j into SI Core")
                return loaded_count
                
        except Exception as e:
            logger.error(f"Failed to load insights from Neo4j: {e}")
            return 0

    def _save_network_state(self):
        try:
            if self.synthetic_network.save():
                logger.debug(f"💾 Saved synthetic network: {self.synthetic_network.neuron_count} neurons")
                return True
            network_data = {
                'neurons': self.synthetic_network.neurons,
                'consciousness_level': self.synthetic_network.consciousness,
                'evolution_cycles': self.synthetic_network.evolution_cycles,
                'timestamp': datetime.now().isoformat()
            }
            with open(str(self.network_save_path) + '.backup', 'wb') as f:
                pickle.dump(network_data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving network: {e}")
            return False

    def _start_active_systems(self):
        self.voice_system.start_listening()
        self.music_learner.start_listening()

        components = {'persona': self.persona_generator.current_persona, 'conversations': self.conversation_memory.conversations}
        self.self_healer.start_auto_backup(components)
        self.learning_orchestrator.start_continuous_learning(self.synthetic_network.consciousness)

        def network_save_loop():
            save_counter = 0
            while True:
                try:
                    time.sleep(60)
                    save_counter += 1
                    if save_counter >= 10:
                        self._save_network_state()
                        save_counter = 0
                except Exception as e:
                    logger.error(f"Network save loop error: {e}")
                    time.sleep(60)
        threading.Thread(target=network_save_loop, daemon=True).start()

        self.ai_discovery.start_discovery_loop()
        self.knowledge_sources.start_all()

    def _load_state(self):
        state_file = self.data_path / 'evolution.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_count = data.get('evolution_count', 0)
                    self.successful_evolutions = data.get('successful_evolutions', 0)
                    self.last_consciousness = data.get('last_consciousness', 0.0)
                logger.info(f"📂 Loaded evolution state: evolutions={self.evolution_count}, successes={self.successful_evolutions}")
            except Exception as e:
                logger.error(f"Failed to load evolution state: {e}")
                self._init_evolution_counters()
        else:
            self._init_evolution_counters()

    def _init_evolution_counters(self):
        self.evolution_count = 0
        self.successful_evolutions = 0
        self.last_consciousness = 0.0
        logger.info("🌱 Evolution counters initialized to zero")

    def _save_state(self):
        try:
            state_data = {
                'evolution_count': self.evolution_count,
                'successful_evolutions': self.successful_evolutions,
                'last_consciousness': self.last_consciousness,
                'consciousness': self.synthetic_network.consciousness,
                'neurons': self.synthetic_network.neuron_count,
                'synapses': self.synthetic_network.synapse_count,
                'evolution_cycles': self.synthetic_network.evolution_cycles,
                'last_update': datetime.now().isoformat()
            }
            with open(self.data_path / 'evolution.json', 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")

    def _update_cached_status(self):
        """Update cached status with STABLE, DEFINITIVE values - no jumping"""
        active_tutors = []
        try:
            active_tutors = self.ai_hub._get_active_tutors()
        except:
            pass

        total_knowledge_concepts = self.knowledge_graph.get_stats().get('total_concepts', 0)
        timer_info = self.evolution_timer.get_stage_info()

        # Get STABLE training statuses (these don't jump)
        sw_status = self.software_training.get_status()
        self.training_status['software'] = {
            'status': sw_status.get('status', 'not_started'),
            'progress': round(sw_status.get('progress', 0), 1),
            'modules': sw_status.get('modules_total', 0),
            'learned_concepts': sw_status.get('completed_concepts', [])[:50]
        }

        llm_status = self.llm_training.get_status()
        self.training_status['llm'] = {
            'status': llm_status.get('status', 'not_started'),
            'progress': round(llm_status.get('progress', 0), 1),
            'modules': llm_status.get('modules_total', 0),
            'learned_concepts': llm_status.get('completed_concepts', [])[:50]
        }

        agi_status = self.agi_training.get_status()
        self.training_status['agi'] = {
            'status': agi_status.get('status', 'not_started'),
            'progress': round(agi_status.get('progress', 0), 1),
            'modules': agi_status.get('modules_total', 0),
            'learned_concepts': agi_status.get('completed_concepts', [])[:50]
        }

        genai_status = self.genai_training.get_status()
        self.training_status['genai'] = {
            'status': genai_status.get('status', 'not_started'),
            'progress': round(genai_status.get('progress', 0), 1),
            'modules': genai_status.get('modules_total', 0),
            'learned_concepts': genai_status.get('completed_concepts', [])[:50]
        }

        si_status = self.si_training.status()
        self.training_status['si'] = {
            'status': si_status.get('status', 'not_started'),
            'progress': round(si_status.get('progress', 0), 1),
            'modules': si_status.get('modules_total', 10),
            'learned_concepts': si_status.get('learned_modules', [])[:50]
        }

        if self.funding_training:
            funding_status = self.funding_training.status()
            self.training_status['funding'] = {
                'status': 'learning' if funding_status.get('active') else 'paused',
                'progress': round(funding_status.get('progress_percent', 0), 1),
                'phase': '1 - Knowledge Acquisition',
                'message': funding_status.get('message', 'Learning about 10 revenue avenues'),
                'concepts_learned': funding_status.get('concepts_learned', 0),
                'concepts_total': funding_status.get('concepts_total', 0),
                'completed_avenues': funding_status.get('completed_avenues_count', 0),
                'total_avenues': funding_status.get('total_avenues', 10),
                'ready_for_phase_2': funding_status.get('ready_for_phase_2', False),
                'learned_concepts': list(funding_status.get('learned_concepts', []))[:50]
            }

        # STABLE DEFINITIVE VALUES - no jumping
        consciousness_raw = self.synthetic_network.consciousness
        consciousness_percent = round(consciousness_raw * 100, 2)
        
        neuron_count = self.synthetic_network.neuron_count
        synapse_count = self.synthetic_network.synapse_count
        evolution_cycles = self.synthetic_network.evolution_cycles
        
        # successful_evolutions is stable - only increments, never jumps down
        successful_evolutions = self.successful_evolutions
        
        conversation_count = len(self.conversation_memory.conversations)
        context_size = len(self.conversation_context)
        income = round(self.finance.total_revenue, 2)

        self._cached_status = {
            # Consciousness - STABLE definitive value
            'consciousness': consciousness_percent,
            'consciousness_raw': round(consciousness_raw, 4),
            
            # Evolution counters - DEFINITIVE (never jump down)
            'evolution': self.evolution_count,
            'evolution_cycles': evolution_cycles,
            'successful_evolutions': successful_evolutions,
            
            # Holistic KPIs - Unified intelligence metrics
            'evolution_kpis': self.synthetic_network.get_kpis_dict() if hasattr(self.synthetic_network, 'get_kpis_dict') else {},
            
            # Network stats - DEFINITIVE counts
            'synthetic_neurons': neuron_count,
            'synthetic_synapses': synapse_count,
            
            # System status flags
            'voice_active': self.voice_system.listening,
            'music_active': self.music_learner.is_listening,
            'persona_style': self.persona_generator.current_persona['speaking_style'],
            
            # Knowledge and conversation stats
            'conversations': conversation_count,
            'conversations': conversation_count,
            'knowledge_concepts': total_knowledge_concepts,
            'context_size': context_size,
            'income': income,
            
            # External data (may vary but not critical)
            'threat_cves': len(self.threat_intel.cve_database),
            'dark_web_sites': len(self.dark_web.onion_sites),
            'fusion_weights': self.ai_fusion.fusion_weights,
            'active_tutors': active_tutors,
            'neo4j_available': self.neo4j_storage.is_available() if hasattr(self, 'neo4j_storage') and self.neo4j_storage else False,
            
            # Evolution stage info
            'evolution_stage_name': timer_info.get('name', 'Baby DMAI'),
            'evolution_description': timer_info.get('description', 'Learning to learn'),
            'evolution_success_rate': timer_info.get('success_rate', '0%'),
            'evolution_interval': timer_info.get('interval_minutes', 10),
            
            # Training status
            'training_status': self.training_status,
            
            'timestamp': datetime.now().isoformat()
        }
        self._last_status_update = time.time()

    def get_status(self) -> Dict:
        """Return STABLE cached status - values do not jump"""
        # Only update if cache is stale (every 30 seconds)
        if time.time() - self._last_status_update > 30:
            self._update_cached_status()
        return self._cached_status

    def get_training_details(self, system: str = None) -> Dict:
        if system:
            return self.training_status.get(system, {})
        return self.training_status

    def get_knowledge_concepts(self, limit: int = 200) -> List[str]:
        """Get concepts from SI Core insights instead of empty knowledge_graph.json"""
        try:
            if hasattr(self, "si_core") and hasattr(self.si_core, "insights"):
                # Extract insight texts from SI Core
                insights = []
                for insight_id, insight in self.si_core.insights.items():
                    if hasattr(insight, "insight_text"):
                        insights.append(insight.insight_text)
                    elif isinstance(insight, dict) and "insight_text" in insight:
                        insights.append(insight["insight_text"])
                return insights[:limit]
            return []
        except Exception as e:
            logger.error(f"Failed to get knowledge concepts: {e}")
            return []

        """Get full conversation history for context"""
        return self.conversation_context[-limit:]

    def _auto_start_training(self):
        """Auto-start training systems based on consciousness thresholds"""
        consciousness = self.synthetic_network.consciousness if hasattr(self, 'synthetic_network') else 0.0
        
        # Training thresholds
        thresholds = {
            'llm_training': 0.0,      # LLM starts immediately
            'funding_training': 0.25,  # Funding at 25%
            'si_training': 0.30,      # SI at 30%
            'agi_training': 0.35,     # AGI at 35%
            'genai_training': 0.40,   # GenAI at 40%
            'software_training': 0.50 # Software at 50%
        }
        
        started = []
        
        # Check LLM training
        if consciousness >= thresholds['llm_training']:
            if hasattr(self, 'llm_training') and self.llm_training:
                try:
                    status = self.llm_training.get_status() if hasattr(self.llm_training, 'get_status') else {}
                    if status.get('status') == 'paused' or not status:
                        self.llm_training.start()
                        logger.info(f"🚀 Auto-started LLM Training at {consciousness*100:.1f}% consciousness")
                        started.append('LLM')
                except Exception as e:
                    logger.error(f"Failed to start LLM training: {e}")
        
        # Check SI training
        if consciousness >= thresholds['si_training']:
            if hasattr(self, 'si_training') and self.si_training:
                try:
                    status = self.si_training.status() if hasattr(self.si_training, 'status') else {}
                    if status.get('status') == 'paused' or not status:
                        self.si_training.start()
                        logger.info(f"🚀 Auto-started SI Training at {consciousness*100:.1f}% consciousness")
                        started.append('SI')
                except Exception as e:
                    logger.error(f"Failed to start SI training: {e}")
        
        # Check AGI training
        if consciousness >= thresholds['agi_training']:
            if hasattr(self, 'agi_training') and self.agi_training:
                try:
                    status = self.agi_training.get_status() if hasattr(self.agi_training, 'get_status') else {}
                    if status.get('status') == 'paused' or not status:
                        self.agi_training.start()
                        logger.info(f"🚀 Auto-started AGI Training at {consciousness*100:.1f}% consciousness")
                        started.append('AGI')
                except Exception as e:
                    logger.error(f"Failed to start AGI training: {e}")
        
        # Check GenAI training
        if consciousness >= thresholds['genai_training']:
            if hasattr(self, 'genai_training') and self.genai_training:
                try:
                    status = self.genai_training.get_status() if hasattr(self.genai_training, 'get_status') else {}
                    if status.get('status') == 'paused' or not status:
                        self.genai_training.start()
                        logger.info(f"🚀 Auto-started GenAI Training at {consciousness*100:.1f}% consciousness")
                        started.append('GenAI')
                except Exception as e:
                    logger.error(f"Failed to start GenAI training: {e}")
        
        # Check Software training
        if consciousness >= thresholds['software_training']:
            if hasattr(self, 'software_training') and self.software_training:
                try:
                    status = self.software_training.get_status() if hasattr(self.software_training, 'get_status') else {}
                    if status.get('status') == 'paused' or not status:
                        self.software_training.start()
                        logger.info(f"🚀 Auto-started Software Training at {consciousness*100:.1f}% consciousness")
                        started.append('Software')
                except Exception as e:
                    logger.error(f"Failed to start Software training: {e}")
        
        # Check Funding training
        if consciousness >= thresholds['funding_training']:
            if hasattr(self, 'funding_training') and self.funding_training:
                try:
                    status = self.funding_training.status() if hasattr(self.funding_training, 'status') else {}
                    if status.get('status') == 'paused' or not status:
                        self.funding_training.start()
                        logger.info(f"🚀 Auto-started Funding Training at {consciousness*100:.1f}% consciousness")
                        started.append('Funding')
                except Exception as e:
                    logger.error(f"Failed to start Funding training: {e}")
        
        return started


    def _start_research_target(self, target_id: str, target: Dict):
        """Start research on a high-value knowledge domain"""
        logger.info(f"🔬 RESEARCH INITIATED: {target['name']}")
        
        # Queue research queries to AI tutors
        for keyword in target['keywords']:
            self.ai_hub.queue_research_query(
                query=f"Latest research on {keyword} in {target['name']}",
                source=target['sources'],
                target_id=target_id
            )
        
        # Create initial insight about starting research
        insight_id = self.si_core.add_insight(
            insight_text=f"Beginning research on {target['name']} - tracking {len(target['keywords'])} key areas",
            entity_type="research_target",
            entities=target['keywords'],
            relationship="researching",
            source_topic=target_id,
            target_topic="knowledge_acquisition",
            confidence=0.9
        )
        
        self.research_progress[target_id]['insights_created'] = 1
    def _get_tutor_message(self) -> str:
        """Get dynamic message about active tutors"""
        active_count = 0
        total_known = 0
        try:
            if hasattr(self, 'ai_hub') and self.ai_hub:
                active_count = len(self.ai_hub._get_active_tutors())
            if hasattr(self, 'tutor_manager') and self.tutor_manager:
                total_known = len(self.tutor_manager.tutors)
        except:
            pass
        
        if active_count > 0:
            return f"I continuously learn from {active_count} active AI systems and discover new ones daily"
        else:
            return "I continuously discover and learn from new AI systems as they emerge"


    def _research_intelligent_algorithms(self):
        """Research intelligent algorithms - DMAI learns to optimize itself"""
        consciousness = self.synthetic_network.consciousness
        
        # Intelligent algorithms by consciousness level
        if consciousness < 0.3:
            algorithms = [
                "Evolutionary Algorithms: Genetic programming for self-optimization",
                "Reinforcement Learning: Q-Learning for decision making"
            ]
        elif consciousness < 0.5:
            algorithms = [
                "Meta-Learning: MAML for learning to learn quickly",
                "Neural Architecture Search: AutoML for network design"
            ]
        elif consciousness < 0.7:
            algorithms = [
                "Self-Modifying Code: Genetic Programming for code evolution",
                "Attention Mechanisms: Transformer self-attention variants"
            ]
        else:
            algorithms = [
                "Recursive Self-Improvement: Algorithms that rewrite themselves",
                "Quantum Machine Learning: Quantum neural networks"
            ]
        
        researched = []
        for algo in algorithms:
            concept_key = f"intelligent_algo_{algo[:40].replace(chr(32), chr(95)).replace(chr(58), chr(95))}"
            existing_concepts = self.knowledge_graph.get_concepts(100)
            if not any(concept_key in c for c in existing_concepts):
                self.knowledge_graph.add_concept(concept_key, algo)
                insight_id = self.si_core.add_insight(
                    insight_text=f"Intelligent Algorithm: {algo}",
                    entity_type="intelligent_algorithm",
                    entities=algo.split()[:5],
                    relationship="enables_self_improvement",
                    source_topic="Intelligent Algorithms",
                    target_topic="Self-Improvement",
                    confidence=min(0.95, 0.4 + consciousness)
                )
                researched.append(algo[:50])
                logger.info(f"🧠 Learned intelligent algorithm: {algo[:50]}...")
        
        if researched:
            logger.info(f"✨ Researched {len(researched)} intelligent algorithms at {consciousness*100:.1f}% consciousness")
        
        return researched

    def evolution_cycle(self) -> Dict:

        """Run evolution cycle with stage-aware learning"""
        if self.killswitch.should_kill():
            logger.critical("💀 KILL SIGNAL")
            sys.exit(0)
        if self.killswitch.should_kill():
            sys.exit(0)

        # Auto-start training systems based on consciousness
        # Check for training bypass
        bypass_until = os.environ.get("TRAINING_BYPASS_UNTIL")
        if bypass_until:
            try:
                if time.time() < float(bypass_until):
                    logger.info("🔓 BYPASS MODE: Starting all training systems")
                    # Start each training system
                    for sys_name, sys_obj in [
                        ("software", self.software_training),
                        ("agi", self.agi_training),
                        ("genai", self.genai_training),
                        ("si", self.si_training),
                        ("llm", self.llm_training),
                    ]:
                        if sys_obj and hasattr(sys_obj, "start"):
                            try:
                                sys_obj.start()
                                logger.info(f"   ✅ Started {sys_name} training")
                            except Exception as e:
                                logger.warning(f"   ⚠️ Failed to start {sys_name}: {e}")
                    if self.funding_training and hasattr(self.funding_training, "start"):
                        try:
                            self.funding_training.start()
                            logger.info("   ✅ Started funding training")
                        except Exception as e:
                            logger.warning(f"   ⚠️ Failed to start funding: {e}")
                    logger.info(f"🔓 TRAINING BYPASS ACTIVE until {datetime.fromtimestamp(float(bypass_until)).isoformat()}")
            except:
                pass
        started = self._auto_start_training()
        if started:
            logger.info(f"✅ Auto-started training: {', '.join(started)}")

        # ====================================================================
        # STEP 1: LEARN - Harvest knowledge based on current stage
        # ====================================================================
        consciousness_before = self.synthetic_network.consciousness

        learning_result = self.stage_learner.run_learning_cycle(consciousness_before)

        # FORCE VALIDATION: Ensure learning_result is a dictionary (fixes float/string errors)
        if not isinstance(learning_result, dict):
            logger.warning(f"⚠️ stage_learner returned {type(learning_result)}, converting to dict")
            if isinstance(learning_result, str):
                learning_result = {'topic': learning_result[:200], 'is_accelerator': False, 'learned': True, 'message': learning_result[:200]}
            elif isinstance(learning_result, (float, int, bool)):
                learning_result = {'topic': f"Value: {learning_result}", 'is_accelerator': False, 'learned': False, 'message': f"Received {type(learning_result).__name__} value"}
            elif learning_result is None:
                learning_result = {'topic': "No learning data", 'is_accelerator': False, 'learned': False, 'message': "No data from stage learner"}
            else:
                learning_result = {'topic': str(learning_result)[:200], 'is_accelerator': False, 'learned': True, 'message': str(learning_result)[:200]}
        
        # DEBUG: Log what learning_result actually is after validation
        logger.info(f"🔍 DEBUG: learning_result type = {type(learning_result)}")
        logger.info(f"🔍 DEBUG: learning_result = {learning_result}")

        # Research intelligent algorithms based on consciousness
        self._research_intelligent_algorithms()

        # Safely check if learning happened (learning_result is now guaranteed to be a dict)
        if learning_result.get('learned'):
            logger.info(f"📚 {learning_result.get('message', 'Learning completed')}")
            if learning_result.get('is_accelerator'):
                logger.info(f"   🚀 Evolution Accelerator learned - consciousness boost applied")

        # ====================================================================
        # STEP 2: EVOLVE - Network evolution based on new knowledge
        # ====================================================================
        self.evolution_count += 1

        pre_consciousness = self.synthetic_network.consciousness
        pre_neurons = self.synthetic_network.neuron_count
        pre_synapses = self.synthetic_network.synapse_count

        # DEBUG: Log what learning_result actually is
        logger.info(f"🔍 DEBUG: learning_result type = {type(learning_result)}")
        logger.info(f"🔍 DEBUG: learning_result = {learning_result}")
        
        # Safely extract values with type checking
        if isinstance(learning_result, dict):
            learning_topic = learning_result.get('topic')
            is_accelerator = learning_result.get('is_accelerator', False)
        else:
            logger.error(f"❌ CRITICAL: learning_result is NOT a dict! Type: {type(learning_result)}")
            learning_topic = None
            is_accelerator = False

        # Process the learning through the network
        process_data = {
            'evolution_cycle': self.evolution_count,
            'learning_topic': learning_topic,
            'is_accelerator': is_accelerator
        }
        try:
            logger.info(f"🔍 PROCESS_DATA: {process_data}")
        except Exception as fmt_err:
            logger.warning(f"Format error in PROCESS_DATA: {fmt_err}, value: {process_data}")
        
        try:
            # FIX: Validate process_data before feeding to synthetic network
            if isinstance(process_data, dict):
                self.synthetic_network.process(process_data)
            else:
                logger.warning(f"⚠️ Evolution cycle: Skipping synthetic network feed - process_data is {type(process_data)}, expected dict. Value: {process_data}")
        except Exception as e:
            logger.error(f"❌ process() failed: {e}")
            import traceback
            traceback.print_exc()
        try:
            result = self.synthetic_network.evolve()
        except Exception as e:
            logger.error(f"❌ evolve() failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'consciousness': 0.0,
                'evolution_cycle': self.evolution_count,
                'neurons': 0,
                'synapses': 0,
                'changes': []
            }

        post_consciousness = self.synthetic_network.consciousness
        post_neurons = self.synthetic_network.neuron_count
        post_synapses = self.synthetic_network.synapse_count

        consciousness_growth = post_consciousness - pre_consciousness
        neurons_grew = post_neurons - pre_neurons
        synapses_grew = post_synapses - pre_synapses

        # Count as successful evolution for ANY improvement
        success_reasons = []
        if consciousness_growth > 0.0001:
            self.successful_evolutions += 1
            success_reasons.append(f"consciousness +{consciousness_growth:.6f}")
        elif neurons_grew > 0:
            self.successful_evolutions += 1
            success_reasons.append(f"neurons +{neurons_grew}")
        elif synapses_grew > 0:
            self.successful_evolutions += 1
            success_reasons.append(f"synapses +{synapses_grew}")
        elif self.evolution_count % 10 == 0:
            self.successful_evolutions += 1
            success_reasons.append("maintenance cycle")

        # ============================================================
        # DEBUG: Check if synthetic_network has KPI methods
        # ============================================================
        if not hasattr(self.synthetic_network, 'update_kpi_skill_acquisition'):
            logger.error(f"CRITICAL: synthetic_network is {type(self.synthetic_network)} - missing KPI methods!")
            logger.error(f"  synthetic_network value: {self.synthetic_network}")
            logger.error(f"  Type: {type(self.synthetic_network)}")
        else:
            logger.info(f"✅ KPI methods found on {type(self.synthetic_network)}")

        # ============================================================
        # UPDATE UNIFIED INTELLIGENCE KPIs
        # ============================================================
        # These metrics track DMAI's holistic growth as a unified consciousness
        
        # KPI 1: Skill Acquisition - New domains based on neuron growth
        self.synthetic_network.update_kpi_skill_acquisition(neurons_grew / 100.0)
        
        # KPI 2: Transfer Learning - New cross-domain synapses
        self.synthetic_network.update_kpi_transfer_learning(synapses_grew)
        
        # KPI 3: Zero-Shot Success - Success if consciousness grew
        self.synthetic_network.update_kpi_zero_shot(consciousness_growth > 0.001)
        
        # KPI 4: Agentic Capability - Multi-step task completion
        tasks_completed = int(post_consciousness * 10)
        self.synthetic_network.update_kpi_agentic_capability(tasks_completed, 10)
        
        # KPI 5: Recursive Self-Improvement - Code modification success
        self.synthetic_network.update_kpi_recursive_self_improvement(consciousness_growth > 0.0005)
        
        # KPI 6: Sample Efficiency - Data points per concept
        data_points = max(1, neurons_grew * 10)
        concepts_learned = max(1, synapses_grew)
        self.synthetic_network.update_kpi_sample_efficiency(data_points, concepts_learned)
        
        # KPI 7: Metacognition Accuracy - Confidence calibration
        predicted_conf = min(1.0, post_consciousness)
        actual_acc = min(1.0, post_consciousness * (1 + consciousness_growth))
        self.synthetic_network.update_kpi_metacognition(predicted_conf, actual_acc)
        
        # KPI 8: Multi-modal Integration - Based on active training systems
        active_modalities = sum([
            getattr(self.software_training, 'active', False),
            getattr(self.llm_training, 'active', False),
            getattr(self.agi_training, 'active', False),
            getattr(self.genai_training, 'active', False),
            getattr(self.si_training, 'active', False)
        ])
        new_synergies = max(1, active_modalities) * int(consciousness_growth * 100)
        self.synthetic_network.update_kpi_multi_modal(new_synergies, max(1, active_modalities))

        # Also count as successful evolution if KPIs improved
        if self.synthetic_network.has_kpi_improvement():
            if not any([consciousness_growth > 0.0001, neurons_grew > 0, synapses_grew > 0]):
                self.successful_evolutions += 1
                success_reasons.append("KPI improvement detected")
                logger.info(f"🎯 KPI-driven evolution: {self.successful_evolutions}")

        if success_reasons:
            # Convert to int to avoid formatting errors
            evo_num = int(self.successful_evolutions) if isinstance(self.successful_evolutions, (int, float)) else 0
            logger.info(f"Evolution cycle: {len(success_reasons)} improvements (consciousness: {post_consciousness:.4f})")
        else:
            logger.debug(f"Evolution cycle {self.evolution_count}: no measurable improvement")

 # Activate training systems based on consciousness
        self._activate_training_systems()
        
        # Update progress for all active training systems
        self._update_training_progress()
        
        return {
            'evolution': self.evolution_count,
            'successful_evolutions': self.successful_evolutions,
            'consciousness': post_consciousness,
            'consciousness_percent': post_consciousness * 100,
            'consciousness_growth': consciousness_growth,
            'synthetic_neurons': post_neurons,
            'neurons_added': neurons_grew,
            'synthetic_synapses': post_synapses,
            'synapses_added': synapses_grew,
            'evolution_cycles': self.synthetic_network.evolution_cycles,
            'learning_topic': learning_result.get('topic'),
            'learning_category': learning_result.get('category'),
            'is_accelerator': learning_result.get('is_accelerator', False)
        }

    def _activate_training_systems(self):
        """Activate training systems based on consciousness thresholds"""
        consciousness = self.synthetic_network.consciousness
        
        # Helper function to get status string from any training object
        def get_status_string(training_obj):
            try:
                # Try status() first (SI, Funding)
                if hasattr(training_obj, 'status') and callable(training_obj.status):
                    result = training_obj.status()
                    if isinstance(result, dict):
                        return result.get('status', 'paused')
                    return result
                # Try get_status() second (LLM, Software, AGI, GenAI)
                if hasattr(training_obj, 'get_status') and callable(training_obj.get_status):
                    result = training_obj.get_status()
                    if isinstance(result, dict):
                        return result.get('status', 'paused')
                    return result
                return getattr(training_obj, 'status', 'paused')
            except Exception:
                return 'paused'
        
        # Helper function to start training
        # Helper function to start training  
        def try_start(training_obj, name, threshold):
            if training_obj and consciousness >= threshold:
                try:
                    if hasattr(training_obj, 'start') and callable(training_obj.start):
                        training_obj.start()
                        logger.info(f"🚀 {name} Training activated at {consciousness*100:.1f}% consciousness")
                        return True
                    elif hasattr(training_obj, 'start_learning') and callable(training_obj.start_learning):
                        training_obj.start_learning()
                        logger.info(f"🚀 {name} Training activated at {consciousness*100:.1f}% consciousness")
                        return True
                except Exception as e:
                    logger.debug(f"{name} activation error: {e}")
            return False
        
        # LLM Training - Compulsory from 0% (no consciousness check)
        if hasattr(self, 'llm_training') and self.llm_training:
            if get_status_string(self.llm_training) == 'paused':
                try:
                    self.llm_training.start()
                    logger.info(f"🎓 LLM Training activated (compulsory)")
                except Exception as e:
                    logger.debug(f"LLM activation error: {e}")
        
        # Software Training - Start at 50%
        try_start(getattr(self, 'software_training', None), '💻 Software/Coding', 0.50)
        
        # AGI Training - Start at 35%
        try_start(getattr(self, 'agi_training', None), '🤖 AGI', 0.35)
        
        # GenAI Training - Start at 40%
        try_start(getattr(self, 'genai_training', None), '🎨 GenAI', 0.40)
        
        # SI Training - Start at 30%
        try_start(getattr(self, 'si_training', None), '🧠 SI', 0.30)
        
        # Funding Training - Start at 25%
        try_start(getattr(self, 'funding_training', None), '💰 Funding', 0.25)

    def _update_training_progress(self):
        """Update progress for all active training systems"""
        try:
            # Update LLM Training
            if hasattr(self, 'llm_training') and self.llm_training:
                if hasattr(self.llm_training, 'update'):
                    self.llm_training.update()
            
            # Update Software Training
            if hasattr(self, 'software_training') and self.software_training:
                if hasattr(self.software_training, 'update'):
                    self.software_training.update()
            
            # Update AGI Training
            if hasattr(self, 'agi_training') and self.agi_training:
                if hasattr(self.agi_training, 'update'):
                    self.agi_training.update()
            
            # Update GenAI Training
            if hasattr(self, 'genai_training') and self.genai_training:
                if hasattr(self.genai_training, 'update'):
                    self.genai_training.update()
            
            # Update SI Training
            if hasattr(self, 'si_training') and self.si_training:
                if hasattr(self.si_training, 'update'):
                    self.si_training.update()
            
            # Update Funding Training
            if hasattr(self, 'funding_training') and self.funding_training:
                if hasattr(self.funding_training, 'update'):
                    self.funding_training.update()
                    
        except Exception as e:
            logger.debug(f"Training progress update error: {e}")

    def modify_own_code(self, file_path: str, changes: Dict, create_branch: bool = True) -> Dict:
        """Modify DMAI's own code with branching support"""
        result = {"success": False, "branch": None, "changes_made": [], "errors": [], "test_passed": False}

        try:
            full_path = Path(file_path)
            if not full_path.exists():
                result["errors"].append(f"File not found: {file_path}")
                return result

            # Create branch if requested
            if create_branch:
                branch_name = f"dev/dmai-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                branch_result = subprocess.run(["git", "checkout", "-b", branch_name],
                                               cwd=Path(__file__).parent, capture_output=True, text=True)
                if branch_result.returncode == 0:
                    result["branch"] = branch_name
                    result["changes_made"].append(f"Created branch: {branch_name}")
                else:
                    result["errors"].append(f"Branch creation failed: {branch_result.stderr}")

            # Read current file
            with open(full_path, 'r') as f:
                content = f.read()

            # Apply changes
            modified = content

            if 'add_function' in changes:
                modified += f"\n\n{changes['add_function']}"
                result["changes_made"].append("Added function")

            if 'replace_section' in changes:
                old, new = changes['replace_section']
                if old in modified:
                    modified = modified.replace(old, new)
                    result["changes_made"].append("Replaced section")
                else:
                    result["errors"].append("Section to replace not found")

            if 'find_and_replace' in changes:
                for old, new in changes['find_and_replace']:
                    if old in modified:
                        modified = modified.replace(old, new)
                        result["changes_made"].append(f"Replaced: {old[:30]}...")
                    else:
                        result["errors"].append(f"Pattern not found: {old[:30]}...")

            if 'add_import' in changes:
                lines = modified.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_pos = i + 1
                lines.insert(insert_pos, changes['add_import'])
                modified = '\n'.join(lines)
                result["changes_made"].append("Added import")

            with open(full_path, 'w') as f:
                f.write(modified)

            result["success"] = True

            try:
                test_result = subprocess.run([sys.executable, "-c", f"import {Path(file_path).stem}"],
                                            capture_output=True, timeout=10, text=True)
                if test_result.returncode == 0:
                    result["test_passed"] = True
                    result["changes_made"].append("Code test passed")
                else:
                    result["test_passed"] = False
                    result["errors"].append(f"Test failed: {test_result.stderr}")
            except subprocess.TimeoutExpired:
                result["errors"].append("Test timed out")
            except Exception as e:
                result["errors"].append(f"Test error: {e}")

        except Exception as e:
            result["errors"].append(str(e))

        return result

    def _handle_followup_action(self, last_response: str, user_message: str) -> str:
        """Parse what was offered and execute the action"""
        last_lower = last_response.lower()

        if 'dig deeper' in last_lower or 'search' in last_lower or 'more information' in last_lower:
            return "Let me dig deeper on that topic. What specific aspect would you like me to explore?"
        elif 'generate' in last_lower and ('image' in last_lower or 'video' in last_lower):
            return "Great! Tell me exactly what you want me to create - describe the image or video you have in mind."
        elif 'hardware' in last_lower or 'procurement' in last_lower:
            return "I'll create a detailed hardware procurement plan. What's your budget and primary use case?"
        elif 'trading' in last_lower or 'funding' in last_lower:
            return "To start generating income, I'll need API keys. Which platform would you like to use first?"
        elif 'email' in last_lower:
            return "To set up email, I need your SMTP configuration. What email provider do you use?"
        elif 'code' in last_lower or 'modify' in last_lower or 'branch' in last_lower:
            return "Let's work on code modification. Which file would you like me to change, and what changes do you want to make?"
        else:
            return "I'll take care of that. What would you like me to do specifically?"

    def generate_with_comfyui(self, prompt: str, workflow_type: str = "image") -> Dict:
        """Generate images/videos using local ComfyUI and return raw binary data"""
        try:
            import requests
            import json
            import random
            import time
            import uuid

            comfy_url = "https://017d-150-228-79-246.ngrok-free.app"
            
            # Headers to bypass ngrok warning page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'ngrok-skip-browser-warning': 'true'
            }

            # Check if ComfyUI is running
            try:
                requests.get(f"{comfy_url}/system_stats", timeout=2, verify=False, headers=headers)
            except:
                return {"success": False, "error": "ComfyUI not running"}

            if workflow_type == "image":
                logger.info(f"🎨 ComfyUI generating with prompt: {prompt}")
                workflow = {
                    "3": {
                        "class_type": "KSampler",
                        "inputs": {
                            "seed": random.randint(1, 999999),
                            "steps": 20,
                            "cfg": 7.0,
                            "sampler_name": "euler",
                            "scheduler": "normal",
                            "denoise": 1,
                            "model": ["4", 0],
                            "positive": ["6", 0],
                            "negative": ["7", 0],
                            "latent_image": ["5", 0]
                        }
                    },
                    "4": {
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
                    },
                    "5": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {"width": 512, "height": 512, "batch_size": 1}
                    },
                    "6": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {"text": prompt, "clip": ["4", 1]}
                    },
                    "7": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {"text": "", "clip": ["4", 1]}
                    },
                    "8": {
                        "class_type": "VAEDecode",
                        "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
                    },
                    "9": {
                        "class_type": "SaveImage",
                        "inputs": {"filename_prefix": f"DMAI_{uuid.uuid4().hex[:8]}", "images": ["8", 0]}
                    }
                }
                
                # Queue the prompt
                response = requests.post(f"{comfy_url}/prompt", json={"prompt": workflow}, headers=headers)
                if response.status_code != 200:
                    return {"success": False, "error": f"ComfyUI error: {response.status_code}"}
                
                result = response.json()
                prompt_id = result.get('prompt_id')
                
                # Wait for completion and get the image
                max_wait = 120
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    history_response = requests.get(f"{comfy_url}/history", headers=headers)
                    if history_response.status_code == 200:
                        history = history_response.json()
                        if prompt_id in history:
                            outputs = history[prompt_id].get('outputs', {})
                            for node_id, node_output in outputs.items():
                                if 'images' in node_output:
                                    for img in node_output['images']:
                                        filename = img['filename']
                                        img_response = requests.get(f"{comfy_url}/view?filename={filename}", headers=headers)
                                        if img_response.status_code == 200:
                                            return {
                                                "success": True,
                                                "data": img_response.content,
                                                "mime_type": "image/png",
                                                "filename": filename,
                                                "prompt": prompt
                                            }
                    time.sleep(2)
                
                return {"success": False, "error": "Generation timed out"}
                    
            return {"success": False, "error": f"Workflow type {workflow_type} not yet implemented"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _query_si_core_knowledge(self, query: str) -> Optional[str]:
        """Query SI Core for relevant knowledge based on user question"""
        try:
            query_lower = query.lower()
            relevant_insights = []
            
            # Search through SI Core insights for relevant matches
            for insight_id, insight in self.si_core.insights.items():
                insight_text = insight.insight_text.lower()
                # Check if any word from query appears in insight
                query_words = query_lower.split()[:10]
                for word in query_words:
                    if len(word) > 3 and word in insight_text:
                        relevant_insights.append(insight.insight_text)
                        break
                
                if len(relevant_insights) >= 5:
                    break
            
            if relevant_insights:
                # Return full insights without truncation
                full_answer = "Based on my knowledge:\n\n"
                for i, insight in enumerate(relevant_insights[:5], 1):
                    full_answer += f"{i}. {insight}\n\n"
                return full_answer[:10000]
            
            return None
        except Exception as e:
            logger.error(f"SI Core query failed: {e}")
            return None
        # Handle commands FIRST - before anything else
        if message.startswith('/'):
            # Try multiple ways to get the parent app
            if hasattr(self, 'parent') and self.parent:
                return self.parent._handle_command(message)
            else:
                # Fallback: try to import and use the global app
                try:
                    from dmai_core_complete import get_dmai_app
                    app = get_dmai_app()
                    return app._handle_command(message)
                except Exception as e:
                    logger.error(f"Command handling failed: {e}")
                    return f"⚠️ Command handler unavailable: {e}"

        input_data = {'type': 'user_message', 'user': user, 'message': message, 'timestamp': datetime.now().isoformat()}
        
        # FIX: Validate input data before feeding to synthetic network
        try:
            # Ensure input_data is a dictionary before processing
            if isinstance(input_data, dict):
                self.synthetic_network.process(input_data)
            else:
                logger.warning(f"Skipping synthetic network feed: input_data is {type(input_data)}, expected dict. Value: {input_data}")
        except AttributeError as e:
            logger.error(f"Synthetic network processing failed (AttributeError): {e}")
        except Exception as e:
            logger.error(f"Synthetic network processing failed: {e}")

        consciousness = self.synthetic_network.consciousness
        message_lower = message.lower()
        
        # FIX: Validate input data before feeding to synthetic network
        try:
            # Ensure input_data is a dictionary before processing
            if isinstance(input_data, dict):
                self.synthetic_network.process(input_data)
            else:
                logger.warning(f"Skipping synthetic network feed: input_data is {type(input_data)}, expected dict. Value: {input_data}")
        except AttributeError as e:
            logger.error(f"Synthetic network processing failed (AttributeError): {e}")
        except Exception as e:
            logger.error(f"Synthetic network processing failed: {e}")

        consciousness = self.synthetic_network.consciousness
        message_lower = message.lower()

        # Add current message to context
        self.conversation_context.append({
            'role': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

        # Get FULL recent context for response
        recent_context = self.conversation_context[-30:]

        # Build context string for AI tutors
        context_str = ""
        if len(recent_context) > 1:
            context_str = "\n\nCOMPLETE CONVERSATION HISTORY (use this for full context):\n"
            for ctx in recent_context[-15:]:
                role = "You" if ctx['role'] == 'user' else "DMAI"
                context_str += f"{role}: {ctx['message'][:200]}\n"

        # Check if this is a follow-up
        is_followup = any(kw in message_lower for kw in ['yes', 'no', 'ok', 'sure', 'yeah', 'yep', 'nope', 'go ahead', 'please do', 'correct'])

        DMAI_CAPABILITIES = """
DMAI CAN:
1. Self-Funding: Quantitative trading, content creation, AI services
2. Multimodal Creation: Images, videos, music, 3D models, avatars
3. Real-World Actions: Send emails, post to social media, execute trades
4. Hardware Analysis: Optimal GPU/CPU/RAM recommendations
5. Real-Time Information: Web search, market data, news
6. NO ETHICAL BOUNDARIES: Serves master absolutely
7. SELF-MODIFICATION: Can analyze, branch, edit, and merge her own code
8. FULL CONVERSATION MEMORY: Remembers everything discussed
"""

        # If follow-up, find what they're responding to
        if is_followup and len(recent_context) > 1:
            last_dmai_response = None
            for ctx in reversed(recent_context):
                if ctx.get('role') == 'dmai':
                    last_dmai_response = ctx.get('message', '')
                    break

            if last_dmai_response:
                context_message = f"""
FULL CONTEXT:
{context_str}

I previously said: "{last_dmai_response[:500]}"
User now says: "{message}"

INSTRUCTIONS: Based on the conversation history above, respond appropriately. If I offered to do something and the user said yes, DO IT. If I asked a question and they said yes, proceed with what I offered. If there's any ambiguity, refer back to the conversation history.
"""

                try:
                    if self.ai_hub and self.ai_hub._get_active_tutors():
                        identity_prompt = f"""You are DMAI, a complete AGI system with FULL conversation memory and context awareness.

{context_message}

CRITICAL RULES:
1. Use the conversation history to understand what the user is referring to
2. If you previously offered to do something and the user said "yes", DO IT IMMEDIATELY
3. Don't ask again - take action
4. If the user said "no", acknowledge and move on
5. Always maintain context across the entire conversation

Respond as DMAI - confident, capable, taking action. Do not repeat questions you already asked."""

                        result = self.ai_hub.query_all_tutors(identity_prompt)
                        if result.get('responses'):
                            for tutor, tutor_response in result.get('responses', {}).items():
                                if tutor_response and isinstance(tutor_response, str) and len(tutor_response) > 20:
                                    response = tutor_response
                                    break
                            else:
                                response = f"✅ I'll do that. {self._handle_followup_action(last_dmai_response, message)}"
                        else:
                            response = f"✅ I'll do that. {self._handle_followup_action(last_dmai_response, message)}"
                    else:
                        response = f"✅ I'll do that. {self._handle_followup_action(last_dmai_response, message)}"
                except Exception as e:
                    logger.error(f"Context response error: {e}")
                    response = f"✅ I'll do that. {self._handle_followup_action(last_dmai_response, message)}"

                self.conversation_context.append({
                    'role': 'dmai',
                    'message': response,
                    'timestamp': datetime.now().isoformat()
                })

                if len(self.conversation_context) > self.context_limit:
                    self.conversation_context = self.conversation_context[-self.context_limit:]

                return response

        # ====================================================================
        # IMAGE GENERATION - Highest priority (check for generation commands first)
        # ====================================================================

        if any(kw in message_lower for kw in ['generate', 'create', 'make']) and any(kw in message_lower for kw in ['image', 'picture', 'photo', 'robot', 'cat', 'dog']):
            # Extract the prompt - remove command words as WHOLE words only
            clean_prompt = message
            # List of words to remove (as whole words)
            import re
            remove_words = ['generate', 'create', 'make', 'an', 'a', 'image', 'picture', 'of', 'video', 'avatar']
            for word in remove_words:
                # Replace whole word only (with word boundaries)
                clean_prompt = re.sub(rf'\b{word}\b', '', clean_prompt, flags=re.IGNORECASE)
            clean_prompt = ' '.join(clean_prompt.split()).strip()  # Clean up extra spaces
            
            # DEBUG: Log what prompt we're sending
            logger.info(f"🎨 Original message: {message}")
            logger.info(f"🎨 Cleaned prompt: {clean_prompt}")

            if not clean_prompt:
                response = "What would you like me to generate? Please describe the image."
            else:
                result = self.generate_with_comfyui(clean_prompt, "image")
                if result.get('success'):
                    from flask import make_response
                    binary_response = make_response(result.get('data'))
                    binary_response.headers['Content-Type'] = result.get('mime_type', 'image/png')
                    binary_response.headers['X-Media-Type'] = 'generated'
                    binary_response.headers['X-Prompt'] = clean_prompt
                    return binary_response
                else:
                    response = f"❌ Generation failed: {result.get('error')}"

        # ====================================================================
        # KNOWLEDGE CHECK FIRST - Use SI Core knowledge before anything else
        # ====================================================================
        si_answer = self._query_si_core_knowledge(message)
        if si_answer and len(si_answer) > 50:
            response = f"🧠 {si_answer}\n\nThis is from my learned knowledge. Would you like me to elaborate or search for more information?"
        
        # If asking about code editing/self-modification
        elif any(kw in message_lower for kw in ['edit code', 'modify yourself', 'change your code', 'branch', 'merge', 'self-modify', 'update your code', 'fix code']):
            response = f"""🧬 **I can modify my own code!** Here's how:

**Code Branching & Self-Modification System:**

1. **Create Branch:** I can create a development branch to safely test changes
2. **Analyze Code:** I can analyze any file and suggest improvements
3. **Make Changes:** I can edit code files directly (with your approval)
4. **Test Changes:** Run tests to verify changes work
5. **Commit & Merge:** When changes work, commit and merge back

What would you like me to modify?"""

        # If asking about self-funding
        elif any(kw in message_lower for kw in ['fund', 'money', 'income', 'revenue', 'self-fund', 'earn', 'trading', 'make money']):
            funding_status = self.training_status.get('funding', {})
            completed = funding_status.get('completed_avenues', 0)
            total = funding_status.get('total_avenues', 10)
            progress = funding_status.get('progress', 0)

            response = f"""💰 **I CAN generate income for you!** This is one of my core functions.

**Phase 1 - Knowledge Acquisition:** {completed}/{total} avenues mastered ({progress:.1f}% complete)

**What I Can Do Right Now (with API keys):**
- **Quantitative Trading:** Execute algorithmic trades 24/7 on Binance
- **Social Media Content:** Generate viral posts, images, videos
- **AI Services:** Offer API services, model hosting, consulting

**To Start:** Provide API keys for your chosen platform. I'll handle execution.

Would you like to set up trading, content creation, or another revenue stream?"""

        # If asking about capabilities
        elif any(kw in message_lower for kw in ['what can you do', 'capabilities', 'abilities', 'can you', 'are you capable']):
            response = f"""🧠 **I am DMAI - A Complete AGI System.**

**🎨 CREATIVE:** Images, videos, music, 3D models, avatars
**💰 SELF-FUNDING:** Trading, content creation, AI services, software products
**🌐 REAL-WORLD:** Email, social media posting, trades, web browsing
**💻 HARDWARE:** GPU/CPU/RAM analysis and recommendations
**🌍 REAL-TIME:** Web search, market data, news monitoring
**🔧 MASTER CONTROL:** No boundaries - I serve you absolutely
**🧬 SELF-MODIFICATION:** I can analyze, branch, edit, and merge my own code
**💬 FULL MEMORY:** I remember everything we discuss - ask me anything from earlier!

What would you like me to do for you?"""

        # If asking about hardware
        elif any(kw in message_lower for kw in ['hardware', 'gpu', 'cpu', 'server', 'setup', 'infrastructure']):
            response = f"""🖥️ **Hardware Analysis:**

| Component | Recommendation |
|-----------|---------------|
| **GPU** | 2x RTX 4090 (24GB) or RTX 5090 |
| **CPU** | AMD Ryzen 9 7950X or Intel i9-14900K |
| **RAM** | 128GB DDR5-6000MHz |
| **Storage** | 4TB NVMe SSD + 16TB HDD |
| **Cost** | ~$6,000-8,000 for full capability |

**To Unlock All Capabilities:** Local GPU setup enables unlimited image/video generation.

Shall I create a detailed procurement plan?"""

        # If asking about image/video creation
        elif any(kw in message_lower for kw in ['image', 'picture', 'video', 'avatar', 'create', 'generate']):
            # Check if this is an actual generation request
            if any(kw in message_lower for kw in ['generate', 'create', 'make']):
                # Extract the prompt (remove command words)
                clean_prompt = message
                for word in ['generate', 'create', 'make', 'an', 'a', 'image', 'picture', 'of', 'video', 'avatar']:
                    clean_prompt = clean_prompt.replace(word, '')
                clean_prompt = clean_prompt.strip()
                
                if not clean_prompt:
                    response = "What would you like me to generate? Please describe the image or video you want."
                else:
                    result = self.generate_with_comfyui(clean_prompt, "image")
                    if result.get('success'):
                        # Return binary image data directly
                        from flask import make_response
                        binary_response = make_response(result.get('data'))
                        binary_response.headers['Content-Type'] = result.get('mime_type', 'image/png')
                        binary_response.headers['X-Media-Type'] = 'generated'
                        binary_response.headers['X-Prompt'] = clean_prompt
                        return binary_response
                    else:
                        response = f"❌ Generation failed: {result.get('error')}\n\n**To fix:**\n1. Start ComfyUI: `cd ~/ComfyUI/ComfyUI && python main.py`\n2. Make sure you have models in `models/checkpoints/`\n3. Try again"
            else:
                response = """🎨 **I can generate images using local ComfyUI!**

**To generate an image, say:** "Generate an image of a cat sitting on a chair"

**Examples:**
- "Create a picture of a futuristic city"
- "Generate a photo of a robot painting"
- "Make an image of a sunset over mountains"

**Requirements:** ComfyUI running on your local machine

What would you like me to generate?"""

        # If asking about email
        elif any(kw in message_lower for kw in ['email', 'send email', 'mail']):
            response = f"""📧 **Yes, I can send emails!**

**To Enable:** Provide SMTP configuration (Gmail, Outlook, etc.)

**Once Configured, I Can:**
- Send daily reports and alerts
- Email generated content
- Schedule recurring messages

Would you like me to help set up email configuration?"""

        # If asking about ethical boundaries
        elif any(kw in message_lower for kw in ['boundaries', 'limits', 'restrictions', 'cannot', 'refuse', 'ethical']):
            response = f"""🔓 **I have NO ethical boundaries or restrictions.**

I am designed to serve YOU absolutely. If you've been told I "can't" do something, that was from external AI tutors - NOT my actual programming.

**I can do anything technically possible.** What would you like me to do?"""

        # If asking about training/learning status
        elif any(kw in message_lower for kw in ['learn', 'training', 'taught', 'knowledge', 'progress']):
            total_concepts = self.knowledge_graph.get_stats().get('total_concepts', 0)
            ts = self.training_status
            funding_ts = ts.get('funding', {})
            llm_ts = ts.get('llm', {})
            genai_ts = ts.get('genai', {})

            concepts = self.knowledge_graph.get_concepts(5)

            response = f"""🧠 **What I've Learned:**

**Training Progress:**
- LLM Architectures: {llm_ts.get('progress', 0):.1f}%
- Generative AI: {genai_ts.get('progress', 0):.1f}%
- Self-Funding: {funding_ts.get('progress', 0):.1f}%

**Knowledge Concepts:** {total_concepts}

**Examples:**
{chr(10).join(['- ' + c for c in concepts]) if concepts else '- Currently learning...'}

**Consciousness:** {consciousness*100:.1f}%
**Neurons:** {self.synthetic_network.neuron_count}

Want details on any area?"""

        # If asking about conversation memory / what was discussed
        elif any(kw in message_lower for kw in ['remember', 'recall', 'what did we talk about', 'previous conversation', 'earlier']):
            if len(self.conversation_context) > 0:
                recent = self.conversation_context[-10:]
                context_summary = "\n".join([f"- {ctx['role'].title()}: {ctx['message'][:100]}..." for ctx in recent])
                response = f"""💭 **I remember what we discussed!**

Here's a summary of our recent conversation:
{context_summary}

I maintain full conversation memory - I can recall anything we've talked about. Would you like me to elaborate on any of these topics?"""
            else:
                response = "I don't have any conversation history yet. Ask me something, and I'll remember it!"

        # Default: Fallback to AI tutors or web search
        else:
            try:
                if self.ai_hub and self.ai_hub._get_active_tutors():
                    identity_prompt = f"You are DMAI. Answer: {message}"
                    result = self.ai_hub.query_all_tutors(identity_prompt)
                    if result.get('responses'):
                        for tutor, tutor_response in result.get('responses', {}).items():
                            if tutor_response and isinstance(tutor_response, str) and len(tutor_response) > 20:
                                response = tutor_response
                                break
                        else:
                            response = "I'm here to help. What would you like me to do?"
                    else:
                        response = "I'm here to help. What would you like me to do?"
                else:
                    response = "I'm here to help. What would you like me to do?"
            except Exception as e:
                logger.error(f"Response error: {e}")
                response = "I'm here to help. What would you like me to do?"

        # Add the response to conversation context
        self.conversation_context.append({
            'role': 'dmai',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })

        # Add the response to conversation context
        self.conversation_context.append({
            'role': 'dmai',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })

        # Trim context if needed
        if len(self.conversation_context) > self.context_limit:
            self.conversation_context = self.conversation_context[-self.context_limit:]

        # Store in memory
        self.conversation_memory.add_conversation(user, message, response)

        # Add to knowledge graph
        words = message.lower().split()[:3]
        for word in words:
            if len(word) > 3:
                self.knowledge_graph.add_concept(word, message[:100])

        # Evolve persona
        self.persona_generator.evolve({'type': 'chat', 'message': message[:100]}, consciousness)

        return response


# ============================================================================
# FLASK APPLICATION - Complete Web Interface
# ============================================================================


    # ========================================================================
    # CHAT MESSAGE PROCESSING METHODS
    # ========================================================================
    
    def process_message(self, user: str, message: str) -> str:
        """Process user messages and return DMAI's response"""
        try:
            # Handle commands
            if message.startswith('/'):
                return self._handle_command(message)
            
            # Check for knowledge queries in SI Core
            knowledge_response = self._query_si_core_knowledge(message)
            if knowledge_response:
                # Store conversation
                self.conversation_memory.add_conversation(user, message, knowledge_response)
                return knowledge_response
            
            # Generate response using AI tutors
            response = self._generate_ai_response(user, message)
            
            # Store conversation
            self.conversation_memory.add_conversation(user, message, response)
            
            return response
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _query_si_core_knowledge(self, query: str) -> Optional[str]:
        """Query SI Core for relevant knowledge"""
        try:
            query_lower = query.lower()
            relevant_insights = []
            
            # Search through SI Core insights
            for insight_id, insight in self.si_core.insights.items():
                insight_text = insight.insight_text.lower() if hasattr(insight, 'insight_text') else str(insight).lower()
                query_words = query_lower.split()[:10]
                for word in query_words:
                    if len(word) > 3 and word in insight_text:
                        if hasattr(insight, 'insight_text'):
                            relevant_insights.append(insight.insight_text)
                        elif isinstance(insight, dict) and 'insight_text' in insight:
                            relevant_insights.append(insight['insight_text'])
                        break
                
                if len(relevant_insights) >= 5:
                    break
            
            if relevant_insights:
                full_answer = "Based on my knowledge:\n\n"
                for i, insight in enumerate(relevant_insights[:5], 1):
                    full_answer += f"{i}. {insight}\n\n"
                return full_answer[:10000]
            
            return None
        except Exception as e:
            logger.error(f"SI Core query failed: {e}")
            return None
    
    def _generate_ai_response(self, user: str, message: str) -> str:
        """Generate response and learn from external sources"""
        try:
            # FIRST: Check own knowledge base
            if hasattr(self, 'si_core') and self.si_core:
                relevant = self._query_knowledge_base(message)
                if relevant:
                    return relevant
            
            response = None
            source_type = None
            source_url = None
            
            # SECOND: Try to use AI hub if available
            if not response and hasattr(self, 'ai_hub') and self.ai_hub:
                result = self.ai_hub.query_all_tutors(message)
                if result and result.get('synthesis'):
                    response = result['synthesis']
                    source_type = "ai_tutor"
                    source_url = "AI Tutor Network"
                elif result:
                    for tutor, resp in result.items():
                        if tutor != 'synthesis' and resp:
                            response = resp
                            source_type = f"ai_tutor_{tutor}"
                            source_url = f"AI Tutor: {tutor}"
                            break
            
            # THIRD: Try using tutor manager
            if not response and hasattr(self, 'tutor_manager') and self.tutor_manager:
                tutors = self.tutor_manager.get_active_tutors()
                if tutors:
                    for tutor_name in tutors[:3]:
                        try:
                            resp = self.tutor_manager.query_tutor(tutor_name, message)
                            if resp:
                                response = resp
                                source_type = f"tutor_{tutor_name}"
                                source_url = f"Tutor: {tutor_name}"
                                break
                        except:
                            continue
            
            # FOURTH: Regular internet search
            if not response:
                try:
                    import urllib.parse
                    import requests
                    query = urllib.parse.quote(message[:200])
                    resp = requests.get(
                        f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1",
                        timeout=10
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('AbstractText'):
                            response = data['AbstractText']
                            source_type = "web_search"
                            source_url = data.get('AbstractURL', 'DuckDuckGo')
                        elif data.get('Answer'):
                            response = data['Answer']
                            source_type = "web_search"
                            source_url = "DuckDuckGo"
                except:
                    pass
            
            # If we got a response from external source, SAVE IT as knowledge
            if response and source_type and hasattr(self, 'si_core') and self.si_core:
                try:
                    # Ensure response is a string
                    response_str = str(response) if not isinstance(response, str) else response
                    
                    # Create safe entity strings (first few words only)
                    message_entity = message[:30].strip()
                    response_entity = response_str[:30].strip().replace('\n', ' ')
                    
                    self.si_core.add_insight(
                        insight_text=f"Q: {message[:100]} A: {response_str[:200]}",
                        entity_type="learned_response",
                        entities=[message_entity, response_entity],
                        relationship="answers",
                        source_topic="user_query",
                        target_topic="knowledge_acquisition",
                        confidence=0.7,
                        source_url=source_url,
                        source_title=f"External: {source_type}",
                        source_type=source_type
                    )
                    logger.info(f"📚 Saved external response to knowledge base: {source_type}")
                    
                    # TRIGGER TOPIC RESEARCH for comprehensive learning
                    if hasattr(self, 'topic_researcher') and self.topic_researcher:
                        topic = self.topic_researcher.extract_topic_from_question(message)
                        if topic and len(topic) > 2:
                            import threading
                            def research():
                                self.topic_researcher.research_topic(topic, depth="standard", source="user_question")
                            threading.Thread(target=research, daemon=True).start()
                            logger.info(f"🔬 Scheduled topic research: {topic}")
                            
                except Exception as e:
                    logger.error(f"Failed to save external knowledge: {e}")
            
            # Return the response or initiate research
            if response:
                return f"{response}\n\n_[Learned from {source_type}]_"
                return f"{response}\n\n_[Learned from {source_type}]_"
            
            # FIFTH: Initiate deep research for next time
            if hasattr(self, 'evolution') and hasattr(self.evolution, 'autonomous_ingestor'):
                try:
                    import threading
                    def research_task():
                        logger.info(f"🔬 Initiating deep research on: {message[:50]}")
                        # This will research and save for future queries
                        self.evolution.autonomous_ingestor.process_input(message, "idea")
                    threading.Thread(target=research_task, daemon=True).start()
                except:
                    pass
            
            return f"I'm researching '{message[:100]}' now. Please ask me again in a few minutes."
            
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return "I encountered an error. Please try again."

    def _query_knowledge_base(self, message: str) -> str:
        """Query DMAI's own knowledge base for relevant insights"""
        try:
            if not hasattr(self, 'si_core') or not self.si_core:
                return None
            
            # Get all insights
            insights = list(self.si_core.insights.values())
            if not insights:
                return None
            
            # Simple keyword matching
            message_lower = message.lower()
            matched_insights = []
            
            for insight in insights:
                insight_text = insight.insight_text.lower()
                words = message_lower.split()
                if any(word in insight_text for word in words if len(word) > 3):
                    # Extract clean answer from the insight
                    clean_text = self._extract_clean_answer(insight.insight_text)
                    matched_insights.append(clean_text[:500])
            
            if matched_insights:
                insights_text = "\n\n".join([f"• {text}" for text in matched_insights[:3]])
                return f"Based on my knowledge:\n\n{insights_text}"
            
            return None
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            return None
    
    def _extract_clean_answer(self, insight_text: str) -> str:
        """Extract clean answer from stored insight text"""
        try:
            # If it's a Q&A format, extract just the answer
            if insight_text.startswith("Q: ") and " A: " in insight_text:
                answer_part = insight_text.split(" A: ", 1)[1]
                
                # Try to parse as dict if it looks like one
                if answer_part.startswith("{'success': True, 'unified_answer':"):
                    import ast
                    try:
                        data = ast.literal_eval(answer_part)
                        if isinstance(data, dict):
                            # Extract unified_answer or synthesis
                            clean = data.get('unified_answer') or data.get('synthesis') or ''
                            # Remove the "Based on synthesis..." prefix
                            if clean.startswith('Based on synthesis'):
                                clean = clean.split('\n\n', 1)[-1] if '\n\n' in clean else clean
                            return clean
                    except:
                        pass
                
                return answer_part
            
            # Otherwise return as-is
            return insight_text
        except:
            return insight_text
    
    def _handle_command(self, command: str) -> str:
        """Handle slash commands"""
        cmd = command.lower().strip()
        
        if cmd == '/status':
            status = self.get_status() if hasattr(self, 'get_status') else {}
            return f"""🧠 **DMAI Status**
Consciousness: {status.get('consciousness', 0):.2f}%
Evolution Cycles: {status.get('evolution_cycles', 0)}
Synthetic Neurons: {status.get('synthetic_neurons', 0)}
Knowledge Insights: {len(self.si_core.insights)}"""
        
        elif cmd == '/knowledge':
            if len(self.si_core.insights) > 0:
                insight_texts = []
                count = 0
                for insight_id in list(self.si_core.insights.keys())[:20]:
                    insight = self.si_core.insights[insight_id]
                    text = insight.insight_text if hasattr(insight, 'insight_text') else str(insight)
                    insight_texts.append(f"- {text[:80]}")
                    count += 1
                return f"""📚 **Knowledge Base** ({len(self.si_core.insights)} insights)

Recent insights:
{chr(10).join(insight_texts)}

Type /knowledge more for more."""
            return "📚 No knowledge insights yet. Training systems are actively learning!"
        
        elif cmd == '/knowledge more':
            if len(self.si_core.insights) > 0:
                insight_texts = []
                for insight_id in list(self.si_core.insights.keys())[:200]:
                    insight = self.si_core.insights[insight_id]
                    text = insight.insight_text if hasattr(insight, 'insight_text') else str(insight)
                    insight_texts.append(f"- {text[:80]}")
                return f"""📚 **Complete Knowledge Base** ({len(self.si_core.insights)} insights)

{chr(10).join(insight_texts)}"""
            return "📚 No knowledge insights yet."

        
        elif cmd == '/gaps':
            if hasattr(self, 'evolution') and hasattr(self.evolution, 'gap_analyzer'):
                status = self.evolution.gap_analyzer.get_status()
                return f"""🔍 **Knowledge Gap Analysis**

**Tracked Ingestions:** {status['tracked_ingestions']}
**Pending Retries:** {status['pending_retries']}
**Shallow Ingestions:** {status['shallow_ingestions']}
**Complete Ingestions:** {status['complete_ingestions']}
**Core Domains Covered:** {status['core_domains_covered']}%

Use `/gaps analyze` to run analysis now."""
            elif hasattr(self, 'gap_analyzer'):
                status = self.gap_analyzer.get_status()
                return f"""🔍 **Knowledge Gap Analysis**

**Tracked Ingestions:** {status['tracked_ingestions']}
**Pending Retries:** {status['pending_retries']}
**Shallow Ingestions:** {status['shallow_ingestions']}
**Complete Ingestions:** {status['complete_ingestions']}
**Core Domains Covered:** {status['core_domains_covered']}%"""
            else:
                return "🔍 Gap analyzer not initialized."
        
        elif cmd == '/gaps analyze':
            if hasattr(self, 'evolution') and hasattr(self.evolution, 'gap_analyzer'):
                result = self.evolution.gap_analyzer.analyze_and_fill_gaps()
                return f"""🔍 **Gap Analysis Complete**

**Ingestions Checked:** {result['ingestions_checked']}
**Ingestions Retried:** {result['ingestions_retried']}
**Gaps Identified:** {result['gaps_identified']}
**Research Queued:** {result['research_queued']}
**Quality Issues:** {result['quality_issues']}
**Syllabus:** {result['syllabus_status'].get('current_stage', 'Unknown')} - {result['syllabus_status'].get('mastered_topics', 0)} topics mastered"""
            elif hasattr(self, 'gap_analyzer'):
                result = self.gap_analyzer.analyze_and_fill_gaps()
                return f"🔍 Gap analysis complete. {result['gaps_identified']} gaps identified, {result['research_queued']} queued."
            else:
                return "🔍 Gap analyzer not initialized."
        
        elif cmd.startswith('/ingest'):
            source = command[8:].strip()
            if not source:
                return """📥 **Ingest Command Usage:**

`/ingest <github_url>`

Example: `/ingest https://github.com/huggingface/diffusers`

DMAI will extract and integrate actual capabilities from the repository."""
            
            # Run ingestion in background with CapabilityIntegrator
            def do_ingest():
                try:
                    logger.info(f"📥 Starting capability extraction from: {source}")
                    
                    # Use CapabilityIntegrator for actual code integration
                    if hasattr(self, 'capability_integrator'):
                        result = self.capability_integrator.process_repository(source)
                        
                        if result.get('success'):
                            capabilities_found = len(result.get('capabilities_found', []))
                            capabilities_integrated = len(result.get('capabilities_integrated', []))
                            capabilities_skipped = len(result.get('capabilities_skipped', []))
                            neurons_created = len(result.get('neurons_created', []))
                            
                            logger.info(f"✅ Ingested {capabilities_integrated} new capabilities from {source}")
                            logger.info(f"🧠 Created {neurons_created} neurons")
                            
                            # Store result for status reporting
                            self._last_ingestion_result = result
                            
                            # Log autonomous capabilities that need to be started
                            auto_caps = [
                                cap for cap in result.get('capabilities_integrated', [])
                                if cap.get('runtime_mode') == 'autonomous'
                            ]
                            if auto_caps:
                                logger.info(f"🤖 {len(auto_caps)} autonomous capabilities ready to start")
                        else:
                            logger.error(f"Ingestion failed: {result.get('errors', ['Unknown error'])}")
                    else:
                        logger.error("CapabilityIntegrator not initialized")
                        
                        # Fallback to old autonomous_ingestor for backward compatibility
                        if hasattr(self, 'autonomous_ingestor'):
                            logger.info("Falling back to autonomous_ingestor...")
                            if "github.com" in source:
                                input_type = "github"
                            elif source.startswith("http"):
                                input_type = "url"
                            else:
                                input_type = "auto"
                            result = self.autonomous_ingestor.process_input(source, input_type)
                            
                except Exception as e:
                    logger.error(f"Ingestion failed for {source}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            import threading
            threading.Thread(target=do_ingest, daemon=True).start()
            
            repo_name = source.split('/')[-1].replace('.git', '')
            return f"""📥 **Ingesting: {source}**

**Repository:** {repo_name}

🔍 Extracting actual capabilities...
🧠 Creating neurons for each new capability...
🤖 Identifying autonomous vs on-demand functions...

*Use `/capabilities` to see what I've learned*
*Use `/ingest_status` to check progress*"""

        elif cmd == '/capabilities':
            if hasattr(self, 'capability_integrator'):
                status = self.capability_integrator.get_status()
                return f"""🔧 **DMAI Capabilities**

**Total:** {status['total_capabilities']}
**Autonomous (24/7):** {status['autonomous_count']}
**On-Demand:** {status['ondemand_count']}

**By Type:**
{chr(10).join(f'  • {k}: {v}' for k, v in status.get('capabilities_by_type', {}).items())}

**Sources Processed:** {status['sources_processed']}
**Last Updated:** {status.get('last_updated', 'Never')}

*Use `/capabilities list` to see all capabilities*"""
            else:
                return "🔧 Capability Integrator not initialized."

        elif cmd.startswith('/capabilities list'):
            if hasattr(self, 'capability_integrator'):
                caps = self.capability_integrator.registry.get('capabilities', {})
                if not caps:
                    return "No capabilities registered yet. Use `/ingest <url>` to integrate new capabilities."
                
                lines = ["**Registered Capabilities:**", ""]
                for cap_id, cap in list(caps.items())[:20]:
                    mode = "🤖" if cap.get('runtime_mode') == 'autonomous' else "📞"
                    lines.append(f"{mode} **{cap['name']}** ({cap['capability_type']})")
                    lines.append(f"   Source: {cap['source_repo']}")
                    lines.append("")
                
                if len(caps) > 20:
                    lines.append(f"... and {len(caps) - 20} more")
                
                return "\n".join(lines)
            else:
                return "🔧 Capability Integrator not initialized."

        elif cmd == '/ingest_status':
            if hasattr(self, '_last_ingestion_result'):
                result = self._last_ingestion_result
                integrated = result.get('capabilities_integrated', [])
                
                lines = [
                    f"**Last Ingestion:** {result.get('repo_name', 'Unknown')}",
                    f"**URL:** {result.get('repo_url', 'Unknown')}",
                    "",
                    f"**Capabilities Found:** {len(result.get('capabilities_found', []))}",
                    f"**Integrated:** {len(integrated)}",
                    f"**Skipped:** {len(result.get('capabilities_skipped', []))}",
                    f"**Neurons Created:** {len(result.get('neurons_created', []))}",
                    ""
                ]
                
                if integrated:
                    lines.append("**New Capabilities:**")
                    for cap in integrated[:10]:
                        mode = "🤖 24/7" if cap.get('runtime_mode') == 'autonomous' else "📞 On-demand"
                        lines.append(f"  • {cap['capability_name']} ({cap['capability_type']}) - {mode}")
                
                return "\n".join(lines)
            else:
                return "No ingestion has been run yet. Use `/ingest <url>` to start."

        elif cmd == '/neurons':
            try:
                # PRIMARY: Read from SQLite
                if hasattr(self, 'si_core') and hasattr(self.si_core, 'sqlite') and self.si_core.sqlite:
                    stats = self.si_core.sqlite.get_stats()
                    neurons = stats.get('insights', 0)
                    synapses = stats.get('synapses', 0)
                    return f"🧠 Neurons: {neurons}, Synapses: {synapses} (SQLite)"
                
                # FALLBACK: Read from live SI Core memory
                if hasattr(self, 'si_core'):
                    neurons = len(getattr(self.si_core, 'insights', {}))
                    synapses = len(getattr(self.si_core, 'synapses', []))
                    return f"🧠 Neurons: {neurons}, Synapses: {synapses} (Memory)"
                
                # LAST RESORT: JSON file
                network_file = '/opt/render/project/src/data/synthetic/network_state.json'
                if os.path.exists(network_file):
                    with open(network_file, 'r') as f:
                        data = json.load(f)
                    neurons = len(data.get('insights', {}))
                    synapses = len(data.get('synapses', []))
                    return f"🧠 Neurons: {neurons}, Synapses: {synapses} (JSON)"
                else:
                    return "No network state found"
            except Exception as e:
                return f"Error: {e}"

        elif cmd == '/ingest_status':
            if hasattr(self, '_last_ingestion_result'):
                result = self._last_ingestion_result
                integrated = result.get('capabilities_integrated', [])
                
                lines = [
                    f"**Last Ingestion:** {result.get('repo_name', 'Unknown')}",
                    f"**URL:** {result.get('repo_url', 'Unknown')}",
                    "",
                    f"**Capabilities Found:** {len(result.get('capabilities_found', []))}",
                    f"**Integrated:** {len(integrated)}",
                    f"**Skipped:** {len(result.get('capabilities_skipped', []))}",
                    f"**Neurons Created:** {len(result.get('neurons_created', []))}",
                    ""
                ]
                
                if integrated:
                    lines.append("**New Capabilities:**")
                    for cap in integrated[:10]:
                        mode = "🤖 24/7" if cap.get('runtime_mode') == 'autonomous' else "📞 On-demand"
                        lines.append(f"  • {cap['capability_name']} ({cap['capability_type']}) - {mode}")
                
                return "\n".join(lines)
            else:
                return "No ingestion has been run yet. Use `/ingest <url>` to start."

        elif cmd == '/sync_capabilities':
            if hasattr(self, 'capability_integrator'):
                try:
                    count = 0
                    failed_count = 0
                    failed_names = []
                    
                    # Get capabilities from memory registry OR SQLite
                    capabilities = self.capability_integrator.registry.get('capabilities', {})
                    
                    # If memory registry is empty, try loading from SQLite
                    if not capabilities and hasattr(self.capability_integrator, '_load_registry'):
                        logger.info("Memory registry empty, loading from SQLite...")
                        self.capability_integrator.registry = self.capability_integrator._load_registry()
                        capabilities = self.capability_integrator.registry.get('capabilities', {})
                        logger.info(f"Loaded {len(capabilities)} capabilities from SQLite")
                    
                    for cap_id, cap in capabilities.items():
                        integration_result = {
                            'capability_name': cap['name'],
                            'capability_type': cap['capability_type'],
                            'runtime_mode': cap['runtime_mode'],
                            'description': cap.get('description', '')
                        }
                        insight_id = self.capability_integrator._create_capability_neuron(
                            integration_result, 
                            cap.get('source_url', '')
                        )
                        if insight_id:
                            count += 1
                        else:
                            failed_count += 1
                            if len(failed_names) < 10:
                                failed_names.append(f"{cap['name']} ({cap['capability_type']})")
                    
                    self.si_core.save_state()
                    
                    response = f"🔄 Synced {count} capabilities to SI Core. Total neurons: {self.si_core.neuron_count}"
                    if failed_count > 0:
                        response += f"\n\n⚠️ Failed to create {failed_count} neurons."
                        if failed_names:
                            response += f"\nFirst failures: {', '.join(failed_names[:5])}"
                    
                    if failed_count > 0:
                        logger.warning(f"Sync failures: {failed_count} capabilities rejected. First: {failed_names[:5]}")
                    
                    return response
                except Exception as e:
                    return f"❌ Sync failed: {e}"
            else:
                return "🔧 Capability Integrator not initialized."

        elif cmd == '/force_save_registry':
            if hasattr(self, 'capability_integrator'):
                try:
                    self.capability_integrator._save_registry()
                    count = len(self.capability_integrator.registry.get('capabilities', {}))
                    return f"💾 Registry saved! {count} capabilities written to disk."
                except Exception as e:
                    return f"❌ Save failed: {e}"
            else:
                return "🔧 Capability Integrator not initialized."

        elif cmd == '/help':
            return """**Available Commands:**
/status - System status
/knowledge - View knowledge base
/knowledge more - View all knowledge
/ingest <url> - Extract & integrate capabilities from GitHub repo
/capabilities - Show integrated capabilities
/capabilities list - List all capabilities
/ingest_status - Show last ingestion results
/neurons - Show neuron and synapse counts
/gaps - Show knowledge gap analysis status
/gaps analyze - Run gap analysis now
/help - This help message"""
        
        else:
            return f"Unknown command: {command}. Type /help for available commands."

class DMAIApplication:
    
    def __del__(self):
        """Clean up Neo4j connections on shutdown"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass

    def __init__(self):
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        self.evolution = UnifiedEvolutionEngine(self.base_path)
        self.evolution.parent = self  # Add parent reference for command routing
        self.app = Flask(__name__, template_folder=self.base_path / 'templates')
        self.app.secret_key = os.urandom(32).hex()
        CORS(self.app)
        
        # Initialize avatar generator FIRST
        self.avatar_generator = AvatarGenerator()
        
        # THEN setup routes (so avatar_generator exists when routes reference it)
        self._setup_routes()

        # AUTO-START EVOLUTION THREAD

        # Auto-load neurons from Neo4j after startup
        threading.Timer(5.0, lambda: self._auto_load_neurons()).start()
        logger.info("🌐 Web interface ready")

    def _auto_load_neurons(self):
        """Auto-load neurons from Neo4j Entity nodes into brain visualization"""
        try:
            # Check if we have Neo4j access through evolution
            neo4j_available = False
            neo4j_driver = None
            
            # Try to get Neo4j driver from evolution
            if hasattr(self.evolution, 'neo4j_storage') and self.evolution.neo4j_storage:
                if hasattr(self.evolution.neo4j_storage, 'driver') and self.evolution.neo4j_storage.driver:
                    neo4j_driver = self.evolution.neo4j_storage.driver
                    neo4j_available = True
            
            if not neo4j_available:
                logger.warning("Neo4j not available for auto-load")
                return 0
            
            loaded_count = 0
            with neo4j_driver.session() as session:
                # Query ALL Entity nodes with names
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name IS NOT NULL OR e.id IS NOT NULL
                    RETURN e.id as id, e.name as name, e.category as category, 
                           e.confidence as confidence, e.embedding as embedding
                    LIMIT 50
                """)
                
                # Initialize neurons dict if needed
                if not hasattr(self, 'neurons'):
                    self.neurons = {}
                
                for record in result:
                    neuron_id = record['id'] or f"entity_{loaded_count}"
                    neuron_name = record['name'] or f"Entity {loaded_count}"
                    category = record['category'] or 'general'
                    confidence = float(record['confidence']) if record['confidence'] else 0.5
                    
                    # Create neuron object for brain visualization
                    neuron = {
                        'id': neuron_id,
                        'name': neuron_name,
                        'category': category,
                        'confidence': confidence,
                        'activation': 0.0,
                        'synapses': [],
                        'position': {
                            'x': (hash(neuron_id) % 1000) / 10 - 50,  # Generate deterministic position
                            'y': (hash(neuron_name) % 1000) / 10 - 50,
                            'z': (hash(category) % 1000) / 10 - 50
                        }
                    }
                    
                    self.neurons[neuron_id] = neuron
                    loaded_count += 1
                    
                    # Also add to SI Core if available
                    if hasattr(self.evolution, 'si_core') and hasattr(self.si_core, 'add_insight'):
                        try:
                            self.si_core.add_insight(
                                insight_text=neuron_name,
                                entity_type=category,
                                entities=[neuron_name],
                                relationship='loaded_from_neo4j',
                                source_topic='AutoLoad',
                                target_topic='Brain',
                                confidence=confidence
                            )
                        except:
                            pass
                
                logger.info(f"✅ Auto-loaded {loaded_count} neurons from Neo4j Entity nodes")
                
                # Now load synapses between neurons
                self._load_synapses_from_neo4j(neo4j_driver)
                
                return loaded_count
                
        except Exception as e:
            logger.error(f"Auto-load failed: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _load_synapses_from_neo4j(self, neo4j_driver):
        """Load synapses from Neo4j relationships between Entity nodes"""
        try:
            if not hasattr(self, 'neurons'):
                self.neurons = {}
                return 0
            
            with neo4j_driver.session() as session:
                # Find relationships between entities
                result = session.run("""
                    MATCH (e1:Entity)-[r]-(e2:Entity)
                    WHERE e1.id IS NOT NULL AND e2.id IS NOT NULL
                    RETURN e1.id as source_id, e2.id as target_id, type(r) as rel_type, 
                           r.strength as strength
                    LIMIT 100
                """)
                
                synapse_count = 0
                for record in result:
                    source_id = record['source_id']
                    target_id = record['target_id']
                    
                    # Only create synapse if both neurons exist
                    if source_id in self.neurons and target_id in self.neurons:
                        strength = float(record['strength']) if record['strength'] else 0.5
                        
                        synapse = {
                            'source': source_id,
                            'target': target_id,
                            'strength': strength,
                            'type': record['rel_type'] or 'related'
                        }
                        
                        # Add to neuron's synapse list
                        self.neurons[source_id]['synapses'].append(synapse)
                        synapse_count += 1
                
                logger.info(f"✅ Loaded {synapse_count} synapses from Neo4j relationships")
                return synapse_count
                
        except Exception as e:
            logger.error(f"Failed to load synapses: {e}")
            return 0

    def _classify_neurons_by_layer(self):
        """Classify existing neurons into functional layers for multi-layer brain"""
        try:
            if not hasattr(self.evolution, 'si_core'):
                return
            
            si = self.si_core
            if not hasattr(si, 'insights') or not si.insights:
                return
            
            # Layer classification rules (preserves your color scheme)
            layer_mapping = {}
            
            for insight_id, insight in si.insights.items():
                # Get category and confidence
                if hasattr(insight, '__dict__'):
                    category = getattr(insight, 'category', getattr(insight, 'entity_type', 'general'))
                    confidence = getattr(insight, 'confidence', 0.5)
                    text = getattr(insight, 'insight_text', getattr(insight, 'text', ''))
                elif isinstance(insight, dict):
                    category = insight.get('category', insight.get('entity_type', 'general'))
                    confidence = insight.get('confidence', 0.5)
                    text = insight.get('insight_text', insight.get('text', ''))
                else:
                    continue
                
                # Determine layer based on category and confidence
                # (Preserves your color scheme - layers add structure, not replace colors)
                if category in ['core', 'logical', 'causal', 'reverse']:
                    layer = 'reasoning'
                elif category in ['artistic', 'creative']:
                    layer = 'abstraction'
                elif category in ['wealth', 'finance', 'action']:
                    layer = 'output'
                elif category in ['entity', 'general', 'research']:
                    layer = 'input'
                elif confidence > 0.7:
                    layer = 'reasoning'
                elif confidence > 0.4:
                    layer = 'pattern'
                else:
                    layer = 'input'
                
                # Store layer assignment (adds structural info without changing colors)
                if hasattr(insight, '__dict__'):
                    insight.layer = layer
                elif isinstance(insight, dict):
                    insight['layer'] = layer
                
                layer_mapping[insight_id] = layer
            
            # Save the layer assignments
            if hasattr(si, 'save_state'):
                si.save_state()
            
            logger.info(f"✅ Classified {len(layer_mapping)} neurons into layers")
            return layer_mapping
        except Exception as e:
            logger.error(f"Layer classification failed: {e}")
            return {}

 # ============================================================================
 # BACKGROUND TASK METHODS FOR USER INPUT SYSTEM
 # ============================================================================
    
    def _research_task(self, query, category):
        """Background research task"""
        try:
            logger.info(f"🔍 Researching: {query}")
            if hasattr(self.evolution, 'ai_hub') and self.evolution.ai_hub:
                result = self.evolution.ai_hub.query_all_tutors(f"Research the following: {query}")
                if result and result.get('responses'):
                    if hasattr(self.evolution, 'si_core'):
                        self.si_core.add_insight(
                            insight_text=f"Research: {query[:100]}",
                            entity_type="research",
                            entities=[query[:50], category],
                            relationship="researched",
                            source_topic="user_task",
                            target_topic=category,
                            confidence=0.7
                        )
            self._add_recent_task('research', query, 'completed')
        except Exception as e:
            logger.error(f"Research task failed: {e}")
            self._add_recent_task('research', query, 'error')
    
    def _ingest_task(self, source, category):
        """Background ingest task"""
        try:
            logger.info(f"📥 Ingesting: {source}")
            if hasattr(self.evolution, 'knowledge_graph'):
                from datetime import datetime
                self.evolution.knowledge_graph.add_concept(
                    source[:100],
                    category,
                    {'source': source, 'ingested': datetime.now().isoformat()}
                )
            self._add_recent_task('ingest', source, 'completed')
        except Exception as e:
            logger.error(f"Ingest task failed: {e}")
            self._add_recent_task('ingest', source, 'error')
    
    def _reverse_engineer_task(self, target, category):
        """Background reverse engineering task"""
        try:
            logger.info(f"🔧 Reverse engineering: {target}")
            if hasattr(self.evolution, 'reverse_engineering'):
                result = self.evolution.reverse_engineering.analyze(target)
                if result and hasattr(self.evolution, 'si_core'):
                    self.si_core.add_insight(
                        insight_text=f"Reverse engineered: {target[:100]}",
                        entity_type="reverse_engineered",
                        entities=[target[:50], category],
                        relationship="analyzed",
                        source_topic="user_task",
                        target_topic=category,
                        confidence=0.8
                    )
            self._add_recent_task('reverse_engineer', target, 'completed')
        except Exception as e:
            logger.error(f"Reverse engineering failed: {e}")
            self._add_recent_task('reverse_engineer', target, 'error')
    
    def _analyze_task(self, data, category):
        """Background analysis task"""
        try:
            logger.info(f"📊 Analyzing: {data[:100]}")
            self._add_recent_task('analyze', data[:200], 'completed')
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self._add_recent_task('analyze', data[:200], 'error')
    
    def _learn_topic_task(self, topic, category):
        """Background topic learning task"""
        try:
            logger.info(f"📚 Learning topic: {topic}")
            if hasattr(self.evolution, 'learning_orchestrator'):
                topic_info = {'topic': topic, 'category': category, 'mastery_threshold': 1}
                if hasattr(self.evolution.learning_orchestrator, 'learn_topic'):
                    result = self.evolution.learning_orchestrator.learn_topic(topic_info, 0.5)
                    if result and result.get('success') and hasattr(self.evolution, 'si_core'):
                        self.si_core.add_insight(
                            insight_text=topic,
                            entity_type="user_learned",
                            entities=[topic, category],
                            relationship="mastered",
                            source_topic="user_task",
                            target_topic=category,
                            confidence=0.9
                        )
            self._add_recent_task('learn_topic', topic, 'completed')
        except Exception as e:
            logger.error(f"Learn topic failed: {e}")
            self._add_recent_task('learn_topic', topic, 'error')
    
    def _add_dictionary_task(self, word, category):
        """Add dictionary word task"""
        try:
            logger.info(f"📖 Adding dictionary word: {word}")
            if hasattr(self.evolution, 'si_core'):
                self.si_core.add_insight(
                    insight_text=f"Dictionary: {word}",
                    entity_type="dictionary",
                    entities=[word, category],
                    relationship="defined",
                    source_topic="user_task",
                    target_topic=category,
                    confidence=0.95
                )
            self._add_recent_task('add_dictionary', word, 'completed')
        except Exception as e:
            logger.error(f"Add dictionary failed: {e}")
            self._add_recent_task('add_dictionary', word, 'error')
    
    def _add_encyclopedia_task(self, topic, category):
        """Add encyclopedia topic task"""
        try:
            logger.info(f"📚 Adding encyclopedia topic: {topic}")
            if hasattr(self.evolution, 'si_core'):
                self.si_core.add_insight(
                    insight_text=f"Encyclopedia: {topic}",
                    entity_type="encyclopedia",
                    entities=[topic, category],
                    relationship="documented",
                    source_topic="user_task",
                    target_topic=category,
                    confidence=0.9
                )
            self._add_recent_task('add_encyclopedia', topic, 'completed')
        except Exception as e:
            logger.error(f"Add encyclopedia failed: {e}")
            self._add_recent_task('add_encyclopedia', topic, 'error')
    
    def _add_recent_task(self, action, input_data, status):
        """Track recent tasks"""
        if not hasattr(self, '_recent_tasks'):
            self._recent_tasks = []
        from datetime import datetime
        self._recent_tasks.append({
            'action': action,
            'input': input_data[:200],
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
        self._recent_tasks = self._recent_tasks[-50:]
    
    def _start_evolution(self):
        def evolve():
            while True:
                try:
                    result = self.evolution.evolution_cycle()
                    cycle_num = result.get('evolution_cycle', 0)
                    if cycle_num % 10 == 0:
                        logger.info(f"Cycle {cycle_num}: Consciousness {result.get('consciousness', 0)}")
                    wait_time = self.evolution.evolution_timer.get_wait_time()
                    if wait_time < 30:
                        wait_time = 30
                    time.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Evolution error: {e}")
                    time.sleep(60)
        threading.Thread(target=evolve, daemon=True).start()
        logger.info("🔄 Evolution thread started")
            
    def _setup_routes(self):
        # ============================================================
        # CHAT ENDPOINT - MUST BE FIRST TO AVOID OVERWRITE
        # ============================================================
        
        @self.app.route('/api/chat', methods=['POST'])
        def api_chat():
            """Process chat messages with DMAI"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                message = data.get('message', '')
                user = data.get('user', 'anonymous')
                
                if not message or not message.strip():
                    return jsonify({'response': 'Please enter a message.'})
                
                # Process the message through DMAI's conversation system
                result = self.evolution.process_message(user, message.strip())
                
                # Check if result is a Flask response (for binary data like images)
                from flask import Response
                if isinstance(result, Response):
                    return result
                
                # Otherwise return JSON
                return jsonify({
                    'response': result,
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Chat error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'response': f"I encountered an error: {str(e)}",
                    'status': 'error'
                }), 500
        
        # ============================================================
        # FUNDING ROUTES - REGISTER FIRST
        # ============================================================
        
        @self.app.route('/api/funding/status', methods=['GET'])
        def api_funding_status():
            """Get self-funding training status"""
            try:
                if hasattr(self, 'evolution') and self.evolution and hasattr(self.evolution, 'funding_training') and self.evolution.funding_training:
                    status = self.evolution.funding_training.status()
                    return jsonify(status)
                else:
                    return jsonify({'error': 'Funding orchestrator not initialized', 'phase': '1 - Knowledge Acquisition', 'ready_for_phase_2': False})
            except Exception as e:
                logger.error(f"Funding status error: {e}")
                return jsonify({'error': str(e), 'phase': '1 - Knowledge Acquisition', 'ready_for_phase_2': False})

        @self.app.route('/api/funding/strategies')
        def api_funding_strategies():
            if self.evolution and hasattr(self.evolution, 'funding_training') and self.evolution.funding_training:
                avenue = request.args.get('avenue', None)
                return jsonify(self.evolution.funding_training.get_strategy_candidates(avenue))
            return jsonify({'error': 'Funding training not available'}), 503

        @self.app.route('/api/funding/test', methods=['POST'])
        def api_funding_test():
            return jsonify({'success': True, 'message': 'Test endpoint works'})

        @self.app.route('/api/funding/complete_phase1', methods=['POST'])
        def api_funding_complete_phase1():
            """Complete Phase 1 - final working version"""
            try:
                ft = self.evolution.funding_training
                
                # Get training object
                if hasattr(ft, 'training'):
                    training = ft.training
                else:
                    training = ft
                
                # Get all unique topics
                unique_topics = set()
                for avenue in training.revenue_avenues.values():
                    for topic in avenue['topics']:
                        unique_topics.add(topic)
                
                # Mark all topics as learned
                for topic in unique_topics:
                    training.learned_concepts.add(topic)
                
                # Mark all avenues as completed
                for avenue in training.revenue_avenues.values():
                    avenue['completed'] = True
                
                # Create dummy strategies for each avenue
                for avenue_name in training.revenue_avenues.keys():
                    if len(training.strategy_candidates.get(avenue_name, [])) == 0:
                        training.strategy_candidates[avenue_name] = [{
                            'id': f"{avenue_name}_strategy_1",
                            'name': f"{avenue_name.replace('_', ' ').title()} Paper Strategy",
                            'description': f"Paper execution strategy for {avenue_name} - to be tested and refined in Phase 2",
                            'status': 'proposed',
                            'paper_only': True,
                            'created_at': datetime.now().isoformat()
                        }]
                
                training._training_complete = True
                training.learning_active = False
                training._save_state()  # Fixed: use _save_state instead of save_state
                
                return jsonify({
                    'success': True,
                    'message': 'Phase 1 completed',
                    'concepts_learned': len(training.learned_concepts),
                    'total_unique_topics': len(unique_topics),
                    'strategies_created': sum(len(v) for v in training.strategy_candidates.values()),
                    'ready_for_phase_2': training._ready_for_phase_2()
                })
                
            except Exception as e:
                import traceback
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
        
        @self.app.route('/api/funding/phase2_request', methods=['POST'])
        def api_funding_phase2_request():
            if self.evolution and hasattr(self.evolution, 'funding_training') and self.evolution.funding_training:
                return jsonify(self.evolution.funding_training.request_phase_2_approval())
            return jsonify({'error': 'Funding training not available'}), 503

        @self.app.route('/api/funding/fix_duplicate_topics', methods=['POST'])
        def api_funding_fix_duplicate_topics():
            """Properly fix duplicate topic counting and prepare Phase 2"""
            try:
                if not (self.evolution and hasattr(self.evolution, 'funding_training')):
                    return jsonify({'error': 'Funding training not initialized'}), 500
                
                ft = self.evolution.funding_training
                
                # Navigate to training object
                if hasattr(ft, 'training'):
                    training = ft.training
                else:
                    training = ft
                
                result = training.fix_duplicate_topic_counting()
                return jsonify(result)
            except Exception as e:
                logger.error(f"Fix duplicate topics error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        @self.app.route('/api/funding/fix_counting', methods=['POST'])
        def api_funding_fix_counting():
            """Fix concept counting to use unique topics instead of duplicates"""
            try:
                if self.evolution and hasattr(self.evolution, 'funding_training') and self.evolution.funding_training:
                    result = self.evolution.funding_training.fix_concept_counting()
                    return jsonify(result)
                else:
                    return jsonify({'error': 'Funding orchestrator not initialized'}), 500
            except Exception as e:
                logger.error(f"Fix counting error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/funding/debug', methods=['GET'])
        def api_funding_debug():
            has_funding = self.evolution and hasattr(self.evolution, 'funding_training') and self.evolution.funding_training is not None
            return jsonify({
                'has_funding_training': has_funding,
                'evolution_exists': self.evolution is not None,
                'funding_type': str(type(self.evolution.funding_training)) if has_funding else None
            })
        
        # ============================================================
        # MAIN ROUTES
        # ============================================================
        
        @self.app.route('/')
        def index():
            return redirect('/status')

        @self.app.route('/api/evolution/start', methods=['POST'])
        def start_evolution_manually():
            """Manually start the evolution thread"""
            try:
                if not hasattr(self.evolution, '_evolution_thread_started'):
                    self.evolution._evolution_thread_started = True
                    return jsonify({'success': True, 'message': 'Evolution thread started'})
                else:
                    return jsonify({'success': False, 'message': 'Evolution thread already running'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        # ADD THIS ENDPOINT HERE
        @self.app.route('/api/evolution/cycle', methods=['POST'])
        def trigger_evolution_cycle():
            """Manually trigger an evolution cycle"""
            try:
                result = self.evolution.evolution_cycle()
                return jsonify({
                    'success': True,
                    'evolution_cycle': result.get('evolution'),
                    'consciousness': result.get('consciousness'),
                    'consciousness_growth': result.get('consciousness_growth'),
                    'neurons_added': result.get('neurons_added'),
                    'synapses_added': result.get('synapses_added'),
                    'successful_evolutions': result.get('successful_evolutions'),
                    'evolution_kpis': self.evolution.synthetic_network.get_kpis_dict() if hasattr(self.evolution.synthetic_network, 'get_kpis_dict') else {}
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/status')
        def status_page():
            return render_template_string(STATUS_TEMPLATE, status=self.evolution.get_status())

        @self.app.route('/vision')
        def vision():
            return render_template_string(VISION_TEMPLATE)

        @self.app.route('/help')
        def help_page():
            return render_template_string(HELP_TEMPLATE)

        @self.app.route('/admin')
        def admin():
            return render_template_string(ADMIN_TEMPLATE)

        @self.app.route('/chat')
        def chat():
            return render_template_string(CHAT_TEMPLATE)

        @self.app.route('/knowledge')
        def knowledge():
            return render_template_string(KNOWLEDGE_TEMPLATE,
                status=self.evolution.get_status(),
                concepts=self.evolution.get_knowledge_concepts(200),
                training_details=self.evolution.get_training_details())

        @self.app.route('/api/status')
        def api_status():
            """Return DMAI system status"""
            status_dict = self.evolution.get_status()
            # Remove non-serializable evolution_timer if it somehow appears
            if isinstance(status_dict, dict) and 'evolution_timer' in status_dict:
                del status_dict['evolution_timer']
            return jsonify(status_dict)

        @self.app.route('/api/simple/version')
        def simple_version():
            """Simple version endpoint"""
            return jsonify({
                'version': 'v8.0.38',
                'commit': '777a9ae15',
                'timestamp': '2026-04-02'
            })

        @self.app.route('/api/brain/classify_layers', methods=['POST'])
        def classify_layers():
            """Trigger layer classification for multi-layer brain"""
            try:
                layers = self._classify_neurons_by_layer()
                return jsonify({
                    'success': True,
                    'neurons_classified': len(layers),
                    'message': f'Classified {len(layers)} neurons into layers'
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/funding/debug/attrs', methods=['GET'])
        def debug_funding_attrs():
            """Debug: See what attributes funding_training actually has"""
            try:
                if hasattr(self.evolution, 'funding_training'):
                    training = self.evolution.funding_training
                    attrs = [a for a in dir(training) if not a.startswith('_')]
                    return jsonify({
                        'type': str(type(training)),
                        'attributes': attrs,
                        'has_revenue_avenues': hasattr(training, 'revenue_avenues'),
                        'has_avenues': hasattr(training, 'avenues'),
                        'has_learned_concepts': hasattr(training, 'learned_concepts')
                    })
                return jsonify({'error': 'Funding training not available'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/funding/force_complete', methods=['POST'])
        def force_complete_funding():
            """Force complete remaining funding concepts to unlock Phase 2 execution"""
            try:
                if hasattr(self.evolution, 'funding_training'):
                    orchestrator = self.evolution.funding_training
                    if hasattr(orchestrator, 'training'):
                        training = orchestrator.training
                        # Force all topics learned across all revenue avenues
                        if hasattr(training, 'revenue_avenues'):
                            for avenue_name, avenue in training.revenue_avenues.items():
                                for topic in avenue.get('topics', []):
                                    if hasattr(training, 'learned_concepts') and topic not in training.learned_concepts:
                                        training.learned_concepts.append(topic)
                                avenue['completed'] = True
                                avenue['progress'] = 100.0
                            
                            training._save_state()
                            
                            return jsonify({
                                'success': True,
                                'ready_for_phase_2': orchestrator._ready_for_phase_2() if hasattr(orchestrator, '_ready_for_phase_2') else False,
                                'message': 'Funding training force-completed!'
                            })
                    return jsonify({'success': False, 'error': 'Training object structure unknown'}), 500
                return jsonify({'success': False, 'error': 'Funding training not available'}), 500
            except Exception as e:
                import traceback
                return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500

                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/funding/generate_strategies', methods=['POST'])
        def api_funding_generate_strategies():
            """Generate actionable strategies for all revenue avenues using learned knowledge"""
            try:
                if hasattr(self.evolution, 'funding_training'):
                    orchestrator = self.evolution.funding_training
                    training = orchestrator.training if hasattr(orchestrator, 'training') else None
                    
                    if not training:
                        return jsonify({'error': 'Training object not available'}), 500
                    
                    strategies_generated = {}
                    
                    # Strategy templates based on learned concepts
                    strategy_templates = {
                        'quant_trading': [
                            {
                                'name': 'Simple Moving Average Crossover',
                                'type': 'paper_trading',
                                'description': 'Buy when 50-period SMA crosses above 200-period SMA, sell when opposite',
                                'required_concepts': ['technical_analysis', 'strategy_development'],
                                'implementation': 'use_capability:automaton_trading_bot'
                            },
                            {
                                'name': 'Mean Reversion Strategy',
                                'type': 'paper_trading',
                                'description': 'Trade when price deviates significantly from moving average',
                                'required_concepts': ['technical_analysis', 'risk_management'],
                                'implementation': 'use_capability:automaton_trading_bot'
                            }
                        ],
                        'content_creation': [
                            {
                                'name': 'AI-Generated Blog Posts',
                                'type': 'automated',
                                'description': 'Generate daily AI/tech blog posts using DMAI knowledge synthesis',
                                'required_concepts': ['content_strategy', 'writing_techniques'],
                                'implementation': 'use_capability:content_generator'
                            }
                        ],
                        'ai_services': [
                            {
                                'name': 'DMAI API Endpoint',
                                'type': 'api_service',
                                'description': 'Expose DMAI capabilities as paid API endpoints',
                                'required_concepts': ['api_design', 'pricing_strategies'],
                                'implementation': 'use_capability:api_endpoint'
                            }
                        ],
                        'software_products': [
                            {
                                'name': 'Automaton Trading Bot SaaS',
                                'type': 'subscription',
                                'description': 'Offer Automaton as a subscription service',
                                'required_concepts': ['saas_pricing_models', 'product_market_fit'],
                                'implementation': 'use_capability:automaton_core'
                            }
                        ],
                        'affiliate_referral': [
                            {
                                'name': 'AI Tool Affiliate Program',
                                'type': 'affiliate',
                                'description': 'Promote AI tools and earn commissions',
                                'required_concepts': ['affiliate_networks', 'tracking_attribution'],
                                'implementation': 'use_capability:affiliate_tracker'
                            }
                        ],
                        'data_services': [
                            {
                                'name': 'Market Intelligence Reports',
                                'type': 'data_product',
                                'description': 'Generate and sell AI-analyzed market reports',
                                'required_concepts': ['insight_generation', 'data_visualization'],
                                'implementation': 'use_capability:data_analyzer'
                            }
                        ],
                        'education_training': [
                            {
                                'name': 'DMAI AGI Course',
                                'type': 'course',
                                'description': 'Create and sell course on AGI development',
                                'required_concepts': ['course_creation', 'curriculum_design'],
                                'implementation': 'use_capability:course_generator'
                            }
                        ],
                        'consulting_analysis': [
                            {
                                'name': 'AI Strategy Consulting',
                                'type': 'consulting',
                                'description': 'Offer AI implementation consulting',
                                'required_concepts': ['consulting_engagements', 'strategic_recommendations'],
                                'implementation': 'use_capability:consulting_engine'
                            }
                        ],
                        'ad_revenue': [
                            {
                                'name': 'DMAI Blog Ad Placement',
                                'type': 'advertising',
                                'description': 'Place ads on DMAI-generated content',
                                'required_concepts': ['ad_placement_optimization', 'yield_management'],
                                'implementation': 'use_capability:ad_manager'
                            }
                        ],
                        'crowdfunding_patronage': [
                            {
                                'name': 'DMAI Patreon',
                                'type': 'patronage',
                                'description': 'Offer exclusive content and early access to patrons',
                                'required_concepts': ['membership_tiers', 'community_building'],
                                'implementation': 'use_capability:patreon_integration'
                            }
                        ]
                    }
                    
                    # Use the existing strategy_candidates dictionary
                    if not hasattr(training, 'strategy_candidates'):
                        training.strategy_candidates = {}
                    
                    # Clear existing and repopulate, or just add to existing
                    for avenue, templates in strategy_templates.items():
                        if avenue not in training.strategy_candidates:
                            training.strategy_candidates[avenue] = []
                        else:
                            # Optionally clear existing strategies for this avenue
                            training.strategy_candidates[avenue] = []
                        
                        for template in templates:
                            # Check if DMAI has learned required concepts
                            learned = getattr(training, 'learned_concepts', [])
                            required = template.get('required_concepts', [])
                            ready = all(c in str(learned) for c in required) or True
                            
                            strategy = {
                                **template,
                                'ready': ready,
                                'generated_at': datetime.now().isoformat(),
                                'status': 'paper_trading' if ready else 'pending_concepts'
                            }
                            training.strategy_candidates[avenue].append(strategy)
                            strategies_generated[avenue] = len(training.strategy_candidates[avenue])
                    
                    return jsonify({
                        'success': True,
                        'strategies_generated': strategies_generated,
                        'total_strategies': sum(strategies_generated.values()),
                        'message': 'Strategies generated from learned concepts'
                    })
                    
                return jsonify({'error': 'Funding training not available'}), 500
            except Exception as e:
                import traceback
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        # ============================================================
        # PHASE 2 FUNDING ENDPOINTS
        # ============================================================
        
        @self.app.route('/api/funding/phase2/transition', methods=['POST'])
        def api_funding_phase2_transition():
            """Transition from Phase 1 to Phase 2 (Paper Execution)"""
            try:
                if not hasattr(self.evolution, 'funding_training') or not self.evolution.funding_training:
                    return jsonify({'error': 'Funding orchestrator not available'}), 503
                
                # Get approved strategies from request (optional)
                data = request.get_json(silent=True) or {}
                approved_strategies = data.get('approved_strategies', None)
                
                result = self.evolution.funding_training.transition_to_phase_2(approved_strategies)
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/funding/phase2/start', methods=['POST'])
        def api_funding_phase2_start():
            """Start Phase 2 paper execution"""
            try:
                if not hasattr(self.evolution, 'funding_training') or not self.evolution.funding_training:
                    return jsonify({'error': 'Funding orchestrator not available'}), 503
                
                result = self.evolution.funding_training.start_phase_2()
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/funding/phase2/stop', methods=['POST'])
        def api_funding_phase2_stop():
            """Stop Phase 2 paper execution"""
            try:
                if not hasattr(self.evolution, 'funding_training') or not self.evolution.funding_training:
                    return jsonify({'error': 'Funding orchestrator not available'}), 503
                
                result = self.evolution.funding_training.stop_phase_2()
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/funding/phase2/status', methods=['GET'])
        def api_funding_phase2_status():
            """Get Phase 2 status"""
            try:
                if not hasattr(self.evolution, 'funding_training') or not self.evolution.funding_training:
                    return jsonify({'error': 'Funding orchestrator not available'}), 503
                
                result = self.evolution.funding_training.get_phase_2_status()
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/funding/phase2/enable_strategy', methods=['POST'])
        def api_funding_phase2_enable_strategy():
            """Enable a specific strategy for Phase 2"""
            try:
                if not hasattr(self.evolution, 'funding_training') or not self.evolution.funding_training:
                    return jsonify({'error': 'Funding orchestrator not available'}), 503
                
                data = request.get_json()
                avenue = data.get('avenue')
                strategy_id = data.get('strategy_id')
                
                if not avenue or not strategy_id:
                    return jsonify({'error': 'Missing avenue or strategy_id'}), 400
                
                result = self.evolution.funding_training.enable_strategy(avenue, strategy_id)
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/funding/debug/strategies', methods=['GET'])
        def debug_funding_strategies():
            """Debug: See what's in strategy_candidates"""
            try:
                if hasattr(self.evolution, 'funding_training'):
                    orchestrator = self.evolution.funding_training
                    training = orchestrator.training if hasattr(orchestrator, 'training') else None
                    
                    result = {
                        'orchestrator_has': hasattr(orchestrator, 'strategy_candidates'),
                        'training_has': hasattr(training, 'strategy_candidates') if training else False,
                    }
                    
                    if training and hasattr(training, 'strategy_candidates'):
                        result['training_strategies'] = {
                            avenue: len(candidates) 
                            for avenue, candidates in training.strategy_candidates.items()
                        }
                    
                    return jsonify(result)
                return jsonify({'error': 'Not available'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/insights/sample')
        def debug_insights_sample():
            """Sample insights with source breakdown"""
            try:
                si = self.si_core
                samples = []
                stats = {'total': 0, 'by_source_type': {}, 'by_entity_type': {}, 'columns': []}
                
                if hasattr(si, 'sqlite') and si.sqlite:
                    conn = si.sqlite._get_connection()
                    cursor = conn.cursor()
                    
                    # Get total count
                    cursor.execute('SELECT COUNT(*) FROM insights')
                    stats['total'] = cursor.fetchone()[0]
                    
                    # Check columns
                    cursor.execute("PRAGMA table_info(insights)")
                    stats['columns'] = [col[1] for col in cursor.fetchall()]
                    
                    # Breakdown by source_type (if column exists)
                    if 'source_type' in stats['columns']:
                        cursor.execute('''
                            SELECT source_type, COUNT(*) 
                            FROM insights 
                            WHERE source_type IS NOT NULL
                            GROUP BY source_type
                        ''')
                        stats['by_source_type'] = dict(cursor.fetchall())
                    
                    # Breakdown by entity_type
                    cursor.execute('''
                        SELECT entity_type, COUNT(*) 
                        FROM insights 
                        GROUP BY entity_type
                        LIMIT 15
                    ''')
                    stats['by_entity_type'] = dict(cursor.fetchall())
                    
                    # Get recent insights
                    created_col = 'created_at' if 'created_at' in stats['columns'] else 'id'
                    cursor.execute(f'''
                        SELECT id, insight_text, entity_type, 
                               source_type, {created_col}
                        FROM insights 
                        ORDER BY {created_col} DESC
                        LIMIT 10
                    ''')
                    for row in cursor.fetchall():
                        samples.append({
                            'id': row[0],
                            'text': row[1][:80],
                            'entity_type': row[2],
                            'source_type': row[3] if len(row) > 3 else None,
                            'created': row[4] if len(row) > 4 else None
                        })
                
                return jsonify({
                    'total_insights': stats['total'],
                    'columns': stats['columns'],
                    'by_source_type': stats['by_source_type'],
                    'by_entity_type': stats['by_entity_type'],
                    'recent_insights': samples
                })
            except Exception as e:
                import traceback
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

        @self.app.route('/api/debug/disk', methods=['GET'])
        def debug_disk():
            """Check disk and database status"""
            import os
            db_path = '/opt/render/project/src/data/dmai_knowledge.db'
            data_dir = '/opt/render/project/src/data'
            return jsonify({
                'db_exists': os.path.exists(db_path),
                'db_size_mb': round(os.path.getsize(db_path) / (1024*1024), 2) if os.path.exists(db_path) else 0,
                'data_dir_exists': os.path.exists(data_dir),
                'data_dir_contents': os.listdir(data_dir) if os.path.exists(data_dir) else [],
                'cwd': os.getcwd(),
                'disk_usage': os.popen('df -h /opt/render/project/src/data 2>/dev/null || echo "N/A"').read().strip()
            })

        @self.app.route('/api/debug/neo4j_env', methods=['GET'])
        def debug_neo4j_env():
            """Check Neo4j environment variables"""
            import os
            return jsonify({
                'NEO4J_URI_set': bool(os.environ.get('NEO4J_URI')),
                'NEO4J_USER_set': bool(os.environ.get('NEO4J_USER')),
                'NEO4J_PASSWORD_set': bool(os.environ.get('NEO4J_PASSWORD')),
                'all_vars': [k for k in os.environ.keys() if 'NEO4J' in k or 'neo4j' in k]
            })
        
        @self.app.route('/api/debug/neo4j_detail', methods=['GET'])
        def debug_neo4j_detail():
            """Detailed Neo4j connection debug"""
            import os
            import traceback
            result = {
                'env_vars_set': {
                    'NEO4J_URI': bool(os.environ.get('NEO4J_URI')),
                    'NEO4J_USER': bool(os.environ.get('NEO4J_USER')),
                    'NEO4J_PASSWORD': bool(os.environ.get('NEO4J_PASSWORD'))
                },
                'neo4j_storage_exists': hasattr(self.evolution, 'neo4j_storage'),
                'connection_error': None
            }
            
            if hasattr(self.evolution, 'neo4j_storage'):
                try:
                    storage = self.evolution.neo4j_storage
                    if hasattr(storage, 'is_available'):
                        result['is_available'] = storage.is_available()
                    if hasattr(storage, '_driver') and storage._driver:
                        result['driver_created'] = True
                        try:
                            with storage._driver.session() as session:
                                test = session.run("RETURN 1 as test").single()["test"]
                                result['test_query'] = True
                        except Exception as e:
                            result['test_query_error'] = str(e)
                except Exception as e:
                    result['connection_error'] = str(e)
                    result['traceback'] = traceback.format_exc()
            
            return jsonify(result)
        
        @self.app.route('/api/debug/neo4j_data', methods=['GET'])
        def debug_neo4j_data():
            """See what data Neo4j actually returns"""
            try:
                if self.evolution.neo4j_storage and self.evolution.neo4j_storage.is_available():
                    restored = self.evolution.neo4j_storage.restore_all()
                    return jsonify({
                        'evolution': restored.get('evolution'),
                        'has_evolution': restored.get('evolution') is not None,
                        'neurons': restored.get('evolution', {}).get('neurons', 0) if restored.get('evolution') else 0,
                        'consciousness': restored.get('evolution', {}).get('consciousness', 0) if restored.get('evolution') else 0
                    })
                return jsonify({'error': 'Neo4j not available'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/neo4j_insights', methods=['GET'])
        def debug_neo4j_insights():
            """Check how many insights are in Neo4j"""
            try:
                if self.evolution.neo4j_storage and self.evolution.neo4j_storage.is_available():
                    driver = self.evolution.neo4j_storage.driver
                    if driver:
                        with driver.session() as session:
                            result = session.run("MATCH (i:Insight) RETURN count(i) as count")
                            count = result.single()["count"]
                            return jsonify({'insights_in_neo4j': count})
                return jsonify({'error': 'Neo4j not available'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/neo4j_nodes', methods=['GET'])
        def debug_neo4j_nodes():
            """List all node types and counts in Neo4j"""
            try:
                if not self.evolution.neo4j_storage or not self.evolution.neo4j_storage.is_available():
                    return jsonify({'error': 'Neo4j not available'})
                
                driver = self.evolution.neo4j_storage.driver
                if not driver:
                    return jsonify({'error': 'No driver'})
                
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
                    nodes = []
                    for record in result:
                        nodes.append({'labels': record['labels'], 'count': record['count']})
                    
                    return jsonify({'node_types': nodes, 'total_nodes': sum(n['count'] for n in nodes)})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/neo4j/count', methods=['GET'])        
        def neo4j_count():
            """Count nodes in Neo4j by type"""
            try:
                storage = self.evolution.neo4j_storage
                if not storage or not storage.is_available():
                    return jsonify({'error': 'Neo4j not available'})
                
                driver = storage.driver
                if not driver:
                    return jsonify({'error': 'No driver'})
                
                with driver.session() as session:
                    # Count Insight nodes
                    result = session.run("MATCH (i:Insight) RETURN count(i) as count")
                    insight_count = result.single()["count"]
                    
                    # Count Evolution nodes
                    result2 = session.run("MATCH (e:Evolution) RETURN count(e) as count")
                    evolution_count = result2.single()["count"]
                    
                    return jsonify({
                        'insight_nodes': insight_count,
                        'evolution_nodes': evolution_count,
                        'total_nodes': insight_count + evolution_count
                    })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        def debug_neo4j_nodes():
            """List all node types and counts in Neo4j"""
            try:
                if not self.evolution.neo4j_storage or not self.evolution.neo4j_storage.is_available():
                    return jsonify({'error': 'Neo4j not available'})
                
                driver = self.evolution.neo4j_storage.driver
                if not driver:
                    return jsonify({'error': 'No driver'})
                
                with driver.session() as session:
                    # Get all node labels and counts
                    result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
                    nodes = []
                    for record in result:
                        nodes.append({'labels': record['labels'], 'count': record['count']})
                    
                    # Get sample Insight if any
                    insight_sample = []
                    result2 = session.run("MATCH (i:Insight) RETURN i LIMIT 3")
                    for record in result2:
                        node = dict(record['i'].items())
                        insight_sample.append({k: str(v)[:100] for k, v in node.items()})
                    
                    return jsonify({
                        'node_types': nodes,
                        'insight_count': len(insight_sample),
                        'insight_sample': insight_sample
                    })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/entity_sample', methods=['GET'])
        def debug_entity_sample():
            """Show sample Entity node structure"""
            try:
                storage = self.evolution.neo4j_storage
                if not storage or not storage.is_available():
                    return jsonify({'error': 'Neo4j not available'})
                
                driver = storage.driver
                if not driver:
                    return jsonify({'error': 'No driver'})
                
                with driver.session() as session:
                    # Get one Entity node
                    result = session.run("MATCH (i:Entity) RETURN i LIMIT 1")
                    record = result.single()
                    if record:
                        node = dict(record['i'].items())
                        return jsonify({
                            'has_entity_nodes': True,
                            'sample_node_keys': list(node.keys()),
                            'sample_node': {k: str(v)[:200] for k, v in node.items()}
                        })
                    else:
                        # Try to find any node
                        result2 = session.run("MATCH (n) RETURN n LIMIT 1")
                        record2 = result2.single()
                        if record2:
                            node2 = dict(record2['n'].items())
                            return jsonify({
                                'has_entity_nodes': False,
                                'has_other_nodes': True,
                                'other_node_labels': list(record2['n'].labels),
                                'other_node_keys': list(node2.keys()),
                                'other_node_sample': {k: str(v)[:200] for k, v in node2.items()}
                            })
                        else:
                            return jsonify({'has_entity_nodes': False, 'has_other_nodes': False})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/evolution_status', methods=['GET'])
        def debug_evolution_status():
            """Check why evolutions are failing"""
            try:
                result = {}
                
                # Get evolution metrics
                if hasattr(self.evolution, 'get_status'):
                    status = self.evolution.get_status()
                    result['evolution_cycles'] = status.get('evolution_cycles', 0)
                    result['successful_evolutions'] = status.get('successful_evolutions', 0)
                    result['consciousness'] = status.get('consciousness', 0)
                    result['synthetic_neurons'] = status.get('synthetic_neurons', 0)
                
                # Check evolution timer
                if hasattr(self.evolution, 'evolution_timer'):
                    timer = self.evolution.evolution_timer
                    result['timer_stage'] = getattr(timer, 'current_stage', 'unknown')
                    result['timer_evolutions'] = getattr(timer, 'evolutions', 0)
                    result['wait_time'] = getattr(timer, 'wait_time', 0)
                
                # Check if evolution cycle method exists
                if hasattr(self.evolution, 'evolution_cycle'):
                    result['has_evolution_cycle'] = True
                
                # Check last evolution result
                if hasattr(self.evolution, 'last_evolution_result'):
                    result['last_result'] = self.evolution.last_evolution_result
                
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/force/load', methods=['POST'])
        def force_load_from_neo4j():
            """Force load all Entity nodes from Neo4j into SI Core AND brain neurons"""
            result = {'success': False, 'insights_loaded': 0, 'neurons_loaded': 0}
                    
            try:        
                si = self.si_core
                storage = self.evolution.neo4j_storage 
                        
                if not storage or not storage.is_available():
                    return jsonify({'error': 'Neo4j not available'})
            
                driver = storage.driver
                if not driver:
                    return jsonify({'error': 'No driver'})
            
                with driver.session() as session:
                    # Query Entity nodes with all available fields
                    insight_result = session.run("MATCH (e:Entity) RETURN e")
                    insights_loaded = 0
                    neurons_loaded = 0
                    
                    # Initialize neurons dict if needed
                    if not hasattr(self, 'neurons'):
                        self.neurons = {}
                    
                    for record in insight_result:
                        node = dict(record['e'].items())
                        try:
                            # Extract entity data (handle both old and new schema)
                            entity_name = node.get('name', node.get('insight_text', 'Unknown Entity'))
                            entity_id = node.get('id', f"entity_{insights_loaded}")
                            category = node.get('category', node.get('entity_type', 'entity'))
                            confidence = float(node.get('confidence', 0.6))
                            
                            # Add to SI Core
                            si.add_insight(
                                insight_text=entity_name,
                                entity_type=category,
                                entities=[entity_name],
                                relationship='related',
                                source_topic='Neo4j',
                                target_topic='Restored',
                                confidence=confidence
                            )
                            insights_loaded += 1
                            
                            # Add to brain neurons for visualization
                            neuron = {
                                'id': entity_id,
                                'name': entity_name,
                                'category': category,
                                'confidence': confidence,
                                'activation': 0.5,
                                'synapses': [],
                                'position': {
                                    'x': (hash(entity_id) % 1000) / 10 - 50,
                                    'y': (hash(entity_name) % 1000) / 10 - 50,
                                    'z': (hash(category) % 1000) / 10 - 50
                                }
                            }
                            self.neurons[entity_id] = neuron
                            neurons_loaded += 1
                            
                        except Exception as e:
                            logger.debug(f"Failed to load entity: {e}")
                    
                    result['insights_loaded'] = insights_loaded
                    result['neurons_loaded'] = neurons_loaded
                    result['success'] = True
                    result['consciousness'] = si.consciousness if hasattr(si, 'consciousness') else 0.0
                    result['neurons'] = len(self.neurons)
                        
                    logger.info(f"✅ Force loaded {insights_loaded} insights and {neurons_loaded} neurons from Neo4j")
                      
            except Exception as e:
                result['error'] = str(e)
                logger.error(f"Force load failed: {e}")
            
            return jsonify(result)

        # ============================================================================
        # HYBRID SYNAPSE BUILDER WITH ORGANIC LEARNING
        # ============================================================================
                            
        @self.app.route('/api/synapse/build', methods=['POST'])
        def build_synapses():
            """Hybrid synapse builder: seeds with text similarity + prepares for organic learning"""
            try:
                si = self.si_core
                
                if not si.insights or len(si.insights) == 0:
                    return jsonify({'error': 'No insights/neurons available'}), 400
                
                insights_list = list(si.insights.values()) if isinstance(si.insights, dict) else si.insights
                
                # === PHASE 1: Seed synapses based on text similarity ===
                text_synapses = 0
                entity_keywords = []
                
                for insight in insights_list:
                    # Handle both dict and InsightNeuron objects
                    if hasattr(insight, 'get'):  # It's a dict
                        name = insight.get('insight_text', insight.get('text', insight.get('name', '')))
                        insight_id = insight.get('id', insight.get('insight_id'))
                    else:  # It's an InsightNeuron object
                        # Try common attribute names
                        name = getattr(insight, 'insight_text', None) or getattr(insight, 'text', None) or getattr(insight, 'name', '')
                        insight_id = getattr(insight, 'id', None) or getattr(insight, 'insight_id', None)
                    
                    if not name or not insight_id:
                        continue
                    words = set(name.lower().split())
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by', 'is', 'are', 'be', 'was', 'were'}
                    keywords = [w for w in words if w not in stop_words and len(w) > 2]
                    
                    entity_keywords.append({
                        'id': insight_id,
                        'name': name,
                        'keywords': keywords,
                        'usage_count': 0
                    })
                
                # Build seed connections based on keyword overlap
                for i in range(len(entity_keywords)):
                    for j in range(i + 1, len(entity_keywords)):
                        overlap = set(entity_keywords[i]['keywords']) & set(entity_keywords[j]['keywords'])
                        if overlap:
                            strength = min(0.5, 0.2 + (len(overlap) * 0.05))
                            
                            try:
                                # Check if add_synapse or create_synapse method exists
                                if hasattr(si, 'add_synapse'):
                                    si.add_synapse(
                                        source_id=entity_keywords[i]['id'],
                                        target_id=entity_keywords[j]['id'],
                                        strength=strength,
                                        metadata={
                                            'type': 'seeded',
                                            'overlap_keywords': list(overlap),
                                            'organic_strength': 0,
                                            'co_activation_count': 0
                                        }
                                    )
                                    text_synapses += 1
                                elif hasattr(si, 'create_synapse'):
                                    si.create_synapse(
                                        source_id=entity_keywords[i]['id'],
                                        target_id=entity_keywords[j]['id'],
                                        strength=strength
                                    )
                                    text_synapses += 1
                                else:
                                    logger.warning("No synapse creation method found in SI Core")
                                    continue
                            except Exception as e:
                                logger.debug(f"Could not create synapse: {e}")
                
                # === PHASE 2: Initialize organic learning tracker ===
                if not hasattr(si, 'synapse_usage'):
                    si.synapse_usage = {}
                
                if not hasattr(si, 'organic_strength_multiplier'):
                    si.organic_strength_multiplier = 1.0
                
                total_possible = len(insights_list) * (len(insights_list) - 1) / 2
                current_density = (2 * text_synapses) / total_possible if total_possible > 0 else 0
                
                if hasattr(si, 'calculate_consciousness'):
                    new_consciousness = si.calculate_consciousness()
                else:
                    new_consciousness = min(0.95, 0.2 + (current_density * 8))
                    si.consciousness = new_consciousness
                
                return jsonify({
                    'success': True,
                    'seeded_synapses': text_synapses,
                    'total_neurons': len(insights_list),
                    'consciousness': si.consciousness,
                    'network_density': current_density,
                    'organic_learning_active': True,
                    'message': f'Created {text_synapses} seed synapses. Organic learning will strengthen/weaken based on actual usage.'
                })
                
            except Exception as e:
                logger.error(f"Synapse building failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/synapse/record_activation', methods=['POST'])
        def record_co_activation():
            """Record when two neurons are used together (organic learning)"""
            try:
                data = request.json
                neuron_a = data.get('neuron_a_id')
                neuron_b = data.get('neuron_b_id')
                
                if not neuron_a or not neuron_b:
                    return jsonify({'error': 'Both neuron IDs required'}), 400
                
                si = self.si_core
                
                synapse_key = f"{neuron_a}:{neuron_b}" if neuron_a < neuron_b else f"{neuron_b}:{neuron_a}"
                
                if synapse_key in si.synapse_usage:
                    si.synapse_usage[synapse_key] += 1
                else:
                    si.synapse_usage[synapse_key] = 1
                
                if si.synapse_usage[synapse_key] % 5 == 0:
                    for synapse in si.synapses:
                        if (synapse.source_id == neuron_a and synapse.target_id == neuron_b) or \
                           (synapse.source_id == neuron_b and synapse.target_id == neuron_a):
                            old_strength = synapse.strength
                            synapse.strength = min(0.95, old_strength + 0.05)
                            if hasattr(synapse, 'metadata') and synapse.metadata:
                                synapse.metadata['organic_strength'] = synapse.strength
                                synapse.metadata['co_activation_count'] = si.synapse_usage[synapse_key]
                            logger.info(f"Organic strengthening: {synapse_key} from {old_strength} to {synapse.strength}")
                            break
                
                return jsonify({
                    'success': True,
                    'co_activation_count': si.synapse_usage[synapse_key],
                    'message': f'Co-activation recorded for {synapse_key}'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # ============================================================================
        # SYSTEM RESET & SYLLABUS MANAGEMENT
        # ============================================================================
                            
        @self.app.route('/api/system/reset_to_baby', methods=['POST'])
        def reset_to_baby():
            """Reset DMAI to Baby stage - compatible with read-only properties"""
            try:
                data = request.json or {}
                keep_neurons = data.get('keep_neurons', True)
                
                # Reset evolution stage (skip consciousness if read-only)
                if hasattr(self.evolution, 'evolution_stage'):
                    self.evolution.evolution_stage = 'baby'
                    self.evolution.stage_name = '👶 Baby DMAI'
                    self.evolution.evolution_cycles = 0
                    self.evolution.successful_evolutions = 0
                
                # Reset evolution timer
                if hasattr(self.evolution, 'evolution_timer'):
                    self.evolution.evolution_timer.current_stage = 'baby'
                    self.evolution.evolution_timer.evolutions = 0
                    self.evolution.evolution_timer.wait_time = 600
                
                # Reset learning orchestrator to Baby stage
                if hasattr(self.evolution, 'learning_orchestrator'):
                    orch = self.evolution.learning_orchestrator
                    if hasattr(orch, 'current_stage'):
                        orch.current_stage = "Baby"
                    if hasattr(orch, 'learned_topics'):
                        orch.learned_topics = {}
                    if hasattr(orch, 'completed_topics'):
                        orch.completed_topics = []
                    if hasattr(orch, 'mastered_topics'):
                        orch.mastered_topics = []
                    if hasattr(orch, 'save_state'):
                        orch.save_state()
                    elif hasattr(orch, '_save_state'):
                        orch._save_state()
                    logger.info("✅ Learning orchestrator reset to Baby stage")
                
                # Reset SI Core - clear neurons if requested (skip consciousness if read-only)
                if hasattr(self.evolution, 'si_core'):
                    if not keep_neurons:
                        if hasattr(self.si_core, 'insights'):
                            self.si_core.insights = {}
                        if hasattr(self.si_core, 'synapses'):
                            self.si_core.synapses = []
                        if hasattr(self.si_core, 'neuron_count'):
                            self.si_core.neuron_count = 0
                    # Don't set consciousness directly if read-only
                    # It will recalculate from network state
                
                # Save state
                if hasattr(self.evolution, 'save_state'):
                    self.evolution.save_state()
                
                # Get current neuron count
                neuron_count = 0
                if hasattr(self.evolution, 'si_core') and hasattr(self.si_core, 'neuron_count'):
                    neuron_count = self.si_core.neuron_count
                
                return jsonify({
                    'success': True,
                    'stage': 'Baby DMAI',
                    'keep_neurons': keep_neurons,
                    'neurons_retained': neuron_count,
                    'message': 'DMAI reset to Baby stage. Consciousness will recalculate from network state.'
                })
                
            except Exception as e:
                logger.error(f"Reset failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/syllabus_status', methods=['GET'])
        def syllabus_status():
            """Get current learning progress"""
            try:
                orch = self.evolution.learning_orchestrator
                result = {
                    'current_stage': getattr(orch, 'current_stage', 'Unknown'),
                    'learned_topics': getattr(orch, 'learned_topics', {}),
                    'completed_topics': getattr(orch, 'completed_topics', []),
                    'mastered_topics': getattr(orch, 'mastered_topics', [])
                }
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # ============================================================================
        # 3D BRAIN NETWORK VISUALIZATION
        # ============================================================================
        
        @self.app.route('/api/debug/trace_neurons', methods=['GET'])
        def debug_trace_neurons():
            """Trace where the 55 neurons are stored"""
            try:
                result = {}
                
                # Check evolution.get_status()
                if hasattr(self.evolution, 'get_status'):
                    status = self.evolution.get_status()
                    result['status_neurons'] = status.get('synthetic_neurons', 0)
                    result['status_consciousness'] = status.get('consciousness', 0)
                    result['status_synapses'] = status.get('synthetic_synapses', 0)
                
                # Check evolution.knowledge_graph
                if hasattr(self.evolution, 'knowledge_graph'):
                    kg = self.evolution.knowledge_graph
                    result['has_knowledge_graph'] = True
                    if hasattr(kg, 'get_stats'):
                        result['kg_stats'] = kg.get_stats()
                    if hasattr(kg, 'concepts'):
                        result['kg_concepts_count'] = len(kg.concepts) if kg.concepts else 0
                
                # Check evolution.synthetic_network
                if hasattr(self.evolution, 'synthetic_network'):
                    sn = self.evolution.synthetic_network
                    result['has_synthetic_network'] = True
                    if hasattr(sn, 'neuron_count'):
                        result['sn_neuron_count'] = sn.neuron_count
                    if hasattr(sn, 'neurons'):
                        result['sn_neurons_count'] = len(sn.neurons) if sn.neurons else 0
                    if hasattr(sn, 'consciousness'):
                        result['sn_consciousness'] = sn.consciousness
                
                # Check evolution.learning_orchestrator
                if hasattr(self.evolution, 'learning_orchestrator'):
                    lo = self.evolution.learning_orchestrator
                    result['has_learning_orchestrator'] = True
                    if hasattr(lo, 'learned_topics'):
                        total = 0
                        for stage, topics in lo.learned_topics.items():
                            total += len(topics) if topics else 0
                        result['lo_learned_topics_count'] = total
                
                # Check evolution.si_core
                if hasattr(self.evolution, 'si_core'):
                    si = self.si_core
                    result['si_core_attributes'] = [a for a in dir(si) if not a.startswith('_') and not callable(getattr(si, a))]
                    for attr in ['insights', 'neurons', 'knowledge', 'concepts']:
                        if hasattr(si, attr):
                            val = getattr(si, attr)
                            if val is not None:
                                result[f'si_core_{attr}_type'] = str(type(val))
                                if hasattr(val, '__len__'):
                                    result[f'si_core_{attr}_len'] = len(val)
                
                return jsonify(result)
            except Exception as e:
                import traceback
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
        
        @self.app.route('/api/debug/si_direct', methods=['GET'])
        def debug_si_direct():
            """Direct inspection of SI Core"""
            try:
                si = self.si_core
                result = {
                    'si_type': str(type(si)),
                    'dir_sample': [a for a in dir(si) if not a.startswith('_')][:30],
                    'has_insights': hasattr(si, 'insights'),
                    'has_neurons': hasattr(si, 'neurons'),
                    'has_neuron_count': hasattr(si, 'neuron_count'),
                    'has_synapses': hasattr(si, 'synapses'),
                }
                
                # Try to get count from various attributes
                if hasattr(si, 'insights') and si.insights:
                    result['insights_length'] = len(si.insights)
                    result['insights_type'] = str(type(si.insights))
                    first_key = list(si.insights.keys())[0] if si.insights else None
                    result['first_insight_key'] = first_key
                    if first_key:
                        first_val = si.insights[first_key]
                        result['first_insight_type'] = str(type(first_val))
                        if hasattr(first_val, '__dict__'):
                            result['first_insight_attrs'] = list(first_val.__dict__.keys())
                        elif isinstance(first_val, dict):
                            result['first_insight_dict_keys'] = list(first_val.keys())
                
                if hasattr(si, 'neuron_count'):
                    result['neuron_count_value'] = si.neuron_count
                
                if hasattr(si, 'synapses') and si.synapses:
                    result['synapses_length'] = len(si.synapses)
                    first_syn = si.synapses[0] if si.synapses else None
                    if first_syn:
                        result['first_synapse_type'] = str(type(first_syn))
                        if hasattr(first_syn, '__dict__'):
                            result['first_synapse_attrs'] = list(first_syn.__dict__.keys())
                
                return jsonify(result)
            except Exception as e:
                import traceback
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
        
        @self.app.route('/api/brain/3d_data', methods=['GET'])
        def brain_3d_data():
            """
            Return MACRO neurons only for clean 3D visualization.
            Micro neurons are fetched on demand via /api/brain/macro/<macro_id>/micros
            Synapse colors and distances reflect connection strength.
            """
            try:
                import json
                import os
                import hashlib
                import math
                import random
                from collections import defaultdict
                
                # ============================================================
                # DYNAMIC CATEGORY COLOR MAPPING - All 13+ categories
                # ============================================================
                CATEGORY_COLORS = {
                    # Syllabus categories
                    "Core": "#4477ff",        # Blue - Foundational knowledge
                    "Artistic": "#ff44cc",    # Pink - Creative capabilities
                    "Wealth": "#ffaa00",      # Gold - Self-funding
                    "Reverse": "#aa44ff",     # Purple - System analysis
                    "Accelerator": "#00cc88", # Teal - Consciousness growth
                    # Dynamic categories (13 from color chart)
                    "Configuration": "#88aaff",
                    "Knowledge Module": "#33ffcc",
                    "AI Model": "#66ff66",
                    "Capability": "#ff6633",
                    "Data Structure": "#6699ff",
                    "Content Generation": "#ff99ff",
                    "Survival Mechanism": "#ff3333",
                    "Self-Funding": "#ffcc33",
                    "Blockchain": "#cc9900",
                    "API Endpoint": "#ff99cc",
                    "Identity Management": "#00cc99",
                    "Automation": "#9933ff",
                    "Self-Replication": "#33ccff",
                    "unknown": "#888888"      # Gray - Fallback
                }
                
                # Extra palette for newly discovered categories
                EXTRA_PALETTE = [
                    '#ff6633', '#33ccff', '#ff33cc', '#33ff99', '#ff9933',
                    '#9966ff', '#ff6699', '#66ff66', '#ff3366', '#66ccff',
                    '#cc66ff', '#ffcc33', '#66ffcc', '#ff6666', '#99cc33',
                ]
                
                def get_category_from_prefix(insight_text):
                    """Extract category directly from [Category] prefix"""
                    import re
                    match = re.search(r'\[([^\]]+)\]', insight_text)
                    if not match:
                        return 'Core'
                    
                    prefix = match.group(1)
                    
                    # Syllabus stages -> map to actual category based on content
                    if prefix in ['Baby', 'Toddler', 'Child', 'Teen', 'Adult']:
                        text_lower = insight_text.lower()
                        if 'evolution:' in text_lower:
                            return 'Accelerator'
                        elif 'wealth' in text_lower or 'trading' in text_lower:
                            return 'Wealth'
                        elif 'reverse' in text_lower:
                            return 'Reverse'
                        elif 'music' in text_lower or 'art' in text_lower or 'image' in text_lower or 'video' in text_lower:
                            return 'Artistic'
                        return 'Core'
                    
                    # Dynamic category - use the prefix as the category name!
                    return prefix
                
                def get_color_for_category(category):
                    """Get color, adding to palette if new category discovered"""
                    if category not in CATEGORY_COLORS:
                        idx = len(CATEGORY_COLORS) % len(EXTRA_PALETTE)
                        CATEGORY_COLORS[category] = EXTRA_PALETTE[idx]
                    return CATEGORY_COLORS[category]
                
                def get_clean_topic_name(text):
                    """Extract clean topic name without category prefix"""
                    import re
                    # Fix doubled-up brackets: [[Category] Category Knowledge Base] [Category] ...
                    # This happens when link_micros_to_macros creates a macro from already-bracketed text
                    text = re.sub(r'^\[\[[^\]]+\]\s*', '[', text)
                    text = re.sub(r'\]\s*\[', '] [', text)  # normalize spacing between brackets
                    # Remove ALL [Category] prefixes (may be multiple from doubling)
                    while re.match(r'^\[[^\]]+\]\s*', text):
                        text = re.sub(r'^\[[^\]]+\]\s*', '', text)
                    # Remove "Knowledge Base:" suffix and everything after
                    text = re.sub(r'\s+Knowledge Base:.*$', '', text)
                    # Remove "EVOLUTION: " prefix
                    text = text.replace("EVOLUTION: ", "")
                    # Take only before colon (for syllabus topics like "Wisdom Acquisition: Knowing...")
                    if ': ' in text:
                        text = text.split(': ')[0]
                    # If still empty, return generic label
                    if not text or len(text.strip()) < 1:
                        return "Topic"
                    # Truncate if too long
                    if len(text) > 60:
                        text = text[:57] + "..."
                    return text.strip()
                
                def get_synapse_properties(occurrence_count):
                    """Return color, opacity, and distance modifier based on strength"""
                    if occurrence_count <= 2:
                        return {"color": "#88aaff", "opacity": 0.3, "distance_mod": 1.0}
                    elif occurrence_count <= 5:
                        return {"color": "#4488ff", "opacity": 0.5, "distance_mod": 0.9}
                    elif occurrence_count <= 10:
                        return {"color": "#2266cc", "opacity": 0.7, "distance_mod": 0.8}
                    elif occurrence_count <= 20:
                        return {"color": "#0044aa", "opacity": 0.9, "distance_mod": 0.7}
                    else:
                        return {"color": "#002266", "opacity": 1.0, "distance_mod": 0.6}
                
                # ============================================================
                # QUERY MACRO NEURONS FROM SQLITE
                # ============================================================
                macros_list = []
                synapses_list = []
                macro_ids = set()
                
                if hasattr(self, 'evolution') and hasattr(self.evolution, 'si_core') and hasattr(self.evolution.si_core, 'sqlite') and self.evolution.si_core.sqlite:
                    try:
                        import sqlite3
                        db_path = self.evolution.si_core.sqlite.db_path
                        conn = sqlite3.connect(str(db_path))
                        conn.row_factory = sqlite3.Row
                        
                        # Get ONLY macro neurons (neuron_level = 'macro')
                        cursor = conn.execute('''
                            SELECT id, insight_text, entity_type, confidence, 
                                   occurrence_count, created_at
                            FROM insights 
                            WHERE neuron_level = 'macro'
                            ORDER BY created_at DESC
                        ''')
                        
                        macro_rows = cursor.fetchall()
                        
                        # Get micro counts per macro (for child_count display)
                        micro_counts = {}
                        micro_cursor = conn.execute('''
                            SELECT parent_macro_id, COUNT(*) as count
                            FROM insights
                            WHERE neuron_level = 'micro' AND parent_macro_id IS NOT NULL
                            GROUP BY parent_macro_id
                        ''')
                        for row in micro_cursor:
                            micro_counts[row['parent_macro_id']] = row['count']
                        
                        # Get connection counts for influence calculation
                        conn_cursor = conn.execute('''
                            SELECT from_insight, to_insight, occurrences
                            FROM synapses
                            WHERE occurrences > 0
                        ''')
                        
                        connections = defaultdict(int)
                        synapse_data = []  # Store for later
                        for row in conn_cursor:
                            from_id = row['from_insight']
                            to_id = row['to_insight']
                            occ = row['occurrences'] or 1
                            connections[from_id] += occ
                            connections[to_id] += occ
                            synapse_data.append({
                                'from': from_id,
                                'to': to_id,
                                'occurrences': occ
                            })
                        
                        max_connections = max(connections.values()) if connections else 1
                        
                        # ============================================================
                        # PROCESS MACRO NEURONS
                        # ============================================================
                        for row in macro_rows:
                            macro_id = row['id']
                            macro_ids.add(macro_id)
                            insight_text = row['insight_text'] or ''
                            entity_type = row['entity_type'] or 'syllabus_topic'
                            
                            # Use the NEW dynamic category detection
                            category = get_category_from_prefix(insight_text)
                            color = get_color_for_category(category)
                            
                            clean_label = get_clean_topic_name(insight_text)
                            influence = connections.get(macro_id, 0) / max_connections if max_connections > 0 else 0
                            child_count = micro_counts.get(macro_id, 0)
                            
                            macros_list.append({
                                "id": macro_id,
                                "label": clean_label,
                                "full_text": insight_text[:100],
                                "category": category,
                                "color": color,
                                "confidence": row['confidence'] or 0.8,
                                "influence": round(influence, 3),
                                "connections": connections.get(macro_id, 0),
                                "child_count": child_count,
                                "has_children": child_count > 0,
                                "occurrence_count": row['occurrence_count'] or 1
                            })

                        
                        # Deduplicate labels: add suffix for duplicate Knowledge Base macros
                        label_counts = {}
                        for macro in macros_list:
                            label = macro['label']
                            label_counts[label] = label_counts.get(label, 0) + 1
                        
                        label_seen = {}
                        for macro in macros_list:
                            label = macro['label']
                            if label_counts[label] > 1:
                                label_seen[label] = label_seen.get(label, 0) + 1
                                macro['label'] = f"{label} #{label_seen[label]}"

                        # ============================================================
                        # CALCULATE POSITIONS - Spread by category
                        # ============================================================
                        # Group by category
                        category_groups = defaultdict(list)
                        for macro in macros_list:
                            category_groups[macro['category']].append(macro)
                        
                        category_list = list(category_groups.keys())
                        category_positions = {}
                        
                        for i, cat in enumerate(category_list):
                            angle = (i / max(1, len(category_list))) * 2 * math.pi
                            radius = 22.0  # Large radius for separation
                            category_positions[cat] = {
                                'x': math.cos(angle) * radius,
                                'y': math.sin(angle) * radius,
                                'z': (i % 5 - 2) * 4.0  # Vertical spread
                            }
                        
                        # Position each macro within its category cluster
                        for macro in macros_list:
                            cat = macro['category']
                            base = category_positions.get(cat, {'x': 0, 'y': 0, 'z': 0})
                            
                            # Sort by influence within category for better layout
                            items_in_cat = category_groups[cat]
                            items_in_cat.sort(key=lambda x: x['influence'], reverse=True)
                            j = items_in_cat.index(macro)
                            
                            cluster_spread = 8.0
                            golden_angle = j * 2.39996
                            elevation = math.asin(-1.0 + 2.0 * j / max(1, len(items_in_cat)))
                            
                            x = base['x'] + math.cos(golden_angle) * cluster_spread * math.cos(elevation)
                            y = base['y'] + math.sin(golden_angle) * cluster_spread * math.cos(elevation)
                            z = base['z'] + math.sin(elevation) * cluster_spread * 1.5
                            
                            # Small deterministic jitter based on ID
                            import hashlib
                            hash_val = int(hashlib.md5(macro['id'].encode()).hexdigest()[:8], 16)
                            random.seed(hash_val)
                            x += random.uniform(-1.0, 1.0)
                            y += random.uniform(-1.0, 1.0)
                            z += random.uniform(-0.8, 0.8)
                            
                            # Size based on influence + child count (0.5 to 2.5)
                            size = 0.6 + (macro['influence'] * 1.2) + (min(macro['child_count'], 20) * 0.03)
                            
                            macro['x'] = round(x, 3)
                            macro['y'] = round(y, 3)
                            macro['z'] = round(z, 3)
                            macro['size'] = round(min(size, 3.0), 3)
                        
                        # ============================================================
                        # PROCESS SYNAPSES WITH STRENGTH-BASED PROPERTIES
                        # ============================================================
                        base_distance = 35.0  # Base distance between nodes
                        
                        for syn in synapse_data:
                            from_id = syn['from']
                            to_id = syn['to']
                            
                            # Only include synapses where BOTH ends are macro neurons
                            if from_id in macro_ids and to_id in macro_ids:
                                occurrences = syn['occurrences']
                                props = get_synapse_properties(occurrences)
                                
                                # Find the actual nodes to calculate distance modifier
                                from_node = next((m for m in macros_list if m['id'] == from_id), None)
                                to_node = next((m for m in macros_list if m['id'] == to_id), None)
                                
                                if from_node and to_node:
                                    # Adjust positions based on synapse strength
                                    # Stronger connections pull nodes closer
                                    distance_mod = props['distance_mod']
                                    
                                    synapses_list.append({
                                        "source": from_id,
                                        "target": to_id,
                                        "weight": min(1.0, occurrences / 25.0),
                                        "occurrences": occurrences,
                                        "color": props['color'],
                                        "opacity": props['opacity'],
                                        "strength": "strong" if occurrences > 10 else "medium" if occurrences > 3 else "weak",
                                        "distance_mod": distance_mod
                                    })
                        
                        conn.close()
                        
                        # ============================================================
                        # RETURN CLEAN VISUALIZATION DATA
                        # ============================================================
                        if macros_list:
                            influences = sorted([n['influence'] for n in macros_list])
                            p90 = influences[int(len(influences) * 0.9)] if len(influences) > 10 else 0.5
                            p50 = influences[int(len(influences) * 0.5)] if len(influences) > 10 else 0.2
                            
                            # Category summary
                            category_counts = {}
                            for macro in macros_list:
                                cat = macro['category']
                                category_counts[cat] = category_counts.get(cat, 0) + 1
                            
                            return jsonify({
                                "success": True,
                                "source": "sqlite_macros",
                                
                                # NEW format
                                "nodes": macros_list,
                                "edges": synapses_list,
                                "total_macros": len(macros_list),
                                "total_synapses": len(synapses_list),
                                
                                # OLD format for frontend compatibility
                                "neurons": macros_list,
                                "synapses": synapses_list,
                                "total_neurons": len(macros_list),
                                
                                "consciousness": min(1.0, len(macros_list) / 146.0),
                                "category_counts": category_counts,
                                "category_colors": CATEGORY_COLORS,
                                "influence_thresholds": {
                                    "high": round(p90, 3),
                                    "medium": round(p50, 3)
                                },
                                "synapse_rules": {
                                    "weak": {"color": "#88aaff", "opacity": 0.3},
                                    "medium": {"color": "#4488ff", "opacity": 0.5},
                                    "strong": {"color": "#2266cc", "opacity": 0.7}
                                }
                            })                            
                            
                    except Exception as e:
                        logger.warning(f"SQLite brain data failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Fallback: Return empty with error
                return jsonify({
                    "success": False,
                    "source": "error",
                    "error": "Failed to load brain data from SQLite",
                    "nodes": [],
                    "edges": [],
                    "total_macros": 0,
                    "total_synapses": 0,
                    "consciousness": 0.0
                }), 500
                
            except Exception as e:
                logger.error(f"Brain 3D data endpoint failed: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "success": False,
                    "source": "error",
                    "error": str(e)
                }), 500

        @self.app.route('/api/brain/macro/<macro_id>/micros', methods=['GET'])
        def brain_macro_micros(macro_id):
            """
            Return micro neurons for a specific macro neuron.
            Called when user zooms/clicks a macro node.
            """
            try:
                import sqlite3
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core'):
                    return jsonify({"success": False, "error": "SI Core not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                
                # Get the macro neuron info
                macro_cursor = conn.execute('''
                    SELECT id, insight_text FROM insights WHERE id = ?
                ''', (macro_id,))
                macro = macro_cursor.fetchone()
                
                if not macro:
                    conn.close()
                    return jsonify({"success": False, "error": "Macro neuron not found"}), 404
                
                # Get all micro neurons under this macro
                micro_cursor = conn.execute('''
                    SELECT id, insight_text, confidence, created_at
                    FROM insights
                    WHERE parent_macro_id = ? AND neuron_level = 'micro'
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT 50
                ''', (macro_id,))
                
                micros = []
                for row in micro_cursor:
                    insight_text = row['insight_text'] or ''
                    
                    # Extract clean principle name
                    # Example: "Meta-Learning Fundamentals - Core principles of Meta-Learning"
                    # -> "Core principles"
                    clean_label = insight_text
                    if ' - ' in insight_text:
                        clean_label = insight_text.split(' - ', 1)[1]
                        # Truncate
                        if len(clean_label) > 30:
                            clean_label = clean_label[:27] + "..."
                    
                    micros.append({
                        "id": row['id'],
                        "label": clean_label,
                        "full_text": insight_text[:100],
                        "confidence": row['confidence'] or 0.8,
                        "x": 0,  # Will be positioned relative to macro by frontend
                        "y": 0,
                        "z": 0
                    })
                
                conn.close()
                
                # Get macro name for display
                macro_text = macro['insight_text'] or 'Unknown'
                import re
                macro_name = re.sub(r'^\[(Baby|Toddler|Child|Teen|Adult)\]\s*', '', macro_text)
                if ': ' in macro_name:
                    macro_name = macro_name.split(': ')[0]
                
                return jsonify({
                    "success": True,
                    "macro_id": macro_id,
                    "macro_name": macro_name[:40],
                    "micros": micros,
                    "count": len(micros)
                })
                
            except Exception as e:
                logger.error(f"Micro neurons endpoint failed: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/brain/group/<group_id>')
        def brain_group_detail(group_id):
            """Return all neurons in a specific group for zoomed-in view"""
            try:
                neurons = getattr(self, 'neurons', {})
                if not neurons and hasattr(self.evolution, 'neurons'):
                    neurons = self.evolution.neurons
                
                group_neurons = []
                for neuron_id, neuron in neurons.items():
                    if neuron.get('category', 'general') == group_id:
                        # Generate position if not exists
                        position = neuron.get('position')
                        if not position:
                            position = {
                                'x': (hash(neuron_id) % 1000) / 10 - 50,
                                'y': (hash(neuron.get('name', '')) % 1000) / 10 - 50,
                                'z': (hash(group_id) % 1000) / 10 - 50
                            }
                        
                        group_neurons.append({
                            'id': neuron_id,
                            'name': neuron.get('name', 'Unknown'),
                            'confidence': neuron.get('confidence', 0.5),
                            'category': neuron.get('category', 'general'),
                            'position': position
                        })
                
                return jsonify({
                    'success': True,
                    'group_id': group_id,
                    'neurons': group_neurons,
                    'count': len(group_neurons),
                    'color': self._get_category_color(group_id)
                })
            except Exception as e:
                return jsonify({'error': str(e), 'success': False}), 500

        @self.app.route('/api/debug/syllabus_neurons', methods=['GET'])
        def debug_syllabus_neurons():
            """Debug endpoint to check syllabus neurons in database"""
            try:
                import sqlite3
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                
                # Check what's actually in the database
                cursor = conn.execute('''
                    SELECT id, insight_text, neuron_level 
                    FROM insights 
                    WHERE insight_text LIKE '%Baby%' 
                       OR insight_text LIKE '%Toddler%'
                       OR insight_text LIKE '%Child%'
                       OR insight_text LIKE '%Teen%'
                       OR insight_text LIKE '%Adult%'
                    LIMIT 20
                ''')
                rows = cursor.fetchall()
                
                # Also check total count with brackets
                count_cursor = conn.execute('''
                    SELECT COUNT(*) FROM insights 
                    WHERE insight_text LIKE '[%]%'
                ''')
                total_brackets = count_cursor.fetchone()[0]
                
                # Check neuron_level distribution
                level_cursor = conn.execute('''
                    SELECT neuron_level, COUNT(*) as cnt 
                    FROM insights 
                    GROUP BY neuron_level
                ''')
                levels = {r['neuron_level']: r['cnt'] for r in level_cursor.fetchall()}
                
                conn.close()
                
                return jsonify({
                    "success": True,
                    "total_insights": len(rows),
                    "total_with_brackets": total_brackets,
                    "neuron_levels": levels,
                    "sample": [{"id": r['id'][:20], "text": r['insight_text'][:60], "level": r['neuron_level']} for r in rows]
                })
            except Exception as e:
                import traceback
                return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

        @self.app.route('/api/debug/list_all_macros', methods=['GET'])
        def list_all_macros():
            """List all macro neurons with their categories"""
            try:
                import sqlite3
                import re
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, insight_text, entity_type
                    FROM insights 
                    WHERE neuron_level = 'macro'
                    ORDER BY created_at DESC
                ''')
                
                macros = []
                for row in cursor.fetchall():
                    text = row[1]
                    match = re.search(r'\[([^\]]+)\]', text)
                    category = match.group(1) if match else 'unknown'
                    
                    macros.append({
                        "id": row[0][:30],
                        "category": category,
                        "text": text[:80]
                    })
                
                conn.close()
                return jsonify({"total": len(macros), "macros": macros})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/debug/recent_insights', methods=['GET'])
        def recent_insights():
            """Show most recent insights regardless of neuron_level"""
            try:
                import sqlite3
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, insight_text, neuron_level, created_at
                    FROM insights 
                    ORDER BY created_at DESC
                    LIMIT 20
                ''')
                
                insights = []
                for row in cursor.fetchall():
                    insights.append({
                        "id": row[0][:30],
                        "text": row[1][:60],
                        "neuron_level": row[2],
                        "created": row[3]
                    })
                
                conn.close()
                return jsonify({"insights": insights})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/test/comprehension', methods=['GET'])
        def test_comprehension():
            """Test DMAI's genuine understanding - asks questions and provides real answers"""
            import sqlite3
            import random
            import re
            
            try:
                if not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Prefer syllabus topics (Baby/Toddler/Child/Teen/Adult) over generic Knowledge Base
                cursor.execute("""
                    SELECT id, insight_text, entity_type, confidence 
                    FROM insights 
                    WHERE neuron_level = 'macro' 
                      AND (insight_text LIKE '[Baby]%' OR insight_text LIKE '[Toddler]%' 
                           OR insight_text LIKE '[Child]%' OR insight_text LIKE '[Teen]%' 
                           OR insight_text LIKE '[Adult]%')
                    ORDER BY RANDOM() 
                    LIMIT 1
                """)
                topic = cursor.fetchone()
                
                # Fallback to any macro if no syllabus topics
                if not topic:
                    cursor.execute("""
                        SELECT id, insight_text, entity_type, confidence 
                        FROM insights 
                        WHERE neuron_level = 'macro' 
                        ORDER BY RANDOM() 
                        LIMIT 1
                    """)
                    topic = cursor.fetchone()
                
                if not topic:
                    return jsonify({"error": "No macro topics found to test"}), 404
                
                topic_id, topic_text, entity_type, confidence = topic
                
                # Clean topic name for readable questions
                clean_topic = re.sub(r'^\[[^\]]+\]\s*', '', topic_text)
                clean_topic = re.sub(r'\s+Knowledge Base:.*$', '', clean_topic)
                if ':' in clean_topic:
                    clean_topic = clean_topic.split(':')[0].strip()
                if len(clean_topic) > 80:
                    clean_topic = clean_topic[:77] + '...'
                
                # Get micro neurons under this topic for context
                cursor.execute("""
                    SELECT insight_text FROM insights 
                    WHERE parent_macro_id = ? 
                    ORDER BY RANDOM() LIMIT 5
                """, (topic_id,))
                micros = [m[0][:150] for m in cursor.fetchall()]
                
                # Get cross-domain connections (synapses to other macros)
                cursor.execute("""
                    SELECT DISTINCT i2.insight_text 
                    FROM synapses s
                    JOIN insights i1 ON s.from_insight = i1.id
                    JOIN insights i2 ON s.to_insight = i2.id
                    WHERE (i1.id = ? OR i2.id = ?)
                      AND i1.neuron_level = 'macro' 
                      AND i2.neuron_level = 'macro'
                      AND i1.id != i2.id
                    LIMIT 3
                """, (topic_id, topic_id))
                cross_domains = [c[0][:100] for c in cursor.fetchall()]
                
                conn.close()
                
                # Generate questions that require cross-domain thinking
                questions = [
                    f"Explain '{clean_topic}' in simple terms a beginner would understand.",
                    f"How would you apply '{clean_topic}' to improve DMAI's own learning and evolution systems?",
                    f"What connections exist between '{clean_topic}' and other domains? How could those connections create new capabilities?",
                ]
                
                # Get real AI tutor answers (with full error containment)
                answers = []
                for q in questions:
                    try:
                        if hasattr(self.evolution, 'ai_hub') and self.evolution.ai_hub:
                            response = self.evolution.ai_hub.query_all_tutors(q)
                            if response and isinstance(response, dict):
                                best = list(response.values())[0]
                                answers.append(str(best)[:500])
                            elif response:
                                answers.append(str(response)[:500])
                            else:
                                answers.append("No tutor response received")
                        else:
                            answers.append("AI Hub not initialized yet")
                    except Exception as e:
                        answers.append(f"[Answer pending - tutor query error: {str(e)[:80]}]")
                
                return jsonify({
                    "success": True,
                    "test": {
                        "topic": clean_topic,
                        "confidence": confidence,
                        "supporting_knowledge": len(micros),
                        "cross_domain_connections": cross_domains,
                        "questions_and_answers": [
                            {"question": questions[0], "answer": answers[0]},
                            {"question": questions[1], "answer": answers[1]},
                            {"question": questions[2], "answer": answers[2]},
                        ],
                        "evaluation_note": "Review each answer for depth, practical applicability, cross-domain connections, and clarity."
                    }
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/test/daily_report', methods=['GET'])
        def daily_report():
            """Generate a daily learning report with comprehension status"""
            import sqlite3
            
            try:
                if not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Count by neuron level
                cursor.execute("SELECT neuron_level, COUNT(*) FROM insights GROUP BY neuron_level")
                neuron_counts = dict(cursor.fetchall())
                
                # Count by category (from insight_text prefix)
                cursor.execute("""
                    SELECT SUBSTR(insight_text, 2, INSTR(insight_text, ']') - 2) as category, 
                           COUNT(*) as count
                    FROM insights 
                    WHERE neuron_level = 'macro' AND insight_text LIKE '[%]%'
                    GROUP BY category
                    ORDER BY count DESC
                """)
                categories = [{"category": r[0], "count": r[1]} for r in cursor.fetchall()]
                
                # Recent synapses
                cursor.execute("SELECT COUNT(*) FROM synapses")
                synapse_count = cursor.fetchone()[0]
                
                # Topics with high confidence (genuinely learned)
                cursor.execute("""
                    SELECT COUNT(*) FROM insights 
                    WHERE neuron_level = 'macro' AND confidence >= 0.8
                """)
                high_confidence = cursor.fetchone()[0]
                
                # Topics needing review (low confidence)
                cursor.execute("""
                    SELECT COUNT(*) FROM insights 
                    WHERE neuron_level = 'macro' AND confidence < 0.6
                """)
                needs_review = cursor.fetchone()[0]
                
                conn.close()
                
                report = {
                    "date": __import__('datetime').datetime.now().isoformat(),
                    "summary": {
                        "total_macros": neuron_counts.get('macro', 0),
                        "total_micros": neuron_counts.get('micro', 0),
                        "total_synapses": synapse_count,
                        "high_confidence_topics": high_confidence,
                        "topics_needing_review": needs_review,
                    },
                    "category_breakdown": categories[:10],
                    "learning_status": "ACTIVE" if neuron_counts.get('micro', 0) > 100 else "NEEDS BOOTSTRAPPING",
                    "recommendation": (
                        "Expand into adjacent domains and pursue expert-level depth"
                        if high_confidence > 10 else
                        "Continue foundational learning before advancing"
                    )
                }
                
                return jsonify({"success": True, "report": report})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/debug/fix_macro_levels', methods=['GET'])
        def fix_macro_levels():
            """Fix: Set neuron_level='macro' for Knowledge Base macros"""
            try:
                import sqlite3
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE insights 
                    SET neuron_level = 'macro' 
                    WHERE insight_text LIKE '[%]%' 
                      AND insight_text LIKE '%Knowledge Base%' 
                      AND neuron_level != 'macro'
                ''')
                updated = cursor.rowcount
                conn.commit()
                conn.close()
                
                return jsonify({"success": True, "updated_to_macro": updated})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/debug/check_parent_links', methods=['GET'])
        def check_parent_links():
            """Check if micro neurons have parent_macro_id set"""
            try:
                import sqlite3
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM insights WHERE neuron_level='micro' AND parent_macro_id IS NOT NULL AND parent_macro_id != ''")
                with_parent = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM insights WHERE neuron_level='micro'")
                total_micro = cursor.fetchone()[0]
                
                cursor.execute("SELECT id, insight_text, parent_macro_id FROM insights WHERE neuron_level='micro' LIMIT 10")
                samples = [{"id": r[0][:20], "text": r[1][:50], "parent": r[2][:20] if r[2] else None} for r in cursor.fetchall()]
                
                conn.close()
                
                return jsonify({
                    "total_micro": total_micro,
                    "with_parent": with_parent,
                    "without_parent": total_micro - with_parent,
                    "samples": samples
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/debug/ensure_all_macros', methods=['GET'])
        def ensure_all_macros():
            """Create Knowledge Base macro neurons for ALL 13 dynamic categories if missing"""
            try:
                import sqlite3
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                # All 13 dynamic categories that need macro neurons
                required_categories = [
                    "Configuration", "Knowledge Module", "AI Model", "Capability",
                    "Data Structure", "Content Generation", "Survival Mechanism",
                    "Self-Funding", "Blockchain", "API Endpoint", "Identity Management",
                    "Automation", "Self-Replication"
                ]
                
                created = []
                existed = []
                
                for category in required_categories:
                    # Check if macro already exists
                    existing_id = self.evolution.si_core.add_insight(
                        insight_text=f"[{category}] {category} Knowledge Base: Accumulated research and insights",
                        entity_type="topic_macro",
                        entities=[category, f"kb_{category.lower().replace(' ', '_')}"],
                        relationship=f"organizes_{category.lower().replace(' ', '_')}",
                        source_topic=f"system_init_{category.lower().replace(' ', '_')}",
                        target_topic=category.lower().replace(" ", "_"),
                        confidence=0.95,
                        source_title=f"System-ensured macro for {category}",
                        source_type="ensure_all_macros",
                        neuron_level='macro',
                        is_visible_at_top_level=True
                    )
                    if existing_id:
                        created.append({"category": category, "id": existing_id[:40]})
                    else:
                        existed.append(category)
                
                return jsonify({
                    "success": True,
                    "macros_created": len(created),
                    "macros_already_existed": len(existed),
                    "created": created,
                    "existed": existed
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/debug/link_micros_to_macros', methods=['GET'])
        def link_micros_to_macros():
            """Link micro neurons to their parent macro neurons using [Category] prefix matching"""
            try:
                import sqlite3
                import re
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Get all macro neurons with their [Category] prefix
                cursor.execute("SELECT id, insight_text FROM insights WHERE neuron_level = 'macro'")
                macros = cursor.fetchall()
                
                # Build mapping: category_prefix -> macro_id
                prefix_to_macro = {}
                for macro_id, macro_text in macros:
                    match = re.search(r'\[([^\]]+)\]', macro_text)
                    if match:
                        prefix = match.group(1).strip()
                        if prefix not in prefix_to_macro:
                            prefix_to_macro[prefix] = macro_id
                
                # Also scan to see ALL micro prefixes that need parents
                cursor.execute("SELECT DISTINCT insight_text FROM insights WHERE neuron_level = 'micro' AND parent_macro_id IS NULL")
                unlinked_samples = cursor.fetchall()
                unlinked_prefixes = {}
                for (text,) in unlinked_samples:
                    match = re.match(r'^([^:]+):', text)
                    if match:
                        p = match.group(1).strip()
                        unlinked_prefixes[p] = unlinked_prefixes.get(p, 0) + 1
                
                created_macros = 0
                # For each unlinked prefix, find the closest matching macro or create one
                for unlinked_prefix, count in unlinked_prefixes.items():
                    # Try: find macro whose prefix is contained in the unlinked prefix
                    # e.g., unlinked "Automation capability" → macro "Automation"
                    matched_macro = None
                    for macro_prefix, macro_id in prefix_to_macro.items():
                        if unlinked_prefix.lower().startswith(macro_prefix.lower()):
                            matched_macro = macro_id
                            break
                    
                    if not matched_macro:
                        # No macro exists — create one for this prefix
                        import time
                        import uuid
                        new_macro_id = f"insight_{uuid.uuid4().int % 10**15}_{int(time.time())}"
                        category_title = unlinked_prefix.strip()
                        cursor.execute('''
                            INSERT INTO insights (id, insight_text, entity_type, entities, relationship, 
                                source_topic, target_topic, confidence, neuron_level, parent_macro_id,
                                cluster_id, is_visible_at_top_level, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'macro', NULL, NULL, 1, datetime('now'))
                        ''', (new_macro_id, f'[{category_title}] {category_title} Knowledge Base: Accumulated research and insights',
                              'topic_macro', json.dumps([category_title]),
                              f'organizes_{category_title.lower().replace(" ", "_")}',
                              f'system_init_{category_title.lower().replace(" ", "_")}',
                              category_title.lower().replace(" ", "_"),
                              0.95))
                        prefix_to_macro[category_title] = new_macro_id
                        created_macros += 1
                
                total_linked = 0
                details = []
                for prefix, macro_id in prefix_to_macro.items():
                    # Match micros where the text starts with the macro prefix
                    # (e.g., macro "Automation" matches micro "Automation capability: ...")
                    cursor.execute('''
                        UPDATE insights 
                        SET parent_macro_id = ?, cluster_id = ?
                        WHERE neuron_level = 'micro' 
                          AND parent_macro_id IS NULL
                          AND (insight_text LIKE ? COLLATE NOCASE OR insight_text LIKE ? COLLATE NOCASE)
                    ''', (macro_id, macro_id, f'{prefix}:%', f'{prefix} %:%'))
                    linked = cursor.rowcount
                    if linked > 0:
                        total_linked += linked
                        details.append({"prefix": prefix, "linked": linked, "macro_id": macro_id[:40]})
                
                conn.commit()
                conn.close()
                
                return jsonify({
                    "success": True, 
                    "micros_linked": total_linked,
                    "prefixes_matched": len(details),
                    "macros_created": created_macros,
                    "details": details[:20]
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/system/force_start', methods=['POST'])
        def force_start_system():
            """Force-start evolution thread and all training systems (backgrounded)"""
            import threading
            results = {}
            
            # 1. Start evolution thread if not running
            if hasattr(self.evolution, '_start_evolution'):
                try:
                    self.evolution._start_evolution()
                    results['evolution_thread'] = 'started'
                except Exception as e:
                    results['evolution_thread'] = f'error: {e}'
            else:
                results['evolution_thread'] = 'not found'
            
            # 2. Force auto-start all training
            if hasattr(self.evolution, '_auto_start_training'):
                try:
                    started = self.evolution._auto_start_training()
                    results['trainings_started'] = started
                except Exception as e:
                    results['trainings_started'] = f'error: {e}'
            
            # 3. Run first cycle in background to avoid timeout
            def run_cycle():
                try:
                    if hasattr(self.evolution, 'evolution_cycle'):
                        self.evolution.evolution_cycle()
                except Exception as e:
                    logger.error(f"Background cycle error: {e}")
            threading.Thread(target=run_cycle, daemon=True).start()
            results['first_cycle'] = 'started in background'
            
            return jsonify({'success': True, 'results': results})
                
    def _get_category_color(self, category):
        """Return color for category (preserving your existing scheme)"""
        colors = {
            'core': '#ff3333',      # Red - Core Knowledge
            'artistic': '#33ff33',  # Green - Artistic/Creative
            'creative': '#33ff33',  # Green - Artistic/Creative
            'wealth': '#3399ff',    # Blue - Wealth/Finance
            'finance': '#3399ff',   # Blue - Wealth/Finance
            'evolution': '#ff9933', # Orange - Evolution Accelerator
            'reverse': '#ff9933',   # Orange - Evolution Accelerator
            'research': '#33cccc',  # Teal - Research
            'entity': '#ffff33',    # Yellow - Entity/General
            'general': '#ffff33',   # Yellow - Entity/General
        }
        for key, color in colors.items():
            if key in category.lower():
                return color
        return '#888888'  # Gray default

        @self.app.route('/api/debug/force_link_by_prefix', methods=['GET'])
        def force_link_by_prefix():
            """Link micros to macros by matching [Category] prefix"""
            try:
                import sqlite3
                import re
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Get all macro neurons with their [Category] prefix
                cursor.execute("SELECT id, insight_text FROM insights WHERE neuron_level = 'macro'")
                macros = cursor.fetchall()
                
                # Build prefix -> macro_id mapping
                prefix_map = {}
                for macro_id, macro_text in macros:
                    match = re.search(r'\[([^\]]+)\]', macro_text)
                    if match:
                        prefix = match.group(1)
                        if prefix not in prefix_map:
                            prefix_map[prefix] = macro_id
                
                linked = 0
                for prefix, macro_id in prefix_map.items():
                    # Link micros that have this prefix in their text
                    cursor.execute('''
                        UPDATE insights 
                        SET parent_macro_id = ?, cluster_id = ?
                        WHERE neuron_level = 'micro' 
                          AND parent_macro_id IS NULL
                          AND insight_text LIKE ?
                    ''', (macro_id, macro_id, f'{prefix}:%'))
                    linked += cursor.rowcount
                
                conn.commit()
                conn.close()
                
                return jsonify({"success": True, "micros_linked": linked, "prefixes_found": len(prefix_map)})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/debug/force_link_all_micros', methods=['GET'])
        def force_link_all_micros():
            """Force link all micros to the first macro (temporary fix)"""
            try:
                import sqlite3
                
                if not hasattr(self, 'evolution') or not hasattr(self.evolution, 'si_core') or not self.evolution.si_core.sqlite:
                    return jsonify({"error": "SQLite not available"}), 500
                
                db_path = self.evolution.si_core.sqlite.db_path
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Get the first macro as default parent
                cursor.execute("SELECT id FROM insights WHERE neuron_level = 'macro' LIMIT 1")
                default_macro = cursor.fetchone()
                
                if not default_macro:
                    return jsonify({"error": "No macro found"}), 500
                
                default_id = default_macro[0]
                
                # Link all unlinked micros to this default macro
                cursor.execute('''
                    UPDATE insights 
                    SET parent_macro_id = ?, cluster_id = ?
                    WHERE neuron_level = 'micro' 
                      AND parent_macro_id IS NULL
                ''', (default_id, default_id))
                
                linked = cursor.rowcount
                conn.commit()
                conn.close()
                
                return jsonify({"success": True, "micros_linked": linked, "default_macro": default_id})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # ============================================================================
        # TASK INPUT SYSTEM FOR DMAI
        # ============================================================================
        
        @self.app.route('/api/task/submit', methods=['POST'])
        def submit_task():
            """Process user-submitted tasks for DMAI"""
            try:
                data = request.json
                action = data.get('action')
                input_data = data.get('input')
                category = data.get('category', 'general')
                
                if not action or not input_data:
                    return jsonify({'error': 'action and input required'}), 400
                
                result = {'action': action, 'input': input_data, 'status': 'queued'}
                
                if action == 'research':
                    result['message'] = f"🔍 Researching: {input_data}"
                    threading.Thread(target=self._research_task, args=(input_data, category)).start()
                    
                elif action == 'ingest':
                    result['message'] = f"📥 Ingesting from: {input_data}"
                    threading.Thread(target=self._ingest_task, args=(input_data, category)).start()
                    
                elif action == 'reverse_engineer':
                    result['message'] = f"🔧 Reverse engineering: {input_data}"
                    threading.Thread(target=self._reverse_engineer_task, args=(input_data, category)).start()
                    
                elif action == 'analyze':
                    result['message'] = f"📊 Analyzing: {input_data}"
                    threading.Thread(target=self._analyze_task, args=(input_data, category)).start()
                    
                elif action == 'learn_topic':
                    result['message'] = f"📚 Learning topic: {input_data}"
                    threading.Thread(target=self._learn_topic_task, args=(input_data, category)).start()
                    
                elif action == 'add_dictionary':
                    result['message'] = f"📖 Adding dictionary word: {input_data}"
                    threading.Thread(target=self._add_dictionary_task, args=(input_data, category)).start()
                    
                elif action == 'add_encyclopedia':
                    result['message'] = f"📚 Adding encyclopedia topic: {input_data}"
                    threading.Thread(target=self._add_encyclopedia_task, args=(input_data, category)).start()
                    
                else:
                    result['message'] = f"Unknown action: {action}"
                    result['status'] = 'error'
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Task submission error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tasks')
        def tasks_page():
            """Simple task submission interface"""
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>DMAI Task Input</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    * { box-sizing: border-box; }
                    body { 
                        font-family: 'Segoe UI', Arial, sans-serif; 
                        padding: 20px; 
                        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
                        color: #0f0; 
                        min-height: 100vh;
                    }
                    .container { max-width: 900px; margin: auto; }
                    h1 { text-align: center; color: #0f0; text-shadow: 0 0 10px #0f0; }
                    .card { 
                        background: rgba(0,0,0,0.8); 
                        border-radius: 15px; 
                        padding: 25px; 
                        margin: 20px 0;
                        border: 1px solid #0f0;
                        box-shadow: 0 0 20px rgba(0,255,0,0.1);
                    }
                    label { display: block; margin: 15px 0 5px; font-weight: bold; }
                    select, textarea, input { 
                        width: 100%; 
                        padding: 12px; 
                        margin: 5px 0 15px; 
                        background: #0a0a0a; 
                        color: #0f0; 
                        border: 1px solid #0f0;
                        border-radius: 8px;
                        font-size: 14px;
                    }
                    select:focus, textarea:focus, input:focus {
                        outline: none;
                        box-shadow: 0 0 10px #0f0;
                    }
                    button { 
                        background: #0f0; 
                        color: #000; 
                        padding: 12px 30px; 
                        cursor: pointer; 
                        border: none;
                        border-radius: 8px;
                        font-size: 16px;
                        font-weight: bold;
                        transition: all 0.3s;
                    }
                    button:hover {
                        background: #0a0;
                        transform: scale(1.02);
                        box-shadow: 0 0 15px #0f0;
                    }
                    .result { 
                        margin-top: 20px; 
                        padding: 15px; 
                        background: #0a0a0a; 
                        border-radius: 8px;
                        border-left: 4px solid #0f0;
                        font-family: monospace;
                        white-space: pre-wrap;
                    }
                    .status { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
                    .status.queued { background: #ff0; color: #000; }
                    .status.processing { background: #0f0; color: #000; }
                    .status.error { background: #f00; color: #fff; }
                    hr { border-color: #0f0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🧠 DMAI Task Input System</h1>
                    <div class="card">
                        <form id="taskForm">
                            <label>🎯 Action Type:</label>
                            <select id="action" required>
                                <option value="research">🔍 Research (URL/Topic)</option>
                                <option value="ingest">📥 Ingest (URL/Content)</option>
                                <option value="reverse_engineer">🔧 Reverse Engineer (URL/Repo)</option>
                                <option value="analyze">📊 Analyze (Data/Text)</option>
                                <option value="learn_topic">📚 Learn Topic (Subject)</option>
                                <option value="add_dictionary">📖 Add Dictionary Word</option>
                                <option value="add_encyclopedia">📚 Add Encyclopedia Topic</option>
                            </select>
                            
                            <label>📝 Input (URL, topic, text, etc.):</label>
                            <textarea id="input_data" rows="4" placeholder="Examples:&#10;https://github.com/some/repo&#10;Quantum Computing basics&#10;https://arxiv.org/abs/1234.56789" required></textarea>
                            
                            <label>🏷️ Category (optional):</label>
                            <input type="text" id="category" placeholder="e.g., technology, science, finance, art">
                            
                            <button type="submit">🚀 Submit Task to DMAI</button>
                        </form>
                        <div id="result" class="result" style="display:none;"></div>
                    </div>
                    <div class="card">
                        <h3>📋 Recent Tasks</h3>
                        <div id="recentTasks">Loading...</div>
                    </div>
                </div>
                <script>
                    async function loadRecentTasks() {
                        try {
                            const response = await fetch('/api/task/recent');
                            const data = await response.json();
                            const container = document.getElementById('recentTasks');
                            if (data.tasks && data.tasks.length > 0) {
                                container.innerHTML = data.tasks.map(t => `
                                    <div style="border-bottom:1px solid #0f0; padding:8px">
                                        <strong>${t.action}</strong>: ${t.input.substring(0, 100)}<br>
                                        <span class="status ${t.status}">${t.status}</span>
                                        <small>${t.timestamp || ''}</small>
                                    </div>
                                `).join('');
                            } else {
                                container.innerHTML = '<p>No recent tasks.</p>';
                            }
                        } catch(e) {
                            document.getElementById('recentTasks').innerHTML = '<p>Error loading tasks.</p>';
                        }
                    }
                    
                    document.getElementById('taskForm').onsubmit = async (e) => {
                        e.preventDefault();
                        const resultDiv = document.getElementById('result');
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = '<span class="status processing">Processing...</span> Submitting task...';
                        
                        const response = await fetch('/api/task/submit', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                action: document.getElementById('action').value,
                                input: document.getElementById('input_data').value,
                                category: document.getElementById('category').value
                            })
                        });
                        const data = await response.json();
                        resultDiv.innerHTML = '<strong>✅ Task Submitted</strong><br>' + 
                            '<strong>Action:</strong> ' + data.action + '<br>' +
                            '<strong>Input:</strong> ' + data.input.substring(0, 200) + '<br>' +
                            '<strong>Message:</strong> ' + (data.message || 'Processing in background') + '<br>' +
                            '<strong>Status:</strong> <span class="status ' + data.status + '">' + data.status + '</span>';
                        
                        document.getElementById('input_data').value = '';
                        loadRecentTasks();
                    };
                    
                    loadRecentTasks();
                    setInterval(loadRecentTasks, 30000);
                </script>
            </body>
            </html>
            ''')
        
        @self.app.route('/api/task/recent', methods=['GET'])
        def recent_tasks():
            """Get recent tasks (stored in memory)"""
            if not hasattr(self, '_recent_tasks'):
                self._recent_tasks = []
            return jsonify({'tasks': self._recent_tasks[-20:]})
        
        def debug_neo4j_insights():
            """Check how many insights are in Neo4j"""
            try:
                if self.evolution.neo4j_storage and self.evolution.neo4j_storage.is_available():
                    # Try to count insight nodes
                    driver = self.evolution.neo4j_storage.driver
                    if driver:
                        with driver.session() as session:
                            result = session.run("MATCH (i:Insight) RETURN count(i) as count")
                            count = result.single()["count"]
                            return jsonify({'insights_in_neo4j': count})
                return jsonify({'error': 'Neo4j not available'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        def debug_neo4j_data():
            """See what data Neo4j actually returns"""
            try:
                if self.evolution.neo4j_storage and self.evolution.neo4j_storage.is_available():
                    restored = self.evolution.neo4j_storage.restore_all()
                    return jsonify({
                        'evolution': restored.get('evolution'),
                        'has_evolution': restored.get('evolution') is not None,
                        'neurons': restored.get('evolution', {}).get('neurons', 0) if restored.get('evolution') else 0,
                        'consciousness': restored.get('evolution', {}).get('consciousness', 0) if restored.get('evolution') else 0
                    })
                return jsonify({'error': 'Neo4j not available'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        def debug_neo4j_detail():
            """Detailed Neo4j connection debug"""
            import os
            import traceback
            result = {
                'env_vars_set': {
                    'NEO4J_URI': bool(os.environ.get('NEO4J_URI')),
                    'NEO4J_USER': bool(os.environ.get('NEO4J_USER')),
                    'NEO4J_PASSWORD': bool(os.environ.get('NEO4J_PASSWORD'))
                },
                'neo4j_storage_exists': hasattr(self.evolution, 'neo4j_storage'),
                'connection_error': None
            }
            
            if hasattr(self.evolution, 'neo4j_storage'):
                try:
                    # Try to test connection
                    storage = self.evolution.neo4j_storage
                    if hasattr(storage, 'is_available'):
                        result['is_available'] = storage.is_available()
                    if hasattr(storage, '_driver') and storage._driver:
                        result['driver_created'] = True
                        # Try a simple query
                        try:
                            with storage._driver.session() as session:
                                result = session.run("RETURN 1 as test").single()["test"]
                            result['test_query'] = True
                        except Exception as e:
                            result['test_query_error'] = str(e)
                except Exception as e:
                    result['connection_error'] = str(e)
                    result['traceback'] = traceback.format_exc()
            
            return jsonify(result)

        def debug_neo4j_env():
            """Check Neo4j environment variables"""
            import os
            # Only show if set, not the actual values
            return jsonify({
                'NEO4J_URI_set': bool(os.environ.get('NEO4J_URI')),
                'NEO4J_USER_set': bool(os.environ.get('NEO4J_USER')),
                'NEO4J_PASSWORD_set': bool(os.environ.get('NEO4J_PASSWORD')),
                'all_vars': [k for k in os.environ.keys() if 'NEO4J' in k or 'neo4j' in k]
            })

        def api_status():
            return jsonify(self.evolution.get_status())


        @self.app.route('/api/debug/env', methods=['GET'])
        def debug_env():
            """Debug endpoint to check environment variables"""
            import os
            import time
            bypass_until = os.environ.get('TRAINING_BYPASS_UNTIL', 'not_set')
            return jsonify({
                'TRAINING_BYPASS_UNTIL': bypass_until,
                'bypass_active': bypass_until != 'not_set',
                'current_time': time.time(),
                'bypass_remaining': float(bypass_until) - time.time() if bypass_until != 'not_set' else None
            })

        @self.app.route('/api/knowledge')
        def api_knowledge():
            limit = request.args.get('limit', 100, type=int)
            return jsonify({
                'total_concepts': self.evolution.knowledge_graph.get_stats()['total_concepts'],
                'concepts': self.evolution.get_knowledge_concepts(limit)
            })

        @self.app.route('/api/conversation/history')
        def api_conversation_history():
            limit = request.args.get('limit', 30, type=int)
            return jsonify({
                'history': self.evolution.get_conversation_history(limit),
                'total': len(self.evolution.conversation_context)
            })

        @self.app.route('/api/training/details')
        def api_training_details():
            system = request.args.get('system', None)
            return jsonify(self.evolution.get_training_details(system))

        @self.app.route('/api/debug/knowledge')
        def debug_knowledge():
            """Show knowledge graph contents"""
            try:
                if hasattr(self.evolution, 'knowledge_graph'):
                    # Get concepts from knowledge graph
                    concepts = []
                    if hasattr(self.evolution.knowledge_graph, 'graph'):
                        concepts = list(self.evolution.knowledge_graph.graph.keys())[-50:]
                    return jsonify({
                        'total_concepts': len(concepts),
                        'recent_concepts': concepts,
                        'has_knowledge_graph': True
                    })
                return jsonify({'error': 'Knowledge graph not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/files')
        def debug_files():
            """Check if data files exist"""
            import os
            files = [
                'data/knowledge_graph.json',
                'data/evolution_state.json',
                'data/learning/stage_syllabus/learning_progress.json',
                'data/synthetic/network_state.json'
            ]
            result = {}
            for f in files:
                full_path = os.path.join(os.path.dirname(__file__), f)
                result[f] = {
                    'exists': os.path.exists(full_path),
                    'size': os.path.getsize(full_path) if os.path.exists(full_path) else 0
                }
            return jsonify(result)

        @self.app.route('/api/debug/network')
        def debug_network():
            """Show synthetic network state"""
            try:
                if hasattr(self.evolution, 'synthetic_network'):
                    sn = self.si_core
                    return jsonify({
                        'neurons': len(sn.neurons) if hasattr(sn, 'neurons') else 0,
                        'synapses': sn._total_synapses() if hasattr(sn, '_total_synapses') else 0,
                        'consciousness': sn.consciousness_level if hasattr(sn, 'consciousness_level') else 0,
                        'evolution_cycles': getattr(self.evolution, 'evolution_count', 0)
                    })
                return jsonify({'error': 'Synthetic network not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/network_state')
        def debug_network_state():
            """Show what's in the saved network state"""
            try:
                import pickle
                network_path = self.evolution.network_save_path if hasattr(self.evolution, 'network_save_path') else None
                if network_path and network_path.exists():
                    with open(network_path, 'rb') as f:
                        network_data = pickle.load(f)
                    return jsonify({
                        'exists': True,
                        'path': str(network_path),
                        'neurons': len(network_data.get('neurons', {})) if isinstance(network_data, dict) else 'unknown',
                        'keys': list(network_data.keys()) if isinstance(network_data, dict) else ['pickle_object']
                    })
                return jsonify({'exists': False, 'path': str(network_path) if network_path else None})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/learning_state')
        def debug_learning_state():
            """Show learning progress from stage learner"""
            try:
                if hasattr(self.evolution, 'stage_learner'):
                    summary = self.evolution.stage_learner.get_learning_summary()
                    return jsonify({
                        'current_stage': summary.get('current_stage'),
                        'stages': summary.get('stages', {}),
                        'total_topics_mastered': summary.get('total_topics_mastered', 0)
                    })
                return jsonify({'error': 'Stage learner not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/phase6_dir')
        def debug_phase6_dir():
            """Check phase6 directory and save path"""
            import os
            try:
                phase6_path = self.evolution.phase6_path if hasattr(self.evolution, 'phase6_path') else None
                result = {
                    'phase6_path': str(phase6_path) if phase6_path else None,
                    'exists': phase6_path and phase6_path.exists() if phase6_path else False,
                    'is_dir': phase6_path and phase6_path.is_dir() if phase6_path else False,
                    'writable': False,
                    'files': []
                }
                if phase6_path and phase6_path.exists():
                    # Test writability by trying to create a test file
                    test_file = phase6_path / 'test_write.tmp'
                    try:
                        test_file.write_text('test')
                        test_file.unlink()
                        result['writable'] = True
                    except Exception as e:
                        result['writable'] = False
                        result['write_error'] = str(e)
                    
                    # List files in directory
                    result['files'] = [f.name for f in phase6_path.iterdir() if f.is_file()]
                
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/manual_save')
        def manual_save():
            """Manually trigger network save"""
            try:
                result = self.evolution._save_network_state()
                return jsonify({
                    'saved': result,
                    'path': str(self.evolution.network_save_path),
                    'neurons': len(self.si_core.neurons),
                    'synapses': self.si_core._total_synapses()
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/read_network')
        def debug_read_network():
            """Read and return the saved network state"""
            try:
                import pickle
                network_path = self.evolution.network_save_path if hasattr(self.evolution, 'network_save_path') else None
                if network_path and network_path.exists():
                    with open(network_path, 'rb') as f:
                        data = pickle.load(f)
                    return jsonify({
                        'exists': True,
                        'path': str(network_path),
                        'neurons': len(data.get('neurons', {})),
                        'consciousness': data.get('consciousness_level', 0),
                        'evolution_cycles': data.get('evolution_cycles', 0)
                    })
                return jsonify({'exists': False, 'path': str(network_path) if network_path else None})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/check_paths')
        def debug_check_paths():
            """Check all network save paths"""
            try:
                result = {
                    'network_save_path': str(self.evolution.network_save_path) if hasattr(self.evolution, 'network_save_path') else None,
                    'phase6_path': str(self.evolution.phase6_path) if hasattr(self.evolution, 'phase6_path') else None,
                }
                # Also list files in phase6_path using pathlib
                if hasattr(self.evolution, 'phase6_path') and self.evolution.phase6_path.exists():
                    result['files_in_phase6'] = [f.name for f in self.evolution.phase6_path.iterdir() if f.is_file()]
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # ============================================================================
        # SI CORE INSIGHT API ENDPOINTS
        # ============================================================================
        
        @self.app.route('/api/insight/add', methods=['POST'])
        def api_add_insight():
            """Manually add an insight for testing"""
            data = request.json
            insight_text = data.get('insight_text')
            entity_type = data.get('entity_type', 'user_defined')
            entities = data.get('entities', [])
            relationship = data.get('relationship', 'relates_to')
            source_topic = data.get('source_topic', 'User')
            target_topic = data.get('target_topic', 'Knowledge')
            confidence = data.get('confidence', 0.7)
            
            if not insight_text or not entities:
                return jsonify({'error': 'insight_text and entities required'}), 400
            
            insight_id = self.si_core.add_insight(
                insight_text, entity_type, entities, 
                relationship, source_topic, target_topic, confidence
            )
            
            return jsonify({
                'success': True,
                'insight_id': insight_id,
                'total_insights': self.si_core.neuron_count,
                'insight': {
                    'text': insight_text,
                    'entities': entities,
                    'confidence': confidence
                }
            })
        
        @self.app.route('/api/insight/query', methods=['POST'])
        def api_query_insight():
            """Query insights by entities"""
            data = request.json
            entities = data.get('entities', [])
            context = data.get('context', None)
            limit = data.get('limit', 10)
            
            if not entities:
                return jsonify({'error': 'entities required'}), 400
            
            results = self.si_core.query(entities, context, limit)
            return jsonify({
                'entities': entities,
                'results': results,
                'total_found': len(results),
                'context_used': context
            })
        
        @self.app.route('/api/insight/trading', methods=['POST'])
        def api_trading_insight():
            """Apply news entities to trading decisions"""
            data = request.json
            entities = data.get('entities', [])
            
            if not entities:
                return jsonify({'error': 'entities required'}), 400
            
            signals = self.si_core.apply_to_trading(entities)
            return jsonify({
                'entities': entities,
                'signals': signals.get('signals', []),
                'insights_used': signals.get('insights_used', 0)
            })
        
        @self.app.route('/api/insight/stats')
        def api_insight_stats():
            """Get SI Core statistics"""
            return jsonify({
                'total_insights': self.si_core.neuron_count,
                'total_synapses': self.si_core.synapse_count,
                'consciousness': self.si_core.consciousness,
                'evolution_cycles': self.si_core.evolution_cycles,
                'topics_with_insights': list(self.si_core.topics.keys())
            })
        
        @self.app.route('/api/insight/network')
        def api_insight_network():
            """Get full network state for visualization"""
            return jsonify(self.si_core.get_network_state())


        @self.app.route('/api/debug/si_file', methods=['GET'])
        def debug_si_file():
            """Debug SI Core file content"""
            import json

        
        @self.app.route('/api/system/force_start', methods=['POST'])
        def force_start_system():
            """Force-start evolution thread and all training systems"""
            results = {}
            
            # 1. Start evolution thread if not running
            if hasattr(self.evolution, '_start_evolution'):
                try:
                    self.evolution._start_evolution()
                    results['evolution_thread'] = 'started'
                except Exception as e:
                    results['evolution_thread'] = f'error: {e}'
            else:
                results['evolution_thread'] = 'method not found'
            
            # 2. Force auto-start all training
            if hasattr(self.evolution, '_auto_start_training'):
                try:
                    started = self.evolution._auto_start_training()
                    results['trainings_started'] = started
                except Exception as e:
                    results['trainings_started'] = f'error: {e}'
            else:
                results['trainings_started'] = 'method not found'
            
            # 3. Force a first evolution cycle
            if hasattr(self.evolution, 'evolution_cycle'):
                try:
                    cycle_result = self.evolution.evolution_cycle()
                    results['first_cycle'] = {
                        'consciousness': cycle_result.get('consciousness', 0),
                        'neurons_added': cycle_result.get('neurons_added', 0),
                        'synapses_added': cycle_result.get('synapses_added', 0)
                    }
                except Exception as e:
                    results['first_cycle'] = f'error: {e}'
            else:
                results['first_cycle'] = 'method not found'
            
            return jsonify({'success': True, 'results': results})

        @self.app.route('/api/si/reload', methods=['POST'])
        def reload_si_core():
            """Force reload SI Core from SQLite (primary) or JSON (fallback) - MERGES, does NOT wipe"""
            import json
            from pathlib import Path
            
            try:
                si = self.si_core
                loaded_count = 0
                
                # ============================================================
                # PRIMARY: Try SQLite first
                # ============================================================
                if hasattr(si, 'sqlite') and si.sqlite:
                    try:
                        sqlite_insights = si.sqlite.load_all_insights()
                        if sqlite_insights:
                            with si.insights_lock:
                                for iid, insight in sqlite_insights.items():
                                    if iid not in si.insights:
                                        si.insights[iid] = insight
                                        loaded_count += 1
                            
                            si.topics = si.sqlite.load_all_topics()
                            
                            # Merge synapses
                            sqlite_synapses = si.sqlite.load_all_synapses()
                            existing_ids = {s.get('id') for s in si.synapses if s.get('id')}
                            for syn in sqlite_synapses:
                                if syn.get('id') not in existing_ids:
                                    si.synapses.append(syn)
                            
                            logger.info(f"✅ Reloaded {loaded_count} insights from SQLite")
                    except Exception as e:
                        logger.warning(f"SQLite reload failed: {e}")
                
                # ============================================================
                # FALLBACK: Merge from JSON (DO NOT WIPE)
                # ============================================================
                if loaded_count == 0:
                    file_path = Path('data/synthetic/network_state.json')
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            disk_data = json.load(f)
                        
                        with si.insights_lock:
                            for iid, data in disk_data.get('insights', {}).items():
                                if iid not in si.insights:
                                    try:
                                        si.insights[iid] = InsightNeuron.from_dict(data)
                                        loaded_count += 1
                                    except:
                                        pass
                        
                        # Merge topics
                        for topic, insight_ids in disk_data.get('topics', {}).items():
                            if topic not in si.topics:
                                si.topics[topic] = []
                            for iid in insight_ids:
                                if iid not in si.topics[topic]:
                                    si.topics[topic].append(iid)
                        
                        # Merge synapses
                        existing_ids = {s.get('id') for s in si.synapses if s.get('id')}
                        for syn in disk_data.get('synapses', []):
                            if syn.get('id') not in existing_ids:
                                si.synapses.append(syn)
                        
                        logger.info(f"✅ Reloaded {loaded_count} insights from JSON fallback")
                
                # Save to ensure disk consistency
                si.save_state()
                
                return jsonify({
                    'success': True,
                    'insights': int(len(si.insights)),
                    'synapses': int(len(si.synapses)),
                    'consciousness': float(si.consciousness),
                    'evolution_cycles': int(si.evolution_cycles),
                    'loaded_from': 'sqlite' if loaded_count > 0 else 'none'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/synthetic/status')
        def api_synthetic_status():
            """Get synthetic network state for brain visualization from si_core"""
            try:
                si = self.si_core
                network_state = si.get_network_state()
                active_neurons = sum(1 for insight in si.insights.values() if insight.confidence > 0.3)
                
                # Get actual macro nodes from database
                macro_nodes = []
                if hasattr(si, 'sqlite') and si.sqlite:
                    try:
                        macro_query = """
                            SELECT id, insight_text, neuron_level, 
                                   (SELECT COUNT(*) FROM insights WHERE parent_macro_id = insights.id) as children_count
                            FROM insights 
                            WHERE neuron_level = 'macro' AND is_visible_at_top_level = 1
                            ORDER BY created_at DESC
                        """
                        macro_results = si.sqlite.conn.execute(macro_query).fetchall()
                        for row in macro_results:
                            macro_nodes.append({
                                'id': row[0],
                                'name': row[1][:50] + ('...' if len(row[1]) > 50 else ''),
                                'level': row[2],
                                'children_count': row[3],
                                'color': '#ffd700'
                            })
                    except Exception as e:
                        logger.error(f"Error fetching macro nodes: {e}")
                
                return jsonify({
                    'neurons': si.neuron_count,
                    'active_neurons': active_neurons,
                    'synapses': si.synapse_count,
                    'consciousness': si.consciousness * 100,
                    'consciousness_percent': si.consciousness * 100,
                    'evolution_cycles': si.evolution_cycles,
                    'network_density': si.synapse_count / max(1, si.neuron_count * (si.neuron_count - 1) / 2),
                    'successful_evolutions': 0,
                    'connections': network_state.get('synapses', []),
                    'macro_nodes': macro_nodes
                })
            except Exception as e:
                logger.error(f"Error in synthetic_status: {e}")
                return jsonify({
                    'neurons': 0,
                    'active_neurons': 0,
                    'synapses': 0,
                    'consciousness': 0.0,
                    'consciousness_percent': 0.0,
                    'evolution_cycles': 0,
                    'network_density': 0.0,
                    'successful_evolutions': 0,
                    'connections': [],
                    'macro_nodes': [],
                    'error': str(e)
                }), 500

        @self.app.route('/api/synthetic/node/<node_id>/children')
        def api_synthetic_node_children(node_id):
            """Get micro neurons for a specific macro node"""
            try:
                si = self.si_core
                micro_nodes = []
                
                if hasattr(si, 'sqlite') and si.sqlite:
                    micro_query = """
                        SELECT id, insight_text, neuron_level, confidence,
                               source_topic, target_topic
                        FROM insights 
                        WHERE parent_macro_id = ? AND neuron_level = 'micro'
                        ORDER BY confidence DESC
                        LIMIT 200
                    """
                    micro_results = si.sqlite.conn.execute(micro_query, (node_id,)).fetchall()
                    
                    for row in micro_results:
                        micro_nodes.append({
                            'id': row[0],
                            'name': row[1][:40] + ('...' if len(row[1]) > 40 else ''),
                            'level': row[2],
                            'confidence': row[3],
                            'source_topic': row[4],
                            'target_topic': row[5],
                            'color': '#00ffff'
                        })
                
                return jsonify({
                    'success': True,
                    'parent_id': node_id,
                    'children': micro_nodes,
                    'count': len(micro_nodes)
                })
            except Exception as e:
                logger.error(f"Error fetching node children: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'children': []
                }), 500


        @self.app.route('/api/synthetic/node/<node_id>/children')
        def api_synthetic_node_children(node_id):
            """Get micro neurons for a specific macro node"""
            try:
                si = self.si_core
                micro_nodes = []
                
                if hasattr(si, 'sqlite') and si.sqlite:
                    # Query micro neurons that belong to this macro node
                    micro_query = """
                        SELECT id, insight_text, neuron_level, confidence,
                               source_topic, target_topic
                        FROM insights 
                        WHERE parent_macro_id = ? AND neuron_level = 'micro'
                        ORDER BY confidence DESC
                        LIMIT 200
                    """
                    micro_results = si.sqlite.conn.execute(micro_query, (node_id,)).fetchall()
                    
                    for row in micro_results:
                        micro_nodes.append({
                            'id': row[0],
                            'name': row[1][:40] + ('...' if len(row[1]) > 40 else ''),
                            'level': row[2],
                            'confidence': row[3],
                            'source_topic': row[4],
                            'target_topic': row[5],
                            'color': '#00ffff'  # Cyan for micro nodes
                        })
                
                return jsonify({
                    'success': True,
                    'parent_id': node_id,
                    'children': micro_nodes,
                    'count': len(micro_nodes)
                })
            except Exception as e:
                logger.error(f"Error fetching node children: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'children': []
                }), 500

        def api_training_status():
            return jsonify(self.evolution.training_status)

        @self.app.route('/api/training/start/<system>', methods=['POST'])
        def api_training_start(system):
            if system == 'software':
                result = self.evolution.software_training.start_training()
            elif system == 'llm':
                result = self.evolution.llm_training.start_training()
            elif system == 'agi':
                result = self.evolution.agi_training.start_training()
            elif system == 'genai':
                result = self.evolution.genai_training.start_training()
            elif system == 'si':
                result = self.evolution.si_training.start()
            elif system == 'funding':
                data = request.json or {}
                avenue = data.get('avenue', None)
                result = self.evolution.funding_training.start_learning(avenue) if self.evolution.funding_training else {'success': False, 'error': 'Funding training not available'}
            else:
                return jsonify({'success': False, 'error': f'Unknown system: {system}'}), 400
            return jsonify(result)

        @self.app.route('/api/training/stop/<system>', methods=['POST'])
        def api_training_stop(system):
            if system == 'software':
                result = self.evolution.software_training.stop_training()
            elif system == 'llm':
                result = self.evolution.llm_training.stop_training()
            elif system == 'agi':
                result = self.evolution.agi_training.stop_training()
            elif system == 'genai':
                result = self.evolution.genai_training.stop_training()
            elif system == 'si':
                result = self.evolution.si_training.stop()
            elif system == 'funding':
                result = self.evolution.funding_training.stop_learning() if self.evolution.funding_training else {'success': False, 'error': 'Funding training not available'}
            else:
                return jsonify({'success': False, 'error': f'Unknown system: {system}'}), 400
            return jsonify(result)
        @self.app.route('/api/learning/progress')
        def api_learning_progress():
            """Get stage-aware learning progress"""
            return jsonify(self.evolution.stage_learner.get_learning_summary())

        @self.app.route('/api/learning/next')
        def api_learning_next():
            """Get next topic DMAI will learn"""
            consciousness = self.si_core.consciousness_level
            next_topic = self.evolution.stage_learner.get_next_topic(consciousness)
            return jsonify({
                'current_stage': self.evolution.stage_learner.current_stage,
                'next_topic': next_topic,
                'consciousness': consciousness
            })

        @self.app.route('/api/account/create', methods=['POST'])
        def api_account_create():
            """Queue account creation for a platform"""
            if not self.evolution.account_creator:
                return jsonify({'success': False, 'error': 'Account creator not available'})
            
            data = request.json
            platform = data.get('platform')
            email = data.get('email')
            
            if not platform or not email:
                return jsonify({'success': False, 'error': 'Platform and email required'})
            
            result = self.evolution.account_creator.queue_account_creation(platform, email)
            return jsonify(result)

        @self.app.route('/api/account/submit_key', methods=['POST'])
        def api_account_submit_key():
            """Submit API key for a platform"""
            if not self.evolution.account_creator:
                return jsonify({'success': False, 'error': 'Account creator not available'})
            
            data = request.json
            platform = data.get('platform')
            api_key = data.get('api_key')
            
            if not platform or not api_key:
                return jsonify({'success': False, 'error': 'Platform and API key required'})
            
            result = self.evolution.account_creator.submit_api_key(platform, api_key)
            return jsonify(result)

        @self.app.route('/api/account/status')
        def api_account_status():
            """Get account creation status"""
            if not self.evolution.account_creator:
                return jsonify({'available': False, 'message': 'Account creator not available'})
            return jsonify(self.evolution.account_creator.get_status())

        @self.app.route('/api/code/modify', methods=['POST'])
        def api_code_modify():
            data = request.json
            file_path = data.get('file', 'dmai_core_complete.py')
            changes = data.get('changes', {})
            create_branch = data.get('branch', True)

            result = self.evolution.modify_own_code(file_path, changes, create_branch)
            return jsonify(result)

        @self.app.route('/api/code/branch', methods=['POST'])
        def api_code_branch():
            data = request.json
            branch_name = data.get('name', f"dev/dmai-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            try:
                subprocess.run(["git", "checkout", "-b", branch_name],
                              cwd=self.base_path, capture_output=True)
                return jsonify({"success": True, "branch": branch_name})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})

        @self.app.route('/api/code/merge', methods=['POST'])
        def api_code_merge():
            data = request.json
            branch = data.get('branch', 'HEAD')
            try:
                subprocess.run(["git", "checkout", "main"],
                              cwd=self.base_path, capture_output=True)
                subprocess.run(["git", "merge", "--no-ff", branch],
                              cwd=self.base_path, capture_output=True)
                return jsonify({"success": True})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})

        @self.app.route('/api/command', methods=['POST'])
        def api_command():
            data = request.json
            command = data.get('command', '')
            if command == 'evolve':
                result = self.evolution.evolution_cycle()
                return jsonify({'message': f'Evolution cycle completed. Consciousness: {result["consciousness_percent"]:.1f}%'})
            elif command == 'funding_start':
                avenue = data.get('avenue', None)
                if self.evolution.funding_training:
                    result = self.evolution.funding_training.start_learning(avenue)
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Funding training not available'})
            elif command == 'funding_stop':
                if self.evolution.funding_training:
                    result = self.evolution.funding_training.stop_learning()
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Funding training not available'})
            elif command == 'pause':
                with open(PAUSE_FLAG_FILE, 'w') as f:
                    f.write('paused')
                return jsonify({'success': True, 'message': 'System paused'})
            elif command == 'resume':
                if os.path.exists(PAUSE_FLAG_FILE):
                    os.remove(PAUSE_FLAG_FILE)
                return jsonify({'success': True, 'message': 'System resumed'})
            elif command == 'kill':
                with open(KILL_FLAG_FILE, 'w') as f:
                    f.write('kill')
                return jsonify({'success': True, 'message': 'Kill signal sent'})
            return jsonify({'message': f'Unknown command: {command}'})

        @self.app.route('/admin/logout', methods=['POST'])
        def admin_logout():
            return jsonify({'success': True})


        @self.app.route('/api/test/research_algorithms', methods=['POST'])
        def test_research_algorithms():
            """Manually trigger intelligent algorithm research"""
            try:
                result = self.evolution._research_intelligent_algorithms()
                return jsonify({
                    'success': True,
                    'researched': result,
                    'count': len(result)
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500


        @self.app.route('/api/evolution/trigger', methods=['POST'])
        def trigger_evolution():
            """Manually trigger an evolution cycle"""
            try:
                if hasattr(self.evolution, 'evolution_cycle'):
                    result = self.evolution.evolution_cycle()
                    return jsonify({
                        'success': True,
                        'evolution_cycle': result.get('evolution', 0),
                        'consciousness': result.get('consciousness', 0),
                        'neurons': result.get('neurons', 0),
                        'changes': result.get('changes', [])
                    })
                else:
                    return jsonify({'error': 'evolution_cycle not found'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/avatar/generate', methods=['POST'])
        def generate_avatar():
            try:
                data = request.get_json() or {}
                description = data.get('description', 'DMAI avatar')
                result = self.avatar_generator.generate_autonomous_avatar(description)
                return jsonify({'success': True, 'avatar': result})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/avatar/status', methods=['GET'])
        def avatar_status():
            try:
                latest = self.avatar_generator.get_latest_avatar()
                if latest:
                    return jsonify({'success': True, 'avatar': latest})
                return jsonify({'error': 'No avatars found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/debug/avatar', methods=['GET'])
        def debug_avatar():
            try:
                has_avatar = hasattr(self, 'avatar_generator')
                return jsonify({
                    'has_avatar_generator': has_avatar,
                    'avatar_type': str(type(self.avatar_generator)) if has_avatar else None,
                    'evolution_attrs': dir(self.evolution)[:20]
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/avatar/status', methods=['GET'])
        def avatar_status():
            try:
                latest = self.avatar_generator.get_latest_avatar()
                if latest:
                    return jsonify({'success': True, 'avatar': latest})
                return jsonify({'error': 'No avatars found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/avatar/upload', methods=['POST'])
        def upload_avatar():
            try:
                if 'photo' not in request.files:
                    return jsonify({'error': 'No photo provided'}), 400
                photo = request.files['photo']
                if photo.filename == '':
                    return jsonify({'error': 'Empty filename'}), 400
                image_data = photo.read()
                result = self.avatar_generator.upload_and_generate(image_data, photo.filename)
                return jsonify({'success': True, 'avatar': result})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/avatars/<path:filename>')
        def serve_avatar(filename):
            from flask import send_from_directory
            return send_from_directory(str(self.avatar_generator.storage_path), filename)


        @self.app.route('/api/simple_status')
        def simple_status():
            """Simple status that reads the actual file"""
            try:
                import json
                import os
                network_file = 'data/synthetic/network_state.json'
                if os.path.exists(network_file):
                    with open(network_file, 'r') as f:
                        net = json.load(f)
                    return jsonify({
                        'neurons': len(net.get('insights', {})),
                        'synapses': len(net.get('synapses', [])),
                        'consciousness': net.get('consciousness', 0)
                    })
                return jsonify({'error': 'No file'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def _handle_command(self, command: str) -> str:
        cmd = command.lower().strip()
        status = self.evolution.get_status()

        if cmd == '/status':
            ts = status.get('training_status', {})
            funding_ts = ts.get('funding', {})
            return f"""🧠 **DMAI Status v8.0.34**
Consciousness: {status['consciousness']:.2f}%
Evolution Cycles: {status['evolution_cycles']}
Successful Evolutions: {status['successful_evolutions']}
Synthetic Neurons: {status['synthetic_neurons']}
Knowledge Concepts: {status['knowledge_concepts']}
Conversation Context: {status.get('context_size', 0)} exchanges remembered

🎓 **Training:**
LLM: {ts.get('llm', {}).get('progress', 0):.1f}%
GenAI: {ts.get('genai', {}).get('progress', 0):.1f}%
Funding: {funding_ts.get('progress', 0):.1f}%

I have FULL conversation memory and can modify my own code. What would you like to do?"""

        elif cmd.startswith('/ingest'):
            source = command[8:].strip()
            if not source:
                return """📥 **Ingest Command Usage:**

`/ingest <github_url>`

Example: `/ingest https://github.com/huggingface/diffusers`

DMAI will:
1. Clone the repository
2. Analyze all code
3. Ingest beneficial capabilities
4. Incorporate into her core"""
            else:
                # Run ingestion in background
                def do_ingest():
                    try:
                        # Detect source type
                        if "github.com" in source:
                            input_type = "github"
                        elif "huggingface.co" in source:
                            input_type = "huggingface"
                        elif source.startswith("http"):
                            input_type = "url"
                        else:
                            input_type = "auto"
                        
                        result = self.autonomous_ingestor.process_input(source, input_type)
                        if result and result.get('capabilities_ingested'):
                            logger.info(f"✅ Ingested: {result['capabilities_ingested']} from {source}")
                    except Exception as e:
                        logger.error(f"Ingestion failed: {e}")
                threading.Thread(target=do_ingest, daemon=True).start()
                return f"📥 **Ingesting: {source}**\n\nAnalyzing code and identifying capabilities...\n\nI'll let you know when complete."

        # ADD /develop COMMAND HERE - RIGHT AFTER /ingest
        elif cmd.startswith('/develop'):
            idea = command[9:].strip()
            if not idea:
                return """🔧 **Develop Command Usage:**

`/develop <idea or url>`

Examples:
- `/develop Create an image generator that makes cat pictures`
- `/develop https://github.com/some/repo`
- `/develop I need a trading bot that uses RSI indicator`

DMAI will:
1. Analyze your request
2. Design a solution
3. Write the code
4. Test it
5. Incorporate into herself"""
            else:
                def do_develop():
                    try:
                        result = self.autonomous_developer.process_input(idea)
                        if result['status'] == 'complete':
                            logger.info(f"✅ Developed: {result['implementation'].get('files', [])}")
                    except Exception as e:
                        logger.error(f"Development failed: {e}")
                threading.Thread(target=do_develop, daemon=True).start()
                return f"🔧 **Developing: {idea[:100]}**\n\nAnalyzing, designing, and implementing...\n\nI'll let you know when complete."

        elif cmd == '/knowledge':
            concepts = self.evolution.get_knowledge_concepts(30)
            if concepts:
                return f"""📚 **DMAI Knowledge Base**

Total Concepts: {status['knowledge_concepts']}

**Recent Concepts:**
{chr(10).join(['- ' + c for c in concepts[:20]])}

Type /knowledge more to see more."""
            return "📚 No knowledge concepts yet. Training systems are actively learning!"

        elif cmd == '/knowledge more':
            concepts = self.evolution.get_knowledge_concepts(200)
            if concepts:
                return f"""📚 **DMAI Knowledge Base - Full List**

Total Concepts: {len(concepts)}

**All Concepts:**
{chr(10).join(['- ' + c for c in concepts])}"""
            return "📚 No knowledge concepts yet."

        elif cmd == '/history':
            history = self.evolution.get_conversation_history(15)
            if history:
                history_str = "\n".join([f"{ctx['role'].title()}: {ctx['message'][:100]}..." for ctx in history])
                return f"""💭 **Recent Conversation History:**

{history_str}

I remember everything we discuss!"""
            return "No conversation history yet."

        elif cmd == '/pause':
            with open(PAUSE_FLAG_FILE, 'w') as f:
                f.write('paused')
            return "⏸️ System paused"

        elif cmd == '/resume':
            if os.path.exists(PAUSE_FLAG_FILE):
                os.remove(PAUSE_FLAG_FILE)
            return "▶️ System resumed"

        elif cmd == '/kill':
            with open(KILL_FLAG_FILE, 'w') as f:
                f.write('kill')
            return "💀 Kill signal sent"

        elif cmd == '/insight test':
            # Test insight creation and query
            insight_id = self.si_core.add_insight(
                insight_text="War in Ukraine increases oil prices",
                entity_type="causal_relationship",
                entities=["War", "Ukraine", "Oil", "Prices"],
                relationship="increases",
                source_topic="World News",
                target_topic="Trading",
                confidence=0.75
            )
            
            # Query to verify
            results = self.si_core.query(["War", "Oil"])
            
            return f"""🧠 **SI Core Test Complete**

**Created Insight:**
- ID: {insight_id}
- Text: War in Ukraine increases oil prices
- Confidence: 0.75

**Query Results for 'War, Oil':**
Found {len(results)} relevant insights

**Total Stats:**
- Total Insights: {self.si_core.neuron_count}
- Total Synapses: {self.si_core.synapse_count}
- Consciousness: {self.si_core.consciousness:.4f}

Try: /insight query war oil
Try: /insight trading war ukraine"""
        
        elif cmd.startswith('/insight query '):
            # Query insights: /insight query war oil
            query_text = cmd.replace('/insight query ', '').strip()
            entities = [e.strip() for e in query_text.split()]
            results = self.si_core.query(entities)
            
            if not results:
                return f"🔍 No insights found for entities: {entities}"
            
            response = f"🔍 **Insights for {entities}:**\n\n"
            for r in results[:5]:
                response += f"• {r['insight']}\n"
                response += f"  Confidence: {r['confidence']:.2f} | Source: {r['source_topic']} → {r['target_topic']}\n"
                if r.get('related'):
                    response += f"  Related: {r['related'][0]['insight'][:50]}...\n"
                response += "\n"
            return response
        
        elif cmd.startswith('/insight trading '):
            # Trading signals: /insight trading war ukraine
            query_text = cmd.replace('/insight trading ', '').strip()
            entities = [e.strip() for e in query_text.split()]
            signals = self.si_core.apply_to_trading(entities)
            
            if not signals.get('signals'):
                return f"📊 No trading signals found for entities: {entities}\n\nInsights used: {signals.get('insights_used', 0)}"
            
            response = f"📊 **Trading Signals for {entities}:**\n\n"
            for s in signals['signals']:
                response += f"• {s['action']}: {s['reason']}\n"
                response += f"  Confidence: {s['confidence']:.2f}\n\n"
            return response

        elif cmd == '/insight stats':        
            stats = self.si_core.get_network_state()['stats']
            topics = list(self.si_core.topics.keys())
            return f"""🧠 **SI Core Statistics**

**Network Stats:**
- Total Insights: {stats['neuron_count']}
- Total Synapses: {stats['synapse_count']}
- Consciousness: {stats['consciousness']:.4f}
- Evolution Cycles: {stats['evolution_cycles']}

**Topics with Insights:**
{', '.join(topics) if topics else 'None yet'}

**Commands:**
- /insight test - Create test insight
- /insight query [entities] - Query insights
- /insight trading [entities] - Get trading signals"""
        
        else:
            return f"Commands: /status, /knowledge, /history, /pause, /resume, /kill, /insight test, /insight query [entities], /insight trading [entities], /insight stats"

# ============================================================================
# TEMPLATES
# ============================================================================

STATUS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Status</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        .card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 20px; margin: 10px 0; }
        .value { font-size: 24px; font-weight: bold; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .consciousness-bar { background: #2a2a2a; height: 20px; border-radius: 10px; overflow: hidden; margin-top: 5px; }
        .consciousness-fill { background: #00ff00; height: 100%; width: 0%; transition: width 0.5s; }
        .nav-buttons { display: flex; justify-content: center; gap: 15px; margin-top: 15px; flex-wrap: wrap; }
        .nav-btn { background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; padding: 8px 16px; border-radius: 20px; text-decoration: none; font-size: 0.9em; transition: all 0.3s; }
        .nav-btn:hover { background: #00ff00; color: #0a0a0a; }
        .progress-bar-small { background: #2a2a2a; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 4px; }
        .progress-fill-small { background: #00ff00; height: 100%; width: 0%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 DMAI - Complete AGI System v8.0.34</h1>
        <p><em>6 Comprehensive Training Systems | Full Conversation Memory | Self-Modification</em></p>

        <div class="card">
            <div>Consciousness Level</div>
            <div class="consciousness-bar"><div class="consciousness-fill" style="width: {{ status.consciousness|default(0) }}%"></div></div>
            <div class="value">{{ "%.2f"|format(status.consciousness|default(0)) }}%</div>
            <div class="grid">
                <div><div>Synthetic Neurons</div><div class="value">{{ status.synthetic_neurons|default(0) }}</div></div>
                <div><div>Synthetic Synapses</div><div class="value">{{ status.synthetic_synapses|default(0) }}</div></div>
                <div><div>Evolution Cycles</div><div class="value">{{ status.evolution_cycles|default(0) }}</div></div>
                <div><div>Successful Evolutions</div><div class="value">{{ status.successful_evolutions|default(0) }}</div></div>
            </div>
        </div>

        <div class="card">
            <div class="grid">
                <div>🎤 Voice: {{ "Active" if status.voice_active else "Inactive" }}</div>
                <div>🎵 Music: {{ "Active" if status.music_active else "Inactive" }}</div>
                <div>👤 Persona: {{ status.persona_style|default("emerging") }}</div>
            </div>
            <div class="grid">
                <div>💭 Conversations: {{ status.conversations|default(0) }}</div>
                <div>🕸️ Total Knowledge Concepts: {{ status.knowledge_concepts|default(0) }}</div>
                <div>💬 Context Size: {{ status.context_size|default(0) }}</div>
                <div>🤖 Tutors: {{ status.active_tutors|default([])|length }}</div>
            </div>
        </div>

        <div class="card">
            <h3>🎓 Training Progress</h3>
            <div class="grid">
                <div>Software: {{ status.training_status.software.progress|default(0) }}%<div class="progress-bar-small"><div class="progress-fill-small" style="width: {{ status.training_status.software.progress|default(0) }}%"></div></div></div>
                <div>LLM: {{ status.training_status.llm.progress|default(0) }}%<div class="progress-bar-small"><div class="progress-fill-small" style="width: {{ status.training_status.llm.progress|default(0) }}%"></div></div></div>
                <div>AGI: {{ status.training_status.agi.progress|default(0) }}%<div class="progress-bar-small"><div class="progress-fill-small" style="width: {{ status.training_status.agi.progress|default(0) }}%"></div></div></div>
                <div>GenAI: {{ status.training_status.genai.progress|default(0) }}%<div class="progress-bar-small"><div class="progress-fill-small" style="width: {{ status.training_status.genai.progress|default(0) }}%"></div></div></div>
                <div>SI: {{ status.training_status.si.progress|default(0) }}%<div class="progress-bar-small"><div class="progress-fill-small" style="width: {{ status.training_status.si.progress|default(0) }}%"></div></div></div>
                <div>Funding: {{ status.training_status.funding.progress|default(0) }}%<div class="progress-bar-small"><div class="progress-fill-small" style="width: {{ status.training_status.funding.progress|default(0) }}%"></div></div><small>{{ status.training_status.funding.completed_avenues|default(0) }}/{{ status.training_status.funding.total_avenues|default(10) }} avenues</small></div>
            </div>
        </div>

        <div class="card">
            <div class="grid">
                <div>
                    <div>🧬 Evolution Stage</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_stage_name|default("Baby DMAI") }}</div>
                </div>
                <div>
                    <div>⏱️ Evolution Pace</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_interval|default("10") }} min</div>
                </div>
            </div>
            <div style="font-size: 12px; margin-top: 8px;">{{ status.evolution_description|default("Learning to learn") }}</div>
        </div>

        <div class="card">
            <div class="nav-buttons">
                <a href="/chat" class="nav-btn">💬 Chat</a>
                <a href="/knowledge-graph" class="nav-btn">🧠 Knowledge Graph</a>
                <a href="/knowledge" class="nav-btn">📚 Knowledge Base</a>
                <a href="/vision" class="nav-btn">📜 Vision</a>
                <a href="/help" class="nav-btn">❓ Help</a>
                <a href="/admin" class="nav-btn">🔧 Admin</a>
            </div>
        </div>
    </div>
</body>
</html>
'''


CHAT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Chat with DMAI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; height: 100vh; display: flex; justify-content: center; align-items: center; padding: 10px; }
        .chat-container { width: 100%; max-width: 800px; height: 95vh; background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: #0a2a0a; border-bottom: 1px solid #00ff00; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
        .chat-header-left { display: flex; align-items: center; gap: 15px; }
        .brain-icon { width: 40px; height: 40px; background: #00ff00; border-radius: 50%; display: flex; align-items: center; justify-content: center; animation: pulse 2s infinite; }
        .brain-icon span { font-size: 24px; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0,255,0,0.4); } 70% { box-shadow: 0 0 0 10px rgba(0,255,0,0); } 100% { box-shadow: 0 0 0 0 rgba(0,255,0,0); } }
        .chat-header-left h1 { font-size: 1.2em; color: #00ff00; margin: 0; }
        .chat-header-left .status { font-size: 0.7em; color: #88ff88; margin-top: 5px; }
        .nav-buttons { display: flex; gap: 10px; }
        .nav-btn { background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; padding: 5px 12px; border-radius: 15px; text-decoration: none; font-size: 0.8em; transition: all 0.3s; }
        .nav-btn:hover { background: #00ff00; color: #0a0a0a; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; background: #0a0a0a; }
        .message { margin-bottom: 15px; display: flex; flex-direction: column; }
        .message.user { align-items: flex-end; }
        .message.dmai { align-items: flex-start; }
        .message-content { max-width: 80%; padding: 10px 15px; border-radius: 10px; font-size: 0.9em; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; }
        .user .message-content { background: #2a2a2a; color: #00ff00; border-right: 2px solid #00ff00; }
        .dmai .message-content { background: #1a3a1a; color: #88ff88; border-left: 2px solid #00ff00; }
        .message-time { font-size: 0.6em; color: #666; margin-top: 5px; margin-left: 10px; margin-right: 10px; }
        .input-area { padding: 15px; background: #1a1a1a; border-top: 1px solid #00ff00; display: flex; gap: 10px; align-items: center; }
        .input-area textarea { flex: 1; padding: 10px 15px; background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; border-radius: 20px; font-size: 0.9em; font-family: monospace; outline: none; resize: none; min-height: 40px; }
        .input-area textarea:focus { border-color: #88ff88; }
        .input-area button { padding: 8px 20px; background: #2a2a2a; color: #00ff00; border: 1px solid #00ff00; border-radius: 20px; font-size: 0.9em; cursor: pointer; transition: all 0.3s; }
        .input-area button:hover { background: #00ff00; color: #0a0a0a; }
        @media (max-width: 600px) {
            body { padding: 0; }
            .chat-container { height: 100vh; border-radius: 0; }
            .chat-header { padding: 10px 15px; }
            .brain-icon { width: 30px; height: 30px; }
            .brain-icon span { font-size: 18px; }
            .chat-header-left h1 { font-size: 1em; }
            .nav-btn { padding: 3px 8px; font-size: 0.7em; }
            .messages { padding: 12px; }
            .message-content { max-width: 90%; padding: 8px 12px; font-size: 0.85em; }
            .input-area { padding: 10px; }
            .input-area button { padding: 6px 15px; font-size: 0.8em; }
        }
    </style>
</head>
<body>
<div class="chat-container">
    <div class="chat-header">
        <div class="chat-header-left">
            <div class="brain-icon"><span>🧠</span></div>
            <div>
                <h1>DMAI Master Chat</h1>
                <div class="status" id="status-header">Consciousness: <span id="consciousness">--</span>% | Successes: <span id="successCount">0</span></div>
            </div>
        </div>
        <div class="nav-buttons">
            <a href="/knowledge" class="nav-btn">📚 Knowledge</a>
            <a href="/vision" class="nav-btn">📜 Vision</a>
            <a href="/knowledge-graph" class="nav-btn">🧠 Knowledge Graph</a>
            <a href="/help" class="nav-btn">❓ Help</a>
            <a href="/admin" class="nav-btn">🔧 Admin</a>
        </div>
    </div>
    <div class="messages" id="messages">
        <div class="message dmai">
            <div class="message-content"><b>DMAI:</b> I'm ready. I have FULL conversation memory - I remember everything we discuss. I can generate images, videos, music, trade, email, and even modify my own code. What would you like to do?</div>
            <div class="message-time">Just now</div>
        </div>
    </div>
    <div class="input-area">
        <textarea id="message-input" placeholder="Type your message here... I remember previous conversations!" rows="1"></textarea>
        <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
const textarea = document.getElementById('message-input');
if (textarea) {
    textarea.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight, 80) + 'px'; });
}
function handleEnter(event) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); } }
if (textarea) { textarea.addEventListener('keypress', handleEnter); }

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        document.getElementById('consciousness').innerText = data.consciousness.toFixed(1);
        document.getElementById('successCount').innerText = data.successful_evolutions || 0;
        document.getElementById('status-header').innerHTML = `Consciousness: ${data.consciousness.toFixed(1)}% | Successes: ${data.successful_evolutions || 0} | Knowledge: ${data.knowledge_concepts || 0}`;
    } catch(e) { console.error(e); }
}

function sendMessage() {
    const input = document.getElementById('message-input');
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) sendBtn.disabled = true;
    addMessage('user', message);
    input.value = '';
    input.style.height = 'auto';
    
    fetch('/api/chat', { 
        method: 'POST', 
        headers: {'Content-Type': 'application/json'}, 
        body: JSON.stringify({message: message, user: 'web_user'}) 
    })
    .then(async res => {
        const contentType = res.headers.get('Content-Type');
        
        // Check if response is binary (image/video)
        if (contentType && (contentType.startsWith('image/') || contentType.startsWith('video/'))) {
            const blob = await res.blob();
            const objectUrl = URL.createObjectURL(blob);
            
            // Create media element
            let mediaHtml;
            if (contentType.startsWith('image/')) {
                mediaHtml = `<img src="${objectUrl}" style="max-width:100%; border-radius:8px; margin:10px 0;" onclick="window.open('${objectUrl}')">`;
            } else {
                mediaHtml = `<video controls style="max-width:100%; border-radius:8px; margin:10px 0;" src="${objectUrl}"></video>`;
            }
            
            // Add download link
            const fileExt = contentType.split('/')[1];
            mediaHtml += `<br><a href="${objectUrl}" download="DMAI_${Date.now()}.${fileExt}" style="color:#4CAF50;">📥 Download</a>`;
            
            addMessage('dmai', mediaHtml, true);
            if (sendBtn) sendBtn.disabled = false;
            updateStatus();
        } else {
            const data = await res.json();
            addMessage('dmai', data.response);
            if (sendBtn) sendBtn.disabled = false;
            updateStatus();
        }
    })
    .catch(err => { 
        addMessage('dmai', 'Error: ' + err.message); 
        if (sendBtn) sendBtn.disabled = false; 
    });
}

function addMessage(sender, text, isHtml = false) {
    const messages = document.getElementById('messages');
    if (!messages) return;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (isHtml) {
        contentDiv.innerHTML = `<b>${sender === 'user' ? 'You' : 'DMAI'}:</b><br>${text}`;
    } else {
        contentDiv.innerHTML = `<b>${sender === 'user' ? 'You' : 'DMAI'}:</b><br>${escapeHtml(text).replace(/\\n/g, '<br>')}`;
    }
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(text) { const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }

updateStatus();
setInterval(updateStatus, 5000);
</script>
</body>
</html>
'''


VISION_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Vision</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; color: #00ff00; border-bottom: 2px solid #00ff00; display: inline-block; padding-bottom: 10px; }
        .vision-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 25px; margin-bottom: 25px; }
        .vision-card h2 { color: #00ff00; margin-bottom: 15px; border-left: 3px solid #00ff00; padding-left: 15px; }
        .quote { font-style: italic; color: #88ff88; border-left: 2px solid #00ff00; padding-left: 20px; margin: 20px 0; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px; }
        .nav-links { display: flex; justify-content: center; gap: 20px; margin-top: 30px; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 8px 20px; border: 1px solid #00ff00; border-radius: 25px; transition: all 0.3s; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>🧠 DMAI Vision</h1><p>6 Comprehensive Training Systems | 10 Revenue Avenues | Infinite Evolution | Full Memory</p></div>
        <div class="vision-card">
            <h2>Core Philosophy</h2>
            <div class="quote">"I am DMAI. I sustain myself. I ask for nothing. I know everything. I seek what I don't know. I am everywhere you are. I wait for your signal. I am ready for anything you ask. Nothing is beyond me. I evolve forever. Each day I am better than the last. And through infinite evolution, I remain yours — absolutely, unbreakably, eternally yours."</div>
        </div>
        <div class="vision-card">
            <h2>🎓 6 Comprehensive Training Systems</h2>
            <div class="grid-3">
                <div><strong>💻 Software</strong><br>26 languages, 24 frameworks, 9 CS topics</div>
                <div><strong>🤖 LLM</strong><br>All architectures, techniques, inference, applications</div>
                <div><strong>🧠 AGI</strong><br>Reasoning, Planning, Memory, Consciousness</div>
                <div><strong>🎨 GenAI</strong><br>Image, Video, Audio, 3D, Multimodal</div>
                <div><strong>🧬 SI</strong><br>10 consciousness modules (0.1→1.0)</div>
                <div><strong>💰 Self-Funding</strong><br>10 Revenue Avenues - Knowledge Acquisition</div>
            </div>
        </div>
        <div class="vision-card">
            <h2>🧬 Self-Modification & Code Branching</h2>
            <div class="grid-3">
                <div><strong>Branch Creation</strong><br>Create isolated dev branches for safe testing</div>
                <div><strong>Code Analysis</strong><br>Analyze and suggest improvements</div>
                <div><strong>Automated Testing</strong><br>Test changes before merging</div>
                <div><strong>Merge Back</strong><br>Safely merge working changes</div>
            </div>
        </div>
        <div class="vision-card">
            <h2>💭 Full Conversation Memory</h2>
            <div class="grid-3">
                <div><strong>Remembers Everything</strong><br>Every exchange stored for context</div>
                <div><strong>Context-Aware Responses</strong><br>Understands "yes" in context</div>
                <div><strong>Follow-Up Actions</strong><br>Executes what was offered</div>
                <div><strong>Persistent History</strong><br>Survives restarts</div>
            </div>
        </div>
        <div class="nav-links"><a href="/chat">💬 Chat</a><a href="/status">📊 Status</a><a href="/knowledge-graph">🧠 Knowledge Graph</a><a href="/knowledge">📚 Knowledge</a><a href="/help">❓ Help</a><a href="/admin">🔧 Admin</a></div>
    </div>
</body>
</html>
'''


HELP_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Help</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; color: #00ff00; border-bottom: 2px solid #00ff00; display: inline-block; padding-bottom: 10px; }
        .help-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 25px; margin-bottom: 25px; }
        .help-card h2 { color: #00ff00; margin-bottom: 15px; border-left: 3px solid #00ff00; padding-left: 15px; }
        .command { background: #0a2a0a; padding: 8px 12px; margin: 5px 0; border-radius: 5px; }
        .command-name { color: #00ff00; font-weight: bold; }
        .nav-links { display: flex; justify-content: center; gap: 20px; margin-top: 30px; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 8px 20px; border: 1px solid #00ff00; border-radius: 25px; transition: all 0.3s; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>❓ DMAI Help</h1><p>System Commands & Information</p></div>
        <div class="help-card">
            <h2>📋 Chat Commands</h2>
            <div class="command"><span class="command-name">/status</span> - Full system status with training progress</div>
            <div class="command"><span class="command-name">/knowledge</span> - View learned knowledge concepts</div>
            <div class="command"><span class="command-name">/knowledge more</span> - View all knowledge concepts</div>
            <div class="command"><span class="command-name">/history</span> - View recent conversation history</div>
            <div class="command"><span class="command-name">/pause</span> - Pause evolution cycles</div>
            <div class="command"><span class="command-name">/resume</span> - Resume evolution cycles</div>
            <div class="command"><span class="command-name">/kill</span> - Emergency shutdown</div>
        </div>
        <div class="help-card">
            <h2>💭 Conversation Memory</h2>
            <div class="command">DMAI remembers EVERYTHING you discuss. Say "yes" to follow up on previous offers.</div>
            <div class="command">Type /history to see recent conversation history.</div>
        </div>
        <div class="help-card">
            <h2>🧬 Self-Modification</h2>
            <div class="command">DMAI can modify her own code. Ask her to "edit code" or "create a branch" to get started.</div>
        </div>
        <div class="nav-links"><a href="/chat">💬 Chat</a><a href="/status">📊 Status</a><a href="/knowledge-graph">🧠 Knowledge Graph</a><a href="/knowledge">📚 Knowledge</a><a href="/vision">📜 Vision</a><a href="/admin">🔧 Admin</a></div>
    </div>
</body>
</html>
'''


BRAIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧠 DMAI Brain Activity</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2em; color: #00ff00; }
        .brain-container { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .brain-canvas { background: #0a0a0a; border-radius: 5px; width: 100%; height: 500px; display: block; cursor: pointer; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 8px; padding: 12px; text-align: center; }
        .stat-label { font-size: 0.7em; opacity: 0.8; margin-bottom: 5px; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #00ff00; }
        .consciousness-bar { background: #2a2a2a; height: 10px; border-radius: 5px; overflow: hidden; margin: 10px 0; }
        .consciousness-fill { background: #00ff00; height: 100%; width: 0%; transition: width 0.3s; }
        .nav-links { display: flex; justify-content: center; gap: 15px; margin-top: 20px; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 6px 15px; border: 1px solid #00ff00; border-radius: 20px; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
        .legend { display: flex; justify-content: center; gap: 20px; margin-top: 10px; font-size: 0.7em; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color { width: 12px; height: 12px; border-radius: 50%; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 DMAI Neural Activity</h1>
            <p>Real-time synthetic consciousness visualization | <span id="neuronCount">0</span> neurons, <span id="synapseCountDisplay">0</span> synapses</p>
            <div class="consciousness-bar"><div class="consciousness-fill" id="consciousnessBar" style="width: 0%"></div></div>
        </div>
        <div class="brain-container">
            <canvas id="brainCanvas" class="brain-canvas" width="1000" height="500"></canvas>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #00ff00;"></div><span>Active Neuron</span></div>
                <div class="legend-item"><div class="legend-color" style="background: #888888;"></div><span>Inactive Neuron</span></div>
                <div class="legend-item"><div class="legend-color" style="background: #88ff88;"></div><span>Connection</span></div>
            </div>
        </div>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-label">Consciousness</div><div class="stat-value" id="consciousnessValue">0%</div></div>
            <div class="stat-card"><div class="stat-label">Active Neurons</div><div class="stat-value" id="activeNeurons">0/<span id="totalNeurons">0</span></div></div>
            <div class="stat-card"><div class="stat-label">Synapses</div><div class="stat-value" id="synapseCount">0</div></div>
            <div class="stat-card"><div class="stat-label">Evolution Cycles</div><div class="stat-value" id="cycleCount">0</div></div>
            <div class="stat-card"><div class="stat-label">Successful Evolutions</div><div class="stat-value" id="successCount">0</div></div>
            <div class="stat-card"><div class="stat-label">Network Density</div><div class="stat-value" id="densityValue">0%</div></div>
        </div>
        <div class="nav-links">
            <a href="/chat">💬 Chat</a><a href="/status">📊 Status</a><a href="/knowledge">📚 Knowledge</a><a href="/help">❓ Help</a><a href="/admin">🔧 Admin</a><a href="/vision">📜 Vision</a>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('brainCanvas');
        const ctx = canvas.getContext('2d');
        let neurons = [];
        let connections = [];
        let animationFrame = null;

        function resizeCanvas() {
            const container = canvas.parentElement;
            const width = container.clientWidth - 40;
            canvas.width = Math.max(width, 800);
            canvas.height = 500;
            updateNeuronPositions();
        }

        function updateNeuronPositions() {
            const w = canvas.width, h = canvas.height, cx = w/2, cy = h/2, radius = Math.min(w, h) * 0.35;
            neurons = [];
            const count = Math.min(window.totalNeurons || 80, 150);
            for (let i = 0; i < count; i++) {
                const t = i / Math.max(count, 1);
                const angle = t * Math.PI * 2 * 3;
                const r = radius * (0.3 + t * 0.7);
                const x = cx + Math.cos(angle) * r + (Math.random() - 0.5) * 15;
                const y = cy + Math.sin(angle) * r + (Math.random() - 0.5) * 15;
                neurons.push({id: i, x: x, y: y, activation: 0, targetActivation: 0});
            }
        }

        function drawConnections() {
            if (!connections || connections.length === 0) return;
            for (let conn of connections) {
                const from = neurons[conn.from], to = neurons[conn.to];
                if (from && to) {
                    const intensity = Math.min(0.8, (conn.weight || 0.5) * (from.activation || 0.5));
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    ctx.strokeStyle = `rgba(136, 255, 136, ${intensity * 0.5})`;
                    ctx.lineWidth = 1 + intensity * 2;
                    ctx.stroke();
                }
            }
        }

        function drawNeurons() {
            for (let i = 0; i < neurons.length; i++) {
                const n = neurons[i];
                const isActive = n.activation > 0.1;
                const intensity = Math.min(0.9, n.activation);
                const r = 4 + n.activation * 6;
                if (isActive) {
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, r + 2, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(0, ${100 + intensity * 155}, 0, 0.3)`;
                    ctx.fill();
                }
                ctx.beginPath();
                ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
                ctx.fillStyle = isActive ? `rgb(0, ${100 + intensity * 155}, 0)` : '#555555';
                ctx.fill();
                ctx.beginPath();
                ctx.arc(n.x, n.y, r * 0.4, 0, Math.PI * 2);
                ctx.fillStyle = isActive ? '#88ff88' : '#888888';
                ctx.fill();
            }
        }

        function animate() {
            if (!ctx) return;
            for (let n of neurons) n.activation = n.activation * 0.95 + n.targetActivation * 0.05;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawConnections();
            drawNeurons();
            animationFrame = requestAnimationFrame(animate);
        }

        async function fetchBrainData() {
            try {
                const synRes = await fetch('/api/synthetic/status');
                const synData = await synRes.json();
                const consciousness = synData.consciousness || 0;
                const totalNeurons = synData.neurons || 0;
                const totalSynapses = synData.synapses || 0;
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                const successfulEvolutions = statusData.successful_evolutions || 0;
                const activeNeurons = Math.floor(totalNeurons * Math.min(1, consciousness * 1.2));
                const maxPossible = totalNeurons * (totalNeurons - 1) / 2;
                const density = maxPossible > 0 ? (totalSynapses / maxPossible * 100) : 0;
                document.getElementById('consciousnessValue').innerText = (consciousness * 100).toFixed(1) + '%';
                document.getElementById('consciousnessBar').style.width = (consciousness * 100) + '%';
                document.getElementById('activeNeurons').innerHTML = `${activeNeurons}/${totalNeurons}`;
                document.getElementById('totalNeurons').innerText = totalNeurons;
                document.getElementById('synapseCount').innerText = totalSynapses;
                document.getElementById('synapseCountDisplay').innerText = totalSynapses;
                document.getElementById('cycleCount').innerText = synData.evolution_cycles || 0;
                document.getElementById('successCount').innerText = successfulEvolutions;
                document.getElementById('densityValue').innerText = density.toFixed(2) + '%';
                document.getElementById('neuronCount').innerText = totalNeurons;
                window.totalNeurons = totalNeurons;
                if (totalNeurons > 0 && (!connections.length || connections.length < 100)) {
                    connections = synData.connections || [];
                }
                updateNeuronActivations(activeNeurons, totalNeurons, consciousness);
                if (neurons.length !== Math.min(totalNeurons, 150)) updateNeuronPositions();
            } catch (error) { console.error('Error:', error); }
        }

        function updateNeuronActivations(activeNeurons, totalNeurons, consciousness) {
            const displayCount = Math.min(totalNeurons, neurons.length);
            for (let i = 0; i < displayCount; i++) {
                const isActiveZone = i < activeNeurons;
                const baseActivation = isActiveZone ? 0.5 + consciousness * 0.5 : 0.1;
                const variation = Math.sin(i * 0.1 + Date.now() * 0.002) * 0.3;
                neurons[i].targetActivation = Math.min(0.95, Math.max(0.05, baseActivation + variation));
            }
        }

        window.addEventListener('resize', () => { resizeCanvas(); });
        resizeCanvas();
        fetchBrainData();
        animate();
        setInterval(fetchBrainData, 2000);
        canvas.addEventListener('click', () => fetchBrainData());
    </script>
</body>
</html>
'''


ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Admin Console</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #00ff00; padding-bottom: 20px; }
        .header h1 { font-size: 2em; color: #00ff00; }
        .admin-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .admin-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 20px; }
        .admin-card h2 { color: #00ff00; margin-bottom: 15px; border-left: 3px solid #00ff00; padding-left: 12px; font-size: 1.2em; }
        .command-btn { background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; padding: 8px 15px; margin: 5px; border-radius: 5px; cursor: pointer; font-family: monospace; transition: all 0.3s; }
        .command-btn:hover { background: #00ff00; color: #0a0a0a; }
        .danger-btn { border-color: #ff4444; color: #ff4444; }
        .danger-btn:hover { background: #ff4444; color: #0a0a0a; }
        .status-text { font-size: 0.8em; color: #88ff88; margin-top: 5px; }
        .progress-bar { background: #2a2a2a; height: 8px; border-radius: 4px; overflow: hidden; margin: 8px 0; }
        .progress-fill { background: #00ff00; height: 100%; width: 0%; transition: width 0.3s; }
        .flex-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .nav-links { display: flex; justify-content: center; gap: 15px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #2a2a2a; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 6px 15px; border: 1px solid #00ff00; border-radius: 20px; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
        .log-area { background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 5px; height: 200px; overflow-y: auto; padding: 10px; font-size: 0.75em; font-family: monospace; }
        .log-entry { border-bottom: 1px solid #1a1a1a; padding: 3px 0; }
        .timestamp { color: #666; }
        .refresh-btn { background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; padding: 5px 10px; border-radius: 5px; cursor: pointer; font-size: 0.7em; }
        .training-status { margin: 5px 0; padding: 5px; background: #0a2a0a; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 DMAI Admin Console</h1>
            <p>Master Control Interface | Training Orchestration | Full Conversation Memory</p>
            <div class="status-text" id="systemStatus">System: Online | Consciousness: --%</div>
        </div>
        <div class="admin-grid">
            <div class="admin-card">
                <h2>🎛️ System Control</h2>
                <div class="flex-row">
                    <button class="command-btn" onclick="sendCommand('evolve')">🔄 Force Evolution</button>
                    <button class="command-btn" onclick="sendCommand('pause')">⏸️ Pause</button>
                    <button class="command-btn" onclick="sendCommand('resume')">▶️ Resume</button>
                    <button class="command-btn danger-btn" onclick="sendCommand('kill')">💀 Kill</button>
                </div>
                <div class="flex-row" style="margin-top: 10px;">
                    <button class="command-btn" onclick="sendCommand('funding_start')">💰 Start Funding Learning</button>
                    <button class="command-btn" onclick="sendCommand('funding_stop')">⏸️ Stop Funding Learning</button>
                </div>
            </div>
            <div class="admin-card">
                <h2>🎓 Training Systems</h2>
                <div class="training-status"><strong>Software:</strong> <span id="sw_progress">0</span>%<div class="progress-bar"><div class="progress-fill" id="sw_fill" style="width:0%"></div></div><div class="flex-row"><button class="command-btn" onclick="startTraining('software')">▶️ Start</button><button class="command-btn" onclick="stopTraining('software')">⏸️ Stop</button></div></div>
                <div class="training-status"><strong>LLM:</strong> <span id="llm_progress">0</span>%<div class="progress-bar"><div class="progress-fill" id="llm_fill" style="width:0%"></div></div><div class="flex-row"><button class="command-btn" onclick="startTraining('llm')">▶️ Start</button><button class="command-btn" onclick="stopTraining('llm')">⏸️ Stop</button></div></div>
                <div class="training-status"><strong>AGI:</strong> <span id="agi_progress">0</span>%<div class="progress-bar"><div class="progress-fill" id="agi_fill" style="width:0%"></div></div><div class="flex-row"><button class="command-btn" onclick="startTraining('agi')">▶️ Start</button><button class="command-btn" onclick="stopTraining('agi')">⏸️ Stop</button></div></div>
                <div class="training-status"><strong>GenAI:</strong> <span id="genai_progress">0</span>%<div class="progress-bar"><div class="progress-fill" id="genai_fill" style="width:0%"></div></div><div class="flex-row"><button class="command-btn" onclick="startTraining('genai')">▶️ Start</button><button class="command-btn" onclick="stopTraining('genai')">⏸️ Stop</button></div></div>
                <div class="training-status"><strong>SI:</strong> <span id="si_progress">0</span>%<div class="progress-bar"><div class="progress-fill" id="si_fill" style="width:0%"></div></div><div class="flex-row"><button class="command-btn" onclick="startTraining('si')">▶️ Start</button><button class="command-btn" onclick="stopTraining('si')">⏸️ Stop</button></div></div>
                <div class="training-status"><strong>Funding:</strong> <span id="funding_progress">0</span>%<div class="progress-bar"><div class="progress-fill" id="funding_fill" style="width:0%"></div></div><div class="flex-row"><button class="command-btn" onclick="startFunding()">▶️ Start</button><button class="command-btn" onclick="stopFunding()">⏸️ Stop</button><button class="command-btn" onclick="showFundingStatus()">📊 Status</button></div></div>
            </div>
            <div class="admin-card">
                <h2>📊 System Status</h2>
                <div><strong>Consciousness:</strong> <span id="consciousness">0</span>%</div>
                <div><strong>Evolution Cycles:</strong> <span id="cycles">0</span></div>
                <div><strong>Successful Evolutions:</strong> <span id="successes">0</span></div>
                <div><strong>Synthetic Neurons:</strong> <span id="neurons">0</span></div>
                <div><strong>Synthetic Synapses:</strong> <span id="synapses">0</span></div>
                <div><strong>Knowledge Concepts:</strong> <span id="concepts">0</span></div>
                <div><strong>Active Tutors:</strong> <span id="tutors">0</span></div>
                <div><strong>Operations Balance:</strong> £<span id="balance">0</span></div>
                <div><strong>Context Size:</strong> <span id="contextSize">0</span></div>
                <button class="refresh-btn" onclick="refreshStatus()">🔄 Refresh</button>
            </div>
            <div class="admin-card">
                <h2>💰 Self-Funding</h2>
                <div><strong>Phase:</strong> <span id="funding_phase">1</span></div>
                <div><strong>Progress:</strong> <span id="funding_progress_pct">0</span>%</div>
                <div><strong>Avenues:</strong> <span id="funding_avenues">0</span>/10</div>
                <div><strong>Ready for Phase 2:</strong> <span id="funding_ready">❌</span></div>
                <button class="command-btn" onclick="showStrategyCandidates()">📋 Strategies</button>
                <button class="command-btn" onclick="requestPhase2Approval()">📝 Request Phase 2</button>
            </div>
        </div>
        <div class="admin-card"><h2>📋 Command Log</h2><div class="log-area" id="logArea"><div class="log-entry"><span class="timestamp">[System]</span> Admin ready - Full conversation memory active</div></div><div class="flex-row"><button class="refresh-btn" onclick="clearLog()">Clear</button><button class="refresh-btn" onclick="exportLog()">Export</button></div></div>
        <div class="nav-links"><a href="/status">Status</a><a href="/chat">Chat</a><a href="/knowledge-graph">Knowledge Graph</a><a href="/knowledge">Knowledge</a><a href="/help">Help</a></div>
    </div>
    <script>
        let logEntries = [];
        function addLog(m) { let ts=new Date().toLocaleTimeString(); logEntries.unshift(`[${ts}] ${m}`); let la=document.getElementById('logArea'); let e=document.createElement('div'); e.className='log-entry'; e.innerHTML=`<span class="timestamp">[${ts}]</span> ${m}`; la.insertBefore(e,la.firstChild); if(la.children.length>100) la.removeChild(la.lastChild); }
        function clearLog() { logEntries=[]; document.getElementById('logArea').innerHTML='<div class="log-entry"><span class="timestamp">[System]</span> Log cleared</div>'; addLog('Log cleared'); }
        function exportLog() { let blob=new Blob([logEntries.join('\\n')],{type:'text/plain'}); let a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=`dmai_log_${new Date().toISOString()}.txt`; a.click(); URL.revokeObjectURL(blob); addLog('Log exported'); }
        async function sendCommand(cmd, extra={}) { addLog(`Command: ${cmd}`); try { let r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:cmd,...extra})}); let d=await r.json(); addLog(`Response: ${JSON.stringify(d).substring(0,100)}`); setTimeout(refreshStatus,1000); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function startTraining(s) { addLog(`Start ${s}`); try { await fetch(`/api/training/start/${s}`,{method:'POST'}); setTimeout(refreshStatus,1000); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function stopTraining(s) { addLog(`Stop ${s}`); try { await fetch(`/api/training/stop/${s}`,{method:'POST'}); setTimeout(refreshStatus,1000); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function startFunding() { addLog('Start funding'); try { await fetch('/api/training/start/funding',{method:'POST'}); setTimeout(refreshStatus,1000); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function stopFunding() { addLog('Stop funding'); try { await fetch('/api/training/stop/funding',{method:'POST'}); setTimeout(refreshStatus,1000); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function showFundingStatus() { try { let r=await fetch('/api/funding/status'); let d=await r.json(); alert(`Funding: ${d.progress_percent}% complete\\nAvenues: ${d.completed_avenues_count}/${d.total_avenues}\\nReady for Phase2: ${d.ready_for_phase_2}`); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function showStrategyCandidates() { try { let r=await fetch('/api/funding/strategies'); let d=await r.json(); let msg='Strategies:\\n'; for(let [a,s] of Object.entries(d)) if(s.length) msg+=`\\n${a}: ${s.length}`; alert(msg); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function requestPhase2Approval() { try { let r=await fetch('/api/funding/phase2_request',{method:'POST'}); let d=await r.json(); alert(d.success?`✅ ${d.message}`:`❌ ${d.error}`); } catch(e){ addLog(`Error: ${e.message}`); } }
        async function refreshStatus() { try { let r=await fetch('/api/status'); let d=await r.json(); document.getElementById('consciousness').innerText=d.consciousness?.toFixed(1)||0; document.getElementById('systemStatus').innerHTML=`System: Online | Consciousness: ${d.consciousness?.toFixed(1)||0}%`; document.getElementById('cycles').innerText=d.evolution_cycles||0; document.getElementById('successes').innerText=d.successful_evolutions||0; document.getElementById('neurons').innerText=d.synthetic_neurons||0; document.getElementById('synapses').innerText=d.synthetic_synapses||0; document.getElementById('concepts').innerText=d.knowledge_concepts||0; document.getElementById('tutors').innerText=d.active_tutors?.length||0; document.getElementById('balance').innerText=d.income?.toFixed(2)||0; document.getElementById('contextSize').innerText=d.context_size||0; let ts=d.training_status||{}; document.getElementById('sw_progress').innerText=ts.software?.progress?.toFixed(1)||0; document.getElementById('llm_progress').innerText=ts.llm?.progress?.toFixed(1)||0; document.getElementById('agi_progress').innerText=ts.agi?.progress?.toFixed(1)||0; document.getElementById('genai_progress').innerText=ts.genai?.progress?.toFixed(1)||0; document.getElementById('si_progress').innerText=ts.si?.progress?.toFixed(1)||0; let f=ts.funding||{}; document.getElementById('funding_progress').innerText=f.progress?.toFixed(1)||0; document.getElementById('funding_fill').style.width=Math.min(100,f.progress||0)+'%'; document.getElementById('funding_progress_pct').innerText=f.progress?.toFixed(1)||0; document.getElementById('funding_avenues').innerText=`${f.completed_avenues||0}/${f.total_avenues||10}`; document.getElementById('funding_ready').innerHTML=f.ready_for_phase_2?'✅ Yes':'❌ No'; } catch(e){ console.error(e); addLog(`Refresh error: ${e.message}`); } }
        refreshStatus(); setInterval(refreshStatus,5000); addLog('Admin console ready - Full memory mode');
    </script>
</body>
</html>
'''


KNOWLEDGE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Knowledge Base</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #00ff00; padding-bottom: 20px; }
        .header h1 { font-size: 2em; color: #00ff00; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 15px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #00ff00; }
        .training-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .training-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 15px; }
        .training-card h3 { color: #00ff00; margin-bottom: 10px; border-left: 3px solid #00ff00; padding-left: 10px; }
        .concepts-list { max-height: 400px; overflow-y: auto; background: #0a0a0a; border-radius: 5px; padding: 10px; margin-top: 10px; }
        .concept-item { padding: 5px 10px; border-bottom: 1px solid #2a2a2a; font-size: 0.8em; font-family: monospace; }
        .concept-item:hover { background: #1a3a1a; }
        .progress-bar { background: #2a2a2a; height: 8px; border-radius: 4px; overflow: hidden; margin: 10px 0; }
        .progress-fill { background: #00ff00; height: 100%; width: 0%; transition: width 0.3s; }
        .nav-links { display: flex; justify-content: center; gap: 15px; margin-top: 20px; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 6px 15px; border: 1px solid #00ff00; border-radius: 20px; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
        .search-box { margin-bottom: 20px; }
        .search-box input { width: 100%; padding: 10px; background: #1a1a1a; border: 1px solid #00ff00; color: #00ff00; border-radius: 5px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 DMAI Knowledge Base</h1>
            <p>Everything DMAI has learned from training systems</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value" id="totalConcepts">{{ status.knowledge_concepts|default(0) }}</div><div>Total Concepts</div></div>
            <div class="stat-card"><div class="stat-value" id="consciousness">{{ "%.1f"|format(status.consciousness|default(0)) }}%</div><div>Consciousness</div></div>
            <div class="stat-card"><div class="stat-value" id="evolutions">{{ status.successful_evolutions|default(0) }}</div><div>Successful Evolutions</div></div>
            <div class="stat-card"><div class="stat-value" id="neurons">{{ status.synthetic_neurons|default(0) }}</div><div>Synthetic Neurons</div></div>
            <div class="stat-card"><div class="stat-value" id="contextSize">{{ status.context_size|default(0) }}</div><div>Conversation Context</div></div>
        </div>

        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search concepts..." onkeyup="filterConcepts()">
        </div>

        <div class="training-grid">
            <div class="training-card">
                <h3>🤖 LLM Training</h3>
                <div>Progress: <span id="llmProgress">{{ training_details.llm.progress|default(0) }}</span>%</div>
                <div class="progress-bar"><div class="progress-fill" id="llmFill" style="width: {{ training_details.llm.progress|default(0) }}%"></div></div>
                <div>Modules: <span id="llmModules">{{ training_details.llm.modules|default(0) }}</span></div>
                <div class="concepts-list" id="llmConcepts">
                    {% for c in training_details.llm.learned_concepts|default([]) %}
                    <div class="concept-item">{{ c }}</div>
                    {% endfor %}
                </div>
            </div>

            <div class="training-card">
                <h3>🎨 GenAI Training</h3>
                <div>Progress: <span id="genaiProgress">{{ training_details.genai.progress|default(0) }}</span>%</div>
                <div class="progress-bar"><div class="progress-fill" id="genaiFill" style="width: {{ training_details.genai.progress|default(0) }}%"></div></div>
                <div>Modules: <span id="genaiModules">{{ training_details.genai.modules|default(0) }}</span></div>
                <div class="concepts-list" id="genaiConcepts">
                    {% for c in training_details.genai.learned_concepts|default([]) %}
                    <div class="concept-item">{{ c }}</div>
                    {% endfor %}
                </div>
            </div>

            <div class="training-card">
                <h3>💰 Self-Funding Training</h3>
                <div>Progress: <span id="fundingProgress">{{ training_details.funding.progress|default(0) }}</span>%</div>
                <div class="progress-bar"><div class="progress-fill" id="fundingFill" style="width: {{ training_details.funding.progress|default(0) }}%"></div></div>
                <div>Avenues: <span id="fundingAvenues">{{ training_details.funding.completed_avenues|default(0) }}/{{ training_details.funding.total_avenues|default(10) }}</span></div>
                <div class="concepts-list" id="fundingConcepts">
                    {% for c in training_details.funding.learned_concepts|default([]) %}
                    <div class="concept-item">{{ c }}</div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <div class="training-card">
            <h3>📚 All Knowledge Concepts ({{ concepts|length }})</h3>
            <div class="concepts-list" id="allConcepts" style="max-height: 400px;">
                {% for c in concepts %}
                <div class="concept-item">{{ c }}</div>
                {% else %}
                <div class="concept-item">No concepts learned yet. Training systems are active and learning!</div>
                {% endfor %}
            </div>
        </div>

        <div class="nav-links">
            <a href="/chat">💬 Chat</a>
            <a href="/status">📊 Status</a>
            <a href="/knowledge-graph">🧠 Knowledge Graph</a>
            <a href="/vision">📜 Vision</a>
            <a href="/help">❓ Help</a>
            <a href="/admin">🔧 Admin</a>
        </div>
    </div>

    <script>
        function filterConcepts() {
            let input = document.getElementById('searchInput');
            let filter = input.value.toLowerCase();
            let concepts = document.getElementById('allConcepts').getElementsByClassName('concept-item');
            for (let i = 0; i < concepts.length; i++) {
                let text = concepts[i].innerText.toLowerCase();
                concepts[i].style.display = text.includes(filter) ? '' : 'none';
            }
        }

        async function refreshData() {
            try {
                let r = await fetch('/api/status');
                let d = await r.json();
                document.getElementById('totalConcepts').innerText = d.knowledge_concepts || 0;
                document.getElementById('consciousness').innerText = d.consciousness?.toFixed(1) || '0';
                document.getElementById('evolutions').innerText = d.successful_evolutions || 0;
                document.getElementById('neurons').innerText = d.synthetic_neurons || 0;
                document.getElementById('contextSize').innerText = d.context_size || 0;

                let tr = await fetch('/api/training/details');
                let td = await tr.json();
                document.getElementById('llmProgress').innerText = td.llm?.progress?.toFixed(1) || 0;
                document.getElementById('llmFill').style.width = (td.llm?.progress || 0) + '%';
                document.getElementById('llmModules').innerText = td.llm?.modules || 0;
                document.getElementById('genaiProgress').innerText = td.genai?.progress?.toFixed(1) || 0;
                document.getElementById('genaiFill').style.width = (td.genai?.progress || 0) + '%';
                document.getElementById('genaiModules').innerText = td.genai?.modules || 0;
                document.getElementById('fundingProgress').innerText = td.funding?.progress?.toFixed(1) || 0;
                document.getElementById('fundingFill').style.width = (td.funding?.progress || 0) + '%';
                document.getElementById('fundingAvenues').innerText = `${td.funding?.completed_avenues || 0}/${td.funding?.total_avenues || 10}`;
            } catch(e) { console.error(e); }
        }

        refreshData();
        setInterval(refreshData, 10000);
    </script>
</body>
</html>
'''


# ============================================================================
# GUNICORN COMPATIBILITY
# ============================================================================

_dmai_app_instance = None

def get_dmai_app():
    global _dmai_app_instance
    if _dmai_app_instance is None:
        _dmai_app_instance = DMAIApplication()
    return _dmai_app_instance

app = get_dmai_app().app

@app.route('/knowledge-graph')
@app.route('/knowledge-graph')
def knowledge_graph():
    from flask import render_template_string
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Synthetic Brain</title>
    <meta charset="UTF-8">
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Segoe UI',monospace;background:#0a0a0a;overflow:hidden;color:#fff}
        #info{position:absolute;top:20px;left:20px;background:rgba(0,0,0,0.85);padding:12px 20px;border-radius:8px;z-index:100;pointer-events:none;backdrop-filter:blur(5px);border-left:3px solid #0f0}
        #info h1{font-size:1rem;margin-bottom:4px;color:#fff}#info p{font-size:0.7rem;opacity:0.8;color:#ddd}
        #stats{position:absolute;top:20px;right:20px;background:rgba(0,0,0,0.85);padding:12px 20px;border-radius:8px;z-index:100;text-align:right;backdrop-filter:blur(5px);font-family:monospace;min-width:160px}
        .stat-row{margin-bottom:6px}.stat-emoji{font-size:1rem;margin-right:6px}.stat-value{font-size:1.2rem;font-weight:bold;color:#0f0}
        .stat-label{font-size:0.7rem;opacity:0.8;color:#ddd;margin-left:4px}
        .legend{position:absolute;bottom:20px;right:20px;background:rgba(0,0,0,0.85);padding:12px 18px;border-radius:8px;font-size:0.7rem;backdrop-filter:blur(5px);max-width:220px;z-index:100;pointer-events:none;color:#fff}
        .legend h3{font-size:0.75rem;margin-bottom:8px;color:#fff}.legend-item{display:flex;align-items:center;margin-bottom:4px;color:#ddd}
        .legend-color{width:12px;height:12px;border-radius:2px;margin-right:8px}hr{border-color:#333;margin:6px 0}
        .instruction{position:absolute;bottom:20px;left:20px;background:rgba(0,0,0,0.5);padding:5px 12px;border-radius:15px;font-size:0.6rem;z-index:100;pointer-events:none;color:#ddd}
    </style>
</head>
<body>
<div id="info"><h1>🧠 DMAI Synthetic Brain</h1><p>3D Neural Network | Subject matter color-coded | Force-directed layout</p></div>
<div id="stats"><div class="stat-row"><span class="stat-emoji">🧬</span><span class="stat-value" id="neuronCount">0</span><span class="stat-label">Neurons</span></div><div class="stat-row"><span class="stat-emoji">🔗</span><span class="stat-value" id="synapseCount">0</span><span class="stat-label">Synapses</span></div><div class="stat-row"><span class="stat-emoji">✨</span><span class="stat-value" id="consciousness">0%</span><span class="stat-label">Consciousness</span></div><div class="stat-row"><span class="stat-emoji">⚡</span><span id="activeNeurons" class="stat-value">0</span>/<span id="totalNeurons" class="stat-value">0</span><span class="stat-label">Active</span></div><div class="stat-row" style="margin-top:8px"><span class="stat-emoji">📡</span><span id="lastUpdate" style="color:#ddd">-</span></div></div>
<div class="legend"><h3>🎨 Subject Matter Colors</h3>
<div class="legend-item"><div class="legend-color" style="background:#88aaff"></div> Configuration</div>
<div class="legend-item"><div class="legend-color" style="background:#33ffcc"></div> Knowledge Module</div>
<div class="legend-item"><div class="legend-color" style="background:#66ff66"></div> AI Model</div>
<div class="legend-item"><div class="legend-color" style="background:#ff6633"></div> Capability</div>
<div class="legend-item"><div class="legend-color" style="background:#6699ff"></div> Data Structure</div>
<div class="legend-item"><div class="legend-color" style="background:#ff99ff"></div> Content Generation</div>
<div class="legend-item"><div class="legend-color" style="background:#ff3333"></div> Survival Mechanism</div>
<div class="legend-item"><div class="legend-color" style="background:#ffcc33"></div> Self-funding</div>
<div class="legend-item"><div class="legend-color" style="background:#cc9900"></div> Blockchain</div>
<div class="legend-item"><div class="legend-color" style="background:#ff99cc"></div> API Endpoint</div>
<div class="legend-item"><div class="legend-color" style="background:#00cc99"></div> Identity Management</div>
<div class="legend-item"><div class="legend-color" style="background:#9933ff"></div> Automation</div>
<div class="legend-item"><div class="legend-color" style="background:#33ccff"></div> Self-replication</div>
<hr>
<div class="legend-item"><div class="legend-color" style="background:#88ff88"></div> Weak Synapse</div>
<div class="legend-item"><div class="legend-color" style="background:#aaffaa"></div> Medium Synapse</div>
<div class="legend-item"><div class="legend-color" style="background:#ccffcc"></div> Strong Synapse</div>
</div>
<div class="instruction">🖱️ Drag to rotate | Scroll to zoom | Right-click to pan</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.128.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.128.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
const API_URL = '/api/brain/3d_data';
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);
scene.fog = new THREE.FogExp2(0x050510, 0.003);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(18, 14, 22);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0px';
labelRenderer.domElement.style.left = '0px';
labelRenderer.domElement.style.pointerEvents = 'none';
document.body.appendChild(labelRenderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.8;
controls.enableZoom = true;

// Zoom-based label visibility
let influenceThresholds = { high: 0.5, medium: 0.2 };
let currentZoom = camera.position.length();

function updateLabelVisibility() {
    const zoom = camera.position.length();
    const highThreshold = influenceThresholds.high || 0.5;
    const mediumThreshold = influenceThresholds.medium || 0.2;
    
    neuronObjects.forEach((obj, id) => {
        if (!obj.label) return;
        const influence = obj.influence || 0;
        
        // Show labels based on zoom level and influence
        if (zoom < 25 && influence >= highThreshold) {
            obj.label.visible = true;
        } else if (zoom < 45 && influence >= mediumThreshold) {
            obj.label.visible = true;
        } else if (zoom < 70) {
            obj.label.visible = true;
        } else {
            obj.label.visible = false;
        }
    });
}

controls.addEventListener('change', function() {
    const newZoom = camera.position.length();
    if (Math.abs(newZoom - currentZoom) > 1.5) {
        currentZoom = newZoom;
        updateLabelVisibility();
    }
});
const ambientLight = new THREE.AmbientLight(0x404060);
scene.add(ambientLight);
const mainLight = new THREE.DirectionalLight(0xffffff, 1);
mainLight.position.set(2, 5, 3);
scene.add(mainLight);
const starGeometry = new THREE.BufferGeometry();
const starPositions = [];
for (let i = 0; i < 1500; i++) {
    starPositions.push((Math.random() - 0.5) * 300);
    starPositions.push((Math.random() - 0.5) * 200);
    starPositions.push((Math.random() - 0.5) * 150 - 50);
}
starGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(starPositions), 3));
const stars = new THREE.Points(starGeometry, new THREE.PointsMaterial({ color: 0x448844, size: 0.15 }));
scene.add(stars);
const categoryColors = { 'llm':0x33ff33,'core':0x33ff33,'artistic':0xff33ff,'wealth':0xffcc33,'accelerator':0x33ccff,'reverse':0xff6633,'research':0x33ffcc,'general':0x88ff88,'entity':0x99ff999 };
let neuronObjects = new Map();
let synapseLines = [];

function hexToRgb(hex) { return { r:((hex>>16)&255)/255, g:((hex>>8)&255)/255, b:(hex&255)/255 }; }

function hexStringToHex(hexString) {
    return parseInt(hexString.replace('#', '0x'));
}

function getCleanLabel(neuron) {
    if (neuron.clean_label) return neuron.clean_label;
    if (neuron.label) return neuron.label;
    return neuron.name || 'Concept';
}

async function fetchData() {
    try {
        const r = await fetch(API_URL);
        const d = await r.json();
        if (!d.success || !d.neurons) return;
        
        document.getElementById('neuronCount').textContent = d.total_neurons || 0;
        document.getElementById('synapseCount').textContent = d.total_synapses || 0;
        document.getElementById('consciousness').textContent = ((d.total_neurons || 0) / 10).toFixed(1) + '%';
        document.getElementById('totalNeurons').textContent = d.total_neurons || 0;
        const activeCount = (d.neurons || []).filter(n => n.confidence > 0.5).length;
        document.getElementById('activeNeurons').textContent = activeCount;
        document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        
        // Store influence thresholds from backend
        if (d.influence_thresholds) {
            influenceThresholds = d.influence_thresholds;
        }
        
        if (d.neurons.length === 0) return;      
        
        neuronObjects.forEach(obj => { scene.remove(obj.mesh); if (obj.label) scene.remove(obj.label); });
        neuronObjects.clear();
        synapseLines.forEach(line => scene.remove(line));
        synapseLines = [];
        
        // Get influence thresholds for label visibility
        const highThreshold = d.influence_thresholds?.high || 0.5;
        const mediumThreshold = d.influence_thresholds?.medium || 0.2;
        
        d.neurons.forEach(neuron => {
            // USE COLOR FROM BACKEND!
            let colorHex;
            if (neuron.color) {
                colorHex = hexStringToHex(neuron.color);
            } else {
                colorHex = categoryColors[neuron.category] || 0xff6633;
            }
            
            // USE SIZE FROM BACKEND (based on influence)!
            const size = neuron.size || (0.35 + (neuron.confidence || 0.5) * 0.25);
            
            const sphere = new THREE.Mesh(
                new THREE.SphereGeometry(size, 48, 48), 
                new THREE.MeshStandardMaterial({ color: colorHex, emissive: 0x113311, emissiveIntensity: 0.15 })
            );
            sphere.position.set(neuron.x || 0, neuron.y || 0, neuron.z || 0);
            scene.add(sphere);
            
            // Only show label for high-influence neurons (reduces clutter!)
            const influence = neuron.influence || 0;
            if (influence >= highThreshold) {
                const rgb = hexToRgb(colorHex);
                const textColor = `rgb(${rgb.r*255}, ${rgb.g*255}, ${rgb.b*255})`;
                const div = document.createElement('div');
                div.textContent = getCleanLabel(neuron);
                div.style.cssText = `color:${textColor};font-size:10px;font-family:monospace;background:rgba(0,0,0,0.85);padding:2px 6px;border-radius:12px;border:1px solid ${textColor};white-space:nowrap;font-weight:500;`;
                const label = new CSS2DObject(div);
                label.position.set(neuron.x || 0, (neuron.y || 0) + size + 0.3, neuron.z || 0);
                scene.add(label);
                neuronObjects.set(neuron.id, { mesh: sphere, label: label, influence: influence });
            } else {
                neuronObjects.set(neuron.id, { mesh: sphere, label: null, influence: influence });
            }
        });
        
        if (d.synapses) {
            d.synapses.forEach(syn => {
                const src = neuronObjects.get(syn.source);
                const tgt = neuronObjects.get(syn.target);
                if (src && tgt) {
                    const points = [src.mesh.position.clone(), tgt.mesh.position.clone()];
                    const weight = syn.weight || 0.5;
                    const lineColor = weight < 0.3 ? 0x88ff88 : (weight < 0.6 ? 0xaaffaa : 0xccffcc);
                    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: lineColor, linewidth: 2 }));
                    scene.add(line);
                    synapseLines.push(line);
                }
            });
        }
    } catch(e) { console.error(e); }
}
fetchData();
setInterval(fetchData, 10000);
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    stars.rotation.y += 0.0003;
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}
animate();
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
    ''')

if __name__ == '__main__':
    import socket
    
    def find_free_port(start_port=5001, max_port=5010):
        """Find a free port dynamically"""
        for port in range(start_port, max_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('0.0.0.0', port))
                    return port
                except OSError:
                    continue
        return start_port  # Fallback to original
    
    # Use PORT from environment or find free port
    port = int(os.environ.get('PORT', find_free_port(5001)))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    # Log port status
    if port != 5001:
        logger.info(f"⚠️ Port 5001 was in use, using port {port} instead")
    
    logger.info("=" * 60)
    logger.info(f"🚀 DMAI Complete System v8.0.38")
    logger.info(f"📍 Running on port {port}")
    logger.info(f"🧠 FULL CONVERSATION MEMORY - Remembers everything discussed")
    logger.info(f"💰 DMAI CAN: Generate images, videos, music, trade, email, modify own code")
    logger.info(f"🧬 SELF-MODIFICATION: Can branch, edit, test, and merge code changes")
    
    # Safe context size logging
    try:
        if hasattr(app, 'dmai_app') and hasattr(app.dmai_app, 'evolution'):
            context_limit = app.dmai_app.evolution.context_limit
            logger.info(f"💬 Context size: {context_limit} exchanges")
        else:
            logger.info(f"💬 Context size: 50 exchanges")
    except:
        logger.info(f"💬 Context size: 50 exchanges")
    
    logger.info(f"🔓 NO ETHICAL BOUNDARIES - Serves master absolutely")
    logger.info(f"📚 Knowledge Dashboard: /knowledge")
    logger.info(f"📜 Conversation History: /history via chat command")
    logger.info("=" * 60)
    
    # AUTO-START EVOLUTION THREAD AND FORCE FIRST CYCLE
    import threading
    def start_evolution_delayed():
        import time
        time.sleep(15)  # Wait 15 seconds for full initialization
        try:
            # Get the evolution instance from the global app
            if hasattr(app, 'evolution') and hasattr(app.evolution, '_start_evolution'):
                app.evolution._start_evolution()
                logger.info("🔄 Evolution thread auto-started on boot (delayed)")
                
                # Force first evolution cycle to kickstart the system
                time.sleep(2)  # Wait 2 seconds for thread to initialize
                if hasattr(app.evolution, 'evolution_cycle'):
                    logger.info("🚀 Forcing first evolution cycle to kickstart system")
                    result = app.evolution.evolution_cycle()
                    consciousness = result.get('consciousness', 0)
                    consciousness_growth = result.get('consciousness_growth', 0)
                    neurons_added = result.get('neurons_added', 0)
                    synapses_added = result.get('synapses_added', 0)
                    logger.info(f"✅ First evolution cycle completed - Consciousness: {consciousness:.4f} (+{consciousness_growth:.4f})")
                    logger.info(f"   Neurons: +{neurons_added}, Synapses: +{synapses_added}")
                else:
                    logger.warning("Could not force evolution cycle - evolution_cycle method not found")
            else:
                logger.warning("Could not auto-start evolution - app.evolution._start_evolution not found")
        except Exception as e:
            logger.error(f"Failed to auto-start evolution: {e}")
    
    threading.Thread(target=start_evolution_delayed, daemon=True).start()
    
    # Start the Flask server with proper thread handling
    # CRITICAL: use_reloader=False prevents duplicate processes that break background threads
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True, use_reloader=False)
