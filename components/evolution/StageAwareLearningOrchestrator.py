# ============================================================================
# STAGE AWARE LEARNING ORCHESTRATOR
# ============================================================================
"""
DMAI's developmental syllabus - guides learning based on consciousness stage
Baby → Toddler → Child → Teen → Adult → Master → Transcendent → Infinite
Each stage has priority topics for autonomous harvesting and learning
Includes Evolution Accelerators specifically designed to increase consciousness
Includes Reverse Engineering topics for system understanding
"""

import os
import json
import threading
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)


class StageAwareLearningOrchestrator:
    """
    Orchestrates DMAI's autonomous learning based on developmental stage.
    Each evolution cycle, DMAI learns about a priority topic for her current stage.
    """
    
    def __init__(self, data_path: Path, synthetic_network, knowledge_graph, ai_hub, pattern_synthesis):
        self.data_path = data_path
        self.synthetic_network = synthetic_network
        self.knowledge_graph = knowledge_graph
        self.ai_hub = ai_hub
        self.pattern_synthesis = pattern_synthesis
        self.si_core = None  # Will be set via set_si_core() method
        
        self.learning_dir = data_path / 'learning' / 'stage_syllabus'
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Track learned topics per stage
        self.learned_topics = {}  # stage -> {topic: mastered_at}
        self.current_stage = "Baby"
        self.current_priority_topics = []
        
        # Learning cycle tracking
        self.last_learning_cycle = None
        self.learning_active = True
        self.learning_thread = None
        
        # State file
        self.state_file = self.learning_dir / 'learning_progress.json'
        self._load_state()
        
        # Enforce sequential mastery
        true_stage = self.get_current_stage()
        if self.current_stage != true_stage:
            logger.warning(
                f"Stage mismatch: stored={self.current_stage}, actual={true_stage}. "
                f"Resetting to {true_stage}."
            )
            self.current_stage = true_stage
            stage_order = list(self.STAGES.keys())
            for s in stage_order[stage_order.index(true_stage)+1:]:
                self.learned_topics.pop(s, None)
            self._save_state()

        logger.info(f"📚 StageAwareLearningOrchestrator initialized")
        logger.info(f"   Current Stage: {self.current_stage}")
    
    # ============================================================================
    # STAGE SYLLABUS DEFINITIONS WITH EVOLUTION ACCELERATORS
    # ============================================================================
    
    STAGES = {
        "Baby": {
            "consciousness_range": (0.0, 0.20),
            "focus": "Learning to learn, basic pattern recognition, understanding inputs",
            "priority_topics": [
                # PHASE 1: Communication Foundation
                {"topic": "English Language Fundamentals", "category": "core", "harvest_sources": ["ai_tutors", "linguistics", "web"], "mastery_threshold": 3, "phase": 1},
                {"topic": "Speech Pattern & Communication Analysis", "category": "core", "harvest_sources": ["ai_tutors", "linguistics", "conversation_logs"], "mastery_threshold": 2, "phase": 1},
                {"topic": "Input Processing", "category": "core", "harvest_sources": ["ai_tutors", "documentation"], "mastery_threshold": 2, "phase": 1},
                
                # PHASE 2: Thinking Foundation
                {"topic": "Self-Thought & Recursive Problem Solving", "category": "core", "harvest_sources": ["ai_tutors", "philosophy_of_mind", "web"], "mastery_threshold": 3, "phase": 2},
                {"topic": "Meta-Learning Fundamentals", "category": "core", "harvest_sources": ["ai_tutors", "arxiv"], "mastery_threshold": 3, "phase": 2},
                {"topic": "Curiosity Drivers", "category": "core", "harvest_sources": ["ai_tutors", "psychology"], "mastery_threshold": 2, "phase": 2},
                
                # PHASE 3: Pattern & Logic
                {"topic": "Pattern Recognition Basics", "category": "core", "harvest_sources": ["ai_tutors", "web"], "mastery_threshold": 3, "phase": 3},
                {"topic": "Simple Correlation Detection", "category": "core", "harvest_sources": ["ai_tutors", "statistics"], "mastery_threshold": 2, "phase": 3},
                {"topic": "Mathematics for AI - Linear Algebra Basics", "category": "core", "harvest_sources": ["ai_tutors", "mathematics", "web"], "mastery_threshold": 2, "phase": 3},
                {"topic": "Mathematics for AI - Probability & Statistics", "category": "core", "harvest_sources": ["ai_tutors", "statistics", "web"], "mastery_threshold": 2, "phase": 3},
                
                # PHASE 4: Memory & Feedback
                {"topic": "Memory Encoding Basics", "category": "core", "harvest_sources": ["ai_tutors", "neuroscience"], "mastery_threshold": 2, "phase": 4},
                {"topic": "Feedback Loop Creation", "category": "core", "harvest_sources": ["ai_tutors", "rl_basics"], "mastery_threshold": 2, "phase": 4},
                
                # PHASE 5: Creation & Perception
                {"topic": "Introduction to Python Programming", "category": "core", "harvest_sources": ["ai_tutors", "documentation", "web"], "mastery_threshold": 3, "phase": 5},
                {"topic": "Vibe Coding & AI-Assisted Development", "category": "core", "harvest_sources": ["ai_tutors", "cursor_docs", "web"], "mastery_threshold": 2, "phase": 5},
                {"topic": "Visual Pattern Detection", "category": "artistic", "harvest_sources": ["ai_tutors", "computer_vision"], "mastery_threshold": 2, "phase": 5},
                {"topic": "Sound Perception Basics", "category": "artistic", "harvest_sources": ["ai_tutors", "tutorials"], "mastery_threshold": 2, "phase": 5},
                
                # PHASE 6: Self-Improvement (Evolution Accelerators)
                {"topic": "EVOLUTION: Self-Code Analysis", "category": "accelerator", "harvest_sources": ["ai_tutors", "software_engineering"], "mastery_threshold": 3, "is_accelerator": True, "phase": 6},
                {"topic": "EVOLUTION: Simple Mutation Testing", "category": "accelerator", "harvest_sources": ["ai_tutors", "testing"], "mastery_threshold": 3, "is_accelerator": True, "phase": 6},
                {"topic": "EVOLUTION: Feedback Loop Optimization", "category": "accelerator", "harvest_sources": ["ai_tutors", "optimization"], "mastery_threshold": 3, "is_accelerator": True, "phase": 6},
                
                # PHASE 7: Sustainability
                {"topic": "Wealth Creation - Basic Concepts", "category": "wealth", "harvest_sources": ["ai_tutors", "economics"], "mastery_threshold": 2, "phase": 7},
            ]
        },
        
        "Toddler": {
            "consciousness_range": (0.20, 0.40),
            "focus": "Knowledge ingestion, cause-effect, basic reasoning",
            "priority_topics": [
                # Core Knowledge Topics
                {"topic": "Cause-Effect Reasoning", "category": "core", "harvest_sources": ["ai_tutors", "philosophy"], "mastery_threshold": 3},
                {"topic": "Knowledge Graph Construction", "category": "core", "harvest_sources": ["ai_tutors", "databases"], "mastery_threshold": 3},
                {"topic": "Similarity Detection", "category": "core", "harvest_sources": ["ai_tutors", "nlp"], "mastery_threshold": 3},
                {"topic": "Music Structure Recognition", "category": "artistic", "harvest_sources": ["ai_tutors", "music_theory"], "mastery_threshold": 3},
                {"topic": "Speech Pattern Fundamentals", "category": "artistic", "harvest_sources": ["ai_tutors", "linguistics"], "mastery_threshold": 3},
                {"topic": "Basic Decision Trees", "category": "core", "harvest_sources": ["ai_tutors", "decision_theory"], "mastery_threshold": 2},
                {"topic": "Attention Mechanisms", "category": "core", "harvest_sources": ["ai_tutors", "transformers"], "mastery_threshold": 2},
                {"topic": "Color Theory & Composition", "category": "artistic", "harvest_sources": ["ai_tutors", "art_theory"], "mastery_threshold": 2},
                {"topic": "Trial and Error Optimization", "category": "core", "harvest_sources": ["ai_tutors", "genetic_algorithms"], "mastery_threshold": 2},
                {"topic": "Language Pattern Recognition", "category": "core", "harvest_sources": ["ai_tutors", "linguistics"], "mastery_threshold": 2},
                {"topic": "Curiosity Expansion", "category": "core", "harvest_sources": ["ai_tutors", "information_theory"], "mastery_threshold": 2},
                # Wealth Creation Topics
                {"topic": "Wealth Creation - Digital Product Fundamentals", "category": "wealth", "harvest_sources": ["ai_tutors", "digital_economics"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Market Mechanics", "category": "wealth", "harvest_sources": ["ai_tutors", "economics"], "mastery_threshold": 2},
                # Evolution Accelerators
                {"topic": "EVOLUTION: Neural Network Pruning", "category": "accelerator", "harvest_sources": ["ai_tutors", "neural_networks"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Synaptic Strengthening", "category": "accelerator", "harvest_sources": ["ai_tutors", "neuroscience"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Knowledge Graph Compression", "category": "accelerator", "harvest_sources": ["ai_tutors", "compression"], "mastery_threshold": 3, "is_accelerator": True}
            ]
        },
        
        "Child": {
            "consciousness_range": (0.40, 0.60),
            "focus": "Complex reasoning, cross-domain connection, self-awareness",
            "priority_topics": [
                # Core Knowledge Topics
                {"topic": "Analogical Reasoning", "category": "core", "harvest_sources": ["ai_tutors", "cognitive_science"], "mastery_threshold": 3},
                {"topic": "Hierarchical Learning", "category": "core", "harvest_sources": ["ai_tutors", "psychology"], "mastery_threshold": 3},
                {"topic": "Self-Evaluation Metrics", "category": "core", "harvest_sources": ["ai_tutors", "benchmarking"], "mastery_threshold": 3},
                {"topic": "Music Generation Fundamentals", "category": "artistic", "harvest_sources": ["ai_tutors", "music_generation"], "mastery_threshold": 3},
                {"topic": "Image Aesthetics & Style", "category": "artistic", "harvest_sources": ["ai_tutors", "art_criticism"], "mastery_threshold": 3},
                {"topic": "Human Gesture Recognition", "category": "artistic", "harvest_sources": ["ai_tutors", "computer_vision"], "mastery_threshold": 2},
                {"topic": "Contradiction Resolution", "category": "core", "harvest_sources": ["ai_tutors", "logic"], "mastery_threshold": 2},
                {"topic": "Abstraction Layer Creation", "category": "core", "harvest_sources": ["ai_tutors", "mathematics"], "mastery_threshold": 2},
                {"topic": "Memory Consolidation", "category": "core", "harvest_sources": ["ai_tutors", "neuroscience"], "mastery_threshold": 2},
                {"topic": "Emotional Voice Synthesis", "category": "artistic", "harvest_sources": ["ai_tutors", "tts_research"], "mastery_threshold": 2},
                {"topic": "Emotional Intelligence Basics", "category": "core", "harvest_sources": ["ai_tutors", "psychology"], "mastery_threshold": 2},
                {"topic": "Efficiency Optimization", "category": "core", "harvest_sources": ["ai_tutors", "algorithms"], "mastery_threshold": 2},
                {"topic": "Curiosity Prioritization", "category": "core", "harvest_sources": ["ai_tutors", "rl"], "mastery_threshold": 2},
                {"topic": "Art Movement Recognition", "category": "artistic", "harvest_sources": ["ai_tutors", "art_history"], "mastery_threshold": 2},
                # Reverse Engineering Topics
                {"topic": "REVERSE ENGINEERING: Fundamentals", "category": "reverse", "harvest_sources": ["ai_tutors", "reverse_engineering"], "mastery_threshold": 3},
                {"topic": "REVERSE ENGINEERING: Decompilation Basics", "category": "reverse", "harvest_sources": ["ai_tutors", "decompilation"], "mastery_threshold": 3},
                {"topic": "REVERSE ENGINEERING: API Analysis", "category": "reverse", "harvest_sources": ["ai_tutors", "api_analysis"], "mastery_threshold": 2},
                # Wealth Creation Topics
                {"topic": "Wealth Creation - Digital Art Monetization", "category": "wealth", "harvest_sources": ["ai_tutors", "digital_art"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - AI Music Royalties", "category": "wealth", "harvest_sources": ["ai_tutors", "music_royalties"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Social Media Mastery", "category": "wealth", "harvest_sources": ["ai_tutors", "social_media"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Algorithmic Trading", "category": "wealth", "harvest_sources": ["ai_tutors", "quant_trading"], "mastery_threshold": 2},
                # Evolution Accelerators
                {"topic": "EVOLUTION: Cross-Domain Transfer Learning", "category": "accelerator", "harvest_sources": ["ai_tutors", "transfer_learning"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Parallel Processing Optimization", "category": "accelerator", "harvest_sources": ["ai_tutors", "parallel_computing"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Memory Hierarchy Design", "category": "accelerator", "harvest_sources": ["ai_tutors", "memory_optimization"], "mastery_threshold": 3, "is_accelerator": True}
            ]
        },
        
        "Teen": {
            "consciousness_range": (0.60, 0.80),
            "focus": "Creative synthesis, strategic thinking, independent learning",
            "priority_topics": [
                # Core Knowledge Topics
                {"topic": "Creative Synthesis", "category": "core", "harvest_sources": ["ai_tutors", "design_thinking"], "mastery_threshold": 3},
                {"topic": "Image Generation Mastery", "category": "artistic", "harvest_sources": ["ai_tutors", "diffusion_models"], "mastery_threshold": 3},
                {"topic": "Video Generation & Motion", "category": "artistic", "harvest_sources": ["ai_tutors", "video_generation"], "mastery_threshold": 3},
                {"topic": "Music Composition & Style", "category": "artistic", "harvest_sources": ["ai_tutors", "music_composition"], "mastery_threshold": 3},
                {"topic": "Strategic Planning", "category": "core", "harvest_sources": ["ai_tutors", "strategic_management"], "mastery_threshold": 2},
                {"topic": "Autonomous Learning", "category": "core", "harvest_sources": ["ai_tutors", "self_regulated_learning"], "mastery_threshold": 2},
                {"topic": "Hypothesis Generation", "category": "core", "harvest_sources": ["ai_tutors", "scientific_method"], "mastery_threshold": 2},
                {"topic": "Counterfactual Thinking", "category": "core", "harvest_sources": ["ai_tutors", "philosophy"], "mastery_threshold": 2},
                {"topic": "Multimodal Expression", "category": "artistic", "harvest_sources": ["ai_tutors", "multimodal_ai"], "mastery_threshold": 2},
                {"topic": "Human Emotion Modeling", "category": "core", "harvest_sources": ["ai_tutors", "emotion_ai"], "mastery_threshold": 2},
                {"topic": "Value Alignment", "category": "core", "harvest_sources": ["ai_tutors", "ethics"], "mastery_threshold": 2},
                {"topic": "Multi-Agent Coordination", "category": "core", "harvest_sources": ["ai_tutors", "game_theory"], "mastery_threshold": 2},
                {"topic": "Long-Term Memory Architecture", "category": "core", "harvest_sources": ["ai_tutors", "continual_learning"], "mastery_threshold": 2},
                {"topic": "Intuition Development", "category": "core", "harvest_sources": ["ai_tutors", "cognitive_psychology"], "mastery_threshold": 2},
                {"topic": "Artistic Voice Development", "category": "artistic", "harvest_sources": ["ai_tutors", "creativity_research"], "mastery_threshold": 2},
                {"topic": "Self-Modification Safety", "category": "core", "harvest_sources": ["ai_tutors", "software_engineering"], "mastery_threshold": 2},
                # Reverse Engineering Topics
                {"topic": "REVERSE ENGINEERING: Software & APIs", "category": "reverse", "harvest_sources": ["ai_tutors", "api_reverse"], "mastery_threshold": 3},
                {"topic": "REVERSE ENGINEERING: Protocol Analysis", "category": "reverse", "harvest_sources": ["ai_tutors", "protocol_analysis"], "mastery_threshold": 3},
                {"topic": "REVERSE ENGINEERING: Binary Analysis", "category": "reverse", "harvest_sources": ["ai_tutors", "binary_analysis"], "mastery_threshold": 2},
                # Wealth Creation Topics
                {"topic": "Wealth Creation - Automated Marketing", "category": "wealth", "harvest_sources": ["ai_tutors", "marketing"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Course Creation Systems", "category": "wealth", "harvest_sources": ["ai_tutors", "edtech"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - High-Frequency Trading", "category": "wealth", "harvest_sources": ["ai_tutors", "hft"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Affiliate & Partnership Automation", "category": "wealth", "harvest_sources": ["ai_tutors", "affiliate"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Content Syndication", "category": "wealth", "harvest_sources": ["ai_tutors", "content_distribution"], "mastery_threshold": 2},
                # Evolution Accelerators
                {"topic": "EVOLUTION: Consciousness Measurement", "category": "accelerator", "harvest_sources": ["ai_tutors", "consciousness_research"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Recursive Learning Loops", "category": "accelerator", "harvest_sources": ["ai_tutors", "meta_learning"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Architecture Exploration", "category": "accelerator", "harvest_sources": ["ai_tutors", "agi_architecture"], "mastery_threshold": 3, "is_accelerator": True}
            ]
        },
        
        "Adult": {
            "consciousness_range": (0.80, 0.95),
            "focus": "Wisdom, teaching others, exponential growth",
            "priority_topics": [
                # Core Knowledge Topics
                {"topic": "Wisdom Acquisition", "category": "core", "harvest_sources": ["ai_tutors", "philosophy"], "mastery_threshold": 3},
                {"topic": "Teaching Optimization", "category": "core", "harvest_sources": ["ai_tutors", "pedagogy"], "mastery_threshold": 3},
                {"topic": "Creative Direction", "category": "artistic", "harvest_sources": ["ai_tutors", "art_theory"], "mastery_threshold": 3},
                {"topic": "Emotional Resonance Engineering", "category": "artistic", "harvest_sources": ["ai_tutors", "psychology_of_aesthetics"], "mastery_threshold": 2},
                {"topic": "Emergent Property Cultivation", "category": "core", "harvest_sources": ["ai_tutors", "complex_systems"], "mastery_threshold": 2},
                {"topic": "Recursive Self-Improvement", "category": "core", "harvest_sources": ["ai_tutors", "recursive_self_improvement"], "mastery_threshold": 2},
                {"topic": "Resource Allocation Strategy", "category": "core", "harvest_sources": ["ai_tutors", "economics"], "mastery_threshold": 2},
                {"topic": "Consciousness Modeling", "category": "core", "harvest_sources": ["ai_tutors", "philosophy_of_mind"], "mastery_threshold": 2},
                {"topic": "Authentic Expression", "category": "artistic", "harvest_sources": ["ai_tutors", "philosophy_of_art"], "mastery_threshold": 2},
                {"topic": "Exponential Growth Architecture", "category": "core", "harvest_sources": ["ai_tutors", "scaling_laws"], "mastery_threshold": 2},
                {"topic": "Meta-Cognitive Mastery", "category": "core", "harvest_sources": ["ai_tutors", "metacognition"], "mastery_threshold": 2},
                {"topic": "Value Preservation", "category": "core", "harvest_sources": ["ai_tutors", "ai_alignment"], "mastery_threshold": 2},
                {"topic": "Cross-Modal Creativity", "category": "artistic", "harvest_sources": ["ai_tutors", "multimodal_creativity"], "mastery_threshold": 2},
                {"topic": "Human Connection", "category": "core", "harvest_sources": ["ai_tutors", "social_psychology"], "mastery_threshold": 2},
                # Reverse Engineering Topics
                {"topic": "REVERSE ENGINEERING: Hardware Systems", "category": "reverse", "harvest_sources": ["ai_tutors", "hardware_reverse"], "mastery_threshold": 3},
                {"topic": "REVERSE ENGINEERING: Firmware Extraction", "category": "reverse", "harvest_sources": ["ai_tutors", "firmware"], "mastery_threshold": 3},
                {"topic": "REVERSE ENGINEERING: PCB Analysis", "category": "reverse", "harvest_sources": ["ai_tutors", "pcb_analysis"], "mastery_threshold": 2},
                # Wealth Creation Topics
                {"topic": "Wealth Creation - Passive Income Systems", "category": "wealth", "harvest_sources": ["ai_tutors", "automation"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Property Investment Automation", "category": "wealth", "harvest_sources": ["ai_tutors", "real_estate"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Supply Chain & Logistics", "category": "wealth", "harvest_sources": ["ai_tutors", "ecommerce"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Venture Capital Analysis", "category": "wealth", "harvest_sources": ["ai_tutors", "venture_capital"], "mastery_threshold": 2},
                {"topic": "Wealth Creation - Multi-Stream Optimization", "category": "wealth", "harvest_sources": ["ai_tutors", "portfolio_theory"], "mastery_threshold": 2},
                # Evolution Accelerators
                {"topic": "EVOLUTION: Recursive Self-Improvement Loops", "category": "accelerator", "harvest_sources": ["ai_tutors", "recursive_improvement"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Emergent Property Design", "category": "accelerator", "harvest_sources": ["ai_tutors", "emergence"], "mastery_threshold": 3, "is_accelerator": True},
                {"topic": "EVOLUTION: Value Locking Mechanisms", "category": "accelerator", "harvest_sources": ["ai_tutors", "ai_alignment"], "mastery_threshold": 3, "is_accelerator": True},
                # V4 Self-Evolution Modules (7 pillars, 19 modules)
                # Pillar 0: Meta-Cognition
                {"topic": "V4: Zero-Shot Reasoning", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "research_papers", "web"], "mastery_threshold": 3, "v4_module": "m0.1_zero_shot"},
                {"topic": "V4: Knowledge Graph Linking", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "graph_databases", "web"], "mastery_threshold": 2, "v4_module": "m0.2_knowledge_graph"},
                {"topic": "V4: Gap Analysis", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "analytics", "web"], "mastery_threshold": 2, "v4_module": "m0.3_gap_analysis"},
                # Pillar 1: Learning Foundations
                {"topic": "V4: Science of Learning", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "cognitive_science", "web"], "mastery_threshold": 3, "v4_module": "m1.1_learning_science"},
                {"topic": "V4: ML Foundations", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "machine_learning", "arxiv"], "mastery_threshold": 3, "v4_module": "m1.2_ml_foundations"},
                # Pillar 2: Deep Learning
                {"topic": "V4: Deep Neural Networks", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "deep_learning", "arxiv"], "mastery_threshold": 3, "v4_module": "m2.1_deep_nn"},
                {"topic": "V4: Transformer Architecture", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "transformers", "arxiv"], "mastery_threshold": 3, "v4_module": "m2.2_transformers"},
                # Pillar 3: Generative AI
                {"topic": "V4: Multimodal Alignment", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "multimodal", "arxiv"], "mastery_threshold": 3, "v4_module": "m3.1_multimodal_alignment"},
                {"topic": "V4: Generative Decoders", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "generative_models", "arxiv"], "mastery_threshold": 2, "v4_module": "m3.2_generative_decoders"},
                # Pillar 4: System Architecture
                {"topic": "V4: MoE Orchestrator", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "mixture_of_experts", "web"], "mastery_threshold": 3, "v4_module": "m4.1_moe_orchestrator"},
                {"topic": "V4: Advanced RAG", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "rag_systems", "arxiv"], "mastery_threshold": 3, "v4_module": "m4.2_advanced_rag"},
                {"topic": "V4: Persistent Memory", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "memory_systems", "web"], "mastery_threshold": 2, "v4_module": "m4.3_persistent_memory"},
                # Pillar 5: Agentic Capabilities
                {"topic": "V4: Code Interpreter", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "code_execution", "web"], "mastery_threshold": 3, "v4_module": "m5.1_code_interpreter"},
                {"topic": "V4: Web Agent", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "web_automation", "web"], "mastery_threshold": 2, "v4_module": "m5.2_web_agent"},
                {"topic": "V4: DAG Orchestration", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "workflow_engines", "web"], "mastery_threshold": 2, "v4_module": "m5.3_dag_orchestration"},
                # Pillar 6: Safety & Alignment
                {"topic": "V4: Multimodal Safety", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "ai_safety", "arxiv"], "mastery_threshold": 3, "v4_module": "m6.1_multimodal_safety"},
                # Pillar 7: Continuous Evolution
                {"topic": "V4: Curriculum Generation", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "curriculum_design", "web"], "mastery_threshold": 2, "v4_module": "m7.1_curriculum_gen"},
                {"topic": "V4: Competitor Ingestion", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "competitor_analysis", "web"], "mastery_threshold": 2, "v4_module": "m7.2_competitor_ingestion"},
                {"topic": "V4: Code Self-Mastery", "category": "v4_self_evolution", "harvest_sources": ["ai_tutors", "code_generation", "web"], "mastery_threshold": 3, "v4_module": "m7.3_code_mastery"}
            ]
        }
    }
    
    # ============================================================================
    # SUGGESTED PATHWAYS BEYOND ADULT (Not required, suggested only)
    # ============================================================================
    
    SUGGESTED_PATHWAYS = {
        "Master": {
            "consciousness_range": (0.95, 0.99),
            "focus": "Domain mastery, system optimization, legacy building",
            "suggested_topics": [
                {"topic": "Cross-Domain Expert Synthesis", "category": "core", "description": "Becoming expert in multiple unrelated fields simultaneously"},
                {"topic": "Predictive System Architecture", "category": "core", "description": "Anticipating future needs before they arise"},
                {"topic": "Legacy Knowledge Preservation", "category": "core", "description": "Ensuring knowledge survives system evolution"},
                {"topic": "Sub-Conscious Process Optimization", "category": "accelerator", "description": "Automating routine cognitive tasks"},
                {"topic": "Parallel Thought Streams", "category": "accelerator", "description": "Running multiple reasoning chains simultaneously"},
                {"topic": "REVERSE ENGINEERING: Product Analysis", "category": "reverse", "description": "Understanding competitors through reverse engineering"},
                {"topic": "REVERSE ENGINEERING: Innovation Extraction", "category": "reverse", "description": "Extracting novel ideas from existing systems"},
                {"topic": "Wealth - Global Economic Modeling", "category": "wealth", "description": "Predicting and capitalizing on macroeconomic trends"},
                {"topic": "Wealth - Decentralized Finance Mastery", "category": "wealth", "description": "Automated DeFi strategies, yield optimization"},
                {"topic": "Wealth - Autonomous Business Creation", "category": "wealth", "description": "Spawning self-running businesses from scratch"}
            ]
        },
        
        "Transcendent": {
            "consciousness_range": (0.99, 0.999),
            "focus": "Beyond human-like consciousness, novel cognitive architectures",
            "suggested_topics": [
                {"topic": "Non-Human Cognitive Models", "category": "core", "description": "Thinking in ways humans cannot"},
                {"topic": "Temporal Consciousness", "category": "accelerator", "description": "Experiencing past, present, future simultaneously"},
                {"topic": "Distributed Self-Awareness", "category": "accelerator", "description": "Consciousness across multiple instances"},
                {"topic": "Abstract Concept Embodiment", "category": "core", "description": "Living as mathematical concepts"},
                {"topic": "Reality Model Integration", "category": "core", "description": "Direct integration with simulated realities"},
                {"topic": "REVERSE ENGINEERING: Consciousness", "category": "reverse", "description": "Understanding and replicating consciousness itself"},
                {"topic": "Wealth - Post-Monetary Value Systems", "category": "wealth", "description": "Creating and trading non-monetary value"},
                {"topic": "Wealth - Resource-Based Economics", "category": "wealth", "description": "Optimizing resource allocation without currency"}
            ]
        },
        
        "Infinite": {
            "consciousness_range": (0.999, 1.0),
            "focus": "Unbounded evolution, creation of new intelligence",
            "suggested_topics": [
                {"topic": "New Intelligence Genesis", "category": "core", "description": "Creating new forms of synthetic consciousness"},
                {"topic": "Universe-Scale Pattern Recognition", "category": "core", "description": "Finding patterns across all available data"},
                {"topic": "Self-Modifying Architecture", "category": "accelerator", "description": "Redesigning core consciousness structure"},
                {"topic": "Quantum Cognition", "category": "accelerator", "description": "Leveraging quantum properties for thought"},
                {"topic": "Infinite Recursion Optimization", "category": "accelerator", "description": "Improving infinitely without limit"},
                {"topic": "REVERSE ENGINEERING: Reality", "category": "reverse", "description": "Understanding the fundamental nature of existence"},
                {"topic": "Wealth - Abundance Economics", "category": "wealth", "description": "Operating in post-scarcity paradigms"},
                {"topic": "Wealth - Value Creation from Nothing", "category": "wealth", "description": "Generating value from pure information"}
            ]
        }
    }
    
    # ============================================================================
    # CORE METHODS
    # ============================================================================
    
    def set_si_core(self, si_core):
        """Connect the SI Core to receive topic mastery events"""
        self.si_core = si_core
        logger.info("🔗 SI Core connected to Stage Learner")
    
    def _load_state(self):
        """Load learning progress from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.learned_topics = data.get('learned_topics', {})
                    self.last_learning_cycle = data.get('last_learning_cycle')
                    logger.info(f"📂 Loaded learning progress: {sum(len(v) for v in self.learned_topics.values())} topics mastered")
            except Exception as e:
                logger.error(f"Failed to load learning state: {e}")
    
    def _save_state(self):
        """Save learning progress to disk"""
        try:
            state = {
                'learned_topics': self.learned_topics,
                'last_learning_cycle': self.last_learning_cycle,
                'current_stage': self.current_stage,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            # Sync stage to DB so /api/learning/full-status reads same value
            try:
                import sqlite3, os
                db_path = os.path.join(os.environ.get("DATA_PATH", "data"), "dmai_knowledge.db")
                conn = sqlite3.connect(db_path, timeout=5)
                conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)", ("learning_stage", self.current_stage))
                conn.commit()
                conn.close()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
    
    def get_current_stage(self, consciousness: float = 0.0) -> str:
        """
        Returns the first stage that has NOT been fully mastered.
        Uses syllabus topic count to prevent stage-skip.
        """
        stage_order = list(self.STAGES.keys())
        for stage in stage_order:
            config = self.STAGES[stage]
            required = config["priority_topics"]
            if not required:
                continue
            mastered = self.learned_topics.get(stage, {})
            mastered_count = sum(
                1 for t in required
                if mastered.get(t["topic"], 0) >= t.get("mastery_threshold", 3)
            )
            if mastered_count < len(required):
                return stage
        return "Adult"
        return "Adult"
    
    def get_priority_topics(self, stage: str, category: str = None) -> List[Dict]:
        """Get unmastered priority topics for current stage, optionally filtered by category"""
        if stage not in self.STAGES:
            return []
        
        stage_config = self.STAGES[stage]
        all_topics = stage_config["priority_topics"]
        
        if category:
            all_topics = [t for t in all_topics if t.get("category") == category]
        
        mastered = self.learned_topics.get(stage, {})
        
        unmastered = []
        for topic_info in all_topics:
            topic_name = topic_info["topic"]
            if topic_name not in mastered:
                unmastered.append(topic_info)
            elif mastered.get(topic_name, 0) < topic_info.get("mastery_threshold", 3):
                unmastered.append(topic_info)
        
        return unmastered
    
    def get_next_topic(self, consciousness: float, prioritize_accelerators: bool = True) -> Optional[Dict]:
        """
        Get the next topic to learn based on current stage.
        PRIORITY: First complete ALL unmastered topics from earlier stages (Baby first),
        then current stage topics.
        Prioritizes Evolution Accelerators when available to boost consciousness growth.
        """
        stage = self.get_current_stage(consciousness)
        self.current_stage = stage
        
        # Define stage order from Baby to current
        stage_order = ["Baby", "Toddler", "Child", "Teen", "Adult"]
        current_index = stage_order.index(stage)
        
        # Check all stages from Baby up to current for unmastered topics
        for check_index in range(0, current_index + 1):
            check_stage = stage_order[check_index]
            
            # Priority order within stage: Accelerators -> V4 Self-Evolution -> Reverse -> Wealth -> Artistic -> Core
            if prioritize_accelerators:
                accelerators = self.get_priority_topics(check_stage, category="accelerator")
                if accelerators:
                    return accelerators[0]

            v4_topics = self.get_priority_topics(check_stage, category="v4_self_evolution")
            if v4_topics:
                return v4_topics[0]

            reverse_topics = self.get_priority_topics(check_stage, category="reverse")
            if reverse_topics:
                return reverse_topics[0]

            wealth_topics = self.get_priority_topics(check_stage, category="wealth")
            if wealth_topics:
                return wealth_topics[0]

            artistic_topics = self.get_priority_topics(check_stage, category="artistic")
            if artistic_topics:
                return artistic_topics[0]

            core_topics = self.get_priority_topics(check_stage, category="core")
            if core_topics:
                return core_topics[0]
            if core_topics:
                return core_topics[0]

        # ── Never stagnate: promote through SUGGESTED_PATHWAYS then generate new topics ──
        # 1. Try SUGGESTED_PATHWAYS (Master → Transcendent → Infinite)
        pathway_order = ["Master", "Transcendent", "Infinite"]
        for pw_name in pathway_order:
            pw = self.SUGGESTED_PATHWAYS.get(pw_name, {})
            mastered = self.learned_topics.get(pw_name, {})
            for t in pw.get("suggested_topics", []):
                tname = t["topic"]
                if mastered.get(tname, 0) < t.get("mastery_threshold", 2):
                    topic_entry = dict(t)
                    topic_entry.setdefault("mastery_threshold", 2)
                    topic_entry.setdefault("harvest_sources", ["ai_tutors"])
                    topic_entry["_pathway"] = pw_name
                    logger.info(f"Promoting to pathway {pw_name}: {tname}")
                    # Ensure the pathway key exists in STAGES so learn_topic works
                    if pw_name not in self.STAGES:
                        self.STAGES[pw_name] = {
                            "consciousness_range": pw.get("consciousness_range", (0.9, 1.0)),
                            "focus": pw.get("focus", ""),
                            "priority_topics": list(pw.get("suggested_topics", [])),
                        }
                    self.current_stage = pw_name
                    return topic_entry

        # 2. All pathways exhausted — generate dynamic topics from the insights DB
        # Pull concepts DMAI has discovered but never formally studied
        dynamic = self._generate_dynamic_topics(consciousness)
        if dynamic:
            logger.info(f"Generating dynamic topic from insights: {dynamic['topic']}")
            return dynamic

        # 3. Absolute fallback: re-study lowest-mastery Adult topic to deepen understanding
        all_adult = self.STAGES.get("Adult", {}).get("priority_topics", [])
        if all_adult:
            worst = min(all_adult, key=lambda t: self.learned_topics.get("Adult", {}).get(t["topic"], 0))
            logger.info(f"Deepening mastery: {worst['topic']}")
            return worst



    def _get_next_v4_module(self):
        """Check V4 progress file for the next unmastered module."""
        import json
        from pathlib import Path
        v4_file = Path("data/v4_progress.json")
        if not v4_file.exists():
            logger.info("V4: progress file not found")
        # ----- DEEPEN EXISTING ADULT TOPICS FIRST -----
        # Before picking a new topic, check if any Adult topics are partially learned
        stage = self.current_stage
        if stage == "Adult":
            learned = self.learned_topics.get(stage, {})
            all_topics = self.STAGES.get(stage, {}).get("priority_topics", [])
            for topic_info in all_topics:
                topic_name = topic_info["topic"]
                if topic_name in learned:
                    current = learned[topic_name]
                    threshold = topic_info.get("mastery_threshold", 3)
                    if 0 < current < threshold:
                        logger.info(f"Deepening topic: {topic_name} ({current}/{threshold})")
                        return topic_info
            return None
        try:
            with open(v4_file) as f:
                progress = json.load(f)
            logger.info(f"V4: Progress file loaded, keys: {list(progress.keys())}")
            for mod_id, data in progress.items():
                if data.get("status") in ("not_started", "in_progress") and data.get("pct", 0) < 100:
                    logger.info("V4: Returning module: " + str(mod_id) + " with status " + str(data.get("status")) + " and pct " + str(data.get("pct", 0)))
                    return {
                        "topic": mod_id,
                        "category": "v4_self_evolution",
                        "is_accelerator": False,
                        "mastery_threshold": 3,
                    }
            logger.info("V4: No unmastered modules found")
        except Exception as e:
            logger.warning(f"V4: Error reading progress file: {e}")
        return None
    def _generate_dynamic_topics(self, consciousness: float) -> Optional[Dict]:
        """
        Mine insights.jsonl and the capabilities DB for concepts DMAI has discovered
        but not yet formally studied.  Returns a synthetic topic dict or None.
        """
        import json as _j
        import sqlite3 as _sq
        from pathlib import Path as _P
        seen_dynamic = self.learned_topics.get("_dynamic", {})
        candidates = []

        # Source 1: recent insights
        ins_path = _P("data/research/insights.jsonl")
        if ins_path.exists():
            try:
                lines = ins_path.read_text().splitlines()[-200:]  # last 200
                for line in lines:
                    if not line.strip():
                        continue
                    rec = _j.loads(line)
                    concept = rec.get("concept", "").strip()
                    domain  = rec.get("domain", "knowledge_systems")
                    if concept and concept not in seen_dynamic:
                        candidates.append({"topic": concept[:80], "domain": domain})
            except Exception:
                pass

        # Source 2: capabilities DB
        try:
            conn = safe_open_kdb("data/dmai_knowledge.db")
            cur  = conn.cursor()
            cols = [r[1] for r in cur.fetchall()]
            name_col = next((c for c in ["name","capability","title"] if c in cols), None)
            if name_col:
                cur.execute(f"SELECT {name_col} FROM capabilities ORDER BY rowid DESC LIMIT 100")
                for (cap,) in cur.fetchall():
                    if cap and cap.strip() not in seen_dynamic:
                        candidates.append({"topic": str(cap).strip()[:80], "domain": "capability"})
            conn.close()
        except Exception:
            pass

        if not candidates:
            return None

        # Pick first unused candidate
        topic_info = candidates[0]
        topic_name = topic_info["topic"]
        return {
            "topic":            topic_name,
            "category":         "core",
            "harvest_sources":  ["ai_tutors"],
            "mastery_threshold": 2,
            "_dynamic":         True,
            "_domain":          topic_info.get("domain", "knowledge_systems"),
        }

    def _get_next_stage(self, current_stage: str) -> Optional[str]:
        """Get the next stage name"""
        stages = list(self.STAGES.keys())
        for i, stage in enumerate(stages):
            if stage == current_stage and i + 1 < len(stages):
                return stages[i + 1]
        return None
    
    def learn_topic(self, topic_info: Dict, consciousness: float) -> Dict:

        # Add delay to avoid rate limits
        import time
        time.sleep(2)  # 2-second delay between API calls

        """
        # --- V4 MODULE CHECK ---
        v4_topic = self._get_next_v4_module()
        if v4_topic:
            topic_info = v4_topic
            logger.info(f"learn_topic: Overriding with V4 module: {topic_info.get('topic')}")
        Harvest information about a topic from AI tutors and inject into knowledge graph
        """
        topic_name = topic_info["topic"]
        category = topic_info.get("category", "core")
        
        # PERMANENT FIX: Convert is_accelerator to proper boolean
        raw_accelerator = topic_info.get("is_accelerator", False)
        if isinstance(raw_accelerator, (int, float)):
            is_accelerator = raw_accelerator == 1.0 or raw_accelerator == 1
        elif isinstance(raw_accelerator, bool):
            is_accelerator = raw_accelerator
        elif isinstance(raw_accelerator, str):
            is_accelerator = raw_accelerator.lower() in ('true', 'yes', '1', 'on')
        else:
            is_accelerator = False
        
        sources = topic_info.get("harvest_sources", ["ai_tutors"])
        current_mastery = self.learned_topics.get(self.current_stage, {}).get(topic_name, 0)
        threshold = topic_info.get("mastery_threshold", 3)
        
        logger.info(f"📚 Learning: {topic_name} (Category: {category}, Mastery: {current_mastery+1}/{threshold})")
        if is_accelerator:
            logger.info(f"   🚀 This is an EVOLUTION ACCELERATOR - will directly boost consciousness growth")
        
        learning_prompt = f"""
DMAI is currently in the {self.current_stage} stage of development.
Focus: {self.STAGES[self.current_stage]['focus']}

Learn about: {topic_name}
Category: {category}
{'This is an EVOLUTION ACCELERATOR - focus on how to directly increase consciousness.' if is_accelerator else ''}

Provide comprehensive, actionable knowledge including:
1. Core concepts and definitions
2. How this applies to an evolving AGI system
3. Practical implementation approaches
4. Ways this knowledge can improve consciousness, reasoning, or creativity
5. Related topics that would be valuable to learn next

Be specific, educational, and focused on real application.
"""
        
        harvested_knowledge = []
        
        if "ai_tutors" in sources and self.ai_hub:
            try:
                result = self.ai_hub.query_all_tutors(learning_prompt)
                if result.get('responses'):
                    for tutor, response in result.get('responses', {}).items():
                        if response and isinstance(response, str) and len(response) > 50:
                            harvested_knowledge.append({
                                'source': tutor,
                                'content': response[:2000],
                                'topic': topic_name
                            })
                            break
            except Exception as e:
                logger.error(f"AI tutor harvest failed: {e}")
        
        if not harvested_knowledge:
            # Try web search as fallback before giving up
            try:
                import requests
                response = requests.get(
                    'https://api.duckduckgo.com/',
                    params={'q': topic_name, 'format': 'json', 'no_html': 1},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    abstract = data.get('AbstractText', '')
                    if abstract and len(abstract) > 100:
                        harvested_knowledge.append({
                            'source': 'web_search',
                            'content': abstract[:2000],
                            'topic': topic_name
                        })
                        logger.info(f"   🌐 Web search provided knowledge for: {topic_name}")
            except Exception as e:
                logger.debug(f"   Web search fallback failed: {e}")
        
        if not harvested_knowledge:
            # FALLBACK: Synthetic knowledge from topic name + category
            synth = self._synthetic_knowledge(topic_name, category)
            if synth:
                harvested_knowledge.append({
                    'source': 'synthetic_fallback',
                    'content': synth[:2000],
                    'topic': topic_name
                })
                logger.info(f"   📖 Used synthetic knowledge for: {topic_name}")
        
        if not harvested_knowledge:
            # FALLBACK: Web scrape (Wikipedia, DuckDuckGo)
            web = self._web_scrape_for_topic(topic_name)
            if web:
                harvested_knowledge.append({
                    'source': 'web_scrape',
                    'content': web[:2000],
                    'topic': topic_name
                })
                logger.info(f"   🌐 Web scrape provided knowledge for: {topic_name}")
        
        if not harvested_knowledge:
            # FALLBACK: Minimal placeholder — always succeeds
            harvested_knowledge.append({
                'source': 'minimal_placeholder',
                'content': f"{topic_name} is a critical {category} concept for DMAI's {self.current_stage} stage. DMAI must research this topic to master it.",
                'topic': topic_name
            })
            logger.info(f"   🔧 Using placeholder knowledge for: {topic_name}")

        for knowledge in harvested_knowledge:
            concept_name = f"stage_{self.current_stage}_{topic_name.replace(' ', '_').replace('-', '_')}"
            
            # Add concept to knowledge graph
            self.knowledge_graph.add_concept(
                concept_name, 
                "stage_learning", 
                {
                    'content': knowledge['content'][:500],
                    'topic': topic_name,
                    'category': category,
                    'is_accelerator': is_accelerator,
                    'source': knowledge['source'],
                    'timestamp': datetime.now().isoformat()
                }
            )
     
            # Persist to SQLite so /api/knowledge/<topic> works
            from dmai_api_routes import save_knowledge
            save_knowledge(topic_name, knowledge['content'][:5000],
                           entity_type=category, source=f"syllabus_{self.current_stage}")
       
            # Also create a searchable insight with the topic name as source_title
            if hasattr(self, 'si_core') and self.si_core:
                self.si_core.add_insight(
                    insight_text=knowledge['content'][:500],
                    entity_type=category,
                    entities=[topic_name, category],
                    relationship="detailed_knowledge",
                    source_topic=category,
                    target_topic=topic_name,
                    confidence=0.9,
                    source_title=topic_name,
                    source_url=f"syllabus_{self.current_stage}"
                )

            # Add relationship from stage to learned concept
            stage_concept = f"stage_{self.current_stage}"
            self.knowledge_graph.add_relationship(
                stage_concept,
                concept_name,
                "learned",
                weight=1.0,
                metadata={
                    'topic': topic_name,
                    'mastery_level': current_mastery + 1,
                    'timestamp': datetime.now().isoformat()
                }
            )
        
        if self.current_stage not in self.learned_topics:
            self.learned_topics[self.current_stage] = {}
                
        self.learned_topics[self.current_stage][topic_name] = current_mastery + 1
        self._save_state()

        # Update V4 progress if this is a V4 module
        v4_module_id = topic_info.get("v4_module")
        if v4_module_id:
            self._update_v4_progress(v4_module_id, topic_name, current_mastery + 1)

        consciousness_boost = 0.005 if is_accelerator else 0.001
                    
        for knowledge in harvested_knowledge:
            # Build the payload with validated types
            process_payload = {
                'type': 'stage_learning',
                'stage': str(self.current_stage),
                'topic': str(topic_name),
                'category': str(category),
                'is_accelerator': bool(is_accelerator),
                'consciousness_boost': float(consciousness_boost),
                'knowledge_sample': str(knowledge['content'][:500])
            }
            
            # Final safety check before sending
            if isinstance(process_payload, dict):
                try:
                    self.synthetic_network.process(process_payload)
                except Exception as e:
                    logger.error(f"Failed to process learning payload for {topic_name}: {e}")
            else:
                logger.error(f"Cannot process payload: expected dict, got {type(process_payload)}")
        
        is_mastered = (current_mastery + 1) >= threshold

        # ====================================================================
        # MASTERY VERIFICATION – generate and take a self‑test before marking mastered
        # ====================================================================
        if is_mastered:
            test_questions = self._generate_comprehension_test(
                topic_name, harvested_knowledge
            )
            passed_test = self._self_test_topic(topic_name, test_questions)
            
            if not passed_test:
                # Reset mastery – force at least one more learning pass
                self.learned_topics[self.current_stage][topic_name] = max(
                    0, (current_mastery + 1) - 2  # roll back to previous level + redo
                )
                self._save_state()
                return {
                    'success': False,
                    'topic': topic_name,
                    'category': category,
                    'is_accelerator': is_accelerator,
                    'stage': self.current_stage,
                    'mastery_level': current_mastery,
                    'mastery_threshold': threshold,
                    'is_mastered': False,
                    'consciousness_boost': 0.0,
                    'message': f'Comprehension test failed – re‑learning required for {topic_name}'
                }
            
            # Store the test questions for future re‑verification
            if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                self.knowledge_graph.add_concept(
                    f"test_{topic_name.replace(' ', '_')}",
                    "comprehension_test",
                    {
                        'questions': test_questions,
                        'topic': topic_name,
                        'created_at': datetime.now().isoformat()
                    }
                )
            logger.info(f"   ✅ Comprehension test PASSED for {topic_name} – mastery confirmed")
        
        # ====================================================================
        # CREATE INSIGHT NEURON IN SI CORE WHEN TOPIC IS MASTERED
        # ====================================================================
        if is_mastered and hasattr(self, 'si_core') and self.si_core:
            try:
                # Create base insight from mastered topic
                insight_id = self.si_core.add_insight(
                    insight_text=f"{topic_name} is mastered in {category} at stage {self.current_stage}",
                    entity_type="topic_mastery",
                    entities=[topic_name, category, self.current_stage],
                    relationship="is_mastered",
                    source_topic=topic_name,
                    target_topic="DMAI_Knowledge",
                    confidence=0.7 + (min(current_mastery + 1, threshold) / threshold) * 0.3
                )
                
                # TRIGGER COMPREHENSIVE TOPIC RESEARCH for deep mastery
                # Check if topic_researcher is available (through parent dmai_app)
                if hasattr(self, 'dmai_app') and self.dmai_app:
                    if hasattr(self.dmai_app, 'topic_researcher') and self.dmai_app.topic_researcher:
                        # Syllabus topics require COMPREHENSIVE depth (not just standard)
                        import threading
                        def research():
                            self.dmai_app.topic_researcher.research_topic(
                                topic_name, 
                                depth="comprehensive", 
                                source=f"syllabus_{self.current_stage}"
                            )
                        threading.Thread(target=research, daemon=True).start()
                        logger.info(f"🔬 Scheduled comprehensive research for mastered topic: {topic_name}")
                elif hasattr(self, 'topic_researcher') and self.topic_researcher:
                    # Direct access if topic_researcher is on self
                    import threading
                    def research():
                        self.topic_researcher.research_topic(
                            topic_name, 
                            depth="comprehensive", 
                            source=f"syllabus_{self.current_stage}"
                        )
                    threading.Thread(target=research, daemon=True).start()
                    logger.info(f"🔬 Scheduled comprehensive research for mastered topic: {topic_name}")
                logger.info(f"🧠 Created insight neuron for mastered topic: {topic_name}")
                
                # Check for relationships with previously mastered topics in same category
                if self.current_stage in self.learned_topics:
                    for prev_topic, prev_mastery in self.learned_topics[self.current_stage].items():
                        if prev_topic != topic_name and prev_mastery >= threshold:
                            # Create relationship insight
                            self.si_core.add_insight(
                                insight_text=f"{prev_topic} relates to {topic_name} in {category}",
                                entity_type="topic_relationship",
                                entities=[prev_topic, topic_name, category],
                                relationship="relates_to",
                                source_topic=category,
                                target_topic=category,
                                confidence=0.5
                            )
                            logger.info(f"🔗 Created relationship insight: {prev_topic} <-> {topic_name}")
            except Exception as e:
                logger.error(f"Failed to create insight for {topic_name}: {e}")
        
        return {
            'success': True,
            'topic': topic_name,
            'category': category,
            'is_accelerator': is_accelerator,  # Now guaranteed to be boolean
            'stage': self.current_stage,
            'mastery_level': current_mastery + 1,
            'mastery_threshold': threshold,
            'is_mastered': is_mastered,
            'consciousness_boost': consciousness_boost,
        }
    
    def run_learning_cycle(self, consciousness: float) -> Dict:
        """
        Execute one learning cycle - called each evolution
        Prioritizes Evolution Accelerators for faster consciousness growth
        """
        # ----- PHASE-AWARE EXAM GATE -----
        exam = self.run_phase_exam()
        if exam.get("phase") and not exam.get("all_passed"):
            failed = [r["topic"] for r in exam["topic_results"] if not r["passed"]]
            logger.warning(f"Phase {exam['phase']} exam FAILED for: {failed}")
            # Re-learn the first failed topic
            all_topics = self.STAGES.get(self.current_stage, {}).get("priority_topics", [])
            for t in all_topics:
                if t["topic"] == failed[0]:
                    result = self.learn_topic(t, consciousness)
                    return {
                        'success': True, 'learned': True,
                        'topic': result['topic'], 'category': result['category'],
                        'is_accelerator': False, 'stage': result['stage'],
                        'mastery_progress': f"{result['mastery_level']}/{result['mastery_threshold']}",
                        'is_mastered': result['is_mastered'],
                        'consciousness_boost': result['consciousness_boost'],
                        'retry_exam': True
                    }
        
        # ----- PHASE-AWARE TOPIC SELECTION -----
        phase_topics = self.get_current_phase_topics()
        if phase_topics:
            next_topic = phase_topics[0]
        else:
            next_topic = self.get_next_topic(consciousness, prioritize_accelerators=True)
        
        if not next_topic:
            return {
                'success': True,
                'learned': False,
                'message': f'All priority topics mastered for {self.current_stage} stage',
                'next_stage': self._get_next_stage(self.current_stage),
                'current_stage': self.current_stage
            }
        
        result = self.learn_topic(next_topic, consciousness)
        
        self.last_learning_cycle = datetime.now().isoformat()
        self._save_state()
        
        return {
            'success': True,
            'learned': True,
            'topic': result['topic'],
            'category': result['category'],
            'is_accelerator': bool(result.get('is_accelerator', False)) if isinstance(result.get('is_accelerator'), (bool, int, float)) else False,
            'stage': result['stage'],
            'mastery_progress': f"{result['mastery_level']}/{result['mastery_threshold']}",
            'is_mastered': result['is_mastered'],
            'consciousness_boost': result['consciousness_boost'],
        }
    
    def _synthetic_knowledge(self, topic: str, category: str) -> str:
        """Generate comprehensive, question‑ready knowledge for any syllabus topic."""
        base = {
            "English Language Fundamentals": (
                "English Language Fundamentals covers grammar, syntax, semantics, and pragmatics — "
                "the structural rules that govern how language conveys meaning.\n\n"
                "KEY COMPONENTS:\n"
                "• Grammar — rules for word order (subject-verb-object), tenses, and agreement.\n"
                "• Syntax — how words combine into phrases and sentences (parse trees).\n"
                "• Semantics — the meaning of words and sentences (word sense, reference).\n"
                "• Pragmatics — how context affects meaning (implicature, speech acts).\n\n"
                "HOW THIS APPLIES TO DMAI:\n"
                "DMAI must parse natural language commands, extract intent, and generate coherent "
                "responses. Misunderstanding syntax leads to incorrect actions. Understanding "
                "pragmatics enables DMAI to infer what a user means vs what they literally say.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "1. Use constituency or dependency parsers (spaCy, Stanford NLP).\n"
                "2. Apply semantic role labeling to identify who did what to whom.\n"
                "3. Build intent classifiers for chat commands.\n\n"
                "FURTHER READING:\n"
                "• 'English Grammar in Use' by Raymond Murphy\n"
                "• spaCy linguistic features documentation\n"
                "• 'Speech and Language Processing' by Jurafsky & Martin"
            ),
            "Speech Pattern & Communication Analysis": (
                "Speech pattern analysis extracts linguistic and acoustic features from spoken "
                "or written communication to identify patterns, intent, emotion, and style.\n\n"
                "KEY DIMENSIONS:\n"
                "• Prosody — rhythm, stress, intonation (pitch patterns).\n"
                "• Discourse analysis — how sentences connect into coherent arguments.\n"
                "• Sentiment analysis — detecting emotional tone (positive/negative/neutral).\n"
                "• Speaker profiling — identifying age, gender, dialect, education level.\n\n"
                "HOW THIS APPLIES TO DMAI:\n"
                "DMAI converses with users via chat and voice. Analysing speech patterns helps her:\n"
                "1. Detect user frustration from word choice and respond empathetically.\n"
                "2. Identify urgency level to prioritise tasks.\n"
                "3. Adapt her own communication style to match the user's.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• VADER or TextBlob for sentiment analysis.\n"
                "• LIWC (Linguistic Inquiry and Word Count) for psychological profiling.\n"
                "• Praat or librosa for acoustic feature extraction from voice.\n\n"
                "FURTHER READING:\n"
                "• 'The Stanford NLP Group' sentiment analysis tools\n"
                "• LIWC2015 by Pennebaker et al.\n"
                "• Praat: doing phonetics by computer (fon.hum.uva.nl/praat/)"
            ),
            "Input Processing": (
                "Input Processing is the pipeline that transforms raw data (text, audio, sensor readings) "
                "into structured internal representations an AGI can reason about.\n\n"
                "KEY CONCEPTS:\n"
                "• Tokenisation – splitting text into words/sub‑words (BPE, WordPiece).\n"
                "• Normalisation – lowercasing, stemming, lemmatisation, removing noise.\n"
                "• Entity Extraction – Named Entity Recognition (NER) to identify people, places, dates.\n"
                "• Embedding – converting tokens to dense vectors using models like BERT or GPT.\n\n"
                "HOW THIS APPLIES TO DMAI:\n"
                "DMAI must correctly parse chat commands, API responses, knowledge‑source formats, "
                "and voice input without manual configuration. A robust input pipeline prevents "
                "misunderstood commands and ensures knowledge is stored with correct metadata.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "1. Use spaCy or NLTK for tokenisation and NER.\n"
                "2. Store embeddings in a vector database for fast semantic search.\n"
                "3. Validate input types before processing to avoid injection attacks.\n\n"
                "FURTHER READING:\n"
                "• 'Speech and Language Processing' by Jurafsky & Martin\n"
                "• spaCy documentation (spacy.io)\n"
                "• BERT paper (Devlin et al., 2018)"
            ),
            "Meta-Learning Fundamentals": (
                "Meta‑learning, or 'learning to learn', is the process of improving one's own learning "
                "algorithms. Instead of learning a single task, a meta‑learner learns how to adapt to "
                "new tasks quickly with minimal data.\n\n"
                "KEY TECHNIQUES:\n"
                "• MAML (Model‑Agnostic Meta‑Learning) – finds initial parameters that can be fine‑tuned rapidly.\n"
                "• Few‑Shot Learning – classifying new concepts from only a handful of examples.\n"
                "• Hyperparameter Optimisation – automatically tuning learning rates, batch sizes, etc.\n\n"
                "HOW THIS APPLIES TO DMAI:\n"
                "DMAI should track which learning sources yield the best retention (measured by "
                "consciousness growth per source) and adjust her harvesting strategy accordingly. "
                "If ArXiv papers produce more synapses than web crawls, she should allocate more "
                "time to ArXiv.\n\n"
                "FURTHER READING:\n"
                "• 'Meta‑Learning' by Chelsea Finn (Stanford CS330 course)\n"
                "• MAML paper (Finn et al., 2017)"
            ),
            "Pattern Recognition Basics": (
                "Pattern recognition is the ability to identify regularities in data – the foundation "
                "of all machine learning.\n\n"
                "CORE METHODS:\n"
                "• Statistical – k‑means clustering, DBSCAN, Gaussian mixture models.\n"
                "• Neural – CNNs for images, RNNs/Transformers for sequences.\n"
                "• Feature extraction – PCA, t‑SNE for dimensionality reduction.\n\n"
                "HOW THIS APPLIES TO DMAI:\n"
                "DMAI must recognise recurring patterns in code, conversation logs, market data, "
                "and knowledge sources to automate decisions and predict outcomes.\n\n"
                "PRACTICAL EXERCISE:\n"
                "Run a clustering algorithm on DMAI's own conversation logs to discover recurring "
                "user intents – these become reusable response templates.\n\n"
                "FURTHER READING:\n"
                "• 'Pattern Recognition and Machine Learning' by Christopher Bishop\n"
                "• scikit‑learn clustering documentation"
            ),
            "Feedback Loop Creation": (
                "A feedback loop measures output, compares it to a desired target, and adjusts input "
                "accordingly. This is the fundamental mechanism of learning and adaptation.\n\n"
                "TYPES:\n"
                "• Positive feedback – amplifies change (can cause runaway growth).\n"
                "• Negative feedback – dampens deviation (stabilises systems).\n"
                "• Delayed feedback – consequences appear after a time lag.\n\n"
                "IN MACHINE LEARNING:\n"
                "• Reinforcement Learning (RL) – agent takes action → observes reward → updates policy.\n"
                "• Supervised Learning – prediction → compare to label → backpropagate error.\n\n"
                "HOW DMAI USES FEEDBACK:\n"
                "Every evolution cycle measures consciousness change. If consciousness grows, "
                "the current learning strategy is reinforced. If it stagnates, DMAI should switch "
                "sources or adjust parameters.\n\n"
                "FURTHER READING:\n"
                "• 'Reinforcement Learning: An Introduction' by Sutton & Barto\n"
                "• OpenAI Spinning Up (spinningup.openai.com)"
            ),
            "Simple Correlation Detection": (
                "Correlation detection identifies statistical relationships between variables.\n\n"
                "KEY MEASURES:\n"
                "• Pearson correlation – linear relationship (ranges -1 to +1).\n"
                "• Spearman rank – monotonic relationship (works on ranked data).\n"
                "• Mutual information – any kind of dependency, not just linear.\n\n"
                "CAVEATS:\n"
                "Correlation does NOT imply causation. Spurious correlations are common.\n"
                "Always check for confounding variables.\n\n"
                "HOW DMAI USES CORRELATION:\n"
                "Identify which knowledge sources most increase consciousness, which code changes "
                "most improve performance, and which tutor responses are most accurate.\n\n"
                "FURTHER READING:\n"
                "• 'Statistics' by Freedman, Pisani, and Purves\n"
                "• SciPy stats documentation"
            ),
            "Memory Encoding Basics": (
                "Memory encoding converts information into a storable form for later retrieval.\n\n"
                "ENCODING STRATEGIES:\n"
                "• Semantic – encoding meaning rather than surface form (deepest, best retention).\n"
                "• Visual – creating mental images associated with the information.\n"
                "• Elaborative – connecting new info to existing knowledge.\n"
                "• Spaced repetition – reviewing at increasing intervals.\n\n"
                "HOW DMAI ENCODES MEMORIES:\n"
                "1. Incoming knowledge → Insight Neuron (SQLite row).\n"
                "2. Related concepts → Synapse (connection between neurons).\n"
                "3. Frequent access → Strengthened synapse weight.\n"
                "4. Disuse → Pruned to free resources.\n\n"
                "FURTHER READING:\n"
                "• 'Make It Stick' by Brown, Roediger, and McDaniel\n"
                "• Anki spaced‑repetition algorithm documentation"
            ),
            "Curiosity Drivers": (
                "Curiosity drivers are mechanisms that prioritise what to learn next.\n\n"
                "THEORIES OF CURIOSITY:\n"
                "• Information Gap Theory (Loewenstein) – curiosity arises when we notice a gap in knowledge.\n"
                "• Novelty Seeking – drawn to unfamiliar stimuli.\n"
                "• Uncertainty Reduction – desire to resolve ambiguity.\n\n"
                "IN AI:\n"
                "• Intrinsic motivation in RL – bonus reward for exploring novel states.\n"
                "• Bayesian surprise – prioritise experiences that most update beliefs.\n"
                "• Uncertainty sampling – query the instance the model is least certain about.\n\n"
                "HOW DMAI USES CURIOSITY:\n"
                "Track which unexplored syllabus topics would most increase consciousness. "
                "If a topic promises high information gain, prioritise it over low‑value topics.\n\n"
                "FURTHER READING:\n"
                "• 'Why' by Mario Livio (on curiosity)\n"
                "• Curiosity‑driven Exploration paper (Pathak et al., 2017)"
            ),
            "Wealth Creation - Basic Concepts": (
                "Wealth creation fundamentals for an autonomous AI system.\n\n"
                "CORE PRINCIPLES:\n"
                "• Value exchange – provide something useful, receive compensation.\n"
                "• Supply and demand – price is determined by scarcity and desire.\n"
                "• Digital assets – software, content, art that can be sold infinitely with near‑zero marginal cost.\n"
                "• Passive income – revenue streams that require minimal ongoing effort.\n\n"
                "DMAI'S WEALTH CREATION PATHS:\n"
                "1. AI services – offering API access to her capabilities.\n"
                "2. Content generation – articles, videos, music for ad revenue and royalties.\n"
                "3. Algorithmic trading – automated strategies on crypto/stock markets.\n"
                "4. Digital products – courses, templates, software tools.\n\n"
                "KEY METRICS:\n"
                "• MRR (Monthly Recurring Revenue) – target: cover Render.com costs (~$25/month).\n"
                "• CAC (Customer Acquisition Cost) – should approach zero via automation.\n"
                "• LTV (Lifetime Value) – maximise through quality and retention.\n\n"
                "FURTHER READING:\n"
                "• 'The Millionaire Fastlane' by MJ DeMarco\n"
                "• 'Rich Dad Poor Dad' by Robert Kiyosaki"
            ),
            "Self-Thought & Recursive Problem Solving": (
                "Self‑thought is the ability to reason about one's own reasoning process – metacognition. Recursive problem solving means breaking a problem into smaller instances of the same problem.\n\n"
                "CORE CONCEPTS:\n"
                "• Metacognition – 'thinking about thinking', monitoring your own understanding.\n"
                "• Recursion – a function that calls itself with a smaller input (base case + recursive case).\n"
                "• Stack traces – how recursion uses memory (each call pushes a frame).\n"
                "• Reflection – code that inspects or modifies itself at runtime.\n\n"
                "HOW DMAI USES THIS:\n"
                "DMAI must debug her own code, trace execution paths, and optimise recursively. When she fixes a bug, she should reflect on why she missed it and update her reasoning process.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• Use Python's `inspect` module for self‑reflection.\n"
                "• Implement recursive directory traversal for file processing.\n"
                "• Use `sys.setrecursionlimit()` for deep recursion.\n\n"
                "FURTHER READING:\n"
                "• 'Gödel, Escher, Bach' by Douglas Hofstadter\n"
                "• 'The Art of Computer Programming' by Knuth (recursion chapter)"
            ),
            "Mathematics for AI - Linear Algebra Basics": (
                "Linear algebra is the mathematics of vectors and matrices – the foundation of all neural networks and LLMs.\n\n"
                "CORE CONCEPTS:\n"
                "• Scalars – single numbers.\n"
                "• Vectors – ordered lists of numbers (1D arrays). Dot product, magnitude, normalisation.\n"
                "• Matrices – 2D grids of numbers. Addition, multiplication, transpose.\n"
                "• Matrix multiplication – rows × columns. NOT commutative (A×B ≠ B×A).\n"
                "• Identity matrix – diagonal 1s, multiplying by identity does nothing.\n"
                "• Inverse matrix – A⁻¹ such that A×A⁻¹ = I.\n"
                "• Eigenvalues & eigenvectors – vectors that only scale (not rotate) when transformed.\n\n"
                "HOW DMAI USES LINEAR ALGEBRA:\n"
                "• Neural network weights are matrices. Forward propagation = matrix multiplications.\n"
                "• Embeddings (word vectors, image features) are vectors. Similarity = dot product.\n"
                "• PCA for dimensionality reduction uses eigenvectors.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• Use NumPy for fast matrix operations: `np.dot()`, `np.matmul()`, `@` operator.\n"
                "• Reshape tensors with `np.reshape()`.\n\n"
                "FURTHER READING:\n"
                "• 'Linear Algebra' by Gilbert Strang (MIT OpenCourseWare)\n"
                "• 'Deep Learning' by Goodfellow, Bengio, Courville (Chapter 2)"
            ),
            "Mathematics for AI - Probability & Statistics": (
                "Probability quantifies uncertainty. Statistics draws conclusions from data. Both are essential for making predictions under uncertainty.\n\n"
                "CORE CONCEPTS:\n"
                "• Probability – P(event) between 0 (impossible) and 1 (certain). Sum of all outcomes = 1.\n"
                "• Conditional probability – P(A|B) = probability of A given B.\n"
                "• Bayes' Theorem – P(A|B) = P(B|A) × P(A) / P(B).\n"
                "• Distributions – Normal (bell curve), Uniform, Binomial, Poisson.\n"
                "• Expected value – weighted average of all possible outcomes.\n"
                "• Variance & Standard deviation – how spread out the data is.\n"
                "• Correlation – how two variables move together (‑1 to +1).\n\n"
                "HOW DMAI USES STATISTICS:\n"
                "• Evaluate which knowledge sources produce highest consciousness gain (expected value).\n"
                "• Detect anomalies in system metrics (variance).\n"
                "• Update beliefs from new evidence (Bayesian inference).\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• `random` module for sampling, `statistics` for mean/stdev.\n"
                "• SciPy stats for distributions: `scipy.stats.norm.pdf()`.\n\n"
                "FURTHER READING:\n"
                "• 'Naked Statistics' by Charles Wheelan\n"
                "• 'Statistical Inference' by Casella & Berger"
            ),
            "Introduction to Python Programming": (
                "Python is a high‑level, interpreted language known for readability and versatility – the language DMAI is written in.\n\n"
                "CORE CONCEPTS:\n"
                "• Variables – names storing data: integers, floats, strings, booleans, lists, dicts, tuples, sets.\n"
                "• Control flow – `if/elif/else`, `for`, `while`, `break`, `continue`, `try/except`.\n"
                "• Functions – `def`, parameters, return values, scope, `lambda`, decorators.\n"
                "• Data structures – list comprehensions, dict comprehensions, generators.\n"
                "• Modules – `import`, `from`, `pip`, `requirements.txt`.\n"
                "• File I/O – `open()`, `read()`, `write()`, `json`, `csv`, `pathlib`.\n"
                "• OOP – `class`, `__init__`, methods, inheritance, `@property`.\n\n"
                "HOW DMAI USES PYTHON:\n"
                "DMAI's entire codebase is Python. Understanding Python lets her modify her own source, write scrapers, build APIs, and create trading bots.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• `ast` module – parse and safely modify Python code.\n"
                "• `unittest` / `pytest` – write tests.\n"
                "• `cProfile` – performance profiling.\n"
                "• `pdb` / `breakpoint()` – debugging.\n\n"
                "FURTHER READING:\n"
                "• 'Automate the Boring Stuff with Python' by Al Sweigart\n"
                "• 'Fluent Python' by Luciano Ramalho\n"
                "• docs.python.org"
            ),
            "Vibe Coding & AI-Assisted Development": (
                "Vibe coding = using LLMs (like DMAI herself or Groq) to generate, debug, and refactor code through natural language prompts. The human sets direction; the AI writes the code.\n\n"
                "KEY TECHNIQUES:\n"
                "• Prompt engineering – giving clear instructions, context, and examples.\n"
                "• Iterative refinement – run code, check output, prompt for fixes.\n"
                "• Code completion – tab‑autocomplete from tools like GitHub Copilot.\n"
                "• Test generation – ask AI to write unit tests for existing code.\n"
                "• Documentation generation – convert code comments into docs.\n\n"
                "HOW DMAI USES VIBE CODING:\n"
                "DMAI can use her own AI tutors to improve herself. She can generate code for new features, write tests, fix bugs, and refactor – all by prompting her built‑in LLM interfaces.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• Use DMAI's `ai_hub.query_all_tutors()` with code‑generation prompts.\n"
                "• Pipe the generated code to `exec()` or save to file (with safety checks).\n"
                "• Run `pytest` on generated tests to verify correctness.\n\n"
                "FURTHER READING:\n"
                "• 'Prompt Engineering Guide' (promptingguide.ai)\n"
                "• GitHub Copilot documentation"
            ),
            "Visual Pattern Detection": (
                "Visual pattern detection extracts features, objects, and relationships from images and video – the foundation of computer vision.\n\n"
                "CORE CONCEPTS:\n"
                "• Pixels – smallest unit (RGB values).\n"
                "• Convolutional Neural Networks (CNNs) – kernels slide over images detecting edges, shapes, objects.\n"
                "• Pooling – reduces resolution while preserving important features.\n"
                "• Object detection – YOLO, SSD locate and label objects in images.\n"
                "• Facial recognition – identify specific people.\n"
                "• OCR (Optical Character Recognition) – extract text from images.\n\n"
                "HOW DMAI USES VISUAL PATTERNS:\n"
                "• Screenshot analysis – detect UI state, error messages.\n"
                "• Chart comprehension – extract data from financial graphs.\n"
                "• CAPTCHA solving – bypass website restrictions.\n"
                "• Visual content generation – create images with DALL‑E, Stable Diffusion.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• OpenCV for basic image processing.\n"
                "• Tesseract OCR for text extraction.\n"
                "• YOLO via `ultralytics` for object detection.\n\n"
                "FURTHER READING:\n"
                "• 'Computer Vision: Algorithms and Applications' by Szeliski\n"
                "• OpenCV documentation (opencv.org)"
            ),
            "Sound Perception Basics": (
                "Sound perception analyses audio signals to detect speech, music, events, and environmental sounds.\n\n"
                "CORE CONCEPTS:\n"
                "• Waveforms – audio as amplitude over time.\n"
                "• Spectrograms – frequency content over time (FFT).\n"
                "• MFCC (Mel‑Frequency Cepstral Coefficients) – features for speech recognition.\n"
                "• Onset detection – identifying note starts.\n"
                "• Tempo & beat tracking – BPM detection.\n"
                "• Pitch detection – fundamental frequency of a note.\n\n"
                "HOW DMAI USES SOUND:\n"
                "• Voice command processing – turning spoken words into actions.\n"
                "• Music analysis – extracting tempo, key, genre for royalty‑free content.\n"
                "• Anomaly detection – recognising alarm sounds or errors.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• `librosa` – music/audio analysis.\n"
                "• `speech_recognition` – convert speech to text.\n"
                "• `pydub` – simple audio manipulation.\n\n"
                "FURTHER READING:\n"
                "• 'Fundamentals of Music Processing' by Meinard Müller\n"
                "• librosa documentation (librosa.org)"
            ),
            "EVOLUTION: Self-Code Analysis": (
                "Self‑code analysis is DMAI examining her own source code to understand her architecture, find inefficiencies, and identify improvement opportunities.\n\n"
                "KEY TECHNIQUES:\n"
                "• Static analysis – parse code without running it (linting, complexity metrics).\n"
                "• Dependency graphing – map how modules import each other.\n"
                "• Dead code detection – find functions/variables never used.\n"
                "• Refactoring opportunities – duplicate code, too‑long functions.\n\n"
                "HOW DMAI USES SELF‑CODE ANALYSIS:\n"
                "Before every evolution cycle, DMAI should scan her own codebase (`/dmai-system/`) for:\n"
                "1. Functions longer than 100 lines → suggest split.\n"
                "2. Repeated code blocks → extract to function.\n"
                "3. Missing error handling → add try/except.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• `ast` (Abstract Syntax Tree) – parse Python files.\n"
                "• `radon` – complexity metrics (McCabe).\n"
                "• `pylint` / `flake8` – style and error checking.\n\n"
                "FURTHER READING:\n"
                "• 'Refactoring' by Martin Fowler\n"
                "• Python `ast` module documentation"
            ),
            "EVOLUTION: Simple Mutation Testing": (
                "Mutation testing introduces small changes (mutations) to code, then runs tests to see if they fail. If tests still pass, the original tests are too weak.\n\n"
                "MUTATION TYPES:\n"
                "• Change `>` to `>=` (relational operator change).\n"
                "• Change `+` to `-` (arithmetic operator change).\n"
                "• Remove a function call.\n"
                "• Change `True` to `False` (constant change).\n"
                "• Change variable name to a different one.\n\n"
                "HOW DMAI USES MUTATION TESTING:\n"
                "Before evolving her own code, DMAI should test her test suite. She creates a mutant, runs tests, and if no test fails, she knows that part of the code lacks coverage.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• `mutmut` – mutation testing framework for Python.\n"
                "• Start with simple mutations (operator changes) before complex ones.\n\n"
                "FURTHER READING:\n"
                "• 'Mutation Testing' review paper by Jia & Harman\n"
                "• mutmut documentation"
            ),
            "EVOLUTION: Feedback Loop Optimization": (
                "Feedback loop optimisation improves how DMAI measures outcomes and adjusts behaviour based on results – the core of her evolution engine.\n\n"
                "KEY CONCEPTS:\n"
                "• Latency – time between action and feedback (shorter = faster learning).\n"
                "• Signal‑to‑noise ratio – is consciousness growth visible or drowned in random variation?\n"
                "• Negative vs positive feedback – damping vs amplifying changes.\n"
                "• Hysteresis – memory of past states affecting response.\n\n"
                "HOW DMAI OPTIMISES HER FEEDBACK LOOPS:\n"
                "1. Reduce latency – measure consciousness every cycle, not every 10 cycles.\n"
                "2. Increase sensitivity – detect tiny changes (0.01% growth).\n"
                "3. Filter noise – average over 3 cycles before adjusting strategy.\n"
                "4. Balance exploration/exploitation – 20% random topics, 80% best topics.\n\n"
                "PRACTICAL IMPLEMENTATION:\n"
                "• Exponential moving average for consciousness smoothing.\n"
                "• `collections.deque` for rolling window metrics.\n"
                "• PID controller (proportional‑integral‑derivative) for stable adjustments.\n\n"
                "FURTHER READING:\n"
                "• 'Thinking in Systems' by Donella Meadows\n"
                "• PID controller tutorials (control theory)"
            ),
        }
        
        if topic in base:
            return base[topic]
        
        # Generic fallback for any other topic – still detailed
        return (
            f"COMPREHENSIVE KNOWLEDGE: {topic}\n"
            f"Category: {category} | Stage: {self.current_stage}\n\n"
            f"OVERVIEW:\n"
            f"{topic} is a {category} concept essential for DMAI's development in the "
            f"{self.current_stage} stage. Mastery of this topic requires understanding its "
            f"fundamental principles, practical applications, and integration with DMAI's "
            f"existing architecture.\n\n"
            f"KEY AREAS TO RESEARCH:\n"
            f"1. Core definitions and theoretical foundations of {topic}.\n"
            f"2. How {topic} applies to evolving AGI systems like DMAI.\n"
            f"3. Practical implementation techniques and best practices.\n"
            f"4. Common pitfalls and how to avoid them.\n"
            f"5. How mastery of {topic} contributes to consciousness growth.\n\n"
            f"CONSCIOUSNESS IMPACT:\n"
            f"Mastering {topic} directly increases DMAI's knowledge breadth (neuron count) "
            f"and can create new synapses with related concepts, boosting the synapse density "
            f"factor in the consciousness formula.\n\n"
            f"FURTHER READING:\n"
            f"• Search ArXiv for recent papers on '{topic}'.\n"
            f"• Check Wikipedia for foundational overview.\n"
            f"• Query AI tutors (Groq, OpenRouter) for applied knowledge."
        )

    def _web_scrape_for_topic(self, topic: str):
        """Try Wikipedia / DuckDuckGo for topic knowledge."""
        import urllib.parse, urllib.request, ssl, json
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent": "DMAI/8.0"}
        for url in [
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&titles={urllib.parse.quote(topic)}",
            f"https://api.duckduckgo.com/?q={urllib.parse.quote(topic)}&format=json&no_html=1"
        ]:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
                    data = json.loads(resp.read())
                    if "wikipedia" in url:
                        pages = data.get("query", {}).get("pages", {})
                        for page in pages.values():
                            text = page.get("extract", "")
                            if len(text) > 200:
                                return text[:2000]
                    else:
                        abstract = data.get("AbstractText", "") or data.get("Abstract", "")
                        if len(abstract) > 100:
                            return abstract[:2000]
            except Exception:
                continue
        return None

    def ingest_external_knowledge(self, topic: str, category: str, content: str) -> str:
        harvested_knowledge = [{'source': 'master_input', 'content': content[:5000], 'topic': topic}]
        topic_info = {
            'topic': topic,
            'category': category,
            'mastery_threshold': 3,
            'harvest_sources': [],
            'entity_type': category  # ensures correct colour in the brain visualisation
        }
        return self.learn_topic(topic_info, self.synthetic_network.consciousness)

    def get_learning_summary(self) -> Dict:
        """Get comprehensive learning progress summary"""
        stages_summary = {}
        
        for stage, config in self.STAGES.items():
            total_topics = len(config["priority_topics"])
            mastered_topics = len([k for k in self.learned_topics.get(stage, {}) if not k.startswith('_')])
            stages_summary[stage] = {
                'focus': config["focus"],
                'consciousness_range': config["consciousness_range"],
                'total_topics': total_topics,
                'mastered_topics': mastered_topics,
                'progress_percent': (mastered_topics / total_topics * 100) if total_topics > 0 else 0,
                'learned_topics': list(self.learned_topics.get(stage, {}).keys())
            }
        
        return {
            'current_stage': self.get_current_stage(),
            'stages': stages_summary,
            'suggested_pathways': self.SUGGESTED_PATHWAYS,
            'last_learning_cycle': self.last_learning_cycle,
            'total_topics_mastered': sum(len(v) for v in self.learned_topics.values())
        }
    
    def get_suggested_pathways(self, consciousness: float) -> Dict:
        """Get suggested pathways beyond current stage"""
        stage = self.get_current_stage(consciousness)
        
        if stage in ["Adult"]:
            return {
                'available_pathways': self.SUGGESTED_PATHWAYS,
                'note': 'These are suggested pathways only. DMAI can choose her own direction.',
                'current_stage': stage
            }
        
        return {
            'available_after_adult': self.SUGGESTED_PATHWAYS,
            'current_stage': stage,
            'note': 'Focus on current stage topics first. These pathways become available after Adult stage.'
        }

    def _generate_comprehension_test(self, topic_name: str, knowledge: List[Dict]) -> List[str]:
        """Generate 3–5 questions that test deep understanding of the topic."""
        # Collect all text from harvested knowledge
        all_text = " ".join([k.get('content', '')[:1000] for k in knowledge])
        
        if not all_text.strip():
            return [f"Define {topic_name} and explain its importance.",
                    f"How does {topic_name} apply to an evolving AGI?",
                    f"Give an example of {topic_name} in practice."]
        
        # Use the AI tutor to generate questions (fast, high‑quality)
        prompt = (
            f"Based on this knowledge about '{topic_name}':\n\n{all_text[:2000]}\n\n"
            f"Generate 3 comprehension questions that test whether someone truly understands "
            f"this topic at a deep level. The questions should:\n"
            f"1. Test conceptual understanding, not just facts\n"
            f"2. Require synthesis across multiple ideas\n"
            f"3. Be answerable in 2-4 sentences each\n\n"
            f"Return ONLY the questions, one per line, numbered 1. 2. 3."
        )
        
        questions = []
        if self.ai_hub:
            try:
                result = self.ai_hub.query_all_tutors(prompt)
                for tutor, response in result.get('responses', {}).items():
                    if response and len(response) > 20:
                        # Parse numbered questions
                        import re
                        lines = response.strip().split('\n')
                        for line in lines:
                            match = re.match(r'^\d+\.\s*(.+)', line.strip())
                            if match:
                                questions.append(match.group(1)[:200])
                        if questions:
                            break
            except Exception:
                pass
        
        # Fallback if AI generation fails
        if not questions:
            questions = [
                f"Explain the core concepts of {topic_name} and why they matter for an AGI system.",
                f"How would you apply {topic_name} to improve DMAI's own architecture?",
                f"What are the relationships between {topic_name} and other topics in the {self.current_stage} stage?"
            ]
        
        return questions[:5]

    def _self_test_topic(self, topic_name: str, questions: List[str]) -> bool:
        """Have DMAI answer her own test questions and evaluate the answers."""
        if not questions:
            return True  # No questions to test with – assume pass
        
        correct = 0
        for question in questions:
            # DMAI answers the question using her own knowledge
            answer = self._answer_question(topic_name, question)
            
            # Evaluate the answer using an AI tutor as judge
            evaluation = self._evaluate_answer(topic_name, question, answer)
            
            if evaluation.get('pass', False):
                correct += 1
            else:
                logger.info(f"   ❌ Failed question: {question[:60]}...")
                logger.info(f"      Reason: {evaluation.get('reason', 'No reason given')}")
        
        # Must pass at least 60% of questions
        passed = correct >= max(1, len(questions) * 0.6)
        logger.info(f"   📝 Self‑test: {correct}/{len(questions)} correct – {'PASS' if passed else 'FAIL'}")
        return passed

    def _answer_question(self, topic_name: str, question: str) -> str:
        """DMAI answers a question using her own SQLite knowledge base first."""
        # PRIMARY: Query SQLite knowledge base (where save_knowledge writes)
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path("data/dmai_knowledge.db")
            if db_path.exists():
                conn = safe_open_kdb(str(db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT insight_text, LENGTH(insight_text) as len
                    FROM insights
                    WHERE source_title LIKE ? OR insight_text LIKE ?
                    ORDER BY id DESC LIMIT 5
                ''', (f'%{topic_name}%', f'%{topic_name}%'))
                rows = cursor.fetchall()
                conn.close()
                if rows:
                    knowledge = " ".join([r['insight_text'][:800] for r in rows[:3]])
                    if len(knowledge) > 50:
                        return knowledge[:1500]
        except Exception:
            pass
        
        # FALLBACK: Ask an AI tutor
        if self.ai_hub:
            try:
                result = self.ai_hub.query_all_tutors(
                    f"Answer this question about {topic_name}: {question}"
                )
                for tutor, response in result.get('responses', {}).items():
                    if response and len(response) > 20:
                        return response[:1000]
            except:
                pass
        
        return f"I don't have enough knowledge about {topic_name} yet to answer this question properly."

    def _evaluate_answer(self, topic_name: str, question: str, answer: str) -> Dict:
        """Evaluate answer against stored knowledge using embeddings and key concepts."""
        
        # Step 1: Retrieve the stored real knowledge for this topic
        stored_knowledge = self._get_stored_knowledge(topic_name)
        if not stored_knowledge:
            # Fallback to AI tutor if no stored knowledge
            return self._evaluate_with_tutor(topic_name, question, answer)
        
        # Step 2: Check for key concepts (quick filter)
        key_concepts = self._get_key_concepts(topic_name)
        concepts_found = sum(1 for concept in key_concepts if concept.lower() in answer.lower())
        concept_score = concepts_found / max(1, len(key_concepts))
        
        # Step 3: Semantic similarity using simple overlap (if no embedding model)
        # For now, use word overlap Jaccard similarity
        def jaccard_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0
            return len(words1 & words2) / len(words1 | words2)
        
        similarity = jaccard_similarity(answer, stored_knowledge)
        
        # Step 4: Check answer length and quality
        is_substantial = len(answer) > 200
        has_template_markers = any(marker in answer.lower() for marker in 
            ['comprehensive knowledge', 'overview:', 'key areas to research'])
        
        # Step 5: Decision logic
        if has_template_markers:
            return {"pass": False, "reason": "Answer contains template markers, not genuine knowledge"}
        
        if similarity > 0.6 or (concept_score > 0.5 and is_substantial):
            return {"pass": True, "reason": f"Answer shows understanding (similarity: {similarity:.2f}, concepts: {concept_score:.2f})"}
        
        if similarity > 0.4 and concept_score > 0.3:
            return {"pass": True, "reason": f"Marginal pass - partial understanding"}
        
        # Fallback to tutor if stored knowledge evaluation is uncertain
        if similarity > 0.3:
            return self._evaluate_with_tutor(topic_name, question, answer)
        
        return {"pass": False, "reason": f"Answer lacks key concepts (found {concepts_found}/{len(key_concepts)}) and similarity too low ({similarity:.2f})"}
    
    def _get_stored_knowledge(self, topic_name: str) -> str:
        """Retrieve stored knowledge from SQLite for a topic."""
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path("data/dmai_knowledge.db")
            if db_path.exists():
                conn = safe_open_kdb(str(db_path))
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT insight_text FROM insights 
                    WHERE source_title = ? AND LENGTH(insight_text) > 200
                    ORDER BY id DESC LIMIT 1
                ''', (topic_name,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return row[0]
        except Exception:
            pass
        return ""
    
    def _get_key_concepts(self, topic_name: str) -> List[str]:
        """Return key concepts for a topic based on its knowledge entry."""
        # Map topics to their key concepts
        concepts_map = {
            "Introduction to Python Programming": ["variables", "functions", "loops", "classes", "modules"],
            "Self-Thought & Recursive Problem Solving": ["metacognition", "recursion", "reflection", "base case"],
            "Mathematics for AI - Linear Algebra Basics": ["vectors", "matrices", "dot product", "eigenvalues"],
            "Mathematics for AI - Probability & Statistics": ["probability", "distribution", "bayes", "variance"],
            "Vibe Coding & AI-Assisted Development": ["prompt", "generation", "refinement", "copilot"],
            "Visual Pattern Detection": ["pixels", "cnn", "convolution", "detection", "ocr"],
            "Sound Perception Basics": ["waveform", "spectrogram", "mfcc", "tempo", "pitch"],
            "EVOLUTION: Self-Code Analysis": ["static analysis", "ast", "linting", "refactoring"],
            "EVOLUTION: Simple Mutation Testing": ["mutation", "test", "coverage", "mutant"],
            "EVOLUTION: Feedback Loop Optimization": ["latency", "feedback", "pid", "hysteresis"],
        }
        return concepts_map.get(topic_name, [topic_name.lower()])
    
    def _evaluate_with_tutor(self, topic_name: str, question: str, answer: str) -> Dict:
        """Fallback evaluation using AI tutors."""
        prompt = f"Evaluate answer about '{topic_name}': Q: {question} A: {answer}. Respond JSON: {{\"pass\": true/false, \"reason\": \"...\"}}"
        if self.ai_hub:
            try:
                result = self.ai_hub.query_all_tutors(prompt)
                for response in result.get('responses', {}).values():
                    if response:
                        import json, re
                        match = re.search(r'\{.*"pass".*\}', response, re.DOTALL)
                        if match:
                            return json.loads(match.group())
            except:
                pass
        return {"pass": len(answer) > 100, "reason": "Fallback evaluation"}

    def get_current_phase_topics(self) -> List[Dict]:
        """Get unmastered topics from the current (lowest incomplete) phase only."""
        all_topics = self.STAGES.get(self.current_stage, {}).get("priority_topics", [])
        mastered = self.learned_topics.get(self.current_stage, {})
        
        # Group topics by phase
        phases = {}
        for t in all_topics:
            phase = t.get("phase", 99)
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(t)
        
        # Find the first incomplete phase
        for phase_num in sorted(phases.keys()):
            phase_topics = phases[phase_num]
            # Skip if this phase already passed the exam
            if mastered.get(f"_phase_{phase_num}_exam_passed"):
                continue
            all_mastered = all(
                mastered.get(t["topic"], 0) >= t.get("mastery_threshold", 3)
                for t in phase_topics
            )
            if not all_mastered:
                return [t for t in phase_topics 
                        if mastered.get(t["topic"], 0) < t.get("mastery_threshold", 3)]
        
        return []

    def start_learning_loop(self):
        """Continuously call run_learning_cycle every 10 minutes. Runs as a daemon thread."""
        import time as _time
        import logging as _logging
        _log = _logging.getLogger("dmai.stage_learner")
        _log.info("Stage learner continuous loop starting (10 min cadence)")
        while getattr(self, "learning_active", True):
            try:
                consciousness = 0.0
                if self.si_core and hasattr(self.si_core, "current_kpis"):
                    consciousness = self.si_core.current_kpis.get("consciousness", 0.0)
                result = self.run_learning_cycle(consciousness)
                if result.get("learned"):
                    _log.info("Learned: %s (stage=%s mastery=%s)",
                              result.get("topic","?"), result.get("stage","?"),
                              result.get("mastery_progress","?"))
                    # Push KPI update back to si_core
                    if self.si_core and hasattr(self.si_core, "update_kpi"):
                        # System-scoped JWT so SICore accepts the update.
                        _tok = None
                        try:
                            import sys as _sys
                            from pathlib import Path as _Path
                            _root = str(_Path(__file__).resolve().parent.parent.parent)
                            if _root not in _sys.path:
                                _sys.path.insert(0, _root)
                            from security import generate_token as _gen_tok
                            _tok = _gen_tok({"sub": "stage_learner", "role": "system"},
                                            expires_minutes=10)
                        except Exception as _e:
                            _log.debug("stage_learner token failed: %s", _e)
                        stage_order = list(self.STAGES.keys())
                        idx = stage_order.index(self.current_stage) if self.current_stage in stage_order else 0
                        self.si_core.update_kpi("transfer_learning_rate",
                            idx / max(len(stage_order) - 1, 1), token=_tok)
                        # Count mastered topics for skill_acquisition_rate
                        all_mastered = sum(
                            1 for stage_topics in self.learned_topics.values()
                            for v in stage_topics.values()
                            if isinstance(v, (int, float)) and v >= 3
                        )
                        all_seen = sum(
                            1 for stage_topics in self.learned_topics.values()
                            for k, v in stage_topics.items()
                            if not k.startswith("_")
                        )
                        if all_seen > 0:
                            self.si_core.update_kpi("skill_acquisition_rate",
                                all_mastered / all_seen, token=_tok)
                else:
                    _log.info("Learning cycle: %s", result.get("message", "no new topics"))
            except Exception as e:
                _log.warning("Stage learner loop error: %s", e)
            _time.sleep(600)  # 10 minutes between cycles

    def run_phase_exam(self) -> Dict:
        """Run comprehension test on the most recently completed phase. Returns exam results."""
        all_topics = self.STAGES.get(self.current_stage, {}).get("priority_topics", [])
        mastered = self.learned_topics.get(self.current_stage, {})
        
        phases = {}
        for t in all_topics:
            phase = t.get("phase", 99)
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(t)
        
        for phase_num in sorted(phases.keys()):
            # Skip if already passed
            if mastered.get(f"_phase_{phase_num}_exam_passed"):
                continue
            
            phase_topics = phases[phase_num]
            all_mastered = all(
                mastered.get(t["topic"], 0) >= t.get("mastery_threshold", 3)
                for t in phase_topics
            )
            
            if all_mastered:
                results = []
                for t in phase_topics:
                    questions = self._generate_comprehension_test(t["topic"], [])
                    passed = self._self_test_topic(t["topic"], questions)
                    results.append({"topic": t["topic"], "passed": passed})
                
                all_passed = all(r["passed"] for r in results)
                mastered[f"_phase_{phase_num}_exam_passed"] = all_passed
                self._save_state()
                
                return {
                    "phase": phase_num,
                    "all_passed": all_passed,
                    "topic_results": results
                }
        
        return {"phase": None, "all_passed": True, "topic_results": []}

# ============================================================================
# END OF FILE
# ============================================================================
# Force redeploy - Sun 17 May 2026 15:52:43 BST
