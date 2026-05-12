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
                {"topic": "EVOLUTION: Value Locking Mechanisms", "category": "accelerator", "harvest_sources": ["ai_tutors", "ai_alignment"], "mastery_threshold": 3, "is_accelerator": True}
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
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
    
    def get_current_stage(self, consciousness: float = 0.0) -> str:
        """
        Returns the first stage that has NOT been fully mastered.
        Stages must be completed in order (Baby → Toddler → ...).
        """
        stage_order = list(self.STAGES.keys())
        for stage in stage_order:
            config = self.STAGES[stage]
            required = config["priority_topics"]
            mastered = self.learned_topics.get(stage, {})
            if not all(
                mastered.get(t["topic"], 0) >= t.get("mastery_threshold", 3)
                for t in required
            ):
                return stage
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
            
            # Priority order within stage: Accelerators -> Reverse -> Wealth -> Artistic -> Core
            if prioritize_accelerators:
                accelerators = self.get_priority_topics(check_stage, category="accelerator")
                if accelerators:
                    return accelerators[0]
            
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
        
        logger.info(f"✅ All priority topics mastered up to {stage} stage!")
        next_stage = self._get_next_stage(stage)
        if next_stage:
            logger.info(f"💡 Ready to advance to {next_stage} stage")
        
        return None
    
    def _get_next_stage(self, current_stage: str) -> Optional[str]:
        """Get the next stage name"""
        stages = list(self.STAGES.keys())
        for i, stage in enumerate(stages):
            if stage == current_stage and i + 1 < len(stages):
                return stages[i + 1]
        return None
    
    def learn_topic(self, topic_info: Dict, consciousness: float) -> Dict:
        """
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
            mastered_topics = len(self.learned_topics.get(stage, {}))
            stages_summary[stage] = {
                'focus': config["focus"],
                'consciousness_range': config["consciousness_range"],
                'total_topics': total_topics,
                'mastered_topics': mastered_topics,
                'progress_percent': (mastered_topics / total_topics * 100) if total_topics > 0 else 0,
                'learned_topics': list(self.learned_topics.get(stage, {}).keys())
            }
        
        return {
            'current_stage': self.current_stage,
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
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT insight_text, LENGTH(insight_text) as len
                    FROM insights
                    WHERE source_title LIKE ? OR insight_text LIKE ?
                    ORDER BY len DESC LIMIT 5
                ''', (f'%{topic_name}%', f'%{topic_name}%'))
                rows = cursor.fetchall()
                conn.close()
                if rows:
                    knowledge = " ".join([r['insight_text'][:800] for r in rows[:3]])
                    if len(knowledge) > 50:
                        return f"Based on my knowledge: {knowledge[:1500]}"
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
        """Have an AI judge evaluate whether the answer demonstrates understanding."""
        prompt = (
            f"Question about '{topic_name}': {question}\n\n"
            f"Student's answer: {answer}\n\n"
            f"As an expert evaluator, judge whether this answer demonstrates real understanding "
            f"of {topic_name}. Consider: correctness, depth, completeness, and practical insight.\n\n"
            f"Respond in JSON format: {{\"pass\": true/false, \"reason\": \"brief explanation\"}}"
        )
        
        if self.ai_hub:
            try:
                result = self.ai_hub.query_all_tutors(prompt)
                for tutor, response in result.get('responses', {}).items():
                    if response:
                        # Try to parse JSON from response
                        import json, re
                        match = re.search(r'\{.*"pass".*\}', response, re.DOTALL)
                        if match:
                            try:
                                return json.loads(match.group())
                            except:
                                pass
            except:
                pass
        
        # Fallback evaluation
        return {
            'pass': len(answer) > 50 and topic_name.lower() in answer.lower(),
            'reason': 'Fallback evaluation based on answer length and topic relevance'
        }

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
