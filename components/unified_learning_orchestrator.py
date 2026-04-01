"""
Unified Learning Orchestrator - Bridges ALL learning sources to SI Core and Evolution Engine
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class UnifiedLearningOrchestrator:
    """
    Central hub that:
    1. Receives ALL learning events from training systems, syllabus, knowledge sources
    2. Creates insights (neurons) in SI Core
    3. Feeds insights to Evolution Engine for self-improvement
    4. Updates Knowledge Graph
    5. Triggers Fresh Blood Injection when evolution candidates are learned
    """
    
    def __init__(self, si_core, evolution_engine, knowledge_graph):
        self.si_core = si_core
        self.evolution_engine = evolution_engine
        self.knowledge_graph = knowledge_graph
        self.evolution_candidates = []  # Queue for concepts that can improve systems
        self.concept_counter = 0
        self.cluster_counter = {}
        
        # Evolution candidate keywords - concepts that can improve DMAI's code/systems
        self.evolution_keywords = [
            'algorithm', 'optimization', 'performance', 'memory', 'efficiency',
            'architecture', 'design pattern', 'refactoring', 'testing',
            'security', 'authentication', 'encryption', 'parallel',
            'concurrency', 'caching', 'database', 'api', 'protocol',
            'compression', 'serialization', 'parsing', 'validation'
        ]
        
        logger.info("🧠 UnifiedLearningOrchestrator initialized")
    
    def on_concept_mastered(self, system: str, concept: str, details: Dict = None) -> Optional[str]:
        """
        Called whenever ANY system masters a concept.
        
        Args:
            system: The source system (software, agi, genai, si, funding, llm, syllabus, etc.)
            concept: The concept/topic name
            details: Additional metadata (category, confidence, etc.)
        
        Returns:
            insight_id if created, None otherwise
        """
        if details is None:
            details = {}
        
        self.concept_counter += 1
        
        # Format insight text
        insight_text = self._format_insight(system, concept, details)
        
        # Determine if this should be a compressed insight or full neuron
        weight, is_compressed = self._determine_weight(system, concept)
        
        logger.info(f"📚 Learning: {system} mastered '{concept}' (weight: {weight})")
        
        try:
            # 1. Create insight neuron in SI Core
            insight_id = self.si_core.add_insight(
                insight_text=insight_text,
                entity_type=system,
                entities=[concept] + details.get('tags', []),
                relationship="mastered",
                source_topic=system,
                target_topic=details.get('category', 'general'),
                confidence=details.get('confidence', 0.85)
            )
            
            # 2. Add to knowledge graph
            self.knowledge_graph.add_concept(
                concept=concept,
                source=system,
                metadata={
                    'insight_id': insight_id,
                    'timestamp': datetime.now().isoformat(),
                    'weight': weight,
                    **details
                }
            )
            
            # 3. Check if this concept can be used for evolution/self-improvement
            if self._is_evolution_candidate(concept, details):
                self.evolution_candidates.append({
                    'insight_id': insight_id,
                    'concept': concept,
                    'source': system,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"🚀 Evolution candidate queued: {concept}")
                
                # Trigger evolution if we have enough candidates
                if len(self.evolution_candidates) >= 5:
                    self._trigger_evolution()
            
            # 4. Update consciousness via SI Core (already does this internally)
            
            return insight_id
            
        except Exception as e:
            logger.error(f"Failed to create insight for {concept}: {e}")
            return None
    
    def on_batch_complete(self, system: str, concepts: List[str], category: str = None):
        """
        Create a compressed neuron from multiple related concepts.
        Used for dictionary, encyclopedia, etc.
        """
        if not concepts:
            return
        
        cluster_key = f"{system}_{category or 'general'}"
        self.cluster_counter[cluster_key] = self.cluster_counter.get(cluster_key, 0) + 1
        cluster_num = self.cluster_counter[cluster_key]
        
        insight_text = f"Knowledge cluster #{cluster_num}: {len(concepts)} {system} concepts"
        if category:
            insight_text += f" in {category}"
        
        insight_id = self.si_core.add_insight(
            insight_text=insight_text,
            entity_type=f"compressed_{system}",
            entities=[system, category] if category else [system],
            relationship="contains",
            source_topic=system,
            target_topic=category or "knowledge_cluster",
            confidence=0.7
        )
        
        logger.info(f"📦 Created compressed insight for {len(concepts)} {system} concepts: {insight_id}")
        return insight_id
    
    def _format_insight(self, system: str, concept: str, details: Dict) -> str:
        """Format the insight text"""
        category = details.get('category', '')
        if category:
            return f"{concept} - {category} concept from {system} training"
        return f"{concept} concept mastered in {system} training"
    
    def _determine_weight(self, system: str, concept: str) -> tuple:
        """
        Determine if this should be a full neuron or compressed.
        Returns (weight, is_compressed)
        """
        # Full neurons for syllabus topics and high-value concepts
        if system in ['syllabus', 'evolution_accelerator']:
            return (1.0, False)
        
        # Compressed for training systems - multiple concepts per neuron
        if system in ['software', 'agi', 'genai', 'llm', 'funding']:
            return (0.3, True)
        
        # Medium weight for SI training
        if system == 'si':
            return (0.5, False)
        
        return (0.3, True)
    
    def _is_evolution_candidate(self, concept: str, details: Dict) -> bool:
        """Determine if a concept can be used for self-improvement"""
        concept_lower = concept.lower()
        text = f"{concept_lower} {str(details).lower()}"
        
        return any(keyword in text for keyword in self.evolution_keywords)
    
    def _trigger_evolution(self):
        """Trigger evolution cycle with queued candidates"""
        if not self.evolution_candidates:
            return
        
        logger.info(f"🔄 Triggering evolution with {len(self.evolution_candidates)} candidates")
        
        # Pass candidates to evolution engine
        if hasattr(self.evolution_engine, 'queue_for_evolution'):
            for candidate in self.evolution_candidates:
                self.evolution_engine.queue_for_evolution(candidate)
        
        self.evolution_candidates = []
    
    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        return {
            'total_concepts': self.concept_counter,
            'evolution_candidates_queued': len(self.evolution_candidates),
            'clusters_created': len(self.cluster_counter)
        }
