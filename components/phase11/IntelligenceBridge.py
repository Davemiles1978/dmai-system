"""
IntelligenceBridge - Connects Phase 11 (AI Tutor Network) to Phase 6 Intelligence Core

This bridge ensures all knowledge from tutors flows into:
1. Phase 6 Knowledge Graph (for relationship storage)
2. Phase 6 Synthetic Neural Network (for consciousness growth)
3. Phase 6 Pattern Synthesis (for pattern detection)
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class IntelligenceBridge:
    """
    Bridge between knowledge acquisition (Phase 11) and intelligence core (Phase 6)
    
    This class:
    - Receives synthesized knowledge from tutors
    - Converts knowledge to neural signals
    - Feeds signals into Synthetic Neural Network
    - Stores knowledge in Knowledge Graph
    """
    
    def __init__(self, intelligence_core=None, knowledge_graph=None, pattern_synthesis=None):
        self.intelligence = intelligence_core
        self.knowledge_graph = knowledge_graph
        self.pattern_synthesis = pattern_synthesis
        self.synthesis_queue = []
        self.bridge_stats = {
            'total_knowledge_packets': 0,
            'total_patterns_fed': 0,
            'total_concepts_added': 0,
            'last_feed': None,
            'queue_size': 0
        }
        
        logger.info("🌉 Intelligence Bridge initialized")
        
    def set_intelligence_core(self, intelligence_core):
        self.intelligence = intelligence_core
        
    def set_knowledge_graph(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        
    def set_pattern_synthesis(self, pattern_synthesis):
        self.pattern_synthesis = pattern_synthesis
        
    def feed_knowledge(self, knowledge_packet: Dict) -> Dict:
        """Feed synthesized knowledge into the intelligence core"""
        result = {
            'success': False,
            'concepts_added': 0,
            'patterns_fed': 0,
            'consciousness_impact': 0.0,
            'errors': []
        }
        
        try:
            # Validate packet type
            if not isinstance(knowledge_packet, dict):
                logger.warning(f"Expected dict packet, got {type(knowledge_packet).__name__}")
                result['errors'].append(f"Invalid packet type: {type(knowledge_packet).__name__}")
                return result
            
            if self.knowledge_graph:
                concepts_added = self._add_to_knowledge_graph(knowledge_packet)
                result['concepts_added'] = concepts_added
                self.bridge_stats['total_concepts_added'] += concepts_added
            
            if self.pattern_synthesis:
                patterns_fed = self._feed_patterns(knowledge_packet)
                result['patterns_fed'] = patterns_fed
                self.bridge_stats['total_patterns_fed'] += patterns_fed
            
            if self.intelligence:
                consciousness_impact = self._feed_to_synthetic_network(knowledge_packet)
                result['consciousness_impact'] = consciousness_impact
            
            self.synthesis_queue.append(knowledge_packet)
            self.bridge_stats['queue_size'] = len(self.synthesis_queue)
            
            if len(self.synthesis_queue) >= 10:
                self._batch_process()
            
            result['success'] = True
            self.bridge_stats['total_knowledge_packets'] += 1
            self.bridge_stats['last_feed'] = datetime.now().isoformat()
            
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Failed to feed knowledge: {e}")
            
        return result
    
    def _add_to_knowledge_graph(self, packet: Dict) -> int:
        """Add concepts and relationships to knowledge graph"""
        concepts_added = 0
        
        # Validate packet type
        if not isinstance(packet, dict):
            logger.warning(f"Cannot add to knowledge graph: packet is {type(packet).__name__}")
            return 0
            
        concepts = packet.get('concepts', {})
        
        if not isinstance(concepts, dict):
            logger.warning(f"Concepts is not a dict: {type(concepts).__name__}")
            return 0
        
        if not self.knowledge_graph:
            return 0
            
        for concept_name, concept_data in concepts.items():
            if isinstance(concept_data, dict):
                try:
                    self.knowledge_graph.add_knowledge(
                        subject=concept_name,
                        predicate='learned_from',
                        object=concept_data.get('source', packet.get('source', 'tutor_network')),
                        metadata={
                            'confidence': concept_data.get('confidence', 0.5),
                            'timestamp': datetime.now().isoformat(),
                            'insights': concept_data.get('insights', []),
                            'importance': packet.get('importance', 0.5)
                        }
                    )
                    concepts_added += 1
                except Exception as e:
                    logger.error(f"Failed to add concept {concept_name}: {e}")
        
        return concepts_added
    
    def _feed_patterns(self, packet: Dict) -> int:
        """Feed discovered patterns to pattern synthesis system"""
        patterns_fed = 0
        
        # Validate packet type
        if not isinstance(packet, dict):
            logger.warning(f"Cannot feed patterns: packet is {type(packet).__name__}")
            return 0
            
        patterns = packet.get('patterns', [])
        
        if not isinstance(patterns, list):
            logger.warning(f"Patterns is not a list: {type(patterns).__name__}")
            return 0
        
        if not self.pattern_synthesis:
            return 0
            
        for pattern in patterns:
            if isinstance(pattern, dict):
                try:
                    self.pattern_synthesis.detect_patterns(
                        [pattern.get('data', {})],
                        pattern.get('context', packet.get('source', 'tutor'))
                    )
                    patterns_fed += 1
                except Exception as e:
                    logger.error(f"Failed to feed pattern: {e}")
        
        return patterns_fed
    
    def _feed_to_synthetic_network(self, packet: Dict) -> float:
        """Convert knowledge to neural signals and feed to synthetic network"""
        consciousness_impact = 0.0
        
        # Guard against incorrect packet type
        if not isinstance(packet, dict):
            logger.warning(f"Expected dict packet for synthetic network, got {type(packet).__name__}: {packet}")
            return 0.0
        
        if not self.intelligence:
            return 0.0
            
        importance = packet.get('importance', 0.5)
        complexity = packet.get('complexity', 1)
        insights = packet.get('insights', [])
        
        # Ensure insights is a list
        if not isinstance(insights, list):
            logger.warning(f"Insights is not a list: {type(insights).__name__}, treating as empty")
            insights = []
            
        insights_count = len(insights)
        
        # Validate and sanitize numeric values
        try:
            importance = float(importance) if importance is not None else 0.5
            complexity = int(complexity) if complexity is not None else 1
        except (TypeError, ValueError):
            logger.warning(f"Invalid numeric values: importance={importance}, complexity={complexity}")
            importance = 0.5
            complexity = 1
        
        signal_strength = importance
        signal_strength += min(0.3, complexity / 100)
        signal_strength += min(0.2, insights_count / 10)
        signal_strength = min(1.0, signal_strength)
        
        try:
            for _ in range(min(5, max(1, complexity))):
                process_result = self.intelligence.process(signal_strength)
                if isinstance(process_result, dict):
                    consciousness_impact += process_result.get('consciousness', 0)
                else:
                    logger.warning(f"process_result is not a dict: {type(process_result).__name__}")
                
            # Safely call evolve - handle different return types
            try:
                evolution_result = self.intelligence.evolve()
                if not isinstance(evolution_result, dict):
                    logger.warning(f"evolve() returned {type(evolution_result).__name__}, expected dict")
                    evolution_result = {'consciousness': 0, 'success': False}
            except TypeError as e:
                logger.warning(f"evolve() called with wrong signature: {e}")
                evolution_result = {'consciousness': 0, 'success': False}
            except Exception as e:
                logger.error(f"evolve() failed: {e}")
                evolution_result = {'consciousness': 0, 'success': False}
            
            # Safely get previous consciousness
            previous_consciousness = 0
            if isinstance(packet, dict):
                previous_consciousness = packet.get('previous_consciousness', 0)
                if not isinstance(previous_consciousness, (int, float)):
                    previous_consciousness = 0
            
            if isinstance(evolution_result, dict):
                consciousness_impact = evolution_result.get('consciousness', 0) - previous_consciousness
            elif isinstance(evolution_result, (int, float)):
                # If evolve() returned a number, treat it as the consciousness value
                consciousness_impact = evolution_result - previous_consciousness
                # SAFE FORMATTING - convert to string first to avoid format errors
                logger.warning(f"evolve() returned number {str(evolution_result)}, treating as consciousness value")
            else:
                logger.warning(f"evolution_result is not a dict or number: {type(evolution_result).__name__}")
                consciousness_impact = 0
                
        except Exception as e:
            logger.error(f"Failed to feed to synthetic network: {str(e)}")
        
        return float(consciousness_impact) if consciousness_impact is not None else 0.0
    
    def _batch_process(self):
        """Process queued knowledge and evolve"""
        if not self.synthesis_queue:
            return
            
        logger.info(f"🌉 Batch processing {len(self.synthesis_queue)} knowledge packets")
        
        all_patterns = []
        all_concepts = {}
        total_importance = 0
        valid_packets = []
        
        # Filter and validate packets
        for packet in self.synthesis_queue:
            if isinstance(packet, dict):
                valid_packets.append(packet)
                all_patterns.extend(packet.get('patterns', []))
                importance = packet.get('importance', 0.5)
                if isinstance(importance, (int, float)):
                    total_importance += importance
                else:
                    total_importance += 0.5
                
                concepts = packet.get('concepts', {})
                if isinstance(concepts, dict):
                    for concept, data in concepts.items():
                        if concept not in all_concepts:
                            all_concepts[concept] = data
        
        if not valid_packets:
            self.synthesis_queue = []
            self.bridge_stats['queue_size'] = 0
            return
        
        batch_packet = {
            'concepts': all_concepts,
            'patterns': all_patterns,
            'importance': total_importance / max(1, len(valid_packets)),
            'complexity': len(all_patterns) + len(all_concepts),
            'insights': [],
            'source': 'batch_processing'
        }
        
        if self.knowledge_graph:
            self._add_to_knowledge_graph(batch_packet)
        
        self.synthesis_queue = []
        self.bridge_stats['queue_size'] = 0
        
        if self.intelligence:
            for _ in range(3):
                try:
                    self.intelligence.evolve()
                except Exception as e:
                    logger.error(f"Batch evolution failed: {e}")
        
        logger.info("🌉 Batch processing complete")
    
    def flush_queue(self):
        """Force immediate batch processing"""
        queue_size = len(self.synthesis_queue)
        self._batch_process()
        return {'flushed': True, 'queue_was_size': queue_size}
    
    def get_bridge_status(self) -> Dict:
        """Get status of the intelligence bridge"""
        consciousness = 0.0
        if self.intelligence and hasattr(self.intelligence, 'consciousness_level'):
            try:
                consciousness = float(self.intelligence.consciousness_level) if self.intelligence.consciousness_level is not None else 0.0
            except (TypeError, ValueError):
                consciousness = 0.0
            
        return {
            'connected_components': {
                'intelligence_core': self.intelligence is not None,
                'knowledge_graph': self.knowledge_graph is not None,
                'pattern_synthesis': self.pattern_synthesis is not None
            },
            'stats': self.bridge_stats,
            'current_consciousness': consciousness,
            'queue_size': len(self.synthesis_queue),
            'status': 'operational' if self.intelligence else 'waiting_for_core'
        }
