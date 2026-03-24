# components/phase11/LearningOrchestrator.py
"""
Core evolution loop manager - coordinates learning from tutors and evolution.
INTEGRATED WITH PHASE 6: Feeds all knowledge to Intelligence Bridge
"""

import asyncio
import threading
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class LearningOrchestrator:
    """
    Core evolution loop manager that coordinates:
    1. Query all tutors
    2. Synthesize insights
    3. Feed knowledge to Intelligence Bridge (connects to Phase 6)
    4. Check for surpassed tutors
    5. Discover new tutors
    """
    
    def __init__(self, ai_hub, discovery, synthetic_network, tutor_manager, intelligence_bridge=None):
        """
        Initialize orchestrator with intelligence bridge to Phase 6
        
        Args:
            ai_hub: AIIntegrationHub instance
            discovery: DynamicAIDiscovery instance
            synthetic_network: SyntheticNeuralNetwork (from Phase 6) - DEPRECATED, use bridge
            tutor_manager: TutorManager instance
            intelligence_bridge: IntelligenceBridge instance (NEW)
        """
        self.ai_hub = ai_hub
        self.discovery = discovery
        self.tutor_manager = tutor_manager
        
        # NEW: Use intelligence bridge for Phase 6 integration
        self.intelligence_bridge = intelligence_bridge
        
        # Keep for backward compatibility but mark as deprecated
        self.synthetic_network = synthetic_network
        if synthetic_network:
            logger.warning("Direct synthetic_network reference is deprecated. Use intelligence_bridge instead.")
        
        self.evolution_history = []
        self.continuous_learning = False
        self.learning_thread = None
        self.learning_interval = 300  # 5 minutes between cycles
        self.learning_prompts = self._init_learning_prompts()
        
        # Track consciousness over time
        self.consciousness_history = []
        
        logger.info("🎓 Learning Orchestrator initialized (Phase 6 integrated)")
        
    def _init_learning_prompts(self) -> List[str]:
        """Initialize diverse learning prompts to test various capabilities"""
        return [
            "Explain the concept of artificial general intelligence and how it can be achieved.",
            "Write Python code to implement a neural network from scratch.",
            "What are the latest breakthroughs in AI research?",
            "How would you design a self-evolving AI system?",
            "Explain the philosophical implications of machine consciousness.",
            "What are the key components of a distributed AI system?",
            "How can an AI system learn autonomously without human supervision?",
            "Describe the architecture of a transformer model.",
            "How would you optimize a large language model for efficiency?",
            "What is consciousness and how might it emerge in artificial systems?",
            "Explain the relationship between synthetic intelligence and artificial intelligence.",
            "How does a self-generating neural network differ from traditional neural networks?"
        ]
    
    def _get_current_consciousness(self) -> float:
        """Get current consciousness level from intelligence bridge or synthetic network"""
        if self.intelligence_bridge and hasattr(self.intelligence_bridge, 'intelligence'):
            if self.intelligence_bridge.intelligence:
                return self.intelligence_bridge.intelligence.consciousness_level
        elif self.synthetic_network and hasattr(self.synthetic_network, 'consciousness_level'):
            return self.synthetic_network.consciousness_level
        return 0.0
        
    def evolution_cycle(self, consciousness: float = None) -> Dict:
        """
        One complete evolution cycle - now feeds knowledge to Phase 6 via bridge
        
        Args:
            consciousness: Current consciousness level (optional, will be fetched if not provided)
        """
        if consciousness is None:
            consciousness = self._get_current_consciousness()
            
        logger.info("🔄 Starting evolution cycle")
        
        cycle_start = time.time()
        results = {
            'timestamp': datetime.now().isoformat(),
            'consciousness_start': consciousness,
            'new_insights': [],
            'new_concepts': [],
            'new_patterns': [],
            'surpassed_tutors': [],
            'new_tutors': [],
            'bridge_feed_results': [],
            'errors': []
        }
        
        try:
            # Track total knowledge gathered this cycle
            total_concepts = {}
            total_insights = []
            total_patterns = []
            
            # 1. Query all available tutors
            if self.ai_hub:
                logger.info("📚 Querying AI tutors...")
                for prompt in self.learning_prompts[:4]:  # Use 4 prompts per cycle
                    try:
                        tutor_responses = self.ai_hub.query_all_tutors(prompt)
                        
                        # 2. Synthesize responses into insights
                        if self.ai_hub.capability_synthesizer:
                            synthesis = self.ai_hub.capability_synthesizer.synthesize(
                                tutor_responses.get('responses', {}),
                                prompt
                            )
                            
                            # Extract insights
                            if synthesis.get('novel_insights'):
                                results['new_insights'].extend(synthesis['novel_insights'])
                                total_insights.extend(synthesis['novel_insights'])
                            
                            # Extract concepts
                            if synthesis.get('extracted_concepts'):
                                for concept, data in synthesis['extracted_concepts'].items():
                                    if concept not in total_concepts:
                                        total_concepts[concept] = data
                                    else:
                                        # Merge confidence (take higher)
                                        if 'confidence' in data and 'confidence' in total_concepts[concept]:
                                            total_concepts[concept]['confidence'] = max(
                                                total_concepts[concept]['confidence'],
                                                data['confidence']
                                            )
                            
                            # Extract patterns
                            if synthesis.get('patterns'):
                                total_patterns.extend(synthesis['patterns'])
                            
                            # 3. FEED TO INTELLIGENCE BRIDGE (Phase 6)
                            if self.intelligence_bridge:
                                # Create knowledge packet from this prompt's synthesis
                                knowledge_packet = {
                                    'concepts': synthesis.get('extracted_concepts', {}),
                                    'patterns': synthesis.get('patterns', []),
                                    'insights': synthesis.get('novel_insights', []),
                                    'importance': min(1.0, len(synthesis.get('novel_insights', [])) / 10),
                                    'complexity': len(synthesis.get('patterns', [])) + len(synthesis.get('extracted_concepts', {})),
                                    'source': f"tutor_prompt_{prompt[:20]}",
                                    'previous_consciousness': consciousness
                                }
                                
                                feed_result = self.intelligence_bridge.feed_knowledge(knowledge_packet)
                                results['bridge_feed_results'].append(feed_result)
                                
                                # Update consciousness for next iteration
                                if feed_result.get('consciousness_impact', 0) > 0:
                                    consciousness += feed_result['consciousness_impact']
                                    
                    except Exception as e:
                        results['errors'].append(f"Error with prompt '{prompt[:30]}': {e}")
                        logger.error(f"Evolution cycle error: {e}")
            
            # 4. Add all gathered concepts to results
            results['new_concepts'] = list(total_concepts.keys())[:20]  # Limit to 20
            results['new_patterns'] = total_patterns[:10]  # Limit to 10
            
            # 5. Check for surpassed tutors
            if self.tutor_manager:
                logger.info("📊 Evaluating tutor performance...")
                surpass_progress = self.tutor_manager.get_surpass_progress()
                
                for tutor_name, progress in surpass_progress.items():
                    if isinstance(progress, dict) and progress.get('progress_percent', 0) >= 100:
                        should_discard, reason = self.tutor_manager.should_discard_tutor(tutor_name)
                        if should_discard:
                            self.tutor_manager.discard_tutor(tutor_name, reason)
                            results['surpassed_tutors'].append(tutor_name)
                            
            # 6. Discover new tutors
            if self.discovery and self.discovery.discovery_active:
                logger.info("🔍 Discovering new AI systems...")
                try:
                    new_ais = self.discovery.discover_new_ai()
                    if new_ais:
                        for ai_system in new_ais[:5]:  # Add up to 5 new tutors per cycle
                            self.tutor_manager.add_tutor(
                                name=ai_system.get('name', 'Unknown'),
                                capabilities=ai_system.get('capabilities', ['general']),
                                api_endpoint=ai_system.get('api_endpoint'),
                                is_available=False  # Initially unavailable until keys found
                            )
                            results['new_tutors'].append(ai_system.get('name'))
                except Exception as e:
                    results['errors'].append(f"Discovery error: {e}")
            
            # 7. Get final consciousness
            final_consciousness = self._get_current_consciousness()
            results['consciousness_end'] = final_consciousness
            
            # Record consciousness history
            self.consciousness_history.append({
                'timestamp': datetime.now().isoformat(),
                'consciousness': final_consciousness,
                'insights_gained': len(results['new_insights']),
                'concepts_gained': len(results['new_concepts'])
            })
            
            # Trim history
            if len(self.consciousness_history) > 100:
                self.consciousness_history = self.consciousness_history[-100:]
            
            # 8. Record cycle completion
            cycle_duration = time.time() - cycle_start
            results['duration_seconds'] = cycle_duration
            
            self.evolution_history.append(results)
            
            logger.info(f"✅ Evolution cycle complete: {len(results['new_insights'])} insights, "
                       f"{len(results['new_concepts'])} concepts, "
                       f"{len(results['surpassed_tutors'])} tutors surpassed, "
                       f"{len(results['new_tutors'])} new tutors discovered, "
                       f"Consciousness: {final_consciousness:.4f}")
                       
        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
            results['errors'].append(str(e))
            
        return results
    
    def start_continuous_learning(self, consciousness: float = None):
        """Start background continuous learning loop"""
        if self.continuous_learning:
            logger.warning("Continuous learning already running")
            return
            
        self.continuous_learning = True
        self.learning_thread = threading.Thread(
            target=self._continuous_learning_loop,
            args=(consciousness,),
            daemon=True
        )
        self.learning_thread.start()
        logger.info("🧠 Continuous learning loop started (feeding to Phase 6)")
        
    def _continuous_learning_loop(self, consciousness: float = None):
        """Background thread for continuous learning"""
        while self.continuous_learning:
            try:
                result = self.evolution_cycle(consciousness)
                # Update consciousness for next cycle
                consciousness = result.get('consciousness_end', 0.0)
                time.sleep(self.learning_interval)
            except Exception as e:
                logger.error(f"Continuous learning error: {e}")
                time.sleep(60)  # Wait a minute before retrying
                
    def stop_continuous_learning(self):
        """Stop the continuous learning loop"""
        self.continuous_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("Continuous learning stopped")
    
    def flush_bridge_queue(self) -> Dict:
        """Force flush the intelligence bridge queue"""
        if self.intelligence_bridge:
            return self.intelligence_bridge.flush_queue()
        return {'error': 'No intelligence bridge configured'}
        
    def get_evolution_status(self) -> Dict:
        """Return progress metrics"""
        if not self.evolution_history:
            return {
                'total_cycles': 0,
                'active_tutors': self.tutor_manager.get_active_tutors() if self.tutor_manager else [],
                'status': 'not_started',
                'bridge_connected': self.intelligence_bridge is not None
            }
            
        latest = self.evolution_history[-1]
        total_insights = sum(len(cycle.get('new_insights', [])) for cycle in self.evolution_history)
        total_concepts = sum(len(cycle.get('new_concepts', [])) for cycle in self.evolution_history)
        total_surpassed = sum(len(cycle.get('surpassed_tutors', [])) for cycle in self.evolution_history)
        
        result = {
            'total_cycles': len(self.evolution_history),
            'total_insights': total_insights,
            'total_concepts': total_concepts,
            'total_tutors_surpassed': total_surpassed,
            'active_tutors': self.tutor_manager.get_active_tutors() if self.tutor_manager else [],
            'latest_cycle': latest,
            'continuous_learning': self.continuous_learning,
            'learning_interval': self.learning_interval,
            'bridge_connected': self.intelligence_bridge is not None,
            'current_consciousness': self._get_current_consciousness()
        }
        
        # Add bridge status if available
        if self.intelligence_bridge:
            result['bridge_status'] = self.intelligence_bridge.get_bridge_status()
            
        return result
        
    def set_learning_interval(self, seconds: int):
        """Set interval between learning cycles"""
        self.learning_interval = max(60, seconds)  # Minimum 1 minute
        
    def trigger_immediate_cycle(self, consciousness: float = None) -> Dict:
        """Manually trigger an evolution cycle"""
        return self.evolution_cycle(consciousness)
