"""
SI Core v2.0 - Self-Improvement Core with Full Network State
Tracks 8 intelligence metrics, consciousness, and network evolution
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

class SICore:
    def __init__(self, data_path: Optional[Path] = None):
        # Set data path for persistence
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path(__file__).parent.parent / 'data'
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # KPI history storage (last 100 cycles)
        self.kpi_history = {
            'skill_acquisition_rate': deque(maxlen=100),
            'transfer_learning_rate': deque(maxlen=100),
            'zero_shot_success_count': deque(maxlen=100),
            'agentic_capability_score': deque(maxlen=100),
            'recursive_self_improvement_rate': deque(maxlen=100),
            'sample_efficiency_trend': deque(maxlen=100),
            'metacognition_accuracy': deque(maxlen=100),
            'multi_modal_integration_score': deque(maxlen=100),
            'consciousness': deque(maxlen=100)
        }
        
        # Current KPI values
        self.current_kpis = {
            'skill_acquisition_rate': 0.0,
            'transfer_learning_rate': 0,
            'zero_shot_success_count': 0,
            'agentic_capability_score': 0.0,
            'recursive_self_improvement_rate': 0.0,
            'sample_efficiency_trend': 0.0,
            'metacognition_accuracy': 0.0,
            'multi_modal_integration_score': 0.0,
            'consciousness': 0.0
        }
        
        # Track attempts for success rate calculations
        self.code_mod_attempts = 0
        self.code_mod_successes = 0
        self.zero_shot_attempts = 0
        self.zero_shot_successes = 0
        
        # Evolution tracking
        self.evolution_cycles = 0
        self.last_topic = None
        self.accelerator_triggered = False
        
        # Dynamic network state
        self._neuron_count = 3533  # Base from Neo4j
        self._synapse_count = 0
        self._consciousness_history = deque(maxlen=20)
        
        # Load saved state
        self.load_state()
    
    def process(self, input_data: Dict) -> Dict:
        """Process input data through the network with improved state tracking"""
        try:
            # Track evolution cycles
            if 'evolution_cycle' in input_data:
                self.evolution_cycles = input_data.get('evolution_cycle', 0)
            
            # Track learning topics
            if 'learning_topic' in input_data:
                self.last_topic = input_data.get('learning_topic')
            
            # Track accelerator flag
            if input_data.get('is_accelerator', False):
                self.accelerator_triggered = True
                # Accelerator gives bonus to consciousness
                self.current_kpis['consciousness'] += 0.01
            
            # Update neurons and synapses based on learning
            if self.last_topic:
                # New learning creates new neurons
                self._neuron_count += 1
                # Create synapses between related concepts
                self._synapse_count += 2
            
            # Cap at reasonable values
            self._neuron_count = min(10000, self._neuron_count)
            self._synapse_count = min(50000, self._synapse_count)
            
            return {'processed': True, 'input': input_data}
        except Exception as e:
            logger.error(f"Process error: {e}")
            return {'processed': False, 'error': str(e)}
    
    def evolve(self) -> Dict:
        """Evolve the network with exponential moving average consciousness"""
        try:
            # Calculate base consciousness from KPIs
            raw_consciousness = 0.0
            
            # Weighted contributions from each KPI
            weights = {
                'skill_acquisition_rate': 0.20,
                'transfer_learning_rate': 0.15,
                'agentic_capability_score': 0.20,
                'metacognition_accuracy': 0.15,
                'multi_modal_integration_score': 0.15,
                'recursive_self_improvement_rate': 0.10,
                'zero_shot_success_count': 0.05
            }
            
            for kpi, weight in weights.items():
                value = self.current_kpis.get(kpi, 0.0)
                if kpi == 'transfer_learning_rate':
                    # Normalize to 0-1 scale (max 100 synapses)
                    normalized = min(1.0, value / 100)
                elif kpi == 'zero_shot_success_count':
                    normalized = min(1.0, value / 10)
                elif kpi == 'recursive_self_improvement_rate':
                    normalized = value / 100
                elif kpi == 'metacognition_accuracy':
                    normalized = value / 100
                else:
                    normalized = value
                
                raw_consciousness += normalized * weight
            
            # Apply exponential moving average for smooth consciousness
            self._consciousness_history.append(raw_consciousness)
            if len(self._consciousness_history) > 0:
                # EMA with alpha=0.3
                alpha = 0.3
                if len(self._consciousness_history) == 1:
                    consciousness = raw_consciousness
                else:
                    prev = self.current_kpis.get('consciousness', 0.0)
                    consciousness = alpha * raw_consciousness + (1 - alpha) * prev
            else:
                consciousness = raw_consciousness
            
            # Cap at 1.0
            consciousness = min(1.0, max(0.0, consciousness))
            
            # Update consciousness
            self.current_kpis['consciousness'] = round(consciousness, 6)
            self.kpi_history['consciousness'].append({
                'value': consciousness,
                'raw': raw_consciousness,
                'timestamp': datetime.now().isoformat()
            })
            
            # Track evolution cycle
            self.evolution_cycles += 1
            
            # Save state periodically (every 10 cycles)
            if self.evolution_cycles % 10 == 0:
                self.save_state()
            
            return {
                'consciousness': consciousness,
                'raw_consciousness': raw_consciousness,
                'evolution_cycle': self.evolution_cycles,
                'neurons': self._neuron_count,
                'synapses': self._synapse_count,
                'changes': self._get_recent_changes()
            }
        except Exception as e:
            logger.error(f"Evolve error: {e}")
            return {
                'consciousness': 0.0,
                'evolution_cycle': 0,
                'neurons': 0,
                'synapses': 0,
                'changes': [],
                'error': str(e)
            }
    
    def _get_recent_changes(self) -> List:
        """Get recent evolution changes"""
        changes = []
        if self.accelerator_triggered:
            changes.append("Accelerator boost applied")
            self.accelerator_triggered = False
        if self.last_topic:
            changes.append(f"Learned: {self.last_topic[:50]}")
        return changes
    
    def save_state(self) -> None:
        """Save network state to disk"""
        try:
            state_file = self.data_path / 'si_core_state.json'
            state = {
                'evolution_cycles': self.evolution_cycles,
                'current_kpis': self.current_kpis,
                'neuron_count': self._neuron_count,
                'synapse_count': self._synapse_count,
                'code_mod_attempts': self.code_mod_attempts,
                'code_mod_successes': self.code_mod_successes,
                'zero_shot_attempts': self.zero_shot_attempts,
                'zero_shot_successes': self.zero_shot_successes,
                'last_saved': datetime.now().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved SI Core state: {self.evolution_cycles} cycles")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self) -> None:
        """Load network state from disk"""
        try:
            state_file = self.data_path / 'si_core_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self.evolution_cycles = state.get('evolution_cycles', 0)
                self.current_kpis.update(state.get('current_kpis', {}))
                self._neuron_count = state.get('neuron_count', 3533)
                self._synapse_count = state.get('synapse_count', 0)
                self.code_mod_attempts = state.get('code_mod_attempts', 0)
                self.code_mod_successes = state.get('code_mod_successes', 0)
                self.zero_shot_attempts = state.get('zero_shot_attempts', 0)
                self.zero_shot_successes = state.get('zero_shot_successes', 0)
                logger.info(f"Loaded SI Core state: {self.evolution_cycles} cycles, {self._neuron_count} neurons")
        except Exception as e:
            logger.warning(f"Failed to load state (starting fresh): {e}")
    
    @property
    def consciousness(self) -> float:
        """Get current consciousness level"""
        return self.current_kpis.get('consciousness', 0.0)
    
    @property
    def neuron_count(self) -> int:
        """Get dynamic neuron count"""
        return self._neuron_count
    
    @property
    def synapse_count(self) -> int:
        """Get dynamic synapse count"""
        return self._synapse_count
    
    # ========== KPI Update Methods ==========
    
    def update_kpi_1_skill_acquisition(self, new_domains_mastered: float) -> None:
        """KPI 1: Track new domains mastered per cycle (precision: 0.001)"""
        self.current_kpis['skill_acquisition_rate'] = round(new_domains_mastered, 3)
        self.kpi_history['skill_acquisition_rate'].append({
            'value': self.current_kpis['skill_acquisition_rate'],
            'timestamp': datetime.now().isoformat()
        })
    
    def update_kpi_2_transfer_learning(self, new_cross_domain_synapses: int) -> None:
        """KPI 2: Track new cross-domain synapses created"""
        self.current_kpis['transfer_learning_rate'] = new_cross_domain_synapses
        self._synapse_count += new_cross_domain_synapses
        self.kpi_history['transfer_learning_rate'].append({
            'value': self.current_kpis['transfer_learning_rate'],
            'timestamp': datetime.now().isoformat()
        })
    
    def update_kpi_3_zero_shot(self, success: bool) -> None:
        """KPI 3: Track zero-shot task successes"""
        self.zero_shot_attempts += 1
        if success:
            self.zero_shot_successes += 1
            self.current_kpis['zero_shot_success_count'] = self.zero_shot_successes
        
        self.kpi_history['zero_shot_success_count'].append({
            'value': self.current_kpis['zero_shot_success_count'],
            'attempts': self.zero_shot_attempts,
            'successes': self.zero_shot_successes,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_kpi_4_agentic_capability(self, multi_step_tasks_completed: int, 
                                        total_attempted: int) -> None:
        """KPI 4: Track multi-step tasks completed autonomously (0-1 scale)"""
        if total_attempted > 0:
            score = multi_step_tasks_completed / total_attempted
            self.current_kpis['agentic_capability_score'] = round(score, 3)
        self.kpi_history['agentic_capability_score'].append({
            'value': self.current_kpis['agentic_capability_score'],
            'completed': multi_step_tasks_completed,
            'attempted': total_attempted,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_kpi_5_recursive_self_improvement(self, success: bool) -> None:
        """KPI 5: Track code self-modification success rate"""
        self.code_mod_attempts += 1
        if success:
            self.code_mod_successes += 1
        
        rate = (self.code_mod_successes / self.code_mod_attempts * 100) if self.code_mod_attempts > 0 else 0
        self.current_kpis['recursive_self_improvement_rate'] = round(rate, 1)
        
        self.kpi_history['recursive_self_improvement_rate'].append({
            'value': self.current_kpis['recursive_self_improvement_rate'],
            'attempts': self.code_mod_attempts,
            'successes': self.code_mod_successes,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_kpi_6_sample_efficiency(self, data_points: int, concepts_learned: int) -> None:
        """KPI 6: Track data points needed per new concept learned"""
        if concepts_learned > 0:
            ratio = data_points / concepts_learned
            self.current_kpis['sample_efficiency_trend'] = round(ratio, 1)
        self.kpi_history['sample_efficiency_trend'].append({
            'value': self.current_kpis['sample_efficiency_trend'],
            'data_points': data_points,
            'concepts_learned': concepts_learned,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_kpi_7_metacognition(self, predicted_confidence: float, actual_accuracy: float) -> None:
        """KPI 7: Track confidence calibration error margin"""
        error_margin = abs(predicted_confidence - actual_accuracy) * 100
        # Score is 100 - error_margin (lower error = better score)
        accuracy_score = max(0, 100 - error_margin)
        self.current_kpis['metacognition_accuracy'] = round(accuracy_score, 1)
        
        self.kpi_history['metacognition_accuracy'].append({
            'value': self.current_kpis['metacognition_accuracy'],
            'predicted': predicted_confidence,
            'actual': actual_accuracy,
            'error_margin': error_margin,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_kpi_8_multi_modal(self, new_synergies_discovered: int, 
                                 total_modalities: int = 5) -> None:
        """KPI 8: Track new modality synergies discovered (0-1 scale)"""
        # Max synergies = n*(n-1)/2 for n modalities
        max_synergies = (total_modalities * (total_modalities - 1)) / 2
        if max_synergies > 0:
            score = min(1.0, new_synergies_discovered / max_synergies)
            self.current_kpis['multi_modal_integration_score'] = round(score, 3)
        
        self.kpi_history['multi_modal_integration_score'].append({
            'value': self.current_kpis['multi_modal_integration_score'],
            'synergies': new_synergies_discovered,
            'max_synergies': max_synergies,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_kpis_for_status(self) -> Dict:
        """Return current KPIs formatted for status page"""
        return {
            'skill_acquisition_rate': self.current_kpis['skill_acquisition_rate'],
            'transfer_learning_rate': self.current_kpis['transfer_learning_rate'],
            'zero_shot_success_count': self.current_kpis['zero_shot_success_count'],
            'agentic_capability_score': self.current_kpis['agentic_capability_score'],
            'recursive_self_improvement_rate': self.current_kpis['recursive_self_improvement_rate'],
            'sample_efficiency_trend': self.current_kpis['sample_efficiency_trend'],
            'metacognition_accuracy': self.current_kpis['metacognition_accuracy'],
            'multi_modal_integration_score': self.current_kpis['multi_modal_integration_score']
        }
    
    def get_kpi_history(self, kpi_name: str, last_n: int = 10) -> List:
        """Get historical values for a KPI"""
        if kpi_name in self.kpi_history:
            return list(self.kpi_history[kpi_name])[-last_n:]
        return []
    
    def calculate_any_improvement(self) -> bool:
        """Check if ANY KPI improved in the last cycle"""
        thresholds = {
            'skill_acquisition_rate': 0.001,
            'transfer_learning_rate': 1,
            'zero_shot_success_count': 1,
            'agentic_capability_score': 0.001,
            'recursive_self_improvement_rate': 0.1,
            'sample_efficiency_trend': -0.1,
            'metacognition_accuracy': 0.1,
            'multi_modal_integration_score': 0.001
        }
        
        improved = False
        for kpi, threshold in thresholds.items():
            value = self.current_kpis[kpi]
            if kpi == 'sample_efficiency_trend':
                if value < threshold:
                    improved = True
            else:
                if value > threshold:
                    improved = True
        
        return improved
