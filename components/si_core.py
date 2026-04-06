"""
SI Core - Self-Improvement Core for Holistic KPI Tracking
Tracks 8 intelligence metrics for evolution success criteria
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from collections import deque

class SICore:
    def __init__(self):
        # KPI history storage (last 100 cycles)
        self.kpi_history = {
            'skill_acquisition_rate': deque(maxlen=100),
            'transfer_learning_rate': deque(maxlen=100),
            'zero_shot_success_count': deque(maxlen=100),
            'agentic_capability_score': deque(maxlen=100),
            'recursive_self_improvement_rate': deque(maxlen=100),
            'sample_efficiency_trend': deque(maxlen=100),
            'metacognition_accuracy': deque(maxlen=100),
            'multi_modal_integration_score': deque(maxlen=100)
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
            'multi_modal_integration_score': 0.0
        }
        
        # Track attempts for success rate calculations
        self.code_mod_attempts = 0
        self.code_mod_successes = 0
        self.zero_shot_attempts = 0
        self.zero_shot_successes = 0
        
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
        # This will be called from evolution_cycle
        # For now, return True if any KPI > threshold
        thresholds = {
            'skill_acquisition_rate': 0.001,
            'transfer_learning_rate': 1,
            'zero_shot_success_count': 1,
            'agentic_capability_score': 0.001,
            'recursive_self_improvement_rate': 0.1,
            'sample_efficiency_trend': -0.1,  # Decreasing is good
            'metacognition_accuracy': 0.1,
            'multi_modal_integration_score': 0.001
        }
        
        improved = False
        for kpi, threshold in thresholds.items():
            value = self.current_kpis[kpi]
            if kpi == 'sample_efficiency_trend':
                if value < threshold:  # Lower is better
                    improved = True
            else:
                if value > threshold:
                    improved = True
        
        return improved
