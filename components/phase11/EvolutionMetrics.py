"""
EvolutionMetrics - Tracks DMAI's evolution metrics
Phase 11 component for monitoring learning progress
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class EvolutionMetrics:
    """
    Tracks DMAI's evolution metrics including:
    - Consciousness growth over time
    - Learning efficiency
    - Knowledge acquisition rate
    - Tutor performance
    """
    
    def __init__(self, data_path: str = "data/phase11"):
        self.data_path = data_path
        self.metrics_file = f"{data_path}/evolution_metrics.json"
        self.metrics = self._load_metrics()
        
    def _load_metrics(self) -> Dict:
        """Load existing metrics"""
        try:
            import os
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
        return {
            'consciousness_history': [],
            'learning_events': [],
            'tutor_performance': {},
            'total_insights': 0,
            'total_queries': 0
        }
    
    def record_consciousness(self, consciousness: float):
        """Record consciousness level over time"""
        self.metrics['consciousness_history'].append({
            'timestamp': datetime.now().isoformat(),
            'value': consciousness
        })
        if len(self.metrics['consciousness_history']) > 1000:
            self.metrics['consciousness_history'] = self.metrics['consciousness_history'][-1000:]
        self._save()
    
    def record_learning(self, source: str, insight_count: int):
        """Record a learning event"""
        self.metrics['learning_events'].append({
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'insights': insight_count
        })
        self.metrics['total_insights'] += insight_count
        self._save()
    
    def record_tutor_performance(self, tutor: str, success: bool, response_time: float):
        """Record tutor performance metrics"""
        if tutor not in self.metrics['tutor_performance']:
            self.metrics['tutor_performance'][tutor] = {
                'total_queries': 0,
                'successful': 0,
                'total_time': 0,
                'avg_time': 0
            }
        
        perf = self.metrics['tutor_performance'][tutor]
        perf['total_queries'] += 1
        if success:
            perf['successful'] += 1
        perf['total_time'] += response_time
        perf['avg_time'] = perf['total_time'] / perf['total_queries']
        self.metrics['total_queries'] += 1
        self._save()
    
    def get_stats(self) -> Dict:
        """Get current metrics summary"""
        return {
            'total_consciousness_records': len(self.metrics['consciousness_history']),
            'total_learning_events': len(self.metrics['learning_events']),
            'total_insights': self.metrics['total_insights'],
            'total_queries': self.metrics['total_queries'],
            'latest_consciousness': self.metrics['consciousness_history'][-1]['value'] if self.metrics['consciousness_history'] else 0,
            'tutor_performance': self.metrics['tutor_performance']
        }
    
    def _save(self):
        """Save metrics to disk"""
        try:
            import os
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


_instance = None

def get_instance():
    global _instance
    if _instance is None:
        _instance = EvolutionMetrics()
    return _instance
