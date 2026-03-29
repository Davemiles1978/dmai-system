# ============================================================================
# EVOLUTION METRICS
# ============================================================================
"""
Tracks evolution success rates and identifies improvement patterns
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EvolutionMetrics:
    """
    Tracks evolution cycles and what makes them successful
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.metrics_file = data_path / 'learning' / 'evolution_metrics.json'
        self.cycles = []
        self._load()
        
        logger.info(f"📈 EvolutionMetrics initialized")
    
    def _load(self):
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.cycles = data.get('cycles', [])
            except:
                pass
    
    def _save(self):
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump({
                'cycles': self.cycles[-1000:],
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def record_cycle(self, cycle_number: int, consciousness_growth: float, neurons_added: int, 
                     synapses_added: int, learning_topic: str, was_successful: bool):
        """Record an evolution cycle"""
        entry = {
            'cycle': cycle_number,
            'timestamp': datetime.now().isoformat(),
            'consciousness_growth': consciousness_growth,
            'neurons_added': neurons_added,
            'synapses_added': synapses_added,
            'learning_topic': learning_topic,
            'was_successful': was_successful
        }
        self.cycles.append(entry)
        self._save()
    
    def get_success_rate(self) -> float:
        """Calculate success rate of evolution cycles"""
        if not self.cycles:
            return 0.0
        
        successful = sum(1 for c in self.cycles if c.get('was_successful', False))
        return successful / len(self.cycles) * 100
