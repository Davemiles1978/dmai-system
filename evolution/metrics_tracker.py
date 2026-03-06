#!/usr/bin/env python3
"""
Metrics Tracker for DMAI Evolution
Tracks quality and diversity of evolution
"""

import json
import time
from datetime import datetime
from pathlib import Path

class MetricsTracker:
    def __init__(self):
        self.metrics_file = "/Users/davidmiles/Desktop/AI-Evolution-System/data/evolution_metrics.json"
        self.load_metrics()
    
    def load_metrics(self):
        if Path(self.metrics_file).exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {"cycles": [], "current": {}}
    
    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def record_cycle(self, cycle_data):
        cycle_data['timestamp'] = datetime.now().isoformat()
        self.metrics['cycles'].append(cycle_data)
        
        # Keep last 100 cycles
        if len(self.metrics['cycles']) > 100:
            self.metrics['cycles'] = self.metrics['cycles'][-100:]
        
        self.save_metrics()
    
    def get_diversity_score(self):
        """Calculate how diverse the evaluations are"""
        if len(self.metrics['cycles']) < 10:
            return 0.5
        
        recent = self.metrics['cycles'][-10:]
        evaluator_pairs = []
        for cycle in recent:
            for eval in cycle.get('evaluations', []):
                pair = (eval.get('evaluator'), eval.get('target'))
                evaluator_pairs.append(pair)
        
        unique_pairs = len(set(evaluator_pairs))
        total_pairs = len(evaluator_pairs)
        
        if total_pairs == 0:
            return 0.5
        
        return unique_pairs / total_pairs

metrics = MetricsTracker()
