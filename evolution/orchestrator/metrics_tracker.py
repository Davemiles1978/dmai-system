#!/usr/bin/env python3
"""
Metrics Tracker for DMAI Evolution System
Tracks evolution quality, diversity, and improvement rates
"""

import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

class MetricsTracker:
    def __init__(self, config_path=None):
        self.evolution_dir = "/Users/davidmiles/Desktop/AI-Evolution-System/evolution"
        self.history_file = f"{self.evolution_dir}/history/evolution_log.json"
        self.metrics_file = f"{self.evolution_dir}/history/metrics_history.json"
        self.config = self.load_config(config_path)
        self.metrics = self.load_metrics()
        
    def load_config(self, config_path):
        if not config_path:
            config_path = f"{self.evolution_dir}/config/evolution_config.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def load_metrics(self):
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {"history": [], "current": {}}
        return {"history": [], "current": {}}
    
    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def calculate_diversity_score(self, versions):
        """Calculate how different the AIs are becoming"""
        if len(versions) < 2:
            return 1.0
        
        # Compare code structure differences
        similarities = []
        for i in range(len(versions)):
            for j in range(i+1, len(versions)):
                if 'hash' in versions[i] and 'hash' in versions[j]:
                    # Different hashes = different code
                    if versions[i]['hash'] != versions[j]['hash']:
                        similarities.append(0)
                    else:
                        similarities.append(1)
        
        if not similarities:
            return 1.0
        
        # Diversity score = 1 - average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def calculate_improvement_rate(self, versions):
        """Calculate if changes are actually improving things"""
        if len(versions) < 2:
            return 0.5
        
        # Look at improvement scores in version history
        scores = [v.get('improvement_score', 0) for v in versions if 'improvement_score' in v]
        if len(scores) < 2:
            return 0.5
        
        # Check if later versions have higher scores
        improvements = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                improvements += 1
        
        return improvements / (len(scores) - 1)
    
    def calculate_external_injection_rate(self, cycle_log):
        """Calculate how many new ideas come from external research"""
        if not cycle_log:
            return 0.0
        
        external_count = cycle_log.get('external_innovations', 0)
        internal_count = cycle_log.get('internal_evolutions', 0)
        total = external_count + internal_count
        
        if total == 0:
            return 0.0
        
        return external_count / total
    
    def check_stagnation(self, target_name, versions):
        """Check if a target has stopped evolving"""
        if len(versions) < self.config.get('metrics', {}).get('stagnation_warning_threshold', 10):
            return False
        
        recent = versions[-self.config['metrics']['stagnation_warning_threshold']:]
        scores = [v.get('improvement_score', 0) for v in recent]
        
        # If no improvement in last N versions
        if all(s <= scores[0] for s in scores[1:]):
            return True
        
        return False
    
    def record_cycle(self, cycle_data):
        """Record metrics for a complete evolution cycle"""
        timestamp = datetime.now().isoformat()
        
        cycle_record = {
            'timestamp': timestamp,
            'cycle_id': cycle_data.get('cycle_id'),
            'external_innovations': cycle_data.get('external_innovations', 0),
            'internal_evolutions': cycle_data.get('internal_evolutions', 0),
            'evaluations_performed': cycle_data.get('evaluations_performed', []),
            'successful_evolutions': cycle_data.get('successful_evolutions', 0),
            'failed_evolutions': cycle_data.get('failed_evolutions', 0),
            'research_sources_used': cycle_data.get('research_sources_used', [])
        }
        
        # Calculate metrics
        cycle_record['external_injection_rate'] = self.calculate_external_injection_rate(cycle_record)
        
        self.metrics['history'].append(cycle_record)
        
        # Update current aggregate metrics
        self.update_aggregate_metrics()
        self.save_metrics()
        
        return cycle_record
    
    def update_aggregate_metrics(self):
        """Update rolling aggregate metrics"""
        history = self.metrics['history'][-50:]  # Last 50 cycles
        
        if not history:
            return
        
        self.metrics['current'] = {
            'avg_external_injection': np.mean([h.get('external_injection_rate', 0) for h in history]),
            'avg_success_rate': np.mean([h.get('successful_evolutions', 0) / max(h.get('evaluations_performed', [1]), 1) for h in history]),
            'total_cycles': len(self.metrics['history']),
            'last_update': datetime.now().isoformat()
        }
    
    def generate_report(self):
        """Generate a human-readable evolution report"""
        report = []
        report.append("\n" + "="*60)
        report.append("📊 DMAI EVOLUTION METRICS REPORT")
        report.append("="*60)
        
        current = self.metrics.get('current', {})
        report.append(f"\n📈 Current Stats:")
        report.append(f"   Total Cycles: {current.get('total_cycles', 0)}")
        report.append(f"   Avg External Injection: {current.get('avg_external_injection', 0):.2%}")
        report.append(f"   Avg Success Rate: {current.get('avg_success_rate', 0):.2%}")
        
        # Recent trends
        recent = self.metrics['history'][-10:] if self.metrics['history'] else []
        if recent:
            report.append(f"\n📉 Last 10 Cycles:")
            external_trend = [h.get('external_injection_rate', 0) for h in recent]
            report.append(f"   External Injection Trend: {', '.join([f'{x:.0%}' for x in external_trend])}")
        
        report.append("\n" + "="*60)
        return "\n".join(report)

if __name__ == "__main__":
    tracker = MetricsTracker()
    print(tracker.generate_report())
