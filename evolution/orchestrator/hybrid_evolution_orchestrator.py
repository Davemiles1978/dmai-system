#!/usr/bin/env python3
"""
Hybrid Multi-AI Evolution Orchestrator
60% External Research / 40% Internal Co-evolution
Every AI evolves every cycle through random pairings
"""

import sys
import os
import json
import random
import time
import subprocess
import importlib.util
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent)))))))

from evolution.orchestrator.metrics_tracker import MetricsTracker
from evolution.orchestrator.version_controller import VersionController

class HybridEvolutionOrchestrator:
    def __init__(self):
        self.evolution_dir = "/Users/davidmiles/Desktop/dmai-system/evolution"
        self.config_path = f"{self.evolution_dir}/config/evolution_config.json"
        self.load_config()
        
        # Initialize components
        self.metrics = MetricsTracker(self.config_path)
        self.version_control = VersionController()
        
        # Track cycle state
        self.current_cycle = 0
        self.cycle_id = None
        self.cycle_results = {
            'external_innovations': 0,
            'internal_evolutions': 0,
            'evaluations_performed': [],
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'research_sources_used': []
        }
        
        # Ensure directories exist
        self.ensure_directories()
        
        print(f"🚀 Hybrid Evolution Orchestrator initialized")
        print(f"   External Research Weight: {self.config['evolution_weights']['external_research']*100}%")
        print(f"   Internal Co-evolution Weight: {self.config['evolution_weights']['internal_coevolution']*100}%")
    
    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def ensure_directories(self):
        dirs = [
            f"{self.evolution_dir}/researchers",
            f"{self.evolution_dir}/evaluators",
            f"{self.evolution_dir}/targets",
            f"{self.evolution_dir}/history/versions",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def get_evaluation_pool(self):
        """Get list of all evaluable targets"""
        return self.config.get('evaluation_pool', [])
    
    def run_external_research(self):
        """Run external research cycle (60% weight)"""
        print("\n🔍 Running EXTERNAL RESEARCH cycle...")
        sources_used = []
        innovations_found = 0
        
        research_sources = self.config.get('research_sources', {})
        
        for source_name, source_config in research_sources.items():
            if not source_config.get('enabled', False):
                continue
            
            # Weighted random selection based on source weights
            if random.random() < source_config.get('weight', 0.1):
                print(f"   📡 Researching: {source_name}")
                sources_used.append(source_name)
                
                # Simulate research (in production, this would call actual scrapers)
                # Each researcher module would be imported and run
                if random.random() < 0.3:  # 30% chance of finding something
                    innovations_found += 1
                    print(f"      ✅ Found innovation from {source_name}")
        
        self.cycle_results['external_innovations'] += innovations_found
        self.cycle_results['research_sources_used'].extend(sources_used)
        
        return innovations_found
    
    def run_internal_coevolution(self):
        """Run internal co-evolution cycle (40% weight)"""
        print("\n🔄 Running INTERNAL CO-EVOLUTION cycle...")
        
        pool = self.get_evaluation_pool()
        if len(pool) < 2:
            print("   ⚠️ Not enough evaluators in pool")
            return 0
        
        # Create random evaluation pairs
        # Each target gets evaluated by a different random evaluator
        evaluators = [e for e in pool if e['enabled']]
        targets = [t for t in pool if t['enabled']]
        
        # Shuffle to ensure randomness
        random.shuffle(evaluators)
        random.shuffle(targets)
        
        evolutions_performed = 0
        
        # Ensure each target gets evaluated
        for i, target in enumerate(targets):
            # Pick a random evaluator that's not the target itself
            available_evaluators = [e for e in evaluators if e['name'] != target['name']]
            if not available_evaluators:
                continue
            
            evaluator = random.choice(available_evaluators)
            
            print(f"\n   🤖 {evaluator['name']} evaluating {target['name']}")
            
            # Record the evaluation pair
            pair = {
                'evaluator': evaluator['name'],
                'target': target['name'],
                'timestamp': datetime.now().isoformat()
            }
            self.cycle_results['evaluations_performed'].append(pair)
            
            # Create version before evolution
            if os.path.exists(target['path']):
                old_version = self.version_control.create_version(
                    target['name'],
                    target['path'],
                    {'evaluator': evaluator['name'], 'stage': 'before'}
                )
                
                # Run evaluation (simulated - would call actual evaluator)
                success = self.run_evaluation(evaluator, target)
                
                if success:
                    evolutions_performed += 1
                    self.cycle_results['successful_evolutions'] += 1
                    
                    # Create version after evolution
                    new_version = self.version_control.create_version(
                        target['name'],
                        target['path'],
                        {
                            'evaluator': evaluator['name'],
                            'stage': 'after',
                            'improvement_score': random.uniform(0.6, 0.95)  # Simulated score
                        }
                    )
                    
                    # Validate improvement
                    if old_version and new_version:
                        if self.version_control.validate_improvement(old_version, new_version):
                            print(f"      ✅ Validated improvement")
                        else:
                            print(f"      ⚠️ No significant improvement - considering rollback")
                            # Could auto-rollback here
                else:
                    self.cycle_results['failed_evolutions'] += 1
        
        self.cycle_results['internal_evolutions'] += evolutions_performed
        return evolutions_performed
    
    def run_evaluation(self, evaluator, target):
        """Run a single evaluation (placeholder - would call actual evaluator)"""
        try:
            # In production, this would import and run the evaluator module
            # For now, simulate with random success
            time.sleep(0.5)  # Simulate work
            return random.random() > 0.2  # 80% success rate
        except Exception as e:
            print(f"      ❌ Evaluation failed: {e}")
            return False
    
    def apply_external_innovations(self, innovations):
        """Apply external research findings to relevant targets"""
        if innovations == 0:
            return
        
        print(f"\n📦 Applying {innovations} external innovations...")
        
        # In production, this would map innovations to specific targets
        # and potentially modify code based on research findings
        
        # For now, just record that innovations were applied
        pass
    
    def run_cycle(self):
        """Run one complete hybrid evolution cycle"""
        self.current_cycle += 1
        self.cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("\n" + "="*70)
        print(f"🔄 HYBRID EVOLUTION CYCLE #{self.current_cycle}")
        print(f"   Cycle ID: {self.cycle_id}")
        print("="*70)
        
        # Reset cycle results
        self.cycle_results = {
            'cycle_id': self.cycle_id,
            'external_innovations': 0,
            'internal_evolutions': 0,
            'evaluations_performed': [],
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'research_sources_used': []
        }
        
        # Decide whether to run external or internal based on weights
        external_weight = self.config['evolution_weights']['external_research']
        
        if random.random() < external_weight:
            # Run external research
            innovations = self.run_external_research()
            if innovations > 0:
                self.apply_external_innovations(innovations)
        else:
            # Run internal co-evolution
            self.run_internal_coevolution()
        
        # Record metrics
        self.metrics.record_cycle(self.cycle_results)
        
        # Generate report
        self.print_cycle_summary()
        
        # Check for stagnation
        self.check_system_health()
    
    def print_cycle_summary(self):
        """Print summary of the cycle"""
        print("\n" + "-"*50)
        print("📊 CYCLE SUMMARY")
        print("-"*50)
        print(f"   External Innovations: {self.cycle_results['external_innovations']}")
        print(f"   Internal Evolutions: {self.cycle_results['internal_evolutions']}")
        print(f"   Successful: {self.cycle_results['successful_evolutions']}")
        print(f"   Failed: {self.cycle_results['failed_evolutions']}")
        print(f"   Research Sources: {', '.join(self.cycle_results['research_sources_used'])}")
        print("-"*50)
    
    def check_system_health(self):
        """Check for stagnation across all targets"""
        stagnation_report = self.version_control.get_stagnation_report()
        
        stagnant = [t for t, data in stagnation_report.items() if data.get('status') == 'STAGNANT']
        
        if stagnant:
            print(f"\n⚠️  STAGNATION WARNING: {len(stagnant)} targets stagnant")
            for target in stagnant:
                print(f"   - {target}")
            print("\n   Recommend: Increase external research weight temporarily")
    
    def run_continuous(self, cycles=None):
        """Run continuous evolution cycles with random timing"""
        print("\n" + "="*70)
        print("🚀 STARTING CONTINUOUS HYBRID EVOLUTION")
        print("="*70)
        
        cycle_count = 0
        while cycles is None or cycle_count < cycles:
            try:
                self.run_cycle()
                cycle_count += 1
                
                # Random wait between cycles (2-6 hours)
                min_wait = self.config['cycle_settings']['cycle_interval_minutes'] * 60
                max_wait = min_wait * 3
                wait_time = random.randint(min_wait, max_wait)
                
                print(f"\n⏰ Next cycle in {wait_time/3600:.1f} hours")
                print(f"   (Press Ctrl+C to stop)")
                
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                print("\n\n⏹️ Evolution stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error in cycle: {e}")
                time.sleep(300)  # Wait 5 minutes on error
        
        print(f"\n✅ Completed {cycle_count} evolution cycles")

if __name__ == "__main__":
    orchestrator = HybridEvolutionOrchestrator()
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        orchestrator.run_cycle()
    elif len(sys.argv) > 2 and sys.argv[1] == "--cycles":
        orchestrator.run_continuous(cycles=int(sys.argv[2]))
    else:
        orchestrator.run_continuous()
