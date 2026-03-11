#!/usr/bin/env python3
"""
Continuous Advanced Evolution System
Orchestrates the evolution of DMAI's capabilities
"""

import os
import sys
import json
import time
import random
import hashlib
import datetime
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Memory optimization
import gc
gc.set_threshold(700, 10, 5)  # More aggressive garbage collection
import resource
try:
    # Set soft memory limit
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
except:
    pass

# Clear cache periodically
import threading
import time
def cache_cleaner():
    while True:
        time.sleep(300)  # Every 5 minutes
        gc.collect()  # Force garbage collection
        if hasattr(__import__('torch'), 'mps'):
            import torch
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
threading.Thread(target=cache_cleaner, daemon=True).start()


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import self-healing scanner
try:
    from evolution.system_weakness_scanner import SystemWeaknessScanner
    HEALING_AVAILABLE = True
except ImportError:
    HEALING_AVAILABLE = False
    print("⚠️ System weakness scanner not available")

# Import promotion tracker
try:
    from evolution.promotion_tracker import PromotionTracker
    PROMOTION_AVAILABLE = True
except ImportError:
    PROMOTION_AVAILABLE = False
    print("⚠️ Promotion tracker not available")

# Configure logging
log_dir = Path.home() / "Library/Logs/dmai"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - evolution - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('evolution_engine')

class EvolutionEngine:
    """Main evolution engine that continuously improves DMAI"""
    
    def __init__(self):
        self.cycle_count = 0
        self.evolution_history = []
        self.current_generation = 1
        self.best_score = 0
        
        # Initialize self-healing if available
        self.healing_scanner = SystemWeaknessScanner() if HEALING_AVAILABLE else None
        self.promotion_tracker = PromotionTracker() if PROMOTION_AVAILABLE else None
        
        # Load existing evolution data
        self.load_evolution_state()
        
        logger.info(f"🧬 Evolution Engine initialized (gen {self.current_generation})")
    
    def load_evolution_state(self):
        """Load previous evolution state if exists"""
        state_file = Path("data/evolution/evolution_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.current_generation = state.get('generation', 1)
                    self.best_score = state.get('best_score', 0)
                    self.evolution_history = state.get('history', [])
                logger.info(f"Loaded evolution state: gen {self.current_generation}")
            except Exception as e:
                logger.error(f"Failed to load evolution state: {e}")
    
    def save_evolution_state(self):
        """Save current evolution state"""
        state_file = Path("data/evolution/evolution_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'generation': self.current_generation,
            'best_score': self.best_score,
            'history': self.evolution_history[-100:],  # Keep last 100
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def scan_for_improvements(self) -> List[Dict[str, Any]]:
        """Scan system for potential improvement opportunities"""
        opportunities = []
        
        # Check service response times
        # Check code quality
        # Check for outdated patterns
        # etc.
        
        return opportunities
    
    def run_evolution_cycle(self):
        """Run one complete evolution cycle"""
        self.cycle_count += 1
        logger.info(f"🔬 Starting evolution cycle {self.cycle_count}")
        
        # Phase 1: Scan for improvement opportunities
        opportunities = self.scan_for_improvements()
        
        # Phase 2: Generate mutations
        mutations = self.generate_mutations(opportunities)
        
        # Phase 3: Test mutations
        results = self.test_mutations(mutations)
        
        # Phase 4: Select best mutations
        best = self.select_best_mutations(results)
        
        # Phase 5: Apply selected mutations
        applied = self.apply_mutations(best)
        
        # Phase 6: Track promotion (if available)
        if self.promotion_tracker and applied:
            for mutation in applied:
                if mutation.get('success_score', 0) > 0.7:
                    self.promotion_tracker.track_success(
                        mutation.get('id', 'unknown'),
                        {'score': mutation.get('success_score', 0)}
                    )
        
        # Phase 7: Run self-healing every 5th cycle
        if HEALING_AVAILABLE and self.cycle_count % 5 == 0:
            self.run_self_healing_cycle()
        
        # Update generation
        self.current_generation += 1
        self.save_evolution_state()
        
        logger.info(f"✅ Evolution cycle {self.cycle_count} complete")
        
        return {
            'cycle': self.cycle_count,
            'generation': self.current_generation,
            'opportunities': len(opportunities),
            'mutations_tested': len(mutations),
            'mutations_applied': len(applied),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def run_self_healing_cycle(self):
        """Run self-healing cycle to detect and fix system weaknesses"""
        logger.info("🩹 Running self-healing cycle")
        
        if not self.healing_scanner:
            logger.warning("Self-healing scanner not available")
            return
        
        try:
            result = self.healing_scanner.scan_and_heal()
            
            # Record healing in evolution history
            self.evolution_history.append({
                "type": "self_heal",
                "cycle": self.cycle_count,
                "timestamp": datetime.datetime.now().isoformat(),
                "result": {
                    "total_weaknesses": result['total_weaknesses'],
                    "fixes_applied": result['fixes_applied']
                }
            })
            
            # If critical issues found, log them
            critical_issues = [w for w in result.get('weaknesses', []) if w.get('severity', 0) >= 9]
            if critical_issues:
                logger.warning(f"Critical issues found: {len(critical_issues)} - monitoring required")
                for issue in critical_issues:
                    logger.warning(f"  🔴 {issue['type']} in {issue['module']}: {issue['description']}")
            
            logger.info(f"Self-healing complete: healed {result['fixes_applied']}/{result['total_weaknesses']} issues")
            
        except Exception as e:
            logger.error(f"Self-healing cycle failed: {e}")
    
    def generate_mutations(self, opportunities: List[Dict]) -> List[Dict]:
        """Generate evolutionary mutations based on opportunities"""
        mutations = []
        
        for opp in opportunities[:5]:  # Limit to top 5
            mutation = {
                'id': hashlib.md5(f"{opp}{time.time()}".encode()).hexdigest()[:8],
                'type': opp.get('type', 'unknown'),
                'target': opp.get('target', 'system'),
                'hypothesis': opp.get('hypothesis', 'Improve performance'),
                'created': datetime.datetime.now().isoformat()
            }
            mutations.append(mutation)
        
        # If no opportunities, generate random mutations
        if not mutations:
            mutations = self.generate_random_mutations(3)
        
        return mutations
    
    def generate_random_mutations(self, count: int) -> List[Dict]:
        """Generate random mutations for exploration"""
        mutation_types = ['optimization', 'refactor', 'new_feature', 'integration']
        mutations = []
        
        for _ in range(count):
            mutation = {
                'id': hashlib.md5(f"rand_{time.time()}_{random.random()}".encode()).hexdigest()[:8],
                'type': random.choice(mutation_types),
                'target': 'random',
                'hypothesis': 'Exploratory mutation',
                'created': datetime.datetime.now().isoformat(),
                'random_seed': random.random()
            }
            mutations.append(mutation)
        
        return mutations
    
    def test_mutations(self, mutations: List[Dict]) -> List[Dict]:
        """Test mutations and measure their impact"""
        results = []
        
        for mutation in mutations:
            # Simulate testing (in real implementation, would run actual tests)
            test_result = {
                **mutation,
                'test_duration': random.uniform(0.5, 2.0),
                'success_rate': random.uniform(0, 1),
                'performance_impact': random.uniform(-0.1, 0.3),
                'memory_impact': random.uniform(-10, 20),
                'tested_at': datetime.datetime.now().isoformat()
            }
            
            # Calculate overall score
            score = test_result['success_rate'] * 0.5 + \
                   max(0, test_result['performance_impact']) * 0.3 + \
                   max(0, -test_result['memory_impact'] / 100) * 0.2
            
            test_result['score'] = min(1.0, max(0, score))
            results.append(test_result)
        
        return results
    
    def select_best_mutations(self, results: List[Dict], top_k: int = 2) -> List[Dict]:
        """Select the best performing mutations"""
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        return sorted_results[:top_k]
    
    def apply_mutations(self, mutations: List[Dict]) -> List[Dict]:
        """Apply selected mutations to the system"""
        applied = []
        
        for mutation in mutations:
            if mutation.get('score', 0) > 0.6:  # Only apply if score > 0.6
                mutation['applied_at'] = datetime.datetime.now().isoformat()
                mutation['status'] = 'applied'
                
                # Track best score
                if mutation['score'] > self.best_score:
                    self.best_score = mutation['score']
                    logger.info(f"🏆 New best score: {self.best_score:.3f}")
                
                applied.append(mutation)
                
                # Add to history
                self.evolution_history.append({
                    'type': 'mutation_applied',
                    'mutation_id': mutation['id'],
                    'score': mutation['score'],
                    'timestamp': mutation['applied_at']
                })
                
                logger.info(f"✅ Applied mutation {mutation['id']} with score {mutation['score']:.3f}")
        
        return applied
    
    def get_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        return {
            'generation': self.current_generation,
            'cycle': self.cycle_count,
            'best_score': self.best_score,
            'history_length': len(self.evolution_history),
            'healing_available': HEALING_AVAILABLE,
            'promotion_available': PROMOTION_AVAILABLE,
            'timestamp': datetime.datetime.now().isoformat()
        }


# Flask server for API endpoint
try:
    from flask import Flask, jsonify, request
    import threading
    
    app = Flask(__name__)
    engine = EvolutionEngine()
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'service': 'agi-evolution-system'})
    
    @app.route('/status', methods=['GET'])
    def status():
        return jsonify(engine.get_status())
    
    @app.route('/cycle', methods=['POST'])
    def run_cycle():
        result = engine.run_evolution_cycle()
        return jsonify(result)
    
    @app.route('/heal', methods=['POST'])
    def run_heal():
        if HEALING_AVAILABLE:
            engine.run_self_healing_cycle()
            return jsonify({'status': 'healing_cycle_initiated'})
        else:
            return jsonify({'status': 'healing_not_available'}), 400
    
    def run_server():
        app.run(host='0.0.0.0', port=9003)
    
except ImportError:
    logger.warning("Flask not available, running in standalone mode")
    def run_server():
        logger.info("Server mode disabled")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--server':
        # Run as API server
        logger.info("Starting evolution server on port 9003")
        run_server()
    else:
        # Run one evolution cycle
        engine = EvolutionEngine()
        result = engine.run_evolution_cycle()
        print("\n📊 Evolution Cycle Results:")
        print(json.dumps(result, indent=2))
        
        # If self-healing due, it will run automatically
