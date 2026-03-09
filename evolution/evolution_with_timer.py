#!/usr/bin/env python3
"""Evolution Engine with Adaptive Timing - DMAI learns to pace herself"""

import time
import subprocess
import sys
from pathlib import Path
from adaptive_timer import AdaptiveEvolutionTimer

class DMAIEvolution:
    """
    DMAI's evolution controller with self-adjusting timing.
    She literally decides how fast to learn based on her success rate.
    """
    
    def __init__(self):
        self.timer = AdaptiveEvolutionTimer()
        self.evolution_script = Path(__file__).parent / "evolution_engine.py"
        
    def run_cycle(self, parent1=None, parent2=None):
        """Run one evolution cycle and record results"""
        print(f"\n🧬 Evolution Cycle Starting...")
        
        # Build command
        cmd = [sys.executable, str(self.evolution_script)]
        if parent1 and parent2:
            cmd.extend(["--parents", f"{parent1},{parent2}"])
        
        # Run evolution
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse result to determine success
        success = False
        quality = 0
        
        if "✅ Improvement passed all tests!" in result.stdout:
            success = True
            # Try to extract quality
            import re
            quality_match = re.search(r'quality[:\s]*([0-9.]+)', result.stdout)
            if quality_match:
                quality = float(quality_match.group(1))
        
        # Record attempt with timer
        wait_time = self.timer.record_attempt(
            parent1 or "DMAI",
            parent2 or "DMAI",
            success=success,
            improvement_quality=quality
        )
        
        # Show status
        self._show_status(success)
        
        return wait_time
    
    def _show_status(self, last_success):
        """Display current evolution status"""
        info = self.timer.get_stage_info()
        
        status = "✅ SUCCESS" if last_success else "❌ No improvement"
        print(f"\n📊 Cycle Result: {status}")
        print(f"📈 Success Rate: {info['success_rate']}")
        print(f"⏱️  Next cycle in: {info['interval_minutes']:.0f} minutes")
        print(f"🧠 Current Stage: {info['name']}")
    
    def continuous_evolution(self):
        """Run evolution continuously with adaptive timing"""
        print("\n" + "="*60)
        print("🚀 DMAI CONTINUOUS EVOLUTION STARTED")
        print("She will automatically adjust her learning pace")
        print("="*60 + "\n")
        
        cycle_count = 0
        
        while True:
            cycle_count += 1
            print(f"\n{'='*60}")
            print(f"Cycle #{cycle_count}")
            print(f"{'='*60}")
            
            # Get recommended parents based on history
            parent1, parent2 = self._select_parents()
            
            # Run evolution cycle
            wait_time = self.run_cycle(parent1, parent2)
            
            # Check if she needs to change strategy
            if self.timer.should_try_new_strategy():
                print("\n🔄 DMAI is changing her evolution strategy...")
                # Could implement strategy switching here
            
            # Wait the adaptive amount
            print(f"\n⏳ Waiting {wait_time/60:.1f} minutes...")
            print("(Press Ctrl+C to pause)\n")
            
            # Smart waiting - check every minute if we should continue
            for _ in range(int(wait_time / 60)):
                time.sleep(60)
                # Could check for external signals here
    
    def _select_parents(self):
        """Intelligently select which systems to cross"""
        info = self.timer.get_stage_info()
        
        # If we have preferred pairs, use the best one
        if info['preferred_pairs']:
            best_pair = info['preferred_pairs'][0]['pair']
            return best_pair.split('⟲')
        
        # Otherwise use default progression
        base_systems = ['gpt_base', 'claude_base', 'grok_base', 'deepseek_base']
        import itertools
        pairs = list(itertools.combinations(base_systems, 2))
        
        # Cycle through pairs based on attempt count
        attempt = self.timer.state['total_attempts']
        return pairs[attempt % len(pairs)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMAI Adaptive Evolution")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--status", action="store_true", help="Show evolution status")
    
    args = parser.parse_args()
    
    evolution = DMAIEvolution()
    
    if args.status:
        info = evolution.timer.get_stage_info()
        print(f"\n🧬 DMAI EVOLUTION STATUS")
        print(f"Stage: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Evolutions: {info['evolutions']}")
        print(f"Success Rate: {info['success_rate']}")
        print(f"Current Interval: {info['interval_minutes']:.0f} minutes")
    elif args.once:
        evolution.run_cycle()
    else:
        try:
            evolution.continuous_evolution()
        except KeyboardInterrupt:
            print("\n\n👋 Evolution paused. Run again to continue where she left off.")
            print(f"📊 Final status: {evolution.timer.get_stage_info()['name']}")
