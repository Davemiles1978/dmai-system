#!/usr/bin/env python3
"""Evolution Timer with Advanced System-Wide Evolution"""

import time
import sys
from pathlib import Path
from advanced_evolution import AdvancedEvolution
from adaptive_timer import AdaptiveEvolutionTimer

class DMAIEvolution:
    """
    DMAI's evolution controller with self-adjusting timing.
    Now uses system-wide evolution for ALL systems!
    """
    
    def __init__(self):
        self.timer = AdaptiveEvolutionTimer()
        self.evolution = AdvancedEvolution()
        
    def run_cycle(self):
        """Run one evolution cycle using the advanced system"""
        print(f"\n🧬 System-Wide Evolution Cycle Starting...")
        
        # Run the advanced evolution
        successes = self.evolution.run_cycle()
        
        # Record with timer
        wait_time = self.timer.record_attempt(
            "system_wide",
            "evolution",
            success=successes > 0
        )
        
        # Show status
        self._show_status(successes > 0, successes)
        
        return wait_time
    
    def _show_status(self, success, num_successes):
        """Display current evolution status"""
        info = self.timer.get_stage_info()
        
        status = "✅ SUCCESS" if success else "❌ No improvement"
        print(f"\n📊 Cycle Result: {status}")
        print(f"   Successful evolutions: {num_successes}")
        print(f"📈 Success Rate: {info['success_rate']}")
        print(f"⏱️  Next cycle in: {info['interval_minutes']:.0f} minutes")
        print(f"🧠 Current Stage: {info['name']}")
    
    def continuous_evolution(self):
        """Run evolution continuously with adaptive timing"""
        print("\n" + "="*70)
        print("🚀 DMAI SYSTEM-WIDE CONTINUOUS EVOLUTION STARTED")
        print("Every cycle evolves ALL systems in the ecosystem")
        print("="*70 + "\n")
        
        cycle_count = 0
        
        while True:
            cycle_count += 1
            print(f"\n{'='*70}")
            print(f"Cycle #{cycle_count}")
            print(f"{'='*70}")
            
            # Run evolution cycle
            wait_time = self.run_cycle()
            
            # Smart waiting - check every minute
            print(f"\n⏳ Waiting {wait_time/60:.1f} minutes...")
            print("(Press Ctrl+C to pause)\n")
            
            for minute in range(int(wait_time / 60)):
                time.sleep(60)
                if minute % 5 == 0:  # Every 5 minutes
                    remaining = wait_time/60 - (minute + 1)
                    print(f"   {remaining:.0f} minutes remaining...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMAI System-Wide Evolution")
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
            print("\n\n👋 Evolution paused. Run again to continue.")
