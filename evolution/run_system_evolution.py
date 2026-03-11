#!/usr/bin/env python3
"""Runner for system-wide evolution"""
import time
from system_wide_evolution import SystemWideEvolution
from adaptive_timer import AdaptiveEvolutionTimer

def main():
    timer = AdaptiveEvolutionTimer()
    evolution = SystemWideEvolution()
    
    print("🚀 Starting System-Wide Evolution")
    print("All systems will evolve together")
    
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"CYCLE #{cycle}")
        print(f"{'='*60}")
        
        # Run evolution cycle
        successes = evolution.run_evolution_cycle()
        
        # Get wait time from adaptive timer
        wait_time = timer.get_wait_time()
        print(f"\n⏱️  Next cycle in {wait_time/60:.1f} minutes")
        
        # Record cycle results
        timer.record_attempt(
            "system_wide",
            "evolution",
            success=successes > 0
        )
        
        # Wait for next cycle
        time.sleep(wait_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Evolution paused")
