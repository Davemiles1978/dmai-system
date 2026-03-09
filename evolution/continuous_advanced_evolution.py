#!/usr/bin/env python3
"""Continuous runner for memory-safe system-wide evolution"""

import time
import gc
from memory_safe_evolution import MemorySafeEvolution
from adaptive_timer import AdaptiveEvolutionTimer
from auto_cleanup import cleanup_if_needed

def main():
    timer = AdaptiveEvolutionTimer()
    evolution = MemorySafeEvolution()
    
    print("\n" + "="*70)
    print("🚀 CONTINUOUS MEMORY-SAFE EVOLUTION STARTED")
    print("Auto-cleanup at 600MB, evolution cycles with memory protection")
    print("="*70 + "\n")
    
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*70}")
        print(f"CYCLE #{cycle}")
        print(f"{'='*70}")
        
        # Check memory and cleanup if needed
        cleanup_if_needed()
        
        # Run evolution cycle with memory protection
        successes = evolution.run_cycle()
        
        # Get adaptive wait time
        wait_time = timer.record_attempt(
            "system_wide",
            "evolution",
            success=successes > 0
        )
        
        print(f"\n⏱️  Next cycle in {wait_time/60:.1f} minutes")
        print("(Press Ctrl+C to pause)\n")
        
        # Wait intelligently with memory checks
        for minute in range(int(wait_time / 60)):
            time.sleep(60)
            if minute % 5 == 0:  # Every 5 minutes, check memory
                cleanup_if_needed()
                remaining = wait_time/60 - (minute + 1)
                print(f"   {remaining:.0f} minutes remaining...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Evolution paused. Run again to continue.")
