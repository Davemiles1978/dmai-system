# trace_main_evolution.py
#!/usr/bin/env python3
"""
Trace what's happening in the main system's evolution cycle
"""

import os
import sys
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("TRACING MAIN SYSTEM EVOLUTION")
print("=" * 60)

# Import the main system
from dmai_core_complete import UnifiedEvolutionEngine
from pathlib import Path

# Create a test instance with a clean data directory
test_data_path = Path("./test_evolution_data")
test_data_path.mkdir(exist_ok=True)

# Create a wrapper to trace the evolution_cycle
class TracingEvolutionEngine(UnifiedEvolutionEngine):
    def __init__(self, base_path):
        super().__init__(base_path)
        self.trace_data = []
    
    def evolution_cycle(self):
        """Wrapped evolution_cycle to trace what's happening"""
        
        # Call the original but capture before/after
        pre_successful = self.successful_evolutions
        pre_consciousness = self.synthetic_network.consciousness_level
        pre_neurons = len(self.synthetic_network.neurons)
        
        print(f"\n📊 BEFORE CYCLE:")
        print(f"   successful_evolutions: {pre_successful}")
        print(f"   consciousness: {pre_consciousness:.4f}")
        print(f"   neurons: {pre_neurons}")
        
        # Call original evolution_cycle
        result = super().evolution_cycle()
        
        post_successful = self.successful_evolutions
        post_consciousness = self.synthetic_network.consciousness_level
        post_neurons = len(self.synthetic_network.neurons)
        
        print(f"\n📊 AFTER CYCLE:")
        print(f"   successful_evolutions: {post_successful} (change: +{post_successful - pre_successful})")
        print(f"   consciousness: {post_consciousness:.4f} (change: +{post_consciousness - pre_consciousness:.4f})")
        print(f"   neurons: {post_neurons} (change: +{post_neurons - pre_neurons})")
        print(f"   result['was_successful']: {result.get('was_successful', 'N/A')}")
        print(f"   result['success_reasons']: {result.get('success_reasons', [])}")
        
        # Check if the if condition in evolution_cycle is being hit
        # In the main system, there's a condition:
        # if was_successful:
        #     self.successful_evolutions += 1
        #     logger.info(...)
        
        if post_successful > pre_successful:
            print(f"\n✅ SUCCESSFUL EVOLUTIONS INCREMENTED!")
        else:
            print(f"\n❌ SUCCESSFUL EVOLUTIONS NOT INCREMENTED!")
            print(f"   This means the 'if was_successful:' condition is NOT being entered.")
            print(f"   Check if was_successful is being set correctly.")
        
        return result


# Create and run
print("\nInitializing evolution engine...")
engine = TracingEvolutionEngine(test_data_path)

print("\n" + "=" * 60)
print("RUNNING 3 EVOLUTION CYCLES")
print("=" * 60)

for i in range(3):
    print(f"\n--- CYCLE {i+1} ---")
    result = engine.evolution_cycle()
    time.sleep(0.5)

print("\n" + "=" * 60)
print("FINAL STATE")
print("=" * 60)
print(f"Total successful_evolutions: {engine.successful_evolutions}")
print(f"Total evolution_count: {engine.evolution_count}")
