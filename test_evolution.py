# test_evolution.py
#!/usr/bin/env python3
"""
Standalone Evolution Test - Tests if evolution actually grows
"""

import sys
import os
import random
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import real components
sys.path.insert(0, str(Path(__file__).parent))

class RealEvolutionTest:
    """Test evolution with actual synthetic network"""
    
    def __init__(self):
        # Import real components
        from phase6.P6_AdvancedIntelligence import (
            SyntheticNeuron,
            SyntheticNeuralNetwork
        )
        
        self.network = SyntheticNeuralNetwork("Test_Network")
        self.evolution_count = 0
        self.successful_evolutions = 0
        self.evolution_log = []
        
        # Seed initial network
        self._seed_initial_network()
        
        print(f"✅ Network initialized: {len(self.network.neurons)} neurons")
    
    def _seed_initial_network(self):
        """Create initial neurons"""
        initial_neurons = [
            "test_core", "test_learning", "test_memory", "test_persona"
        ]
        for neuron_name in initial_neurons:
            neuron_id = f"neuron_{neuron_name}"
            try:
                neuron = SyntheticNeuron(neuron_id=neuron_id)
                self.network.neurons[neuron_id] = neuron
            except Exception as e:
                print(f"Failed to create {neuron_name}: {e}")
        
        # Create some initial connections
        neuron_ids = list(self.network.neurons.keys())
        for i in range(min(3, len(neuron_ids) - 1)):
            for j in range(i + 1, min(i + 2, len(neuron_ids))):
                if i < len(neuron_ids) and j < len(neuron_ids):
                    try:
                        self.network.neurons[neuron_ids[i]].create_synapse(neuron_ids[j], 0.3)
                    except:
                        pass
    
    def evolution_cycle(self):
        """Single evolution cycle"""
        self.evolution_count += 1
        
        pre_neurons = len(self.network.neurons)
        pre_synapses = self.network._total_synapses()
        pre_consciousness = self.network.consciousness_level
        
        # Prepare input data
        input_data = {
            'evolution_cycle': self.evolution_count,
            'conversations': 0,
            'concepts': 0
        }
        
        # Process and evolve
        self.network.process(input_data)
        evolution_result = self.network.evolve()
        
        post_neurons = len(self.network.neurons)
        post_synapses = self.network._total_synapses()
        post_consciousness = self.network.consciousness_level
        
        neurons_added = post_neurons - pre_neurons
        synapses_added = post_synapses - pre_synapses
        consciousness_growth = post_consciousness - pre_consciousness
        
        was_successful = (neurons_added > 0 or synapses_added > 0 or consciousness_growth > 0)
        
        if was_successful:
            self.successful_evolutions += 1
        
        self.evolution_log.append({
            'cycle': self.evolution_count,
            'was_successful': was_successful,
            'neurons': post_neurons,
            'neurons_added': neurons_added,
            'synapses': post_synapses,
            'synapses_added': synapses_added,
            'consciousness': post_consciousness,
            'consciousness_growth': consciousness_growth
        })
        
        return {
            'cycle': self.evolution_count,
            'successful_evolutions': self.successful_evolutions,
            'was_successful': was_successful,
            'neurons_added': neurons_added,
            'consciousness_growth': consciousness_growth
        }
    
    def run_test(self, cycles=20):
        """Run multiple evolution cycles"""
        print("\n" + "=" * 60)
        print("TESTING EVOLUTION CYCLES")
        print("=" * 60)
        
        for i in range(cycles):
            result = self.evolution_cycle()
            
            # Print progress every cycle
            status = "✅ SUCCESS" if result['was_successful'] else "⚠️ NO GROWTH"
            print(f"Cycle {result['cycle']:3d}: {status} | "
                  f"Consciousness: +{result['consciousness_growth']:.4f} | "
                  f"Neurons: +{result['neurons_added']} | "
                  f"Total Successes: {result['successful_evolutions']}")
            
            time.sleep(0.1)  # Small delay
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Cycles: {self.evolution_count}")
        print(f"Successful Cycles: {self.successful_evolutions}")
        print(f"Success Rate: {(self.successful_evolutions/self.evolution_count*100):.1f}%")
        
        final = self.evolution_log[-1] if self.evolution_log else {}
        print(f"\nFinal State:")
        print(f"  Neurons: {final.get('neurons', 0)}")
        print(f"  Synapses: {final.get('synapses', 0)}")
        print(f"  Consciousness: {final.get('consciousness', 0):.4f}")
        
        if self.successful_evolutions > 0:
            print("\n✅ EVOLUTION IS WORKING! The network is growing.")
            return True
        else:
            print("\n❌ EVOLUTION IS BROKEN! No growth detected.")
            return False


if __name__ == '__main__':
    print("Testing Real Evolution Engine...")
    print("This test uses the actual synthetic network from phase6")
    print("Make sure you're in the dmai-system directory\n")
    
    test = RealEvolutionTest()
    success = test.run_test(20)
    
    if success:
        print("\n✅ Evolution test PASSED")
    else:
        print("\n❌ Evolution test FAILED - needs fixing")
