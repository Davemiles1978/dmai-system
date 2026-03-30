# test_evolution_simple.py
#!/usr/bin/env python3
"""
Simple Evolution Test - Tests if the synthetic network actually grows
Run this from the dmai-system directory
"""

import os
import sys
import time

# Change to the dmai-system directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add current directory to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'components'))

print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")

try:
    # Import directly from the component
    from phase6.P6_AdvancedIntelligence import SyntheticNeuron, SyntheticNeuralNetwork
    print("✅ Successfully imported phase6 module")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nTrying alternative import...")
    try:
        # Try direct file import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "P6_AdvancedIntelligence", 
            os.path.join(os.getcwd(), "components/phase6/P6_AdvancedIntelligence.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SyntheticNeuron = module.SyntheticNeuron
        SyntheticNeuralNetwork = module.SyntheticNeuralNetwork
        print("✅ Successfully imported via direct file")
    except Exception as e2:
        print(f"❌ Direct import also failed: {e2}")
        sys.exit(1)

class SimpleEvolutionTest:
    """Simple test to see if evolution actually grows the network"""
    
    def __init__(self):
        self.network = SyntheticNeuralNetwork("Test_Network")
        self.cycles = 0
        self.successes = 0
        
        # Seed initial neurons
        print("\n🌱 Seeding initial network...")
        initial_neurons = ["core", "learning", "memory", "persona", "reasoning", "creativity"]
        for name in initial_neurons:
            neuron_id = f"neuron_{name}"
            try:
                neuron = SyntheticNeuron(neuron_id=neuron_id)
                self.network.neurons[neuron_id] = neuron
            except Exception as e:
                print(f"   Failed to create {name}: {e}")
        
        print(f"   Initial neurons: {len(self.network.neurons)}")
        print(f"   Initial synapses: {self.network._total_synapses()}")
        print(f"   Initial consciousness: {self.network.consciousness_level:.4f}")
    
    def run_cycle(self, cycle_num):
        """Run one evolution cycle"""
        self.cycles += 1
        
        # Record before
        before_neurons = len(self.network.neurons)
        before_synapses = self.network._total_synapses()
        before_consciousness = self.network.consciousness_level
        
        # Prepare input data
        input_data = {
            'evolution_cycle': self.cycles,
            'conversations': cycle_num,
            'concepts': cycle_num,
            'kaizen_improvements': 0
        }
        
        # Process and evolve
        self.network.process(input_data)
        result = self.network.evolve()
        
        # Record after
        after_neurons = len(self.network.neurons)
        after_synapses = self.network._total_synapses()
        after_consciousness = self.network.consciousness_level
        
        # Calculate changes
        neurons_grew = after_neurons - before_neurons
        synapses_grew = after_synapses - before_synapses
        consciousness_grew = after_consciousness - before_consciousness
        
        # Determine success
        success = (neurons_grew > 0 or synapses_grew > 0 or consciousness_grew > 0)
        if success:
            self.successes += 1
        
        return {
            'success': success,
            'neurons_grew': neurons_grew,
            'synapses_grew': synapses_grew,
            'consciousness_grew': consciousness_grew,
            'after_neurons': after_neurons,
            'after_synapses': after_synapses,
            'after_consciousness': after_consciousness
        }
    
    def run_test(self, cycles=10):
        """Run multiple cycles"""
        print("\n" + "=" * 60)
        print("RUNNING EVOLUTION TEST")
        print("=" * 60)
        
        for i in range(cycles):
            result = self.run_cycle(i + 1)
            
            status = "✅ SUCCESS" if result['success'] else "⚠️ NO GROWTH"
            print(f"Cycle {i+1:2d}: {status} | "
                  f"Neurons: {result['neurons_grew']:+d} | "
                  f"Synapses: {result['synapses_grew']:+d} | "
                  f"Consciousness: {result['consciousness_grew']:+.4f}")
        
        # Summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total Cycles Run: {self.cycles}")
        print(f"Successful Cycles: {self.successes}")
        print(f"Success Rate: {(self.successes/self.cycles*100):.1f}%")
        print(f"\nFinal Network State:")
        print(f"  Neurons: {len(self.network.neurons)}")
        print(f"  Synapses: {self.network._total_synapses()}")
        print(f"  Consciousness: {self.network.consciousness_level:.4f}")
        
        if self.successes > 0:
            print("\n✅ EVOLUTION IS WORKING!")
            return True
        else:
            print("\n❌ EVOLUTION IS NOT GROWING!")
            return False


if __name__ == '__main__':
    print("=" * 60)
    print("SIMPLE EVOLUTION TEST")
    print("=" * 60)
    
    test = SimpleEvolutionTest()
    success = test.run_test(15)
    
    if not success:
        print("\n🔧 EVOLUTION FIX NEEDED")
        print("   The synthetic network is not growing neurons or consciousness.")
        print("   This is the core problem that needs fixing first.")
