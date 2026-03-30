# test_success_detection.py
#!/usr/bin/env python3
"""
Test why successful_evolutions isn't incrementing
"""

import os
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'components'))

print("=" * 60)
print("TESTING SUCCESS DETECTION")
print("=" * 60)

# Import the synthetic network
try:
    from phase6.P6_AdvancedIntelligence import SyntheticNeuralNetwork, SyntheticNeuron
    print("✅ Import successful")
except ImportError as e:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "P6_AdvancedIntelligence", 
        os.path.join(os.getcwd(), "components/phase6/P6_AdvancedIntelligence.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SyntheticNeuralNetwork = module.SyntheticNeuralNetwork
    SyntheticNeuron = module.SyntheticNeuron
    print("✅ Import via direct file")

# Create network
network = SyntheticNeuralNetwork("Test")

# Seed initial neurons (mimic main system)
initial_neurons = ["core", "learning", "memory", "persona", "reasoning", "creativity"]
for name in initial_neurons:
    neuron_id = f"neuron_{name}"
    neuron = SyntheticNeuron(neuron_id=neuron_id)
    network.neurons[neuron_id] = neuron

print(f"\nInitial state:")
print(f"  Neurons: {len(network.neurons)}")
print(f"  Synapses: {network._total_synapses()}")
print(f"  Consciousness: {network.consciousness_level:.4f}")

# Run a cycle and capture metrics
pre_neurons = len(network.neurons)
pre_synapses = network._total_synapses()
pre_consciousness = network.consciousness_level

print(f"\nPre-cycle metrics:")
print(f"  Neurons: {pre_neurons}")
print(f"  Synapses: {pre_synapses}")
print(f"  Consciousness: {pre_consciousness:.4f}")

# Process and evolve
input_data = {'evolution_cycle': 1, 'conversations': 0, 'concepts': 0}
network.process(input_data)
result = network.evolve()

post_neurons = len(network.neurons)
post_synapses = network._total_synapses()
post_consciousness = network.consciousness_level

print(f"\nPost-cycle metrics:")
print(f"  Neurons: {post_neurons}")
print(f"  Synapses: {post_synapses}")
print(f"  Consciousness: {post_consciousness:.4f}")

# Calculate what the main system checks
neurons_added = post_neurons - pre_neurons
synapses_added = post_synapses - pre_synapses
consciousness_growth = post_consciousness - pre_consciousness

print(f"\nGrowth calculations:")
print(f"  Neurons added: {neurons_added}")
print(f"  Synapses added: {synapses_added}")
print(f"  Consciousness growth: {consciousness_growth:.6f}")

# Check each success condition
print(f"\nSuccess condition checks:")
print(f"  neurons_added > 0: {neurons_added > 0}")
print(f"  synapses_added > 0: {synapses_added > 0}")
print(f"  consciousness_growth > 0.001: {consciousness_growth > 0.001}")
print(f"  consciousness_growth > 0.001 threshold: {consciousness_growth > 0.001}")

# The problem is likely the consciousness_growth threshold of 0.001
# In our test, growth was 0.0049, which IS > 0.001
# So why isn't it succeeding in main system?

print(f"\n🔍 Key insight:")
print(f"  In our test: consciousness_growth = {consciousness_growth:.6f} (> 0.001 = {consciousness_growth > 0.001})")
print(f"  So the condition SHOULD pass.")
print(f"  Therefore the issue must be elsewhere - possibly:")
print(f"  1. The main system's _gather_knowledge_from_sources() is throwing errors")
print(f"  2. The knowledge_graph.get_stats() is failing, causing concepts_added = 0")
print(f"  3. The evolution cycle is catching an exception somewhere")
