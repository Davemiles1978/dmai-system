# diagnose_evolution.py
#!/usr/bin/env python3
"""
Diagnose why evolution works in isolation but not in main system
"""

import os
import sys
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'components'))

print("=" * 60)
print("DIAGNOSING EVOLUTION DATA FLOW")
print("=" * 60)

# Import correctly
try:
    from phase6.P6_AdvancedIntelligence import SyntheticNeuralNetwork, SyntheticNeuron
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    # Try direct import
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

# Create network and seed it
print("\n1. Creating and seeding synthetic network...")
network = SyntheticNeuralNetwork("Test")

# Manually add neurons (same way main system seeds)
initial_neurons = ["core", "learning", "memory", "persona"]
for name in initial_neurons:
    neuron_id = f"neuron_{name}"
    neuron = SyntheticNeuron(neuron_id=neuron_id)
    network.neurons[neuron_id] = neuron

print(f"   Initial neurons: {len(network.neurons)}")
print(f"   Initial synapses: {network._total_synapses()}")
print(f"   Initial consciousness: {network.consciousness_level:.4f}")

# Test what happens when we call evolve() directly
print("\n2. Testing evolve() directly (no process call)...")
result = network.evolve()
print(f"   Evolve result: {result}")
print(f"   After evolve - neurons: {len(network.neurons)}")
print(f"   After evolve - synapses: {network._total_synapses()}")
print(f"   After evolve - consciousness: {network.consciousness_level:.4f}")

# Test with process call (what main system does)
print("\n3. Testing process() + evolve() (main system pattern)...")
network2 = SyntheticNeuralNetwork("Test2")
for name in initial_neurons:
    neuron_id = f"neuron_{name}"
    neuron = SyntheticNeuron(neuron_id=neuron_id)
    network2.neurons[neuron_id] = neuron

input_data = {
    'evolution_cycle': 1,
    'conversations': 10,
    'concepts': 5,
    'kaizen_improvements': 2
}

print(f"   Input data: {input_data}")
network2.process(input_data)
result2 = network2.evolve()
print(f"   Result: {result2}")
print(f"   Final neurons: {len(network2.neurons)}")
print(f"   Final synapses: {network2._total_synapses()}")
print(f"   Final consciousness: {network2.consciousness_level:.4f}")

# Test with the exact data pattern from main system's evolution_cycle
print("\n4. Testing with main system's exact data pattern...")
network3 = SyntheticNeuralNetwork("Test3")
for name in initial_neurons:
    neuron_id = f"neuron_{name}"
    neuron = SyntheticNeuron(neuron_id=neuron_id)
    network3.neurons[neuron_id] = neuron

# This is what main system passes
main_input = {
    'evolution_cycle': 1,
    'conversations': len([]),
    'concepts': 0,
    'kaizen_improvements': 0,
    'cves': 0,
    'iocs': 0
}
print(f"   Main system input: {main_input}")
network3.process(main_input)
result3 = network3.evolve()
print(f"   Result: {result3}")
print(f"   Final neurons: {len(network3.neurons)}")
print(f"   Final synapses: {network3._total_synapses()}")
print(f"   Final consciousness: {network3.consciousness_level:.4f}")

# Test if knowledge from knowledge_graph is being passed
print("\n5. Checking if knowledge_graph data is being accessed...")
print("   The main system's _gather_knowledge_from_sources() should be:")
print("   - Getting insights from AI tutors")
print("   - Adding concepts to knowledge_graph")
print("   - Then passing concepts count to synthetic_network.process()")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("The synthetic network itself IS capable of growth when fed data.")
print("The issue is likely:")
print("1. _gather_knowledge_from_sources() is not getting real insights")
print("2. Or the concepts count is not increasing (knowledge_graph.add_concept failing)")
print("3. Or the synthetic_network.process() is not receiving updated concept counts")
