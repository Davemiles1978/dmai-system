#!/usr/bin/env python3
"""
Test script for KnowledgeGraph class
Tests all attributes that the API Harvester expects
"""

import sys
import os
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, '/Users/davidmiles/Desktop/dmai-system')

# Import the KnowledgeGraph class from our file
from dmai_core_complete import KnowledgeGraph

print("=" * 60)
print("Testing KnowledgeGraph Class")
print("=" * 60)

# Create a test data path
test_path = Path('/tmp/dmai_test_data')
test_path.mkdir(exist_ok=True)

# Initialize KnowledgeGraph
print("\n1. Creating KnowledgeGraph instance...")
kg = KnowledgeGraph(test_path)
print("   ✅ Instance created")

# Test 1: Check if local_graph attribute exists
print("\n2. Testing local_graph attribute...")
if hasattr(kg, 'local_graph'):
    print(f"   ✅ local_graph exists: {type(kg.local_graph)}")
    print(f"   local_graph content: {kg.local_graph}")
else:
    print("   ❌ local_graph does NOT exist")

# Test 2: Check if we can assign to local_graph
print("\n3. Testing local_graph assignment...")
try:
    kg.local_graph = {'nodes': ['test1', 'test2'], 'edges': []}
    print(f"   ✅ Can assign to local_graph")
    print(f"   New local_graph: {kg.local_graph}")
except Exception as e:
    print(f"   ❌ Cannot assign to local_graph: {e}")

# Test 3: Check dictionary-style access
print("\n4. Testing dictionary-style access...")
try:
    kg['local_graph'] = {'nodes': ['dict_test'], 'edges': []}
    print("   ✅ Can set via __setitem__")
    result = kg.get('local_graph')
    print(f"   ✅ Can get via .get(): {result}")
    has_local = 'local_graph' in kg
    print(f"   ✅ 'local_graph' in kg: {has_local}")
except Exception as e:
    print(f"   ❌ Dictionary access failed: {e}")

# Test 4: Test add_concept method
print("\n5. Testing add_concept method...")
try:
    kg.add_concept("Artificial Intelligence", "AI is the simulation of human intelligence")
    print("   ✅ add_concept succeeded")
    print(f"   Nodes now: {kg._nodes}")
except Exception as e:
    print(f"   ❌ add_concept failed: {e}")

# Test 5: Test get_stats method
print("\n6. Testing get_stats method...")
try:
    stats = kg.get_stats()
    print(f"   ✅ get_stats succeeded: {stats}")
except Exception as e:
    print(f"   ❌ get_stats failed: {e}")

# Test 6: Test that API Harvester can access attributes
print("\n7. Simulating API Harvester access patterns...")
try:
    # This is what the API Harvester does
    if hasattr(kg, 'local_graph'):
        lg = kg.local_graph
        print(f"   ✅ Can access kg.local_graph")
        if isinstance(lg, dict):
            print(f"   ✅ kg.local_graph is a dict")
            if 'nodes' in lg:
                print(f"   ✅ kg.local_graph has 'nodes' key")
            if 'edges' in lg:
                print(f"   ✅ kg.local_graph has 'edges' key")
    
    # Test direct attribute access
    if hasattr(kg, '_nodes'):
        print(f"   ✅ kg._nodes exists with {len(kg._nodes)} items")
    
    if hasattr(kg, '_edges'):
        print(f"   ✅ kg._edges exists with {len(kg._edges)} items")
        
except Exception as e:
    print(f"   ❌ Harvester simulation failed: {e}")

# Test 7: Test saving and loading
print("\n8. Testing save_graph and load_graph...")
try:
    kg.save_graph()
    print("   ✅ save_graph succeeded")
    
    # Create a new instance to test loading
    kg2 = KnowledgeGraph(test_path)
    kg2.load_graph()
    print("   ✅ load_graph succeeded")
    print(f"   Loaded nodes: {kg2._nodes}")
    
    # Verify data persisted
    if kg2._nodes == kg._nodes:
        print("   ✅ Data persisted correctly")
    else:
        print(f"   ⚠️ Data mismatch: original={kg._nodes}, loaded={kg2._nodes}")
        
except Exception as e:
    print(f"   ❌ Save/load failed: {e}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)

# Clean up
import shutil
shutil.rmtree(test_path, ignore_errors=True)
