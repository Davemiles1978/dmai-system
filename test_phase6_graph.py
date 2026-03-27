#!/usr/bin/env python3
import sys
import os
from pathlib import Path

sys.path.insert(0, '/Users/davidmiles/Desktop/dmai-system')
from components.phase6.P6_AdvancedIntelligence import KnowledgeGraph as RealKnowledgeGraph

print("=" * 60)
print("Testing RealKnowledgeGraph from Phase 6")
print("=" * 60)

# Create a Phase 6 graph
print("\n1. Creating RealKnowledgeGraph instance...")
kg = RealKnowledgeGraph()

# Check if it has local_graph
print(f"\n2. Checking attributes:")
print(f"   Has local_graph: {hasattr(kg, 'local_graph')}")
if hasattr(kg, 'local_graph'):
    print(f"   local_graph type: {type(kg.local_graph)}")
    print(f"   local_graph content: {kg.local_graph}")

# Try to add knowledge
print(f"\n3. Testing add_knowledge...")
try:
    kg.add_knowledge("Artificial Intelligence", "is_a", "field of study")
    print("   ✅ add_knowledge succeeded")
except Exception as e:
    print(f"   ❌ add_knowledge failed: {e}")
    import traceback
    traceback.print_exc()

# Check if local_graph was updated
print(f"\n4. After add_knowledge:")
if hasattr(kg, 'local_graph'):
    print(f"   local_graph nodes: {len(kg.local_graph.get('nodes', []))}")
    print(f"   nodes: {kg.local_graph.get('nodes', [])[:3]}")

print("\n" + "=" * 60)
print("Test Complete")
