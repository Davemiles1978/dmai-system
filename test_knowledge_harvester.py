#!/usr/bin/env python3
"""
Test script to simulate API Harvester's knowledge graph access
This will help us isolate and fix the local_graph issue
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test')

# Import the KnowledgeGraph class
sys.path.insert(0, os.path.dirname(__file__))
from dmai_core_complete import KnowledgeGraph, RealKnowledgeGraph
from components.phase6.P6_AdvancedIntelligence import KnowledgeGraph as RealKnowledgeGraph

class SimulatedAPIHarvester:
    """Simulates the API Harvester's knowledge graph operations"""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.test_concepts = [
            "Artificial Intelligence", "Machine Learning", "Neural Networks",
            "Deep Learning", "Reinforcement Learning", "Computer Vision",
            "Natural Language Processing", "Robotics", "Expert Systems",
            "Knowledge Graph", "Synthetic Intelligence", "Consciousness"
        ]
    
    def batch_process(self):
        """Simulate the batch processing that happens in the real Harvester"""
        logger.info("🌉 Batch processing knowledge packets...")
        
        for concept in self.test_concepts[:5]:
            try:
                # This is what the real Harvester does
                self.knowledge_graph.add_concept(concept, f"Information about {concept}")
                
                # Try direct local_graph access (this is what's failing)
                if hasattr(self.knowledge_graph, 'local_graph'):
                    logger.info(f"✅ local_graph exists: {type(self.knowledge_graph.local_graph)}")
                    logger.info(f"   local_graph nodes: {len(self.knowledge_graph.local_graph.get('nodes', []))}")
                else:
                    logger.error(f"❌ local_graph does NOT exist!")
                    # Try to create it on the fly
                    self.knowledge_graph.local_graph = {'nodes': [], 'edges': []}
                    logger.info(f"   Created local_graph on the fly")
                
                # Try to add to local_graph directly
                if concept not in self.knowledge_graph.local_graph.get('nodes', []):
                    self.knowledge_graph.local_graph.setdefault('nodes', []).append(concept)
                    logger.info(f"✅ Added concept to local_graph: {concept}")
                    
            except Exception as e:
                logger.error(f"Failed to add concept {concept}: {e}")
                logger.error(f"   Error type: {type(e).__name__}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Check final state
        logger.info(f"📊 Final local_graph: {self.knowledge_graph.local_graph}")


def test_knowledge_graph():
    """Main test function"""
    print("\n" + "="*70)
    print("🧪 TESTING KNOWLEDGE GRAPH API HARVESTER COMPATIBILITY")
    print("="*70)
    
    # Create a test data directory
    test_path = Path("/tmp/kg_test_data")
    test_path.mkdir(exist_ok=True)
    
    # Initialize KnowledgeGraph
    print("\n1. Creating KnowledgeGraph instance...")
    kg = KnowledgeGraph(test_path)
    
    # Check attributes
    print("\n2. Checking attributes...")
    attrs = ['local_graph', '_nodes', '_edges', 'nodes', 'edges']
    for attr in attrs:
        if hasattr(kg, attr):
            print(f"   ✅ has {attr}: {type(getattr(kg, attr))}")
        else:
            print(f"   ❌ MISSING {attr}")
            # Add missing attribute
            if attr == 'local_graph':
                kg.local_graph = {'nodes': [], 'edges': []}
            elif attr == '_nodes':
                kg._nodes = []
            elif attr == '_edges':
                kg._edges = []
            elif attr == 'nodes':
                kg.nodes = kg._nodes
            elif attr == 'edges':
                kg.edges = kg._edges
    
    # Simulate API Harvester
    print("\n3. Simulating API Harvester batch processing...")
    harvester = SimulatedAPIHarvester(kg)
    harvester.batch_process()
    
    # Check final stats
    print("\n4. Final stats:")
    stats = kg.get_stats()
    print(f"   Total concepts: {stats.get('total_concepts', 0)}")
    
    # Save and reload test
    print("\n5. Testing save and reload...")
    kg.save_graph()
    kg2 = KnowledgeGraph(test_path)
    kg2.load_graph()
    stats2 = kg2.get_stats()
    print(f"   Reloaded concepts: {stats2.get('total_concepts', 0)}")
    
    # Clean up
    import shutil
    shutil.rmtree(test_path, ignore_errors=True)
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    
    return kg

if __name__ == "__main__":
    test_knowledge_graph()
