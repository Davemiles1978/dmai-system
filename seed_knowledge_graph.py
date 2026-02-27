#!/usr/bin/env python3
"""
Seed the knowledge graph with initial concepts
"""
import asyncio
from knowledge_graph import KnowledgeGraph

async def seed():
    kg = KnowledgeGraph()
    
    # Add core concepts
    kg.add_concept("evolution", "core_process", {"description": "System evolution"})
    kg.add_concept("mutation", "core_operation", {"description": "Code mutation"})
    kg.add_concept("selection", "core_operation", {"description": "Fitness selection"})
    kg.add_concept("capability", "core_concept", {"description": "System capability"})
    
    # Add relationships
    kg.add_relationship("evolution", "mutation", "uses")
    kg.add_relationship("evolution", "selection", "uses")
    kg.add_relationship("mutation", "capability", "creates")
    
    # Save
    kg.save()
    print("✅ Knowledge graph seeded with initial concepts")

if __name__ == "__main__":
    asyncio.run(seed())
