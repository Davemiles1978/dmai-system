"""
Microfish Prediction Engine (vendored + reverse-engineered).

Original: https://github.com/666ghj/MiroFish (Zep + OASIS/CAMEL-AI + Neo4j stack).
This port preserves the core prediction pipeline:
    seed_data --> entity_extraction --> knowledge_graph (SQLite) -->
    agent_persona_generation --> sequential_swarm_simulation --> verdict

Differences from upstream:
- Zep Cloud removed -> SQLite-backed graph store (mf_entities, mf_relations).
- OASIS/CAMEL-AI removed -> lightweight sequential agent loop using DMAI's LLM waterfall.
- Neo4j removed -> SQLite only (project-wide constraint).
- LLM client wired to DMAI's 13-provider _direct_provider_chat waterfall.

Public API:
    from components.microfish import PredictionEngine
    engine = PredictionEngine()
    verdict = engine.predict(requirement="Will X happen?", seed_data="...", max_rounds=3, agent_count=5)
    # -> {"id": ..., "verdict": "likely/unlikely/uncertain", "confidence": 0..1, "signals": [...], "agents": [...], "timeline": [...]}
"""
from .prediction_engine import PredictionEngine
from .graph_store import GraphStore
from .llm_client import MicrofishLLM

__all__ = ["PredictionEngine", "GraphStore", "MicrofishLLM"]
