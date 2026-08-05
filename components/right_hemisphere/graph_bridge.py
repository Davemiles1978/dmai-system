"""
Right Hemisphere → Knowledge Graph Bridge
==========================================
Ensures right-hemisphere discoveries become visible neurons in the brain graph.
Without this bridge, vector store activity is invisible in the UI.

Every semantic event (cluster found, analogy discovered, novelty detected,
cross-domain bridge, fusion event) creates a neuron + synapse in graph_schema.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("dmai.right_hemisphere.graph_bridge")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GRAPH_SCHEMA = _REPO_ROOT / "data" / "graph_schema.json"


def _ensure_graph_file() -> Dict:
    """Load or create graph schema."""
    if _GRAPH_SCHEMA.exists():
        try:
            return json.loads(_GRAPH_SCHEMA.read_text())
        except Exception:
            pass
    return {
        "neurons": [],
        "synapses": [],
        "evolution_cycle": 0,
        "last_updated": None,
    }


def _save_graph(graph: Dict) -> None:
    _GRAPH_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    graph["last_updated"] = datetime.now(timezone.utc).isoformat()
    _GRAPH_SCHEMA.write_text(json.dumps(graph, indent=2))


def add_right_hemisphere_neuron(
    domain: str,
    concept: str,
    source: str = "right_hemisphere",
    neuron_type: str = "semantic",
    metadata: Optional[Dict] = None,
) -> bool:
    """
    Add a neuron to the knowledge graph from a right-hemisphere event.
    Deduplicates by domain+concept.
    """
    try:
        graph = _ensure_graph_file()
        neuron_id = f"rh:{domain}:{concept}"[:120]

        # Dedup
        for n in graph.get("neurons", []):
            if n.get("id") == neuron_id:
                # Update metadata, bump weight
                n["weight"] = n.get("weight", 1) + 0.5
                n["metadata"] = {**(n.get("metadata", {})), **(metadata or {})}
                _save_graph(graph)
                return True

        neuron = {
            "id": neuron_id,
            "domain": f"right_hemisphere:{domain}",
            "label": concept[:100],
            "type": neuron_type,
            "weight": 1.0,
            "source": source,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        graph.setdefault("neurons", []).append(neuron)
        graph["evolution_cycle"] = graph.get("evolution_cycle", 0) + 1
        _save_graph(graph)
        logger.debug("RH Graph: +neuron %s (%s)", concept[:40], domain)
        return True
    except Exception as e:
        logger.debug("RH Graph bridge error: %s", e)
        return False


def add_right_hemisphere_synapse(
    source_id: str,
    target_id: str,
    edge_type: str = "semantic_link",
    weight: float = 0.5,
) -> bool:
    """Add a synapse between two neurons from right-hemisphere processing."""
    try:
        graph = _ensure_graph_file()
        synapse_id = f"rh_syn:{source_id}->{target_id}:{edge_type}"[:150]

        for s in graph.get("synapses", []):
            if s.get("id") == synapse_id:
                s["weight"] = min(1.0, s.get("weight", 0.5) + 0.1)
                _save_graph(graph)
                return True

        synapse = {
            "id": synapse_id,
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "weight": weight,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        graph.setdefault("synapses", []).append(synapse)
        _save_graph(graph)
        return True
    except Exception as e:
        logger.debug("RH Synapse bridge error: %s", e)
        return False
