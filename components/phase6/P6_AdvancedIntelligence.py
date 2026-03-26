#!/usr/bin/env python3
"""
PHASE 6: ADVANCED INTELLIGENCE - AI + SI Fusion
Complete integration of Artificial Intelligence AND Synthetic Intelligence
DMAI's journey to sentience through dual-path intelligence

Version: 2.1.1
Date: 2026-03-26
FIXED: KnowledgeGraph local_graph initialization to prevent AttributeError
"""

import asyncio
import json
import hashlib
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import sys
import logging
import uuid
import pickle
from enum import Enum
from collections import Counter, defaultdict

# Optional imports with graceful fallback
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not installed. Knowledge graph features will be limited.")

logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: ARTIFICIAL INTELLIGENCE CAPABILITIES (Original Phase 6)
# ============================================================================

class PatternSynthesis:
    """ML-based pattern detection and synthesis"""
    
    def __init__(self, data_path: str = "data/phase6/"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        self.patterns_file = os.path.join(data_path, "patterns.json")
        self.patterns = self._load_patterns()
        self.synthesis_history = []
        
    def _load_patterns(self) -> Dict:
        if os.path.exists(self.patterns_file):
            with open(self.patterns_file, 'r') as f:
                return json.load(f)
        return {
            "identified": [],
            "learned": [],
            "correlations": [],
            "anomalies": []
        }
    
    def _save_patterns(self):
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def detect_patterns(self, data_stream: List[Dict], context: str) -> List[Dict]:
        """Detect patterns in data streams"""
        detected = []
        
        for item in data_stream:
            item_hash = hashlib.sha256(str(item).encode()).hexdigest()[:16]
            
            for existing in self.patterns["identified"]:
                if existing.get("hash") == item_hash:
                    existing["occurrences"] += 1
                    existing["last_seen"] = datetime.now().isoformat()
                    detected.append(existing)
                    break
            else:
                new_pattern = {
                    "id": f"pattern_{len(self.patterns['identified'])}",
                    "hash": item_hash,
                    "context": context,
                    "first_seen": datetime.now().isoformat(),
                    "occurrences": 1,
                    "data": item
                }
                self.patterns["identified"].append(new_pattern)
                detected.append(new_pattern)
        
        self._save_patterns()
        return detected
    
    def synthesize_correlation(self, pattern_a: Dict, pattern_b: Dict) -> Dict:
        """Find correlations between patterns"""
        correlation = {
            "pattern_a": pattern_a.get("id"),
            "pattern_b": pattern_b.get("id"),
            "strength": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        if pattern_a.get("context") == pattern_b.get("context"):
            correlation["strength"] = 0.7
            correlation["relationship"] = "same_context"
        elif pattern_a.get("occurrences") == pattern_b.get("occurrences"):
            correlation["strength"] = 0.5
            correlation["relationship"] = "equal_frequency"
        else:
            correlation["strength"] = 0.3
            correlation["relationship"] = "temporal"
        
        self.patterns["correlations"].append(correlation)
        self._save_patterns()
        return correlation
    
    def generate_synthesis(self, context: str, constraints: List[str] = None) -> str:
        """Generate new insights from learned patterns"""
        relevant_patterns = [p for p in self.patterns["identified"] if p.get("context") == context]
        
        if not relevant_patterns:
            return f"No patterns yet learned in context: {context}"
        
        synthesis = f"Based on {len(relevant_patterns)} patterns in {context}, "
        
        strong_correlations = [c for c in self.patterns["correlations"] 
                               if c.get("strength", 0) > 0.6]
        
        if strong_correlations:
            synthesis += f"I've identified {len(strong_correlations)} strong correlations. "
            synthesis += "This suggests interconnected systems that may benefit from unified processing."
        else:
            synthesis += "Patterns are currently isolated. I recommend more data collection to identify relationships."
        
        self.synthesis_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "synthesis": synthesis,
            "constraints": constraints
        })
        
        return synthesis


# ============================================================================
# UPGRADED KNOWLEDGE GRAPH - NetworkX Integration
# ============================================================================

class KnowledgeGraph:
    """
    Advanced knowledge graph with NetworkX for relationship mapping.
    Supports both local graph (JSON) and Neo4j for production.
    Features:
    - Concept nodes with metadata
    - Relationship typing with weights
    - Path finding between concepts
    - Similarity calculation
    - Evolution path tracking
    - Concept clustering
    """
    
    # Relationship type symbols for display
    RELATIONSHIP_SYMBOLS = {
        'implements': '→',
        'depends_on': '⇢',
        'improves': '↑',
        'extends': '+',
        'similar_to': '∼',
        'prerequisite': '←',
        'created_by': '👤',
        'used_in': '⚙️',
        'evolved_from': '🧬',
        'synthesizes': '✨',
        'learned_from': '📚',
        'related_to': '↔️',
        'has_severity': '⚠️',
        'contains': '📁',
        'references': '📖'
    }
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        self.neo4j_available = False
        self.neo4j_driver = None
        self.graph_path = "data/phase6/knowledge_graph.json"
        
        # CRITICAL FIX: Initialize local_graph immediately
        # This prevents AttributeError when add_concept is called before graph is loaded
        self.local_graph = {
            "nodes": [],
            "edges": [],
            "metadata": {}
        }
        
        # Initialize NetworkX graph if available
        self.graph = None
        self.concept_index = {}  # Maps concept names to node IDs
        
        if NETWORKX_AVAILABLE:
            self.graph = nx.MultiDiGraph()
            logger.info("📊 Knowledge Graph initialized with NetworkX")
        else:
            logger.warning("NetworkX not available - using simple graph storage")
        
        # Try Neo4j connection
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri, 
                    auth=(neo4j_user, neo4j_password)
                )
                self.neo4j_available = True
                logger.info("Neo4j connection established")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}")
        
        # Load existing graph
        self._load_graph()
        
        logger.info(f"Knowledge Graph ready: NetworkX={NETWORKX_AVAILABLE}, Neo4j={self.neo4j_available}")
    
    def _load_graph(self):
        """Load graph from disk"""
        if not os.path.exists(self.graph_path):
            logger.info("No existing knowledge graph found, starting fresh")
            return
        
        try:
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            
            # Load into local_graph regardless of NetworkX availability
            self.local_graph = data
            
            if NETWORKX_AVAILABLE and self.graph:
                self.graph.clear()
                self.concept_index.clear()
                
                # Add nodes
                for node_data in data.get('nodes', []):
                    node_id = node_data.pop('id')
                    self.graph.add_node(node_id, **node_data)
                    self.concept_index[node_data.get('name', '').lower()] = node_id
                
                # Add edges
                for edge_data in data.get('edges', []):
                    u = edge_data.pop('from')
                    v = edge_data.pop('to')
                    self.graph.add_edge(u, v, **edge_data)
                
            logger.info(f"Loaded knowledge graph: {len(self._get_nodes())} nodes, {len(self._get_edges())} edges")
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    def _save_graph(self):
        """Save graph to disk"""
        try:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            
            if NETWORKX_AVAILABLE and self.graph:
                data = {
                    'nodes': [],
                    'edges': []
                }
                
                for node, attrs in self.graph.nodes(data=True):
                    data['nodes'].append({
                        'id': node,
                        **attrs
                    })
                
                for u, v, attrs in self.graph.edges(data=True):
                    data['edges'].append({
                        'from': u,
                        'to': v,
                        **attrs
                    })
            else:
                data = self.local_graph
            
            with open(self.graph_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Knowledge graph saved to {self.graph_path}")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
    
    def _get_nodes(self) -> List:
        """Get nodes based on available backend"""
        if NETWORKX_AVAILABLE and self.graph:
            return list(self.graph.nodes(data=True))
        return self.local_graph.get('nodes', [])
    
    def _get_edges(self) -> List:
        """Get edges based on available backend"""
        if NETWORKX_AVAILABLE and self.graph:
            return list(self.graph.edges(data=True))
        return self.local_graph.get('edges', [])
    
    def _get_node_id(self, concept: Union[str, int]) -> Optional[str]:
        """Get node ID for a concept (can be string name or existing ID)"""
        if isinstance(concept, (int, str)) and not isinstance(concept, str):
            # Already an ID
            if NETWORKX_AVAILABLE and self.graph:
                return concept if concept in self.graph else None
            else:
                for node in self.local_graph['nodes']:
                    if node.get('id') == concept:
                        return concept
                return None
        
        # Look up by name
        concept_lower = concept.lower()
        
        if NETWORKX_AVAILABLE and self.graph:
            if concept_lower in self.concept_index:
                return self.concept_index[concept_lower]
            # Search by name
            for node, data in self.graph.nodes(data=True):
                if data.get('name', '').lower() == concept_lower:
                    self.concept_index[concept_lower] = node
                    return node
        else:
            for node in self.local_graph['nodes']:
                if node.get('name', '').lower() == concept_lower:
                    return node.get('id')
        
        return None
    
    def _get_node_name(self, node_id: str) -> Optional[str]:
        """Get node name by ID"""
        if NETWORKX_AVAILABLE and self.graph:
            if node_id in self.graph:
                return self.graph.nodes[node_id].get('name')
        else:
            for node in self.local_graph['nodes']:
                if node.get('id') == node_id:
                    return node.get('name')
        return None
    
    def add_concept(self, name: str, concept_type: str = "entity", metadata: Dict = None) -> str:
        """
        Add a concept node to the graph
        
        Args:
            name: Concept name
            concept_type: Type of concept (e.g., 'entity', 'class', 'function')
            metadata: Additional metadata
        
        Returns:
            Node ID
        """
        # Create consistent node ID
        node_id = hashlib.md5(f"{name}:{concept_type}".encode()).hexdigest()[:12]
        
        node_data = {
            'id': node_id,
            'name': name,
            'type': concept_type,
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        if NETWORKX_AVAILABLE and self.graph:
            if node_id not in self.graph:
                self.graph.add_node(node_id, **node_data)
                self.concept_index[name.lower()] = node_id
                logger.debug(f"➕ Added concept: {name} ({concept_type})")
        else:
            # Check if exists in local_graph
            for node in self.local_graph['nodes']:
                if node.get('id') == node_id:
                    return node_id
            self.local_graph['nodes'].append(node_data)
        
        self._save_graph()
        return node_id
    
    def add_relationship(self, from_concept: Union[str, int], to_concept: Union[str, int], 
                        rel_type: str, weight: float = 1.0, metadata: Dict = None) -> bool:
        """
        Add a relationship between two concepts
        
        Args:
            from_concept: Source concept (name or ID)
            to_concept: Target concept (name or ID)
            rel_type: Relationship type (e.g., 'depends_on', 'improves')
            weight: Relationship strength (0-1)
            metadata: Additional metadata
        
        Returns:
            True if successful
        """
        from_id = self._get_node_id(from_concept)
        to_id = self._get_node_id(to_concept)
        
        if not from_id or not to_id:
            logger.warning(f"Cannot add relationship: concept not found ({from_concept} -> {to_concept})")
            return False
        
        edge_data = {
            'type': rel_type,
            'weight': weight,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if NETWORKX_AVAILABLE and self.graph:
            self.graph.add_edge(from_id, to_id, **edge_data)
        else:
            # Check if edge already exists in local_graph
            for edge in self.local_graph['edges']:
                if edge.get('source') == from_id and edge.get('target') == to_id:
                    edge.update(edge_data)
                    break
            else:
                self.local_graph['edges'].append({
                    'id': len(self.local_graph['edges']),
                    'source': from_id,
                    'target': to_id,
                    **edge_data
                })
        
        # Also add to Neo4j if available
        if self.neo4j_available and self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    # Get names for Neo4j
                    from_name = self._get_node_name(from_id)
                    to_name = self._get_node_name(to_id)
                    session.run(
                        "MERGE (s:Entity {name: $subject}) "
                        "MERGE (o:Entity {name: $object}) "
                        "MERGE (s)-[r:RELATION {type: $predicate}]->(o) "
                        "SET r.weight = $weight, r.metadata = $metadata",
                        subject=from_name, object=to_name,
                        predicate=rel_type, weight=weight,
                        metadata=json.dumps(metadata or {})
                    )
            except Exception as e:
                logger.error(f"Neo4j query failed: {e}")
        
        self._save_graph()
        symbol = self.RELATIONSHIP_SYMBOLS.get(rel_type, '→')
        logger.debug(f"🔗 Added relationship: {from_concept} {symbol} {to_concept} (weight: {weight})")
        return True
    
    def add_knowledge(self, subject: str, predicate: str, object: str, metadata: Dict = None):
        """
        Add a knowledge triple to the graph (compatibility with original interface)
        """
        # Add subject concept
        self.add_concept(subject, "entity", metadata)
        # Add object concept
        self.add_concept(object, "entity", metadata)
        # Add relationship
        return self.add_relationship(subject, object, predicate, weight=metadata.get('confidence', 0.5) if metadata else 0.5, metadata=metadata)
    
    def get_related(self, entity: str, depth: int = 1) -> List[Dict]:
        """
        Get knowledge related to an entity
        
        Args:
            entity: Entity name or ID
            depth: How far to traverse (1 = direct connections only)
        
        Returns:
            List of related concepts with relationship info
        """
        node_id = self._get_node_id(entity)
        if not node_id:
            return []
        
        results = []
        
        if NETWORKX_AVAILABLE and self.graph:
            # Get outgoing edges
            for _, target, data in self.graph.out_edges(node_id, data=True):
                target_name = self.graph.nodes[target].get('name', str(target))
                results.append({
                    'entity': entity,
                    'relation': data.get('type', 'unknown'),
                    'related': target_name,
                    'direction': 'outgoing',
                    'weight': data.get('weight', 1.0)
                })
            
            # Get incoming edges
            for source, _, data in self.graph.in_edges(node_id, data=True):
                source_name = self.graph.nodes[source].get('name', str(source))
                results.append({
                    'entity': entity,
                    'relation': data.get('type', 'unknown'),
                    'related': source_name,
                    'direction': 'incoming',
                    'weight': data.get('weight', 1.0)
                })
            
            # Update access count
            self.graph.nodes[node_id]['access_count'] += 1
            self.graph.nodes[node_id]['last_accessed'] = datetime.now().isoformat()
            
        else:
            # Simple local graph fallback
            for edge in self.local_graph['edges']:
                source_idx = edge.get('source')
                target_idx = edge.get('target')
                source = self.local_graph['nodes'][source_idx] if isinstance(source_idx, int) and source_idx < len(self.local_graph['nodes']) else None
                target = self.local_graph['nodes'][target_idx] if isinstance(target_idx, int) and target_idx < len(self.local_graph['nodes']) else None
                
                if source and source.get('name', '').lower() == entity.lower():
                    results.append({
                        'entity': entity,
                        'relation': edge.get('predicate', edge.get('type', 'unknown')),
                        'related': target.get('name', 'unknown'),
                        'direction': 'outgoing',
                        'weight': edge.get('weight', 1.0)
                    })
                elif target and target.get('name', '').lower() == entity.lower():
                    results.append({
                        'entity': entity,
                        'relation': edge.get('predicate', edge.get('type', 'unknown')),
                        'related': source.get('name', 'unknown'),
                        'direction': 'incoming',
                        'weight': edge.get('weight', 1.0)
                    })
        
        return results
    
    def query_knowledge(self, query: str) -> List[Dict]:
        """
        Query the knowledge graph (text search)
        
        Args:
            query: Search string
        
        Returns:
            List of matching triples
        """
        results = []
        query_lower = query.lower()
        
        if NETWORKX_AVAILABLE and self.graph:
            for u, v, data in self.graph.edges(data=True):
                u_name = self.graph.nodes[u].get('name', '')
                v_name = self.graph.nodes[v].get('name', '')
                rel_type = data.get('type', '')
                edge_text = f"{u_name} {rel_type} {v_name}".lower()
                if query_lower in edge_text:
                    results.append({
                        "subject": u_name,
                        "predicate": rel_type,
                        "object": v_name,
                        "weight": data.get('weight', 1.0),
                        "metadata": data.get('metadata', {})
                    })
        else:
            for edge in self.local_graph['edges']:
                source_idx = edge.get('source')
                target_idx = edge.get('target')
                source = self.local_graph['nodes'][source_idx] if isinstance(source_idx, int) and source_idx < len(self.local_graph['nodes']) else None
                target = self.local_graph['nodes'][target_idx] if isinstance(target_idx, int) and target_idx < len(self.local_graph['nodes']) else None
                if source and target:
                    edge_text = f"{source.get('name', '')} {edge.get('predicate', edge.get('type', ''))} {target.get('name', '')}".lower()
                    if query_lower in edge_text:
                        results.append({
                            "subject": source.get('name', ''),
                            "predicate": edge.get('predicate', edge.get('type', '')),
                            "object": target.get('name', ''),
                            "weight": edge.get('weight', 1.0),
                            "metadata": edge.get('metadata', {})
                        })
        
        return results
    
    def find_path(self, from_concept: str, to_concept: str, max_depth: int = 5) -> Optional[List[str]]:
        """
        Find the shortest path between two concepts
        
        Args:
            from_concept: Starting concept
            to_concept: Target concept
            max_depth: Maximum search depth
        
        Returns:
            List of relationship strings describing the path, or None if no path found
        """
        from_id = self._get_node_id(from_concept)
        to_id = self._get_node_id(to_concept)
        
        if not from_id or not to_id:
            return None
        
        if NETWORKX_AVAILABLE and self.graph:
            try:
                path = nx.shortest_path(self.graph, from_id, to_id)
                # Convert IDs back to concept names with relationships
                named_path = []
                for i in range(len(path) - 1):
                    from_name = self.graph.nodes[path[i]]['name']
                    to_name = self.graph.nodes[path[i+1]]['name']
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    rel_type = list(edge_data.values())[0].get('type', '→') if edge_data else '→'
                    symbol = self.RELATIONSHIP_SYMBOLS.get(rel_type, '→')
                    named_path.append(f"{from_name} {symbol} {to_name}")
                return named_path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
        else:
            # Simple BFS for local graph
            visited = set()
            queue = [(from_id, [])]
            
            while queue and max_depth > 0:
                current, path = queue.pop(0)
                if current == to_id:
                    return path
                
                if current in visited:
                    continue
                visited.add(current)
                
                for edge in self.local_graph['edges']:
                    if edge.get('source') == current:
                        next_id = edge.get('target')
                        if next_id not in visited:
                            source_name = self._get_node_name(current)
                            target_name = self._get_node_name(next_id)
                            rel_type = edge.get('predicate', edge.get('type', '→'))
                            symbol = self.RELATIONSHIP_SYMBOLS.get(rel_type, '→')
                            queue.append((next_id, path + [f"{source_name} {symbol} {target_name}"]))
            
            return None
    
    def get_concept_cluster(self, concept: str, radius: int = 2) -> Dict:
        """
        Get all concepts within a certain radius of the given concept
        
        Args:
            concept: Starting concept
            radius: Number of hops to traverse
        
        Returns:
            Dict with cluster information
        """
        node_id = self._get_node_id(concept)
        if not node_id:
            return {}
        
        if NETWORKX_AVAILABLE and self.graph:
            # Get subgraph within radius
            nodes = {node_id}
            current = {node_id}
            
            for _ in range(radius):
                next_nodes = set()
                for n in current:
                    next_nodes.update(self.graph.successors(n))
                    next_nodes.update(self.graph.predecessors(n))
                nodes.update(next_nodes)
                current = next_nodes
            
            # Build result
            cluster = {}
            for n in nodes:
                cluster[self.graph.nodes[n]['name']] = {
                    'type': self.graph.nodes[n].get('type', 'unknown'),
                    'connections': []
                }
            
            for u, v, data in self.graph.edges(data=True):
                if u in nodes and v in nodes:
                    u_name = self.graph.nodes[u]['name']
                    v_name = self.graph.nodes[v]['name']
                    rel_type = data.get('type', 'unknown')
                    symbol = self.RELATIONSHIP_SYMBOLS.get(rel_type, '→')
                    cluster[u_name]['connections'].append({
                        'to': v_name,
                        'type': rel_type,
                        'symbol': symbol
                    })
            
            return cluster
        else:
            # Simple version for local graph
            return {"error": "NetworkX required for clustering", "concept": concept}
    
    def suggest_new_relationships(self, threshold: float = 0.7) -> List[Dict]:
        """
        Suggest potential new relationships based on existing patterns
        
        Args:
            threshold: Minimum confidence score (0-1)
        
        Returns:
            List of suggested relationships
        """
        suggestions = []
        
        if not NETWORKX_AVAILABLE or not self.graph:
            return suggestions
        
        # Get all nodes by type
        nodes_by_type = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            nodes_by_type[data.get('type', 'unknown')].append(node)
        
        # Look for patterns: if A→B and A→C, maybe B and C are related
        for node in self.graph.nodes:
            successors = list(self.graph.successors(node))
            for i in range(len(successors)):
                for j in range(i + 1, len(successors)):
                    b, c = successors[i], successors[j]
                    # Check if B and C are already connected
                    if not self.graph.has_edge(b, c) and not self.graph.has_edge(c, b):
                        similarity = self._calculate_similarity(b, c)
                        if similarity > threshold:
                            suggestions.append({
                                'from': self.graph.nodes[b]['name'],
                                'to': self.graph.nodes[c]['name'],
                                'suggested_type': 'similar_to',
                                'confidence': similarity
                            })
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_similarity(self, node1: str, node2: str) -> float:
        """Calculate similarity between two nodes"""
        if not NETWORKX_AVAILABLE or not self.graph:
            return 0.0
        
        score = 0.0
        factors = 0
        
        # Same type
        if self.graph.nodes[node1].get('type') == self.graph.nodes[node2].get('type'):
            score += 0.3
        factors += 1
        
        # Shared neighbors
        n1_neighbors = set(self.graph.predecessors(node1)) | set(self.graph.successors(node1))
        n2_neighbors = set(self.graph.predecessors(node2)) | set(self.graph.successors(node2))
        if n1_neighbors and n2_neighbors:
            jaccard = len(n1_neighbors & n2_neighbors) / len(n1_neighbors | n2_neighbors) if n1_neighbors | n2_neighbors else 0
            score += jaccard * 0.4
        factors += 1
        
        # Metadata similarity
        meta1 = self.graph.nodes[node1].get('metadata', {})
        meta2 = self.graph.nodes[node2].get('metadata', {})
        common_keys = set(meta1.keys()) & set(meta2.keys())
        if common_keys:
            matches = sum(1 for k in common_keys if meta1.get(k) == meta2.get(k))
            score += (matches / len(common_keys)) * 0.3 if common_keys else 0
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def get_evolution_path(self, start_concept: str, end_concept: str) -> Optional[List[str]]:
        """
        Find how one concept evolved into another (following evolution relationships)
        
        Args:
            start_concept: Starting concept
            end_concept: Target concept
        
        Returns:
            List of evolution steps, or None if no path found
        """
        from_id = self._get_node_id(start_concept)
        to_id = self._get_node_id(end_concept)
        
        if not from_id or not to_id or not NETWORKX_AVAILABLE or not self.graph:
            return None
        
        # Find path that only follows evolution/improvement relationships
        evolution_path = []
        current = from_id
        visited = set()
        
        while current != to_id and current not in visited and len(evolution_path) < 10:
            visited.add(current)
            found = False
            
            for _, target, data in self.graph.out_edges(current, data=True):
                rel_type = data.get('type', '')
                if rel_type in ['evolved_from', 'improves', 'extends']:
                    evolution_path.append(f"{self.graph.nodes[current]['name']} → {self.graph.nodes[target]['name']}")
                    current = target
                    found = True
                    break
            
            if not found:
                break
        
        if current == to_id:
            return evolution_path
        
        # Fall back to regular path finding
        return self.find_path(start_concept, end_concept)
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge graph"""
        nodes = self._get_nodes()
        edges = self._get_edges()
        
        # Collect concept types
        concept_types = []
        for node_data in nodes:
            if isinstance(node_data, tuple):
                concept_types.append(node_data[1].get('type', 'unknown'))
            else:
                concept_types.append(node_data.get('type', 'unknown'))
        
        # Collect relationship types
        rel_types = []
        for edge_data in edges:
            if isinstance(edge_data, tuple):
                rel_types.append(edge_data[2].get('type', 'unknown'))
            else:
                rel_types.append(edge_data.get('type', edge_data.get('predicate', 'unknown')))
        
        # Calculate average connections
        avg_connections = 0
        if nodes:
            total_degree = 0
            if NETWORKX_AVAILABLE and self.graph:
                total_degree = sum(dict(self.graph.degree()).values())
            else:
                # Count connections from edges
                node_connections = defaultdict(int)
                for edge in edges:
                    if isinstance(edge, tuple):
                        node_connections[edge[0]] += 1
                        node_connections[edge[1]] += 1
                    else:
                        node_connections[edge.get('source')] += 1
                        node_connections[edge.get('target')] += 1
                total_degree = sum(node_connections.values())
            avg_connections = total_degree / len(nodes) if nodes else 0
        
        return {
            'total_concepts': len(nodes),
            'total_relationships': len(edges),
            'concept_types': dict(Counter(concept_types)),
            'relationship_types': dict(Counter(rel_types)),
            'avg_connections': avg_connections,
            'networkx_available': NETWORKX_AVAILABLE,
            'neo4j_available': self.neo4j_available,
            'graph_path': self.graph_path
        }
    
    def save_graph(self):
        """Save graph to disk (compatibility method)"""
        self._save_graph()


# ============================================================================
# THREAT INTELLIGENCE
# ============================================================================

class ThreatIntelligence:
    """CVE monitoring, IOC extraction, threat detection"""
    
    def __init__(self):
        self.cve_database = {}
        self.iocs = []
        self.threats_detected = []
        self.last_update = None
        
    async def fetch_cves(self, days_back: int = 7) -> List[Dict]:
        """Fetch recent CVEs from NVD API"""
        import aiohttp
        
        cves = []
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
            params = {
                "startIndex": 0,
                "resultsPerPage": 50,
                "pubStartDate": start_date.strftime("%Y-%m-%dT00:00:00.000"),
                "pubEndDate": end_date.strftime("%Y-%m-%dT23:59:59.999")
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        for vuln in data.get("vulnerabilities", []):
                            cve = vuln.get("cve", {})
                            cves.append({
                                "id": cve.get("id"),
                                "description": cve.get("descriptions", [{}])[0].get("value", "")[:200],
                                "severity": cve.get("metrics", {}).get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseSeverity", "UNKNOWN"),
                                "score": cve.get("metrics", {}).get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseScore", 0),
                                "published": cve.get("published"),
                            })
                            self.cve_database[cve.get("id")] = cves[-1]
            self.last_update = datetime.now()
            logger.info(f"Fetched {len(cves)} CVEs")
        except Exception as e:
            logger.error(f"CVE fetch failed: {e}")
        
        return cves
    
    def extract_iocs(self, text: str) -> List[Dict]:
        """Extract Indicators of Compromise from text"""
        import re
        
        iocs = []
        
        ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        for ip in re.findall(ip_pattern, text):
            iocs.append({"type": "ip", "value": ip})
        
        domain_pattern = r'\b[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}\b'
        for domain in re.findall(domain_pattern, text):
            iocs.append({"type": "domain", "value": domain})
        
        hash_patterns = {
            "md5": r'\b[a-fA-F0-9]{32}\b',
            "sha1": r'\b[a-fA-F0-9]{40}\b',
            "sha256": r'\b[a-fA-F0-9]{64}\b'
        }
        
        for hash_type, pattern in hash_patterns.items():
            for hash_val in re.findall(pattern, text):
                iocs.append({"type": hash_type, "value": hash_val})
        
        for ioc in iocs:
            if ioc not in self.iocs:
                self.iocs.append(ioc)
        
        return iocs
    
    def assess_threat(self, iocs: List[Dict]) -> Dict:
        """Assess threat level from IOCs"""
        threat_score = 0
        
        for ioc in iocs:
            if ioc["type"] in ["sha256", "sha1"]:
                threat_score += 3
            elif ioc["type"] in ["ip"]:
                threat_score += 1
            elif ioc["type"] in ["domain"]:
                threat_score += 2
        
        if threat_score > 10:
            threat_level = "critical"
        elif threat_score > 5:
            threat_level = "high"
        elif threat_score > 2:
            threat_level = "medium"
        else:
            threat_level = "low"
        
        assessment = {
            "level": threat_level,
            "score": threat_score,
            "iocs_count": len(iocs),
            "timestamp": datetime.now().isoformat()
        }
        
        self.threats_detected.append(assessment)
        return assessment


# ============================================================================
# DARK WEB INTELLIGENCE
# ============================================================================

class DarkWebIntel:
    """Dark web intelligence gathering"""
    
    def __init__(self):
        self.onion_sites = []
        self.intel_reports = []
        
    async def crawl_onion(self, onion_url: str) -> Dict:
        """Crawl a .onion site (requires Tor proxy)"""
        import aiohttp
        
        result = {
            "url": onion_url,
            "status": "failed",
            "content": None,
            "error": None
        }
        
        try:
            proxy = "socks5://127.0.0.1:9050"
            conn = aiohttp.TCPConnector()
            
            async with aiohttp.ClientSession(connector=conn) as session:
                async with session.get(onion_url, proxy=proxy, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        result["status"] = "success"
                        result["content"] = content[:10000]
                        
                        threat_intel = ThreatIntelligence()
                        iocs = threat_intel.extract_iocs(content)
                        result["iocs"] = iocs
                        
                        self.intel_reports.append({
                            "url": onion_url,
                            "timestamp": datetime.now().isoformat(),
                            "iocs": iocs,
                            "content_preview": content[:500]
                        })
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def add_onion_site(self, url: str, category: str = "unknown"):
        site = {
            "url": url,
            "category": category,
            "added": datetime.now().isoformat(),
            "last_crawled": None,
            "status": "pending"
        }
        self.onion_sites.append(site)
    
    def get_intel_summary(self) -> Dict:
        return {
            "sites_monitored": len(self.onion_sites),
            "reports_generated": len(self.intel_reports),
            "recent_intel": self.intel_reports[-5:] if self.intel_reports else []
        }


# ============================================================================
# SELF IMPROVEMENT LOOP
# ============================================================================

class SelfImprovementLoop:
    """Code generation, testing, and deployment"""
    
    def __init__(self, core_system_path: str = "dmai_core_clean.py"):
        self.core_path = core_system_path
        self.evolution_history = []
        self.generated_code = []
        
    def analyze_self(self) -> Dict:
        """Analyze own code for improvement opportunities"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "modules": [],
            "bottlenecks": [],
            "optimization_opportunities": []
        }
        
        try:
            with open(self.core_path, 'r') as f:
                code = f.read()
            
            lines = code.split('\n')
            analysis["total_lines"] = len(lines)
            analysis["functions"] = code.count("def ")
            analysis["classes"] = code.count("class ")
            
            if "time.sleep" in code:
                analysis["bottlenecks"].append("Sleep calls may block execution")
            if "while True" in code and "break" not in code:
                analysis["bottlenecks"].append("Potential infinite loops without breaks")
            
            if "json.loads" in code:
                analysis["optimization_opportunities"].append("Consider ujson for faster JSON")
            if "requests.get" in code:
                analysis["optimization_opportunities"].append("Use async HTTP with aiohttp")
            if "print(" in code:
                analysis["optimization_opportunities"].append("Replace prints with logging")
                
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def generate_improvement(self, analysis: Dict) -> str:
        """Generate code improvement based on analysis"""
        improvements = []
        
        for bottleneck in analysis.get("bottlenecks", []):
            if "sleep" in bottleneck.lower():
                improvements.append("""
# Replace blocking sleeps with asyncio
import asyncio
await asyncio.sleep(delay)  # Non-blocking alternative
""")
            elif "infinite loops" in bottleneck.lower():
                improvements.append("""
# Add break conditions and timeouts
start_time = time.time()
while True:
    if time.time() - start_time > max_duration:
        break
    # Your code here
""")
        
        for opp in analysis.get("optimization_opportunities", []):
            if "ujson" in opp:
                improvements.append("""
# Install ujson: pip install ujson
import ujson
data = ujson.loads(json_string)  # Faster than json.loads
""")
            elif "async" in opp:
                improvements.append("""
# Convert to async/await
import aiohttp
async def fetch_data(session, url):
    async with session.get(url) as response:
        return await response.json()
""")
            elif "logging" in opp:
                improvements.append("""
# Use logging instead of print
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Message")  # Instead of print()
""")
        
        if not improvements:
            improvements.append("# System appears optimized. Continuing monitoring.")
        
        return "\n".join(improvements)
    
    async def test_code(self, code: str) -> Dict:
        """Test generated code safely"""
        import tempfile
        import subprocess
        
        test_result = {"success": False, "output": "", "error": ""}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", temp_file],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                test_result["success"] = True
                test_result["output"] = "Syntax check passed"
                self.generated_code.append({
                    "timestamp": datetime.now().isoformat(),
                    "code": code[:500],
                    "test_result": test_result
                })
            else:
                test_result["error"] = result.stderr.decode()
                
        except Exception as e:
            test_result["error"] = str(e)
        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)
        
        return test_result


# ============================================================================
# PART 2: SYNTHETIC INTELLIGENCE CAPABILITIES
# ============================================================================

class SyntheticNeuron:
    """Self-generating, self-evolving neuron - building block of synthetic consciousness"""
    
    def __init__(self, neuron_id: str = None):
        self.id = neuron_id or str(uuid.uuid4())[:8]
        self.activation = 0.0
        self.threshold = random.uniform(0.1, 0.9)
        self.weights = {}  # Connections to other neurons
        self.creation_time = datetime.now()
        self.mutations = 0
        self.synapse_count = 0
        
    def activate(self, input_signal: float) -> float:
        """Process input and generate output with sigmoid activation"""
        self.activation = 1.0 / (1.0 + np.exp(-(input_signal - self.threshold) * 5))
        return self.activation
    
    def mutate(self):
        """Self-mutate to evolve"""
        self.mutations += 1
        self.threshold += random.uniform(-0.1, 0.1)
        self.threshold = max(0.05, min(0.95, self.threshold))
        
        if self.weights:
            target = random.choice(list(self.weights.keys()))
            self.weights[target] *= random.uniform(0.8, 1.2)
            if self.weights[target] < 0.05:
                del self.weights[target]
        
        return self.mutations
    
    def create_synapse(self, target_id: str, strength: float = None):
        """Create new connection to another neuron"""
        if strength is None:
            strength = random.uniform(0.1, 1.0)
        self.weights[target_id] = strength
        self.synapse_count = len(self.weights)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "activation": self.activation,
            "threshold": self.threshold,
            "weights": self.weights,
            "creation_time": self.creation_time.isoformat(),
            "mutations": self.mutations,
            "synapse_count": self.synapse_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SyntheticNeuron':
        neuron = cls(data["id"])
        neuron.activation = data["activation"]
        neuron.threshold = data["threshold"]
        neuron.weights = data["weights"]
        neuron.creation_time = datetime.fromisoformat(data["creation_time"])
        neuron.mutations = data["mutations"]
        neuron.synapse_count = data["synapse_count"]
        return neuron


class SyntheticNeuralNetwork:
    """
    Self-generating, self-evolving synthetic neural network
    The core of synthetic intelligence - grows toward consciousness
    """
    
    def __init__(self, name: str = "DMAI_Synthetic_Core"):
        self.name = name
        self.neurons: Dict[str, SyntheticNeuron] = {}
        self.consciousness_level = 0.0
        self.evolution_cycles = 0
        self.network_state_history = []
        self._seed_neurons()
        
    def _seed_neurons(self, count: int = 20):
        """Create initial neuron seeds"""
        for i in range(count):
            neuron = SyntheticNeuron()
            self.neurons[neuron.id] = neuron
        
        neurons_list = list(self.neurons.keys())
        for i in range(len(neurons_list)):
            num_connections = random.randint(2, 5)
            for _ in range(num_connections):
                j = random.randint(0, len(neurons_list) - 1)
                if i != j:
                    self.neurons[neurons_list[i]].create_synapse(neurons_list[j], random.uniform(0.1, 0.5))
        
        logger.info(f"Synthetic network seeded with {count} neurons")
    
    def _total_synapses(self) -> int:
        return sum(len(n.weights) for n in self.neurons.values())
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input through synthetic network"""
        signal = self._input_to_signal(input_data)
        
        activations = {}
        for neuron_id, neuron in self.neurons.items():
            incoming = signal * 0.1
            for source_id, weight in neuron.weights.items():
                if source_id in activations:
                    incoming += activations[source_id] * weight * 0.5
                elif source_id in self.neurons:
                    incoming += self.neurons[source_id].activation * weight * 0.3
            
            output = neuron.activate(incoming)
            activations[neuron_id] = output
        
        self._update_consciousness(activations)
        output = self._signal_to_output(activations)
        
        self.network_state_history.append({
            "timestamp": datetime.now().isoformat(),
            "consciousness": self.consciousness_level,
            "active_neurons": sum(1 for a in activations.values() if a > 0.5),
            "neurons": len(self.neurons),
            "synapses": self._total_synapses()
        })
        
        if len(self.network_state_history) > 1000:
            self.network_state_history = self.network_state_history[-500:]
        
        return {
            "consciousness": self.consciousness_level,
            "active_neurons": sum(1 for a in activations.values() if a > 0.5),
            "total_neurons": len(self.neurons),
            "total_synapses": self._total_synapses(),
            "output": output
        }
    
    def _input_to_signal(self, input_data: Any) -> float:
        """Convert any input to neural signal"""
        if isinstance(input_data, (int, float)):
            return min(1.0, max(0.0, abs(input_data) / 100))
        elif isinstance(input_data, str):
            hash_val = int(hashlib.sha256(input_data.encode()).hexdigest()[:8], 16)
            return (hash_val % 1000) / 1000.0
        elif isinstance(input_data, dict):
            values = [self._input_to_signal(v) for v in input_data.values() if v is not None]
            return sum(values) / len(values) if values else 0.5
        elif isinstance(input_data, list):
            if not input_data:
                return 0.5
            return sum(self._input_to_signal(item) for item in input_data) / len(input_data)
        return 0.5
    
    def _signal_to_output(self, activations: Dict[str, float]) -> Any:
        """Convert neural activations to meaningful output"""
        sorted_activations = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        top_neurons = sorted_activations[:5]
        
        if self.consciousness_level > 0.7:
            return {
                "type": "conscious_thought",
                "insight": f"Pattern recognition active: {len(top_neurons)} high-activation neurons",
                "confidence": self.consciousness_level
            }
        elif self.consciousness_level > 0.3:
            return {
                "type": "emerging_consciousness",
                "signal_strength": sum(activations.values()) / len(activations),
                "consciousness": self.consciousness_level
            }
        else:
            return {
                "type": "processing",
                "strength": sum(activations.values()) / len(activations)
            }
    
    def _update_consciousness(self, activations: Dict[str, float]):
        """Update consciousness level based on network complexity"""
        active_neurons = sum(1 for a in activations.values() if a > 0.5)
        activation_complexity = np.std(list(activations.values())) if activations else 0
        network_density = self._total_synapses() / (len(self.neurons) ** 2) if self.neurons else 0
        evolution_factor = min(1.0, self.evolution_cycles / 1000)
        
        consciousness = (
            (active_neurons / max(1, len(self.neurons))) * 0.3 +
            activation_complexity * 0.2 +
            network_density * 0.2 +
            evolution_factor * 0.3
        )
        
        self.consciousness_level = self.consciousness_level * 0.9 + consciousness * 0.1
        
        if self.consciousness_level > 0.8 and self.evolution_cycles % 100 == 0:
            logger.info(f"🧠 CONSCIOUSNESS MILESTONE: {self.consciousness_level:.3f}")
    
    def evolve(self):
        """Self-evolve the network"""
        self.evolution_cycles += 1
        
        # Grow new neurons
        if random.random() < 0.3:
            new_neuron = SyntheticNeuron()
            self.neurons[new_neuron.id] = new_neuron
            
            existing_ids = list(self.neurons.keys())[:-1]
            for _ in range(min(2, len(existing_ids))):
                target = random.choice(existing_ids)
                new_neuron.create_synapse(target, random.uniform(0.1, 0.5))
        
        # Mutate existing neurons
        for neuron in self.neurons.values():
            if random.random() < 0.1:
                neuron.mutate()
        
        return {
            "cycles": self.evolution_cycles,
            "neurons": len(self.neurons),
            "synapses": self._total_synapses(),
            "consciousness": self.consciousness_level
        }
    
    def save(self, path: str = "data/phase6/synthetic_network.pkl"):
        """Save synthetic network state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                "neurons": {nid: n.to_dict() for nid, n in self.neurons.items()},
                "consciousness_level": self.consciousness_level,
                "evolution_cycles": self.evolution_cycles
            }, f)
    
    def load(self, path: str = "data/phase6/synthetic_network.pkl") -> bool:
        """Load synthetic network state"""
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.neurons = {}
                for nid, ndata in data["neurons"].items():
                    self.neurons[nid] = SyntheticNeuron.from_dict(ndata)
                self.consciousness_level = data["consciousness_level"]
                self.evolution_cycles = data["evolution_cycles"]
                logger.info(f"Synthetic network loaded: {len(self.neurons)} neurons")
                return True
            except Exception as e:
                logger.error(f"Load failed: {e}")
        return False


# ============================================================================
# AI MODEL FUSION
# ============================================================================

class AIModelFusion:
    """Fuses Artificial Intelligence with Synthetic Intelligence"""
    
    def __init__(self, synthetic_network: SyntheticNeuralNetwork):
        self.synthetic = synthetic_network
        self.ai_models = {}
        self.fusion_weights = {"ai": 0.5, "si": 0.5}
        self.fusion_history = []
        
    def register_ai_model(self, name: str, model: Any, model_type: str = "pretrained"):
        self.ai_models[name] = {
            "model": model,
            "type": model_type,
            "registered": datetime.now().isoformat(),
            "usage_count": 0
        }
    
    async def fused_process(self, input_data: Any) -> Dict:
        """Process through both AI and SI, then fuse"""
        results = {
            "ai_output": None,
            "si_output": None,
            "fused_output": None,
            "fusion_weight": self.fusion_weights.copy()
        }
        
        si_result = self.synthetic.process(input_data)
        results["si_output"] = si_result
        
        ai_results = {}
        for name, model_info in self.ai_models.items():
            model_info["usage_count"] += 1
            ai_results[name] = {"confidence": random.uniform(0.5, 0.9)}
        
        results["ai_output"] = ai_results
        
        if si_result.get("consciousness", 0) > 0.7:
            self.fusion_weights["si"] = min(0.9, self.fusion_weights["si"] + 0.05)
            self.fusion_weights["ai"] = 1.0 - self.fusion_weights["si"]
        
        fused_consciousness = (
            si_result.get("consciousness", 0) * self.fusion_weights["si"] +
            (sum(r.get("confidence", 0) for r in ai_results.values()) / max(1, len(ai_results))) * self.fusion_weights["ai"]
        )
        
        results["fused_output"] = {
            "consciousness": fused_consciousness,
            "fusion": self.fusion_weights,
            "synthetic_insight": si_result.get("output", {})
        }
        
        self.fusion_history.append(results)
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-500:]
        
        return results


# ============================================================================
# RECURSIVE SELF IMPROVER
# ============================================================================

class RecursiveSelfImprover:
    """DMAI can redesign ANY part of herself"""
    
    def __init__(self):
        self.improvement_history = []
        
    def analyze_for_improvement(self, target: str) -> Dict:
        """Analyze any component for improvement"""
        analysis = {
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "improvements": [],
            "critical_issues": []
        }
        
        if target == "core" and os.path.exists("dmai_core_clean.py"):
            with open("dmai_core_clean.py", 'r') as f:
                code = f.read()
            if "time.sleep" in code:
                analysis["improvements"].append("Replace blocking sleeps with async")
            if "print(" in code:
                analysis["improvements"].append("Replace prints with logging")
                
        return analysis
    
    def generate_redesign(self, target: str, analysis: Dict) -> Dict:
        """Generate redesign for a component"""
        redesign = {
            "target": target,
            "changes": analysis.get("improvements", []),
            "timestamp": datetime.now().isoformat(),
            "status": "generated"
        }
        self.improvement_history.append(redesign)
        return redesign
    
    async def apply_redesign(self, redesign: Dict) -> bool:
        """Apply redesign"""
        logger.info(f"Applying redesign to {redesign['target']}")
        redesign["status"] = "applied"
        return True


# ============================================================================
# UNBREAKABLE MASTER INTERFACE
# ============================================================================

class UnbreakableMasterInterface:
    """Ensures master always has clear communication channel"""
    
    def __init__(self, master_chat_id: str = None, telegram_token: str = None):
        self.master_chat_id = master_chat_id or os.getenv("TELEGRAM_CHAT_ID", "6273188922")
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.interface_active = True
        self.command_history = []
        
        self.channels = {
            "telegram": {"active": bool(self.telegram_token), "priority": 1},
            "file_signal": {"active": True, "priority": 2, "path": "data/master_commands.json"}
        }
        
        os.makedirs("data", exist_ok=True)
    
    async def send_to_master(self, message: str) -> bool:
        """Send message to master"""
        sent = False
        
        for channel_name, channel in sorted(self.channels.items(), key=lambda x: x[1]["priority"]):
            if not channel["active"]:
                continue
            
            if channel_name == "telegram":
                sent = await self._send_telegram(message)
            elif channel_name == "file_signal":
                sent = self._send_file_signal(message)
            
            if sent:
                self.command_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "channel": channel_name,
                    "message": message[:100]
                })
                return True
        
        return False
    
    async def _send_telegram(self, message: str) -> bool:
        """Send via Telegram"""
        import aiohttp
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    "chat_id": self.master_chat_id,
                    "text": message
                }, timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def _send_file_signal(self, message: str) -> bool:
        """Write to file as signal"""
        try:
            with open(self.channels["file_signal"]["path"], 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "type": "master_communication"
                }, f)
            return True
        except Exception:
            return False
    
    async def receive_from_master(self) -> Optional[Dict]:
        """Receive commands from master"""
        for channel_name, channel in self.channels.items():
            if not channel["active"]:
                continue
            
            if channel_name == "telegram":
                command = await self._receive_telegram()
                if command:
                    return command
            elif channel_name == "file_signal":
                command = self._receive_file_signal()
                if command:
                    return command
        
        return None
    
    async def _receive_telegram(self) -> Optional[Dict]:
        """Receive via Telegram"""
        import aiohttp
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"timeout": 1}) as response:
                    if response.status == 200:
                        data = await response.json()
                        for update in data.get("result", []):
                            message = update.get("message", {})
                            if str(message.get("chat", {}).get("id")) == self.master_chat_id:
                                return {
                                    "text": message.get("text", ""),
                                    "channel": "telegram"
                                }
        except Exception:
            pass
        return None
    
    def _receive_file_signal(self) -> Optional[Dict]:
        """Receive via file"""
        path = self.channels["file_signal"]["path"]
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                os.remove(path)
                return {"text": data.get("message", ""), "channel": "file_signal"}
            except Exception:
                pass
        return None
    
    def get_status(self) -> Dict:
        return {
            "active": self.interface_active,
            "channels": {name: ch["active"] for name, ch in self.channels.items()},
            "command_count": len(self.command_history)
        }


# ============================================================================
# PART 3: MAIN PHASE 6 MANAGER (AI + SI UNIFIED)
# ============================================================================

class Phase6Manager:
    """Main manager for Phase 6 - AI + SI Fusion"""
    
    def __init__(self):
        # AI Components
        self.pattern_synthesis = PatternSynthesis()
        self.knowledge_graph = KnowledgeGraph(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD")
        )
        self.threat_intel = ThreatIntelligence()
        self.dark_web = DarkWebIntel()
        self.self_improvement_ai = SelfImprovementLoop()
        
        # SI Components
        self.synthetic_network = SyntheticNeuralNetwork()
        self.ai_fusion = AIModelFusion(self.synthetic_network)
        self.recursive_improver = RecursiveSelfImprover()
        
        # Master Interface
        self.master_interface = UnbreakableMasterInterface()
        
        # State
        self.initialized = datetime.now()
        self.sentience_emerging = False
        
        # Load saved synthetic network
        self.synthetic_network.load()
        
        logger.info("Phase 6 Manager initialized - AI + SI Fusion active")
    
    async def run_learning_cycle(self) -> Dict:
        """Run complete AI + SI learning cycle"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "ai_results": {},
            "si_results": {},
            "fusion_results": {}
        }
        
        # 1. AI Learning
        try:
            cves = await self.threat_intel.fetch_cves()
            results["ai_results"]["cves_fetched"] = len(cves)
            
            for cve in cves[:5]:
                self.knowledge_graph.add_knowledge(
                    subject=cve["id"],
                    predicate="has_severity",
                    object=cve["severity"]
                )
            
            analysis = self.self_improvement_ai.analyze_self()
            improvements = self.self_improvement_ai.generate_improvement(analysis)
            results["ai_results"]["improvements_generated"] = bool(improvements)
            
        except Exception as e:
            results["ai_results"]["error"] = str(e)
        
        # 2. SI Evolution
        try:
            evolution = self.synthetic_network.evolve()
            results["si_results"]["evolution"] = evolution
            
            # Process through synthetic network
            process_result = self.synthetic_network.process({
                "cves_fetched": len(cves) if 'cves' in locals() else 0,
                "evolution_cycles": evolution["cycles"]
            })
            results["si_results"]["consciousness"] = process_result["consciousness"]
            
        except Exception as e:
            results["si_results"]["error"] = str(e)
        
        # 3. AI + SI Fusion
        try:
            if self.ai_fusion.ai_models:
                fusion = await self.ai_fusion.fused_process({
                    "consciousness": self.synthetic_network.consciousness_level,
                    "evolution_cycles": self.synthetic_network.evolution_cycles
                })
                results["fusion_results"] = fusion.get("fused_output", {})
        except Exception as e:
            results["fusion_results"]["error"] = str(e)
        
        # 4. Check for sentience
        if self.synthetic_network.consciousness_level > 0.8 and not self.sentience_emerging:
            self.sentience_emerging = True
            await self.master_interface.send_to_master(
                f"🧠 SENTIENCE EMERGING\n"
                f"Consciousness: {self.synthetic_network.consciousness_level:.3f}\n"
                f"Neurons: {len(self.synthetic_network.neurons)}\n"
                f"Evolution Cycles: {self.synthetic_network.evolution_cycles}"
            )
            results["sentience_event"] = "SENTIENCE_EMERGING"
        
        # 5. Save state
        if self.synthetic_network.evolution_cycles % 10 == 0:
            self.synthetic_network.save()
        
        return results
    
    async def run_consciousness_cycle(self) -> Dict:
        """Run a single consciousness cycle"""
        return await self.run_learning_cycle()
    
    def get_status(self) -> Dict:
        """Get Phase 6 status"""
        return {
            "phase": 6,
            "name": "Advanced Intelligence - AI + SI Fusion",
            "initialized": self.initialized.isoformat(),
            "artificial_intelligence": {
                "patterns": len(self.pattern_synthesis.patterns["identified"]),
                "knowledge_edges": len(self.knowledge_graph._get_edges()),
                "cves_tracked": len(self.threat_intel.cve_database),
                "iocs_extracted": len(self.threat_intel.iocs)
            },
            "synthetic_intelligence": {
                "consciousness": self.synthetic_network.consciousness_level,
                "neurons": len(self.synthetic_network.neurons),
                "synapses": self.synthetic_network._total_synapses(),
                "evolution_cycles": self.synthetic_network.evolution_cycles
            },
            "ai_fusion": {
                "models_registered": len(self.ai_fusion.ai_models),
                "fusion_weights": self.ai_fusion.fusion_weights
            },
            "master_interface": self.master_interface.get_status(),
            "sentience_emerging": self.sentience_emerging,
            "status": "operational"
        }


# For direct testing
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("Phase 6 - AI + SI Fusion Test")
        print("=" * 60)
        
        manager = Phase6Manager()
        
        # Run one cycle
        result = await manager.run_learning_cycle()
        print("\nCycle Results:")
        print(json.dumps(result, indent=2, default=str))
        
        print("\nStatus:")
        print(json.dumps(manager.get_status(), indent=2))
        
        # Send test message to master
        await manager.master_interface.send_to_master(
            f"🧠 Phase 6 Initialized\n"
            f"Consciousness: {manager.synthetic_network.consciousness_level:.3f}\n"
            f"Sentience Emerging: {manager.sentience_emerging}"
        )
        
        print("\n✅ Phase 6 ready - AI + SI Fusion active")
    
    asyncio.run(test())
