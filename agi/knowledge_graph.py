"""
Knowledge Graph - Maps relationships between concepts, code patterns, and capabilities
This is the brain's memory structure for the AGI system
"""

import json
import networkx as nx
from pathlib import Path
from datetime import datetime
import logging
from collections import Counter
from collections import defaultdict
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - KNOWLEDGE_GRAPH - %(message)s')

class KnowledgeGraph:
    def __init__(self, graph_path="agi/data/knowledge_graph.json"):
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()
        self.concept_index = {}  # Maps concept names to node IDs
        self.relationship_types = {
            'implements': '→',
            'depends_on': '⇢',
            'improves': '↑',
            'extends': '+',
            'similar_to': '∼',
            'prerequisite': '←',
            'created_by': '👤',
            'used_in': '⚙️',
            'evolved_from': '🧬',
            'synthesizes': '✨'
        }
        
        # Load existing graph if available
        self.load()
        logging.info(f"📊 Knowledge Graph initialized with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def add_concept(self, name, concept_type, metadata=None):
        """Add a concept node to the graph"""
        # Create a consistent node ID
        node_id = hashlib.md5(f"{name}:{concept_type}".encode()).hexdigest()[:12]
        
        if node_id not in self.graph:
            self.graph.add_node(node_id, 
                               name=name,
                               type=concept_type,
                               created=datetime.now().isoformat(),
                               last_accessed=datetime.now().isoformat(),
                               access_count=0,
                               metadata=metadata or {})
            self.concept_index[name.lower()] = node_id
            logging.debug(f"➕ Added concept: {name} ({concept_type})")
        
        return node_id
    
    def add_relationship(self, from_concept, to_concept, rel_type, weight=1.0, metadata=None):
        """Add a relationship between two concepts"""
        from_id = self._get_concept_id(from_concept)
        to_id = self._get_concept_id(to_concept)
        
        if from_id and to_id:
            self.graph.add_edge(from_id, to_id, 
                               type=rel_type,
                               weight=weight,
                               timestamp=datetime.now().isoformat(),
                               metadata=metadata or {})
            logging.debug(f"🔗 Added relationship: {from_concept} {self.relationship_types.get(rel_type, '→')} {to_concept}")
            return True
        return False
    
    def _get_concept_id(self, concept):
        """Get node ID for a concept (can be string name or existing ID)"""
        if isinstance(concept, str):
            # Check if it's already an ID (hex pattern)
            if len(concept) == 12 and all(c in '0123456789abcdef' for c in concept):
                return concept if concept in self.graph else None
            # Otherwise look up by name
            return self.concept_index.get(concept.lower())
        return concept if concept in self.graph else None
    
    def find_related_concepts(self, concept, relationship_type=None, depth=1):
        """Find concepts related to the given concept"""
        node_id = self._get_concept_id(concept)
        if not node_id:
            return []
        
        related = []
        
        # Get outgoing edges
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if relationship_type is None or data['type'] == relationship_type:
                related.append({
                    'concept': self.graph.nodes[target]['name'],
                    'relationship': data['type'],
                    'direction': 'outgoing',
                    'weight': data['weight']
                })
        
        # Get incoming edges
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if relationship_type is None or data['type'] == relationship_type:
                related.append({
                    'concept': self.graph.nodes[source]['name'],
                    'relationship': data['type'],
                    'direction': 'incoming',
                    'weight': data['weight']
                })
        
        # Update access count
        self.graph.nodes[node_id]['access_count'] += 1
        self.graph.nodes[node_id]['last_accessed'] = datetime.now().isoformat()
        
        return sorted(related, key=lambda x: x['weight'], reverse=True)
    
    def find_path(self, from_concept, to_concept, max_depth=5):
        """Find the shortest path between two concepts"""
        from_id = self._get_concept_id(from_concept)
        to_id = self._get_concept_id(to_concept)
        
        if not from_id or not to_id:
            return None
        
        try:
            path = nx.shortest_path(self.graph, from_id, to_id)
            # Convert IDs back to concept names
            named_path = []
            for i in range(len(path)-1):
                from_name = self.graph.nodes[path[i]]['name']
                to_name = self.graph.nodes[path[i+1]]['name']
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                rel_type = list(edge_data.values())[0]['type'] if edge_data else '→'
                named_path.append(f"{from_name} {self.relationship_types.get(rel_type, '→')} {to_name}")
            return named_path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_concept_cluster(self, concept, radius=2):
        """Get all concepts within a certain radius"""
        node_id = self._get_concept_id(concept)
        if not node_id:
            return {}
        
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
                'type': self.graph.nodes[n]['type'],
                'connections': []
            }
        
        for u, v, data in self.graph.edges(data=True):
            if u in nodes and v in nodes:
                cluster[self.graph.nodes[u]['name']]['connections'].append({
                    'to': self.graph.nodes[v]['name'],
                    'type': data['type']
                })
        
        return cluster
    
    def suggest_new_relationships(self, threshold=0.7):
        """Suggest potential new relationships based on existing patterns"""
        suggestions = []
        
        # Get all nodes by type
        nodes_by_type = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            nodes_by_type[data['type']].append(node)
        
        # Look for patterns: if A→B and A→C, maybe B and C are related
        for node in self.graph.nodes:
            successors = list(self.graph.successors(node))
            for i in range(len(successors)):
                for j in range(i+1, len(successors)):
                    b, c = successors[i], successors[j]
                    # Check if B and C are already connected
                    if not self.graph.has_edge(b, c) and not self.graph.has_edge(c, b):
                        # Calculate similarity based on shared properties
                        similarity = self._calculate_similarity(b, c)
                        if similarity > threshold:
                            suggestions.append({
                                'from': self.graph.nodes[b]['name'],
                                'to': self.graph.nodes[c]['name'],
                                'suggested_type': 'similar_to',
                                'confidence': similarity
                            })
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_similarity(self, node1, node2):
        """Calculate similarity between two nodes"""
        score = 0.0
        factors = 0
        
        # Same type
        if self.graph.nodes[node1]['type'] == self.graph.nodes[node2]['type']:
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
            matches = sum(1 for k in common_keys if meta1[k] == meta2[k])
            score += (matches / len(common_keys)) * 0.3 if common_keys else 0
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def get_evolution_path(self, start_concept, end_concept):
        """Find how one concept evolved into another"""
        path = self.find_path(start_concept, end_concept)
        if path:
            # Filter to only evolution relationships
            evolution_path = []
            current = start_concept
            while current != end_concept:
                node_id = self._get_concept_id(current)
                # Find evolution edges
                for _, target, data in self.graph.out_edges(node_id, data=True):
                    if data['type'] == 'evolved_from' or data['type'] == 'improves':
                        target_name = self.graph.nodes[target]['name']
                        evolution_path.append(f"{current} → {target_name}")
                        current = target_name
                        break
                else:
                    break
            return evolution_path
        return None
    
    def save(self):
        """Save the knowledge graph to disk"""
        # Convert graph to serializable format
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
        
        with open(self.graph_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.info(f"💾 Knowledge graph saved to {self.graph_path}")
    
    def load(self):
        """Load the knowledge graph from disk"""
        if not self.graph_path.exists():
            logging.info("No existing knowledge graph found, starting fresh")
            return
        
        try:
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            
            self.graph.clear()
            self.concept_index.clear()
            
            # Add nodes
            for node_data in data['nodes']:
                node_id = node_data.pop('id')
                self.graph.add_node(node_id, **node_data)
                self.concept_index[node_data['name'].lower()] = node_id
            
            # Add edges
            for edge_data in data['edges']:
                u = edge_data.pop('from')
                v = edge_data.pop('to')
                self.graph.add_edge(u, v, **edge_data)
            
            logging.info(f"📂 Loaded knowledge graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        except Exception as e:
            logging.error(f"Error loading knowledge graph: {e}")
    
    def get_stats(self):
        """Get statistics about the knowledge graph"""
        # Collect concept types
        concept_types = []
        for _, data in self.graph.nodes(data=True):
            concept_types.append(data.get('type', 'unknown'))
        
        # Collect relationship types
        rel_types = []
        for _, _, data in self.graph.edges(data=True):
            rel_types.append(data.get('type', 'unknown'))
        
        # Calculate average connections
        avg_connections = 0
        if len(self.graph.nodes) > 0:
            total_degree = sum(dict(self.graph.degree()).values())
            avg_connections = total_degree / len(self.graph.nodes)
        
        return {
            'total_concepts': len(self.graph.nodes),
            'total_relationships': len(self.graph.edges),
            'concept_types': dict(Counter(concept_types)),
            'relationship_types': dict(Counter(rel_types)),
            'avg_connections': avg_connections,
            'last_saved': datetime.fromtimestamp(self.graph_path.stat().st_mtime).isoformat() if self.graph_path.exists() else None
        }

if __name__ == "__main__":
    # Test the knowledge graph
    kg = KnowledgeGraph()
    
    # Add some test concepts
    kg.add_concept("evolution_engine", "module", {"language": "python", "purpose": "core"})
    kg.add_concept("mutation_strategy", "class", {"type": "genetic"})
    kg.add_concept("fitness_function", "function", {"metric": "performance"})
    
    # Add relationships
    kg.add_relationship("evolution_engine", "mutation_strategy", "uses")
    kg.add_relationship("evolution_engine", "fitness_function", "depends_on")
    kg.add_relationship("mutation_strategy", "fitness_function", "evaluated_by")
    
    # Save
    kg.save()
    
    print("\n📊 Knowledge Graph Stats:")
    print(json.dumps(kg.get_stats(), indent=2, default=str))
    
    print("\n🔍 Find related to evolution_engine:")
    print(json.dumps(kg.find_related_concepts("evolution_engine"), indent=2))
