#!/usr/bin/env python3
"""
P0T1_Fix_Knowledge_Graph_health_warnings.py
Knowledge Graph Health Monitor - Scans for warnings and auto-repairs issues
Full-featured component for DMAI evolution system
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_graph_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KnowledgeGraphHealth')

class KnowledgeGraphFixer:
    """
    Knowledge Graph Health Monitor - Scans for warnings and auto-repairs issues
    Full-featured component with graph analysis and repair capabilities
    """
    
    def __init__(self):
        self.name = "Knowledge Graph Health Fixer"
        self.component_id = "P0T1"
        self.version = "1.0.0"
        self.health_status = {
            'healthy': False,
            'warnings': [],
            'errors': [],
            'fixes_applied': [],
            'last_check': None,
            'node_count': 0,
            'edge_count': 0,
            'orphan_nodes': 0,
            'broken_edges': 0,
            'missing_metadata': []
        }
        self.fix_history = []
        self.running = False
        self.graph = None
        self.graph_available = False
        
    def run(self, continuous=False, interval=300):
        """
        Main execution method - called by evolution engine
        
        Args:
            continuous: Whether to run continuously
            interval: Check interval in seconds
        """
        logger.info(f"🚀 Starting Knowledge Graph Health Fixer v{self.version}")
        self.running = True
        
        try:
            # Initialize graph connection
            self._init_knowledge_graph()
            
            if continuous:
                logger.info(f"Continuous mode: checking every {interval} seconds")
                while self.running:
                    self.check_and_fix()
                    time.sleep(interval)
            else:
                # Single run
                result = self.check_and_fix()
                
            logger.info("✅ Knowledge Graph Health Fixer completed")
            return self.get_status()
            
        except Exception as e:
            logger.error(f"❌ Error in health fixer: {e}")
            logger.error(traceback.format_exc())
            self.health_status['errors'].append(str(e))
            return self.health_status
    
    def evolve(self):
        """
        Evolution method - called when component needs to evolve
        """
        logger.info(f"🧬 Evolving {self.name}")
        self.version = f"1.0.{len(self.fix_history) + 1}"
        return {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': self.version,
            'fixes_applied': len(self.fix_history)
        }
    
    def execute(self, command=None, **kwargs):
        """
        Execute method - runs specific commands
        """
        logger.info(f"⚙️ Executing command: {command}")
        
        if command == 'fix':
            return self.check_and_fix()
        elif command == 'reset':
            self.health_status = {
                'healthy': False,
                'warnings': [],
                'errors': [],
                'fixes_applied': [],
                'last_check': None,
                'node_count': 0,
                'edge_count': 0,
                'orphan_nodes': 0,
                'broken_edges': 0,
                'missing_metadata': []
            }
            return {'status': 'reset', 'component': self.component_id}
        else:
            return self.get_status()
    
    def process(self, data=None):
        """
        Process method - handles data processing
        """
        logger.info(f"🔄 Processing data for {self.name}")
        
        if data and isinstance(data, dict):
            # Process input data
            if 'graph_data' in data:
                self._process_graph_data(data['graph_data'])
            if 'health_check' in data:
                return self.check_and_fix()
        
        return {
            'component': self.component_id,
            'processed': True,
            'status': self.health_status
        }
    
    def generate(self):
        """
        Generate method - produces output/report
        """
        logger.info(f"📊 Generating report for {self.name}")
        
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'health_status': self.health_status,
            'fix_history': self.fix_history[-5:],  # Last 5 fixes
            'timestamp': datetime.now().isoformat()
        }
    
    def query(self, question=None):
        """
        Query method - answers questions about component state
        """
        logger.info(f"❓ Querying {self.name}")
        
        if question == 'health':
            return {
                'component': self.component_id,
                'healthy': self.health_status.get('healthy', False),
                'methods': ['run', 'evolve', 'execute', 'process', 'generate', 'query']
            }
        elif question == 'warnings':
            return {
                'component': self.component_id,
                'warnings': self.health_status.get('warnings', [])
            }
        elif question == 'fixes':
            return {
                'component': self.component_id,
                'total_fixes': len(self.fix_history),
                'recent_fixes': self.fix_history[-5:]
            }
        else:
            return self.get_status()
    
    def _init_knowledge_graph(self):
        """Initialize connection to knowledge graph"""
        try:
            # Try to get graph from DMAI core if available
            if hasattr(self, 'dmai') and hasattr(self.dmai, 'knowledge_graph'):
                self.graph = self.dmai.knowledge_graph
                self.graph_available = True
                logger.info("✅ Connected to DMAI core knowledge graph")
                return
            
            # Try to import knowledge graph module
            try:
                # Add parent directory to path
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from knowledge_graph import KnowledgeGraph
                self.graph = KnowledgeGraph()
                self.graph_available = True
                logger.info("✅ Connected to Knowledge Graph module")
                return
            except ImportError:
                logger.warning("⚠️ KnowledgeGraph module not available")
            
            # Create simple in-memory graph as fallback
            try:
                import networkx as nx
                self.graph = nx.DiGraph()
                self.graph_available = True
                
                # Add some test data
                self.graph.add_node("root", type="system", created_at=datetime.now().isoformat())
                self.graph.add_node("knowledge_base", type="collection")
                self.graph.add_edge("root", "knowledge_base", type="contains")
                
                logger.info("⚠️ Using fallback NetworkX graph with test data")
                return
            except ImportError:
                logger.error("❌ No graph library available")
                self.graph_available = False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge graph: {e}")
            self.graph_available = False
    
    def _process_graph_data(self, graph_data):
        """Process incoming graph data"""
        if not graph_data:
            return
        
        if isinstance(graph_data, dict):
            if 'nodes' in graph_data:
                self.health_status['node_count'] = len(graph_data['nodes'])
            if 'edges' in graph_data:
                self.health_status['edge_count'] = len(graph_data['edges'])
    
    def check_and_fix(self):
        """
        Perform health check and apply fixes
        """
        logger.info("🔍 Scanning knowledge graph for health warnings...")
        
        try:
            # Reset warnings for this check
            self.health_status['warnings'] = []
            self.health_status['errors'] = []
            
            # Run health checks
            self._check_graph_integrity()
            self._check_node_health()
            self._check_edge_health()
            self._check_metadata_health()
            
            # Apply fixes for identified issues
            self._apply_fixes()
            
            # Update status
            self.health_status['last_check'] = datetime.now().isoformat()
            self.health_status['healthy'] = len(self.health_status['warnings']) == 0
            
            # Log results
            logger.info(f"📊 Health check complete: {len(self.health_status['warnings'])} warnings, "
                       f"{len(self.health_status['fixes_applied'])} fixes applied")
            
            # Save check results
            self._save_check_results()
            
            return self.health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            self.health_status['errors'].append(str(e))
            return self.health_status
    
    def _check_graph_integrity(self):
        """Check basic graph integrity"""
        if not self.graph_available:
            self.health_status['warnings'].append("Knowledge graph not available")
            return
            
        try:
            # Get node count
            if hasattr(self.graph, 'number_of_nodes'):
                node_count = self.graph.number_of_nodes()
            elif hasattr(self.graph, 'nodes'):
                node_count = len(self.graph.nodes())
            else:
                node_count = 0
                
            # Get edge count
            if hasattr(self.graph, 'number_of_edges'):
                edge_count = self.graph.number_of_edges()
            elif hasattr(self.graph, 'edges'):
                edge_count = len(self.graph.edges())
            else:
                edge_count = 0
            
            self.health_status['node_count'] = node_count
            self.health_status['edge_count'] = edge_count
            
            # Check for empty graph
            if node_count == 0:
                self.health_status['warnings'].append("Knowledge graph is empty")
            elif edge_count == 0 and node_count > 0:
                self.health_status['warnings'].append("Graph has nodes but no edges")
            elif edge_count > node_count * 10:
                self.health_status['warnings'].append(f"High edge density: {edge_count} edges for {node_count} nodes")
                
        except Exception as e:
            self.health_status['warnings'].append(f"Graph integrity check failed: {e}")
    
    def _check_node_health(self):
        """Check node health and find orphans"""
        if not self.graph_available:
            return
            
        orphan_count = 0
        
        try:
            # Get nodes
            if hasattr(self.graph, 'nodes'):
                nodes = list(self.graph.nodes())
            else:
                nodes = []
            
            # Check for orphan nodes (no connections)
            for node in nodes:
                try:
                    if hasattr(self.graph, 'degree'):
                        if self.graph.degree(node) == 0:
                            orphan_count += 1
                except:
                    pass
                    
            self.health_status['orphan_nodes'] = orphan_count
            
            if orphan_count > 0:
                self.health_status['warnings'].append(f"Found {orphan_count} orphan nodes")
                
        except Exception as e:
            self.health_status['warnings'].append(f"Node health check failed: {e}")
    
    def _check_edge_health(self):
        """Check edge health and find broken connections"""
        if not self.graph_available:
            return
            
        broken_edges = 0
        
        try:
            # Get edges
            if hasattr(self.graph, 'edges'):
                edges = list(self.graph.edges())
            else:
                edges = []
            
            # Check for broken edges (missing endpoint nodes)
            for u, v in edges:
                try:
                    if hasattr(self.graph, 'has_node'):
                        if not self.graph.has_node(u) or not self.graph.has_node(v):
                            broken_edges += 1
                except:
                    pass
                    
            self.health_status['broken_edges'] = broken_edges
            
            if broken_edges > 0:
                self.health_status['warnings'].append(f"Found {broken_edges} broken edges")
                
        except Exception as e:
            self.health_status['warnings'].append(f"Edge health check failed: {e}")
    
    def _check_metadata_health(self):
        """Check metadata consistency"""
        missing = []
        
        try:
            # Check graph metadata if available
            if hasattr(self.graph, 'graph'):
                metadata = self.graph.graph
                required_fields = ['created_at', 'version', 'last_modified']
                
                for field in required_fields:
                    if field not in metadata:
                        missing.append(field)
                        
            self.health_status['missing_metadata'] = missing
            
            if missing:
                self.health_status['warnings'].append(f"Missing metadata: {', '.join(missing)}")
                
        except Exception as e:
            self.health_status['warnings'].append(f"Metadata check failed: {e}")
    
    def _apply_fixes(self):
        """Apply fixes to identified issues"""
        fixes_applied = []
        
        if not self.graph_available:
            self.health_status['fixes_applied'] = fixes_applied
            return
        
        # Fix orphan nodes by connecting them to a hub
        if self.health_status.get('orphan_nodes', 0) > 0:
            fixed = self._fix_orphan_nodes()
            if fixed > 0:
                fixes_applied.append(f"Connected {fixed} orphan nodes to health hub")
        
        # Fix broken edges by removing them
        if self.health_status.get('broken_edges', 0) > 0:
            fixed = self._fix_broken_edges()
            if fixed > 0:
                fixes_applied.append(f"Removed {fixed} broken edges")
        
        # Add missing metadata
        if self.health_status.get('missing_metadata', []):
            self._add_missing_metadata()
            fixes_applied.append("Added missing metadata fields")
        
        # Record fixes
        self.health_status['fixes_applied'] = fixes_applied
        
        if fixes_applied:
            self.fix_history.append({
                'timestamp': datetime.now().isoformat(),
                'fixes': fixes_applied,
                'warnings_before': self.health_status['warnings'].copy()
            })
            logger.info(f"✅ Applied fixes: {fixes_applied}")
    
    def _fix_orphan_nodes(self):
        """Connect orphan nodes to a central health hub"""
        fixed = 0
        
        try:
            # Find orphan nodes
            orphans = []
            if hasattr(self.graph, 'nodes'):
                nodes = list(self.graph.nodes())
                for node in nodes:
                    try:
                        if hasattr(self.graph, 'degree') and self.graph.degree(node) == 0:
                            orphans.append(node)
                    except:
                        pass
            
            if orphans:
                # Create or get health hub
                hub = 'health_monitor_hub'
                if not self.graph.has_node(hub):
                    self.graph.add_node(hub, 
                                       type='health_hub',
                                       created_at=datetime.now().isoformat(),
                                       description='Central hub for health-monitored nodes')
                
                # Connect orphans to hub
                for orphan in orphans:
                    try:
                        self.graph.add_edge(orphan, hub, 
                                          type='health_connection',
                                          fixed_at=datetime.now().isoformat())
                        fixed += 1
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error fixing orphan nodes: {e}")
            
        return fixed
    
    def _fix_broken_edges(self):
        """Remove broken edges"""
        fixed = 0
        
        try:
            edges_to_remove = []
            if hasattr(self.graph, 'edges'):
                edges = list(self.graph.edges())
                for u, v in edges:
                    try:
                        if not self.graph.has_node(u) or not self.graph.has_node(v):
                            edges_to_remove.append((u, v))
                    except:
                        pass
                
                for u, v in edges_to_remove:
                    try:
                        self.graph.remove_edge(u, v)
                        fixed += 1
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error fixing broken edges: {e}")
            
        return fixed
    
    def _add_missing_metadata(self):
        """Add missing metadata to graph"""
        try:
            if hasattr(self.graph, 'graph'):
                now = datetime.now().isoformat()
                
                if 'created_at' not in self.graph.graph:
                    self.graph.graph['created_at'] = now
                if 'version' not in self.graph.graph:
                    self.graph.graph['version'] = self.version
                if 'last_modified' not in self.graph.graph:
                    self.graph.graph['last_modified'] = now
                if 'last_health_check' not in self.graph.graph:
                    self.graph.graph['last_health_check'] = now
                if 'health_status' not in self.graph.graph:
                    self.graph.graph['health_status'] = 'monitored'
                    
        except Exception as e:
            logger.error(f"Error adding metadata: {e}")
    
    def _save_check_results(self):
        """Save check results to file"""
        try:
            # Create reports directory
            os.makedirs('health_reports', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"health_reports/kg_health_{timestamp}.json"
            
            # Save results
            with open(filename, 'w') as f:
                json.dump({
                    'component': self.component_id,
                    'version': self.version,
                    'timestamp': datetime.now().isoformat(),
                    'health_status': self.health_status,
                    'fix_history': self.fix_history[-10:]  # Last 10 fixes
                }, f, indent=2)
                
            logger.info(f"✅ Health report saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status.copy()
    
    def stop(self):
        """Stop continuous monitoring"""
        self.running = False
        logger.info("🛑 Knowledge Graph Health Fixer stopped")

# Guard clause ensures code only runs when script is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 KNOWLEDGE GRAPH HEALTH FIXER (P0T1)")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Knowledge Graph Health Fixer')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=300, help='Check interval (seconds)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    fixer = KnowledgeGraphFixer()
    
    try:
        if args.once:
            print("\n📋 Running single health check...")
            result = fixer.run(continuous=False)
            print(f"\n📊 Health Status: {'✅ HEALTHY' if result.get('healthy') else '⚠️ ISSUES FOUND'}")
            print(json.dumps(result, indent=2))
        else:
            continuous = args.continuous or True
            print(f"\n🔄 Running continuous mode (interval: {args.interval}s)")
            print("Press Ctrl+C to stop\n")
            fixer.run(continuous=continuous, interval=args.interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        fixer.stop()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
