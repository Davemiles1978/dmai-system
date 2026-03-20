#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Bridge between Local DMAI and Render Services
Connects to Render PostgreSQL and APIs
"""

import os
import json
import requests
import psycopg2
import psycopg2.extras
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - RENDER_BRIDGE - %(message)s')
logger = logging.getLogger(__name__)

class RenderBridge:
    """Connects local system to Render services"""
    
    def __init__(self):
        # Load Render connection details from environment
        self.db_url = os.getenv('RENDER_DATABASE_URL')
        self.harvester_url = os.getenv('RENDER_API_URL', 'https://api-harvester.onrender.com')
        self.validator_url = os.getenv('RENDER_VALIDATOR_URL', 'https://api-validator.onrender.com')
        
        # Test database connection
        self.test_connection()
    
    def test_connection(self):
        """Test connection to Render PostgreSQL"""
        if not self.db_url:
            logger.error("❌ RENDER_DATABASE_URL not set in .env")
            return False
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()
            logger.info(f"✅ Connected to Render PostgreSQL: {version[0][:50]}...")
            cur.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Render: {e}")
            return False
    
    def get_harvester_discoveries(self, limit=100):
        """Fetch API discoveries from Render harvester database"""
        if not self.db_url:
            return []
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            cur.execute("""
                SELECT id, source, discovery_type, name, url, description, 
                       raw_data, discovered_at, last_seen
                FROM api_discoveries
                ORDER BY last_seen DESC
                LIMIT %s
            """, (limit,))
            
            results = []
            for row in cur.fetchall():
                results.append(dict(row))
            
            cur.close()
            conn.close()
            
            logger.info(f"📊 Retrieved {len(results)} discoveries from Render")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error fetching discoveries: {e}")
            return []
    
    def get_endpoints_for_api(self, discovery_id):
        """Get endpoints for a specific API"""
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            cur.execute("""
                SELECT endpoint, method, auth_type
                FROM api_endpoints
                WHERE discovery_id = %s
                ORDER BY id
            """, (discovery_id,))
            
            results = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error fetching endpoints: {e}")
            return []
    
    def get_harvest_stats(self):
        """Get harvest statistics from Render"""
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("SELECT COUNT(*) FROM api_discoveries")
            total_discoveries = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM api_endpoints")
            total_endpoints = cur.fetchone()[0]
            
            cur.execute("""
                SELECT source, COUNT(*) 
                FROM api_discoveries 
                GROUP BY source 
                ORDER BY COUNT(*) DESC
            """)
            sources = cur.fetchall()
            
            cur.close()
            conn.close()
            
            stats = {
                'total_discoveries': total_discoveries,
                'total_endpoints': total_endpoints,
                'sources': dict(sources),
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Stats: {total_discoveries} discoveries, {total_endpoints} endpoints")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error fetching stats: {e}")
            return {}
    
    def sync_to_local(self, output_file="render_discoveries.json"):
        """Sync Render discoveries to local file"""
        discoveries = self.get_harvester_discoveries()
        
        if discoveries:
            with open(output_file, 'w') as f:
                json.dump({
                    'synced_at': datetime.now().isoformat(),
                    'count': len(discoveries),
                    'discoveries': discoveries
                }, f, indent=2, default=str)
            
            logger.info(f"✅ Synced {len(discoveries)} discoveries to {output_file}")
            
            # Also update evolution's knowledge
            self.update_evolution_knowledge(discoveries)
    
    def update_evolution_knowledge(self, discoveries):
        """Feed discoveries to evolution system"""
        # This would update the knowledge graph or evolution's research
        try:
            # Update knowledge graph if available
            from knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            
            for d in discoveries[:10]:  # Top 10
                # Add to knowledge graph
                node_id = f"api_{d['id']}"
                kg.graph.add_node(node_id, 
                                  type='api_discovery',
                                  name=d['name'],
                                  source=d['source'],
                                  url=d['url'])
                
                # Connect to existing nodes
                kg.graph.add_edge('evolution', node_id, 
                                 relation='can_learn_from')
            
            # Save knowledge graph
            kg.save()
            logger.info("✅ Updated knowledge graph with Render discoveries")
            
        except Exception as e:
            logger.error(f"❌ Failed to update knowledge graph: {e}")
    
    def run_continuous_sync(self, interval_minutes=15):
        """Continuously sync with Render"""
        logger.info("🚀 Starting continuous sync with Render")
        
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"🔄 Sync Cycle #{cycle}")
            
            try:
                # Get stats
                stats = self.get_harvest_stats()
                
                # Sync discoveries
                self.sync_to_local(f"render_discoveries_cycle_{cycle}.json")
                
                logger.info(f"⏰ Next sync in {interval_minutes} minutes")
                
                # Sleep
                for i in range(interval_minutes, 0, -5):
                    logger.info(f"⏳ {i} minutes until next sync")
                    time.sleep(60 * 5)
                    
            except KeyboardInterrupt:
                logger.info("👋 Sync stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in sync cycle: {e}")
                time.sleep(60)

if __name__ == "__main__":
    import time
    import sys
    
    bridge = RenderBridge()
    
    if "--continuous" in sys.argv:
        bridge.run_continuous_sync()
    elif "--stats" in sys.argv:
        stats = bridge.get_harvest_stats()
        print(json.dumps(stats, indent=2))
    else:
        bridge.sync_to_local()
