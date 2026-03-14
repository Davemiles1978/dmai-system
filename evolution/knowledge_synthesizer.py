#!/usr/bin/env python3
"""
DMAI Knowledge Synthesizer - Central hub that processes ALL learning
Transforms raw data into knowledge and triggers evolution
Runs 24/7 on Render
"""
import os
import sys
import json
import time
import glob
import sqlite3
import threading
from datetime import datetime
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KNOWLEDGE_SYNTHESIZER")

# Track service start time
start_time = datetime.now()

class HealthHandler(BaseHTTPRequestHandler):
    """Health check endpoint for cron-job.org"""
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "knowledge-synthesizer",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - start_time)
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return  # Suppress logs

def run_health_server():
    """Run health check server on port specified by Render"""
    port = int(os.environ.get('PORT', 8081))
    try:
        server = HTTPServer(('0.0.0.0', port), HealthHandler)
        logger.info(f"✅ Health check server running on port {port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Health server error: {e}")

class KnowledgeSynthesizer:
    """
    Synthesizes all learning sources into actionable knowledge
    Triggers DMAI evolution when new patterns are discovered
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Learning sources
        self.learning_dirs = {
            'api_keys': os.path.join(self.base_dir, 'api-harvester', 'dmai_local.db'),
            'terminal_sessions': os.path.join(self.base_dir, 'learning', 'terminal_sessions'),
            'commands': os.path.join(self.base_dir, 'learning', 'commands'),
            'files_changed': os.path.join(self.base_dir, 'learning', 'files_changed'),
            'domain_knowledge': os.path.join(self.base_dir, 'learning', 'domain_knowledge'),
            'code_patterns': os.path.join(self.base_dir, 'learning', 'code_patterns'),
            'api_patterns': os.path.join(self.base_dir, 'learning', 'api_patterns'),
            'research': os.path.join(self.base_dir, 'research_data')
        }
        
        # Evolution engine directories
        self.evolution_dir = os.path.join(self.base_dir, 'evolution')
        os.makedirs(self.evolution_dir, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            'total_learning_items': 0,
            'last_synthesis': None,
            'evolution_triggers': 0,
            'patterns_found': 0
        }
        
        logger.info("=" * 60)
        logger.info("🧠 DMAI KNOWLEDGE SYNTHESIZER INITIALIZED")
        logger.info(f"Monitoring {len(self.learning_dirs)} learning sources")
        logger.info("=" * 60)
    
    def start(self):
        """Main loop - runs continuously"""
        logger.info("🚀 Starting Knowledge Synthesizer - Processing all learning sources")
        
        while True:
            try:
                # Synthesize all knowledge
                synthesis = self.synthesize_all()
                
                # Generate insights
                insights = self.generate_insights(synthesis)
                
                # Trigger evolution if significant learning occurred
                if insights['should_evolve']:
                    self.trigger_evolution(insights)
                
                # Log progress
                self.log_stats(synthesis)
                
                # Wait before next cycle
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in synthesis cycle: {e}")
                time.sleep(60)
    
    def synthesize_all(self):
        """Gather and count all learning materials"""
        synthesis = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'total_items': 0,
            'new_items_since_last': 0
        }
        
        # Check each learning source
        for source_name, source_path in self.learning_dirs.items():
            count = 0
            
            if source_name == 'api_keys':
                # Count keys in database
                if os.path.exists(source_path):
                    try:
                        conn = sqlite3.connect(source_path)
                        c = conn.cursor()
                        c.execute("SELECT COUNT(*) FROM service_keys")
                        count = c.fetchone()[0]
                        conn.close()
                    except:
                        count = 0
            else:
                # Count files in directory
                if os.path.exists(source_path):
                    files = glob.glob(os.path.join(source_path, '**', '*'), recursive=True)
                    count = len([f for f in files if os.path.isfile(f)])
            
            synthesis['sources'][source_name] = count
            synthesis['total_items'] += count
        
        # Calculate new items (simple diff from last run)
        if self.stats['last_synthesis']:
            synthesis['new_items_since_last'] = max(0, synthesis['total_items'] - self.stats['total_learning_items'])
        
        self.stats['total_learning_items'] = synthesis['total_items']
        self.stats['last_synthesis'] = synthesis['timestamp']
        
        return synthesis
    
    def generate_insights(self, synthesis):
        """Generate insights from the synthesized data"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'should_evolve': False,
            'reason': '',
            'metrics': {},
            'patterns': []
        }
        
        # Check for significant learning
        if synthesis['new_items_since_last'] > 10:
            insights['should_evolve'] = True
            insights['reason'] = f"High volume of new learning: {synthesis['new_items_since_last']} items"
        
        # Check for specific patterns
        if synthesis['sources'].get('api_keys', 0) > self.stats.get('last_key_count', 0):
            insights['should_evolve'] = True
            insights['reason'] = "New API keys discovered"
        
        # Store metrics
        insights['metrics'] = {
            'learning_velocity': synthesis['new_items_since_last'] / 5 if synthesis['new_items_since_last'] > 0 else 0,
            'total_knowledge_base': synthesis['total_items'],
            'sources_active': sum(1 for v in synthesis['sources'].values() if v > 0)
        }
        
        self.stats['last_key_count'] = synthesis['sources'].get('api_keys', 0)
        
        return insights
    
    def trigger_evolution(self, insights):
        """Trigger DMAI evolution with new knowledge"""
        try:
            self.stats['evolution_triggers'] += 1
            
            # Create evolution trigger file
            trigger_data = {
                'timestamp': datetime.now().isoformat(),
                'source': 'knowledge_synthesizer',
                'reason': insights['reason'],
                'metrics': insights['metrics'],
                'trigger_number': self.stats['evolution_triggers']
            }
            
            # Write to evolution trigger file
            trigger_file = os.path.join(self.evolution_dir, 'evolve_now.signal')
            with open(trigger_file, 'w') as f:
                json.dump(trigger_data, f, indent=2)
            
            # Also save insights for evolution engine
            insights_file = os.path.join(self.evolution_dir, 'synthesized_insights.json')
            with open(insights_file, 'w') as f:
                json.dump(insights, f, indent=2)
            
            logger.info(f"🎯 Evolution triggered: {insights['reason']}")
            
        except Exception as e:
            logger.error(f"Failed to trigger evolution: {e}")
    
    def log_stats(self, synthesis):
        """Log current statistics"""
        logger.info("=" * 50)
        logger.info("📊 KNOWLEDGE SYNTHESIZER STATISTICS")
        logger.info(f"Total learning items: {synthesis['total_items']}")
        logger.info(f"New items: {synthesis['new_items_since_last']}")
        logger.info(f"Evolution triggers: {self.stats['evolution_triggers']}")
        logger.info("\n📚 Learning Sources:")
        for source, count in synthesis['sources'].items():
            if count > 0:
                logger.info(f"  • {source}: {count} items")
        logger.info("=" * 50)

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     DMAI KNOWLEDGE SYNTHESIZER - Central Learning Hub       ║
    ║     Transforms ALL data into knowledge for DMAI's mind      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Start health check server
    threading.Thread(target=run_health_server, daemon=True).start()
    
    synthesizer = KnowledgeSynthesizer()
    synthesizer.start()
