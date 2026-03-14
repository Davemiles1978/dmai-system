#!/usr/bin/env python3
"""
DMAI Learning Bridge - Connects API Harvester to DMAI's Evolution Engine
Every code found, every API key discovered feeds DMAI's learning
"""
import os
import sys
import json
import time
import sqlite3
import threading
from datetime import datetime
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DMAI_BRIDGE")

# Track service start time
start_time = datetime.now()

class HealthCheckHandler(BaseHTTPRequestHandler):
    """Health check endpoint for cron-job.org"""
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "dmai-learning-bridge",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - start_time)
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return  # Suppress logs

def start_health_server():
    """Start health check server on port specified by Render"""
    port = int(os.environ.get('PORT', 8080))
    try:
        server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
        logger.info(f"✅ Health check server running on port {port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")

class DMAILearningBridge:
    """
    Bridges the API Harvester with DMAI's evolution engine
    Every discovery becomes a learning opportunity for DMAI
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.harvester_db = os.path.join(self.base_dir, 'api-harvester', 'dmai_local.db')
        self.learning_dir = os.path.join(self.base_dir, 'learning')
        self.evolution_dir = os.path.join(self.base_dir, 'evolution')
        self.code_patterns_dir = os.path.join(self.learning_dir, 'code_patterns')
        self.api_patterns_dir = os.path.join(self.learning_dir, 'api_patterns')
        
        # Create directories
        for d in [self.code_patterns_dir, self.api_patterns_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Track processed items
        self.processed_keys = set()
        self.processed_code = set()
        
        logger.info("🚀 DMAI Learning Bridge initialized")
    
    def start(self):
        """Start monitoring harvester discoveries and feeding to DMAI"""
        logger.info("Starting DMAI Learning Bridge - Monitoring harvester discoveries")
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._monitor_new_keys, daemon=True),
            threading.Thread(target=self._feed_evolution_engine, daemon=True)
        ]
        
        for t in threads:
            t.start()
        
        # Keep running
        try:
            while True:
                time.sleep(60)
                self._log_stats()
        except KeyboardInterrupt:
            logger.info("Stopping DMAI Learning Bridge")
    
    def _monitor_new_keys(self):
        """Monitor for new API keys discovered"""
        last_check = datetime.now()
        
        while True:
            try:
                if os.path.exists(self.harvester_db):
                    conn = sqlite3.connect(self.harvester_db)
                    c = conn.cursor()
                    
                    # Check for new keys
                    c.execute("""
                        SELECT id, service, api_key, source, discovered_at 
                        FROM service_keys 
                        WHERE discovered_at > ?
                        ORDER BY discovered_at DESC
                    """, (last_check.isoformat(),))
                    
                    new_keys = c.fetchall()
                    conn.close()
                    
                    for key in new_keys:
                        key_id, service, api_key, source, discovered = key
                        
                        if key_id not in self.processed_keys:
                            self.processed_keys.add(key_id)
                            logger.info(f"📚 DMAI learning from new {service} API key")
                    
                    last_check = datetime.now()
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error monitoring keys: {e}")
                time.sleep(30)
    
    def _feed_evolution_engine(self):
        """Feed discoveries to DMAI's evolution engine"""
        while True:
            try:
                # Create evolution trigger file
                evolution_data = {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'harvester_bridge',
                    'new_keys_processed': len(self.processed_keys),
                    'learning_ready': True
                }
                
                # Signal evolution engine
                trigger_file = os.path.join(self.evolution_dir, 'harvester_learning.json')
                with open(trigger_file, 'w') as f:
                    json.dump(evolution_data, f, indent=2)
                
                # Create immediate evolution signal
                signal_file = os.path.join(self.evolution_dir, 'evolve_now.signal')
                with open(signal_file, 'w') as f:
                    f.write(f"Harvester learning ready at {datetime.now().isoformat()}")
                
                time.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error feeding evolution: {e}")
                time.sleep(30)
    
    def _log_stats(self):
        """Log bridge statistics"""
        logger.info("=" * 50)
        logger.info("DMAI LEARNING BRIDGE STATISTICS")
        logger.info(f"Keys processed: {len(self.processed_keys)}")
        logger.info("=" * 50)

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     DMAI LEARNING BRIDGE - Connecting Harvester to DMAI     ║
    ║     Every API key, every line of code feeds DMAI's mind     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Start health check server in background thread
    threading.Thread(target=start_health_server, daemon=True).start()
    
    bridge = DMAILearningBridge()
    bridge.start()
