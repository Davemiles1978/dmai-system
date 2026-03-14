#!/usr/bin/env python3
"""
DMAI Learning Bridge - Connects API Harvester to DMAI's Evolution Engine
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DMAI_BRIDGE")

class DMAILearningBridge:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.harvester_db = os.path.join(self.base_dir, 'api-harvester', 'dmai_local.db')
        self.learning_dir = os.path.join(self.base_dir, 'learning')
        self.evolution_dir = os.path.join(self.base_dir, 'evolution')
        
        os.makedirs(self.learning_dir, exist_ok=True)
        os.makedirs(self.evolution_dir, exist_ok=True)
        
        self.processed_keys = set()
        logger.info("🚀 DMAI Learning Bridge initialized")
    
    def start(self):
        logger.info("Starting DMAI Learning Bridge - Monitoring harvester discoveries")
        while True:
            self.check_for_new_keys()
            time.sleep(60)
    
    def check_for_new_keys(self):
        try:
            if os.path.exists(self.harvester_db):
                conn = sqlite3.connect(self.harvester_db)
                c = conn.cursor()
                c.execute("SELECT id, service, api_key, source, discovered_at FROM service_keys ORDER BY discovered_at DESC LIMIT 10")
                new_keys = c.fetchall()
                conn.close()
                
                for key in new_keys:
                    key_id = key[0]
                    if key_id not in self.processed_keys:
                        self.processed_keys.add(key_id)
                        logger.info(f"📚 DMAI learning from new {key[1]} API key")
                        
                        # Trigger evolution
                        trigger_file = os.path.join(self.evolution_dir, 'evolve_now.signal')
                        with open(trigger_file, 'w') as f:
                            f.write(f"New key learned at {datetime.now().isoformat()}")
        except Exception as e:
            logger.error(f"Error checking keys: {e}")

if __name__ == "__main__":
    print("🚀 DMAI Learning Bridge Starting...")
    bridge = DMAILearningBridge()
    bridge.start()
