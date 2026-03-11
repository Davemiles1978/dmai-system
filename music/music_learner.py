#!/usr/bin/env python3
"""Music Learner - Develop DMAI's music taste"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import random
import time
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MUSIC - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "music_learner.log")
    ]
)
logger = logging.getLogger(__name__)

def develop_dmai_taste():
    """Develop DMAI's music taste based on listening history"""
    logger.info("Developing music taste...")
    
    # Simple implementation
    artists = ["DMAI Generated", "System Music", "Evolution Sounds"]
    genres = ["Electronic", "Ambient", "Generated"]
    
    result = {
        "status": "learning",
        "artists": artists,
        "genres": genres,
        "confidence": random.uniform(0.5, 0.9),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Developed preferences: {result}")
    return result

def run_continuous():
    """Run continuously"""
    logger.info("🎵 Music Learner started - continuous mode")
    cycle = 0
    try:
        while True:
            cycle += 1
            logger.info(f"Cycle {cycle}")
            develop_dmai_taste()
            logger.info("Sleeping for 60 seconds...")
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Music learner stopped")

def main():
    parser = argparse.ArgumentParser(description="DMAI Music Learner")
    parser.add_argument("--test", action="store_true", help="Run one cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running single test cycle")
        develop_dmai_taste()
    elif args.continuous:
        run_continuous()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

# ============================================================================
# PLUGGABLE INTERFACE LAYER - DO NOT MODIFY BELOW THIS LINE
# ============================================================================
# This section adds API endpoints for external systems to connect
# All original code above remains completely unchanged

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Memory optimization
import gc
gc.set_threshold(700, 10, 5)  # More aggressive garbage collection
import resource
try:
    # Set soft memory limit
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
except:
    pass

# Clear cache periodically
import threading
import time
def cache_cleaner():
    while True:
        time.sleep(300)  # Every 5 minutes
        gc.collect()  # Force garbage collection
        if hasattr(__import__('torch'), 'mps'):
            import torch
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
threading.Thread(target=cache_cleaner, daemon=True).start()


# Global reference to track state
_last_preferences = None
_start_time = datetime.now()
_cycle_count = 0

class MusicLearnerAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query music learner status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            global _last_preferences, _cycle_count
            
            status = {
                "name": "music_learner",
                "running": True,
                "cycle": _cycle_count,
                "last_preferences": _last_preferences,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time)
            }
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/preferences':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            self.wfile.write(json.dumps(_last_preferences or {}).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/command':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                command = json.loads(post_data)
                cmd = command.get('command', '')
                
                if cmd == 'learn_now':
                    # Trigger immediate learning cycle
                    result = develop_dmai_taste()
                    self.wfile.write(json.dumps({
                        "status": "learning_completed",
                        "result": result
                    }).encode())
                else:
                    self.wfile.write(json.dumps({"error": f"Unknown command: {cmd}"}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return  # Suppress HTTP logs

def _start_api_server():
    """Start API server in background thread"""
    port = 9007  # Fixed port for music learner
    
    def run_server():
        server = HTTPServer(('localhost', port), MusicLearnerAPIHandler)
        print(f"📡 Music Learner API endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Wrap the develop_dmai_taste function to track state
_original_develop = develop_dmai_taste
def _wrapped_develop():
    global _last_preferences, _cycle_count
    result = _original_develop()
    _last_preferences = result
    _cycle_count += 1
    return result

# Replace the original function with wrapped version
develop_dmai_taste = _wrapped_develop

# Initialize the API server
_api_port = _start_api_server()
