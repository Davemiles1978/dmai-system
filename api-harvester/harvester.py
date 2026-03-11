#!/usr/bin/env python3
"""DMAI API Harvester - Finds and manages API keys"""

import os
import sys
import time
import json
import logging
import signal
import threading
import traceback
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('harvester.log')
    ]
)
logger = logging.getLogger("harvester")

class Harvester:
    def __init__(self):
        self.running = True
        self.cycle_count = 0
        self.start_time = datetime.now()
        
        # Load config
        self.config = self.load_config()
        
        # Initialize components
        self.github_scraper = None
        self.storage = None
        self.db = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("=" * 50)
        logger.info("DMAI API Harvester initializing...")
        logger.info("=" * 50)
        
        self.init_components()
        
    def load_config(self):
        """Load configuration"""
        config = {}
        config_file = Path("config.json")
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ Loaded config from {config_file.absolute()}")
                logger.info(f"📄 Config keys: {list(config.keys())}")
            except Exception as e:
                logger.error(f"❌ Error loading config: {e}")
        
        # Default config
        config.setdefault('github_token', os.environ.get('GITHUB_TOKEN'))
        config.setdefault('redis_host', 'localhost')
        config.setdefault('redis_port', 6379)
        config.setdefault('database_url', 'sqlite:///harvester.db')
        config.setdefault('check_interval', 3600)  # 1 hour
        
        # Log token status (without revealing the full token)
        if config.get('github_token'):
            token = config['github_token']
            logger.info(f"🔑 GitHub token found: {token[:4]}...{token[-4:] if len(token) > 8 else ''}")
        else:
            logger.warning("⚠️ No GitHub token found in config or environment")
        
        return config
    
    def init_components(self):
        """Initialize all scraper components"""
        try:
            # Initialize GitHub scraper
            from scrapers.github_scraper import GitHubScraper
            
            # Get token from config
            token = self.config.get('github_token')
            
            if token:
                logger.info(f"🔑 Passing token to GitHub scraper: {token[:4]}...{token[-4:]}")
            else:
                logger.warning("⚠️ No token available for GitHub scraper")
            
            # GitHubScraper expects a config object
            github_config = {
                'github_token': self.config.get('github_token'),
                'timeout': 30,
                'retry_count': 3
            }
            
            logger.info(f"📦 GitHub config: { {k: v[:4] + '...' if k == 'token' and v else v for k, v in github_config.items()} }")
            
            self.github_scraper = GitHubScraper(config=github_config)
            logger.info("✅ GitHub scraper initialized")
            
            # TODO: Initialize other scrapers as needed
            # self.reddit_scraper = RedditScraper(self.config)
            # self.darkweb_scraper = DarkWebScraper(self.config)
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🚨 SHUTDOWN SIGNAL RECEIVED: {signum}")
        logger.info(f"Stack trace at shutdown:")
        traceback.print_stack(frame)
        logger.info(f"Current cycle: #{self.cycle_count}")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down harvester...")
        self.running = False
        
        # Close connections
        if self.db:
            self.db.close()
        
        logger.info("Harvester shutdown complete")
        sys.exit(0)
    
    def run_cycle(self):
        """Run one harvesting cycle"""
        self.cycle_count += 1
        cycle_start = time.time()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"=== Starting harvesting cycle #{self.cycle_count} ===")
        logger.info(f"{'='*50}")
        
        total_keys = 0
        
        # Run GitHub scraper
        if self.github_scraper:
            logger.info("\n🔍 Starting GitHub scraping...")
            try:
                github_count = self.github_scraper.search_code()
                total_keys += github_count
                logger.info(f"✅ GitHub scraping complete: found {github_count} potential keys")
            except Exception as e:
                logger.error(f"GitHub scraper error: {e}")
                logger.error(traceback.format_exc())
        
        # TODO: Run other scrapers
        # if self.reddit_scraper:
        #     reddit_count = self.reddit_scraper.scrape()
        #     total_keys += reddit_count
        
        cycle_time = time.time() - cycle_start
        logger.info(f"\n{'='*50}")
        logger.info(f"=== Cycle #{self.cycle_count} complete ===")
        logger.info(f"Total keys found: {total_keys}")
        logger.info(f"Cycle time: {cycle_time:.1f} seconds")
        logger.info(f"{'='*50}\n")
        
        return total_keys
    
    def run(self):
        """Main run loop"""
        logger.info("=" * 50)
        logger.info(f"DMAI API Harvester started - Continuous Mode")
        logger.info(f"PID: {os.getpid()}")
        logger.info("=" * 50)
        
        while self.running:
            try:
                self.run_cycle()
                
                # Wait before next cycle
                wait_time = self.config.get('check_interval', 3600)
                logger.info(f"⏰ Waiting {wait_time} seconds until next cycle...")
                
                # Sleep in chunks to allow for graceful shutdown
                for _ in range(wait_time):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.shutdown()
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.error(traceback.format_exc())
                logger.info("Waiting 60 seconds before retry...")
                time.sleep(60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DMAI API Harvester")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--port", type=int, default=8081, help="API port (default: 8081)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    
    args = parser.parse_args()
    
    harvester = Harvester()
    
    if args.once:
        logger.info("Running single cycle")
        harvester.run_cycle()
    elif args.daemon:
        harvester.run()
    else:
        parser.print_help()

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


# Global reference to the harvester instance
_harvester_instance = None
_start_time = datetime.now()

class HarvesterAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query harvester status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                "name": "harvester_daemon",
                "running": True,
                "cycle": 0,
                "total_keys_found": 0,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time)
            }
            
            # Try to get real data if harvester instance exists
            if _harvester_instance:
                try:
                    status["cycle"] = getattr(_harvester_instance, 'cycle_count', 0)
                    # We don't store total keys, but could be added
                except:
                    pass
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            config_info = {
                "has_github_token": False,
                "check_interval": 3600
            }
            
            if _harvester_instance and hasattr(_harvester_instance, 'config'):
                config = _harvester_instance.config
                config_info["has_github_token"] = config.get('github_token') is not None
                config_info["check_interval"] = config.get('check_interval', 3600)
            
            self.wfile.write(json.dumps(config_info).encode())
            
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
                
                if cmd == 'harvest_now':
                    # Trigger immediate harvest cycle
                    if _harvester_instance:
                        # Run in a separate thread to not block the API
                        def run_harvest():
                            _harvester_instance.run_cycle()
                        thread = threading.Thread(target=run_harvest, daemon=True)
                        thread.start()
                        self.wfile.write(json.dumps({"status": "harvest_triggered"}).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Harvester not initialized"}).encode())
                        
                elif cmd == 'get_stats':
                    if _harvester_instance:
                        stats = {
                            "cycle": getattr(_harvester_instance, 'cycle_count', 0),
                            "running": getattr(_harvester_instance, 'running', False),
                            "uptime": str(datetime.now() - getattr(_harvester_instance, 'start_time', datetime.now()))
                        }
                        self.wfile.write(json.dumps(stats).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Harvester not initialized"}).encode())
                        
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
    port = 9001  # Fixed port for harvester daemon
    
    def run_server():
        server = HTTPServer(('localhost', port), HarvesterAPIHandler)
        print(f"📡 Harvester API endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the API server when this module is imported
_api_port = _start_api_server()

# Store reference to harvester instance when created
_original_init = Harvester.__init__
def _wrapped_init(self, *args, **kwargs):
    global _harvester_instance
    _original_init(self, *args, **kwargs)
    _harvester_instance = self

Harvester.__init__ = _wrapped_init
