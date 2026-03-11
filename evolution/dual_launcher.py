#!/usr/bin/env python3
"""Launcher that runs both evolution and telegram in parallel with health endpoint"""
import threading
import time
import sys
import traceback
import os
import socket
import json
from pathlib import Path
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


# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evolution"))

print(f"📂 Project root: {PROJECT_ROOT}")
print(f"📂 Python path: {sys.path}")

# Global variables to track service status
evolution_running = False
telegram_running = False
evolution_pid = None
telegram_thread = None

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health' or self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'healthy',
                'service': 'dual_launcher',
                'evolution_running': evolution_running,
                'telegram_running': telegram_running,
                'evolution_pid': evolution_pid,
                'timestamp': str(__import__('datetime').datetime.now())
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress log messages
        pass

def run_health_server():
    """Run a simple health check server on port 9009"""
    server = HTTPServer(('localhost', 9009), HealthHandler)
    print(f"✅ Health server running on port 9009")
    try:
        server.serve_forever()
    except Exception as e:
        print(f"❌ Health server error: {e}")

def run_evolution():
    """Run the memory-safe evolution"""
    global evolution_running, evolution_pid
    print("🚀 Starting Evolution System...")
    try:
        evolution_pid = os.getpid()
        # Import evolution engine properly
        from continuous_advanced_evolution import EvolutionEngine
        engine = EvolutionEngine()
        evolution_running = True
        
        # Run in server mode
        if '--server' not in sys.argv:
            sys.argv.append('--server')
        
        # Import and run the server function
        from continuous_advanced_evolution import run_server
        run_server()
        
    except Exception as e:
        evolution_running = False
        print(f"❌ Evolution error: {e}")
        traceback.print_exc()

def run_telegram():
    """Run the telegram bot"""
    global telegram_running
    print("📱 Starting Telegram Bot...")
    try:
        # Import the adapter
        sys.path.insert(0, str(PROJECT_ROOT))
        from telegram_adapter import DMAITelegramReporter
        
        bot = DMAITelegramReporter()
        telegram_running = True
        print("✅ Telegram adapter initialized")
        
        # Run the bot (this will block in its own thread)
        bot.run()
        
    except Exception as e:
        telegram_running = False
        print(f"❌ Telegram error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Start health server in a background thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    print("✅ Health server started in background thread")
    
    # Start telegram in a background thread
    telegram_thread = threading.Thread(target=run_telegram, daemon=True)
    telegram_thread.start()
    print("✅ Telegram bot started in background thread")
    
    # Small delay to let services start
    time.sleep(2)
    
    # Run evolution in the main thread
    run_evolution()
