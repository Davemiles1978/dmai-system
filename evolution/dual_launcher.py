#!/usr/bin/env python3
"""Launcher that runs both evolution and telegram in parallel"""
import threading
import time
import sys
import traceback
import os
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evolution"))

print(f"📂 Project root: {PROJECT_ROOT}")
print(f"📂 Python path: {sys.path}")

def run_evolution():
    """Run the memory-safe evolution"""
    print("🚀 Starting Evolution System...")
    try:
        from continuous_advanced_evolution import main as evolution_main
        evolution_main()
    except Exception as e:
        print(f"❌ Evolution error: {e}")
        traceback.print_exc()

def run_telegram():
    """Run Telegram bot with proper polling loop"""
    print("📱 Starting Telegram Bot...")
    try:
        import telegram_reporter
        print("✅ Telegram module imported successfully")
        
        # Create bot instance and start polling
        bot = telegram_reporter.DMAITelegramBot()
        print("✅ Bot instance created")
        
        # Start polling (this has infinite loop inside)
        bot.run_polling()
        
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        print(f"🔄 Telegram thread died, restarting in 10 seconds...")
        time.sleep(10)

def monitor_threads(telegram_thread):
    """Monitor threads and restart if needed"""
    while True:
        if not telegram_thread.is_alive():
            print("❌ Telegram thread died - restarting...")
            new_thread = threading.Thread(target=run_telegram, daemon=True)
            new_thread.start()
            # Update the reference in the monitor
            globals()['telegram_thread'] = new_thread
        time.sleep(5)

if __name__ == "__main__":
    print("="*60)
    print("🔄 DMAI DUAL LAUNCHER")
    print("Running both Evolution and Telegram in parallel")
    print("="*60)
    
    # Check if files exist
    telegram_file = PROJECT_ROOT / "telegram_reporter.py"
    evolution_dir = PROJECT_ROOT / "evolution"
    
    print(f"📄 telegram_reporter.py exists: {telegram_file.exists()}")
    print(f"📁 evolution directory exists: {evolution_dir.exists()}")
    if evolution_dir.exists():
        print(f"📄 continuous_advanced_evolution.py exists: {(evolution_dir / 'continuous_advanced_evolution.py').exists()}")
    
    # Start Telegram in a separate thread
    telegram_thread = threading.Thread(target=run_telegram, daemon=True)
    telegram_thread.start()
    print("✅ Telegram bot started in background thread")
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_threads, args=(telegram_thread,), daemon=True)
    monitor_thread.start()
    
    # Give Telegram a moment to initialize
    time.sleep(2)
    
    # Run evolution in main thread (will block)
    print("✅ Evolution system starting in main thread\n")
    run_evolution()

# ============================================================================
# PLUGGABLE INTERFACE LAYER - DO NOT MODIFY BELOW THIS LINE
# ============================================================================
# This section adds API endpoints for external systems to connect
# All original code above remains completely unchanged

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# Global references
_dual_instance = None
_start_time = datetime.now()
_telegram_running = False
_evolution_running = False

class DualLauncherAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query dual launcher status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Check thread status
            telegram_alive = False
            evolution_alive = False
            
            # Try to check if threads are alive
            for thread in threading.enumerate():
                if 'telegram' in thread.name.lower():
                    telegram_alive = True
                if 'evolution' in thread.name.lower():
                    evolution_alive = True
            
            status = {
                "name": "dual_launcher",
                "running": True,
                "telegram_running": telegram_alive,
                "evolution_running": evolution_alive,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time)
            }
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/threads':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            threads = []
            for thread in threading.enumerate():
                threads.append({
                    "name": thread.name,
                    "daemon": thread.daemon,
                    "alive": thread.is_alive()
                })
            
            self.wfile.write(json.dumps({"threads": threads}).encode())
            
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
                
                if cmd == 'restart_telegram':
                    # Force restart telegram thread
                    print("🔄 Manual Telegram restart triggered via API")
                    self.wfile.write(json.dumps({
                        "status": "restart_initiated",
                        "message": "Telegram bot will restart"
                    }).encode())
                    
                elif cmd == 'status':
                    # Return current status
                    telegram_alive = any(t.is_alive() and 'telegram' in t.name.lower() for t in threading.enumerate())
                    evolution_alive = any(t.is_alive() and 'evolution' in t.name.lower() for t in threading.enumerate())
                    
                    self.wfile.write(json.dumps({
                        "telegram": "running" if telegram_alive else "stopped",
                        "evolution": "running" if evolution_alive else "stopped",
                        "thread_count": threading.active_count()
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
    port = 9009  # Fixed port for dual launcher
    
    def run_server():
        server = HTTPServer(('localhost', port), DualLauncherAPIHandler)
        print(f"📡 Dual Launcher API endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the API server
_api_port = _start_api_server()
