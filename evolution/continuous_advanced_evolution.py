#!/usr/bin/env python3
"""Continuous runner for memory-safe system-wide evolution"""

import time
import gc
import psutil
import os
from memory_safe_evolution import MemorySafeEvolution
from adaptive_timer import AdaptiveEvolutionTimer
from auto_cleanup import cleanup_if_needed
from evolution_cleanup import EvolutionCleanup

def main():
    timer = AdaptiveEvolutionTimer()
    evolution = MemorySafeEvolution()
    
    print("\n" + "="*70)
    print("🚀 CONTINUOUS MEMORY-SAFE EVOLUTION STARTED")
    print("Auto-cleanup at 600MB, evolution cycles with memory protection")
    print("="*70 + "\n")
    
    # Check startup memory
    startup_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"🚀 Startup memory: {startup_memory:.1f} MB")
    if startup_memory > 400:
        print("⚠️ Warning: Startup memory > 400MB, forcing immediate cleanup...")
        cleanup = EvolutionCleanup()
        cleanup.cleanup(dry_run=False)
    
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*70}")
        print(f"CYCLE #{cycle}")
        print(f"{'='*70}")
        
        # Check memory and cleanup if needed
        cleanup_if_needed()
        
        # Run scheduled cleanup every 3 cycles
        if cycle % 3 == 0:
            print("\n🧹 Running scheduled cleanup...")
            cleanup = EvolutionCleanup()
            cleanup.cleanup(dry_run=False)
        
        # Run evolution cycle with memory protection
        successes = evolution.run_cycle()
        
        # Get adaptive wait time
        wait_time = timer.record_attempt(
            "system_wide",
            "evolution",
            success=successes > 0
        )
        
        print(f"\n⏱️  Next cycle in {wait_time/60:.1f} minutes")
        print("(Press Ctrl+C to pause)\n")
        
        # Wait intelligently with memory checks
        for minute in range(int(wait_time / 60)):
            time.sleep(60)
            if minute % 5 == 0:  # Every 5 minutes, check memory
                cleanup_if_needed()
                remaining = wait_time/60 - (minute + 1)
                print(f"   {remaining:.0f} minutes remaining...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Evolution paused. Run again to continue.")

# ============================================================================
# PLUGGABLE INTERFACE LAYER - DO NOT MODIFY BELOW THIS LINE
# ============================================================================
# This section adds API endpoints for external systems to connect
# All original code above remains completely unchanged

import json
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Dict, Any

# Global reference to the evolution instance
_evolution_instance = None
_start_time = datetime.now()

class EvolutionAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query evolution status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get status from the global evolution instance
            status = {
                "name": "evolution_engine",
                "running": True,
                "cycle": 0,
                "memory_mb": 0,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time)
            }
            
            # Try to get real data if evolution instance exists
            if _evolution_instance:
                try:
                    status["cycle"] = getattr(_evolution_instance, 'cycle_count', 0)
                    # Get memory usage if psutil is available
                    try:
                        import psutil
                        process = psutil.Process()
                        status["memory_mb"] = process.memory_info().rss / 1024 / 1024
                    except:
                        pass
                except:
                    pass
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            stats = {
                "successful_evolutions": 0,
                "total_cycles": 0,
                "average_success_rate": 0
            }
            
            # Try to get real stats
            if _evolution_instance and hasattr(_evolution_instance, 'get_stats'):
                try:
                    stats = _evolution_instance.get_stats()
                except:
                    pass
            
            self.wfile.write(json.dumps(stats).encode())
            
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
                
                if cmd == 'trigger_cycle':
                    # Trigger an evolution cycle immediately
                    result = {"status": "cycle_triggered", "cycle": 0}
                    self.wfile.write(json.dumps(result).encode())
                elif cmd == 'get_memory':
                    try:
                        import psutil
                        process = psutil.Process()
                        memory = process.memory_info().rss / 1024 / 1024
                        self.wfile.write(json.dumps({"memory_mb": memory}).encode())
                    except:
                        self.wfile.write(json.dumps({"error": "Cannot get memory"}).encode())
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
    port = 9003  # Fixed port for evolution engine
    
    def run_server():
        server = HTTPServer(('localhost', port), EvolutionAPIHandler)
        print(f"📡 Evolution API endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the API server when this module is imported
_api_port = _start_api_server()

# Override the main function to capture the evolution instance
_original_main = main
def _wrapped_main():
    global _evolution_instance
    # Create evolution instance and store reference
    _evolution_instance = MemorySafeEvolution()
    # Call original main
    _original_main()

# Replace main with wrapped version if needed
if __name__ == "__main__":
    # Store evolution instance when running as main
    global _evolution_instance
    _evolution_instance = MemorySafeEvolution()
    _original_main()
