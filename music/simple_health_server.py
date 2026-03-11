#!/usr/bin/env python3
"""
Simple HTTP health check server for music_learner
Run this alongside music_learner to provide health checks
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import sys
import os
from pathlib import Path

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health' or self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Check if music_learner is running
            music_pid = None
            try:
                result = subprocess.run(['pgrep', '-f', 'music_learner.py'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    music_pid = result.stdout.strip()
            except:
                pass
            
            response = {
                'status': 'healthy' if music_pid else 'degraded',
                'service': 'music_learner_health',
                'music_learner_running': bool(music_pid),
                'music_learner_pid': music_pid,
                'timestamp': str(__import__('datetime').datetime.now())
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress log messages
        pass

if __name__ == '__main__':
    port = 9007  # Same port as music_learner
    server = HTTPServer(('localhost', port), HealthHandler)
    print(f"✅ Music learner health server running on port {port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down health server")
        server.shutdown()
