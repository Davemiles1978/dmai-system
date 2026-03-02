#!/usr/bin/env python3
"""
Simple HTTP server for AGI Evolution System health checks
"""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Get generation from checkpoint if possible
            gen = 4
            checkpoint_dir = Path('/var/data/shared_checkpoints')
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('gen_*'))
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    try:
                        gen = latest.name.split('_')[1]
                    except:
                        pass
            
            health_data = {
                'status': 'healthy',
                'generation': int(gen),
                'service': 'agi-evolution-system',
                'timestamp': str(datetime.now().isoformat())
            }
            self.wfile.write(json.dumps(health_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        # Suppress log messages
        pass

def run_health_server(port=8080):
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    print(f"🌐 Health server running on port {port}")
    server.serve_forever()

if __name__ == '__main__':
    from datetime import datetime
    run_health_server()
