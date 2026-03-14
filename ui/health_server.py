#!/usr/bin/env python3
"""
Simple health check server for dmai-cloud-ui
Runs alongside the static file server to provide /health endpoint for cron-job.org
"""
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
from datetime import datetime

# Track start time
start_time = datetime.now()

class HealthHandler(BaseHTTPRequestHandler):
    """Health check endpoint for cron-job.org"""
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "healthy",
                "service": "dmai-cloud-ui",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - start_time)
            }
            
            self.wfile.write(json.dumps(response).encode())
        else:
            # For any other path, serve a simple message
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>DMAI Cloud UI</h1><p>Health check available at <a href='/health'>/health</a></p></body></html>")
    
    def log_message(self, format, *args):
        return  # Suppress logs

def run_server():
    """Run the health check server"""
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    print(f"✅ Health check server running on port {port}")
    print(f"   Health endpoint: http://0.0.0.0:{port}/health")
    server.serve_forever()

if __name__ == "__main__":
    print("🚀 Starting DMAI Cloud UI Health Server")
    run_server()
