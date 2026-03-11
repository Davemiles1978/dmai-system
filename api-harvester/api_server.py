#!/usr/bin/env python3
"""Simple API server for harvester to handle requests from evolution engine"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import logging
from datetime import datetime
from pathlib import Path

app = FastAPI(title="DMAI Harvester API")
logger = logging.getLogger("harvester_api")

class KeyRequest(BaseModel):
    system: str
    priority: int
    source: str
    timestamp: str

class KeyResponse(BaseModel):
    system: str
    status: str
    key_found: bool
    message: str

@app.get("/")
async def root():
    return {"status": "DMAI Harvester API running"}

@app.get("/status")
async def get_status():
    return {
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "harvester_pid": get_harvester_pid()
    }

@app.post("/request_key")
async def request_key(request: KeyRequest):
    logger.info(f"🔑 Key request received: {request.system} (priority: {request.priority})")
    
    # Log the request
    log_request(request)
    
    # Here we would trigger the harvester to prioritize this system
    return KeyResponse(
        system=request.system,
        status="queued",
        key_found=False,
        message=f"Searching for {request.system} API keys (priority {request.priority})"
    )

@app.get("/keys/found")
async def get_found_keys():
    """Return all found API keys"""
    keys_file = Path("keys/found_keys.json")
    if keys_file.exists():
        with open(keys_file, 'r') as f:
            return json.load(f)
    return {"keys": []}

def get_harvester_pid():
    """Get harvester process ID"""
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'harvester.py.*daemon'], 
                               capture_output=True, text=True)
        return result.stdout.strip() or None
    except:
        return None

def log_request(request):
    """Log API key request"""
    log_file = Path("logs/key_requests.log")
    log_file.parent.mkdir(exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "system": request.system,
            "priority": request.priority,
            "source": request.source
        }) + "\n")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)

# ============================================================================
# PLUGGABLE INTERFACE LAYER - DO NOT MODIFY BELOW THIS LINE
# ============================================================================
# This section adds additional API endpoints for external systems to connect
# All original code above remains completely unchanged

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests

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


# Global reference
_api_start_time = datetime.now()

class HarvesterAPIHandler(BaseHTTPRequestHandler):
    """Additional API for external systems to query harvester status"""
    
    def do_GET(self):
        if self.path == '/api_status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                "name": "harvester_api",
                "running": True,
                "fastapi_port": 8081,
                "uptime": str(datetime.now() - _api_start_time)
            }
            
            # Try to check if FastAPI is responding
            try:
                response = requests.get("http://localhost:8081/status", timeout=2)
                if response.status_code == 200:
                    status["fastapi_status"] = "responding"
                else:
                    status["fastapi_status"] = "error"
            except:
                status["fastapi_status"] = "unreachable"
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            stats = {
                "total_requests": 0,
                "keys_found": 0
            }
            
            # Count log entries
            log_file = Path("logs/key_requests.log")
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        stats["total_requests"] = sum(1 for _ in f)
                except:
                    pass
            
            # Count found keys
            keys_file = Path("keys/found_keys.json")
            if keys_file.exists():
                try:
                    with open(keys_file, 'r') as f:
                        keys = json.load(f)
                        stats["keys_found"] = len(keys.get("keys", []))
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
                
                if cmd == 'trigger_key_search':
                    system = command.get('system', 'general')
                    priority = command.get('priority', 5)
                    
                    # Forward to FastAPI
                    try:
                        response = requests.post(
                            "http://localhost:8081/request_key",
                            json={
                                "system": system,
                                "priority": priority,
                                "source": "internal_api",
                                "timestamp": datetime.now().isoformat()
                            },
                            timeout=5
                        )
                        if response.status_code == 200:
                            self.wfile.write(json.dumps({
                                "status": "forwarded",
                                "response": response.json()
                            }).encode())
                        else:
                            self.wfile.write(json.dumps({
                                "error": f"FastAPI returned {response.status_code}"
                            }).encode())
                    except Exception as e:
                        self.wfile.write(json.dumps({
                            "error": f"Failed to forward to FastAPI: {e}"
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

def _start_additional_api_server():
    """Start additional API server in background thread"""
    port = 9002  # Fixed port for harvester API
    
    def run_server():
        server = HTTPServer(('localhost', port), HarvesterAPIHandler)
        print(f"📡 Harvester API (secondary) endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the additional API server
_additional_api_port = _start_additional_api_server()
