#!/usr/bin/env python3
"""Dark Web Researcher - DMAI - Deep web learning for AI innovations"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import json
import logging
import requests
from datetime import datetime
from core.paths import ROOT

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("DARK_RESEARCHER")

class DarkResearcher:

    # Memory-optimized chunked processing
    def process_in_chunks(self, data, chunk_size=100):
        """Process data in chunks to save memory"""
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_results = self._process_chunk(chunk)
            results.extend(chunk_results)
            
            # Free memory after each chunk
            del chunk
            import gc
            gc.collect()
        
        return results

    def __init__(self):
        self.root = ROOT
        self.research_dir = self.root / "data" / "research" / "dark"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.findings_file = self.research_dir / "findings.json"
        self.tor_available = self.check_tor()
        self.load_findings()
        logger.info(f"🌑 Dark Web Researcher initialized")
        logger.info(f"📂 Research dir: {self.research_dir}")
        logger.info(f"🔌 Tor available: {self.tor_available}")
        
    def check_tor(self):
        """Check if Tor is available"""
        try:
            # Test Tor proxy
            proxies = {
                'http': 'socks5h://127.0.0.1:9050',
                'https': 'socks5h://127.0.0.1:9050'
            }
            test_url = 'http://check.torproject.org'
            response = requests.get(test_url, proxies=proxies, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def load_findings(self):
        """Load existing research findings"""
        try:
            if self.findings_file.exists():
                with open(self.findings_file, 'r') as f:
                    self.findings = json.load(f)
                logger.info(f"📚 Loaded {len(self.findings)} previous findings")
            else:
                self.findings = {}
        except Exception as e:
            logger.error(f"Error loading findings: {e}")
            self.findings = {}
    
    def save_findings(self):
        """Save research findings"""
        try:
            with open(self.findings_file, 'w') as f:
                json.dump(self.findings, f, indent=2)
            logger.info(f"💾 Saved {len(self.findings)} findings")
        except Exception as e:
            logger.error(f"Error saving findings: {e}")
    
    def search_dark_web(self):
        """Search dark web for AI innovations (if Tor available)"""
        if not self.tor_available:
            logger.warning("⚠️ Tor not available - skipping dark web research")
            return []
        
        proxies = {
            'http': 'socks5h://127.0.0.1:9050',
            'https': 'socks5h://127.0.0.1:9050'
        }
        
        # Dark web sources (onion sites)
        sources = [
            {
                "name": "Darknet AI Forum",
                "url": "http://ai4dark.onion/forum",
                "type": "forum"
            },
            {
                "name": "Hidden AI Research",
                "url": "http://research7.onion/latest",
                "type": "research"
            },
            {
                "name": "Dark ML Repository",
                "url": "http://mlhub.onion/models",
                "type": "models"
            }
        ]
        
        findings = []
        for source in sources:
            try:
                logger.info(f"🔍 Researching dark source: {source['name']}")
                response = requests.get(source['url'], proxies=proxies, timeout=60)
                
                if response.status_code == 200:
                    finding = {
                        "source": source['name'],
                        "url": source['url'],
                        "type": source['type'],
                        "timestamp": datetime.now().isoformat(),
                        "data": response.text[:2000],
                        "status": "success"
                    }
                    findings.append(finding)
                    logger.info(f"✅ Found data from {source['name']}")
                else:
                    logger.warning(f"⚠️ {source['name']} returned {response.status_code}")
                
                time.sleep(5)  # Be extra polite to onion sites
                
            except Exception as e:
                logger.error(f"❌ Error researching {source['name']}: {e}")
                finding = {
                    "source": source['name'],
                    "url": source['url'],
                    "type": source['type'],
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "status": "failed"
                }
                findings.append(finding)
        
        return findings
    
    def search_clearnet_alternatives(self):
        """Search clearnet for AI innovations (fallback when Tor unavailable)"""
        sources = [
            {
                "name": "AI Security Papers",
                "url": "https://arxiv.org/list/cs.CR/recent",
                "type": "arxiv"
            },
            {
                "name": "Privacy Preserving ML",
                "url": "https://github.com/topics/federated-learning",
                "type": "github"
            },
            {
                "name": "Encrypted AI Models",
                "url": "https://huggingface.co/models?search=encrypted",
                "type": "huggingface"
            }
        ]
        
        findings = []
        for source in sources:
            try:
                logger.info(f"🔍 Researching clearnet: {source['name']}")
                headers = {'User-Agent': 'DMAI/1.0 (Research Bot; +http://dmai.local)'}
                response = requests.get(source['url'], timeout=30, headers=headers)
                
                if response.status_code == 200:
                    finding = {
                        "source": source['name'],
                        "url": source['url'],
                        "type": source['type'],
                        "timestamp": datetime.now().isoformat(),
                        "data": response.text[:2000],
                        "status": "success",
                        "note": "Clearnet alternative (Tor unavailable)"
                    }
                    findings.append(finding)
                    logger.info(f"✅ Found data from {source['name']}")
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error researching {source['name']}: {e}")
        
        return findings
    
    def run_once(self):
        """Run one research cycle"""
        logger.info("🔬 Starting dark web research cycle")
        
        # Try dark web first if Tor available
        if self.tor_available:
            findings = self.search_dark_web()
        else:
            logger.info("🌐 Using clearnet alternatives (Tor not available)")
            findings = self.search_clearnet_alternatives()
        
        # Store findings
        cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.findings[cycle_id] = {
            "timestamp": datetime.now().isoformat(),
            "tor_available": self.tor_available,
            "findings_count": len(findings),
            "findings": findings
        }
        
        # Keep only last 50 cycles
        if len(self.findings) > 50:
            oldest = sorted(self.findings.keys())[0]
            del self.findings[oldest]
        
        self.save_findings()
        
        logger.info(f"📊 Research complete: {len(findings)} sources found")
        return len(findings) > 0
    
    def run_continuous(self):
        """Run continuously"""
        logger.info("🌑 Dark Web Researcher started in continuous mode")
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"🔄 Research cycle {cycle}")
            self.run_once()
            logger.info("⏰ Next research in 1 hour")
            time.sleep(3600)  # 1 hour

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMAI Dark Web Researcher")
    parser.add_argument("--test", action="store_true", help="Run one cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    args = parser.parse_args()
    
    researcher = DarkResearcher()
    
    if args.test:
        logger.info("🧪 Running in TEST mode")
        researcher.run_once()
    elif args.continuous:
        researcher.run_continuous()
    else:
        parser.print_help()

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


# Global reference to the researcher instance
_dark_instance = None
_start_time = datetime.now()

class DarkResearcherAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query dark researcher status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get status
            status = {
                "name": "dark_researcher",
                "running": True,
                "findings_count": 0,
                "tor_available": False,
                "last_cycle": None,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time)
            }
            
            # Try to get real data if researcher instance exists
            if _dark_instance:
                try:
                    status["findings_count"] = len(getattr(_dark_instance, 'findings', {}))
                    status["tor_available"] = getattr(_dark_instance, 'tor_available', False)
                    if _dark_instance.findings:
                        last_key = sorted(_dark_instance.findings.keys())[-1]
                        status["last_cycle"] = _dark_instance.findings[last_key].get('timestamp')
                except:
                    pass
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/tor_status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            tor_status = {
                "available": False,
                "message": "Tor not available"
            }
            
            if _dark_instance:
                tor_status["available"] = _dark_instance.tor_available
                tor_status["message"] = "Tor is available" if _dark_instance.tor_available else "Tor not available"
            
            self.wfile.write(json.dumps(tor_status).encode())
            
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
                
                if cmd == 'research_now':
                    # Trigger immediate research
                    if _dark_instance:
                        result = _dark_instance.run_once()
                        self.wfile.write(json.dumps({
                            "status": "research_completed", 
                            "success": result
                        }).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Researcher not initialized"}).encode())
                elif cmd == 'check_tor':
                    if _dark_instance:
                        # Force recheck Tor
                        _dark_instance.tor_available = _dark_instance.check_tor()
                        self.wfile.write(json.dumps({
                            "tor_available": _dark_instance.tor_available
                        }).encode())
                    else:
                        self.wfile.write(json.dumps({"tor_available": False}).encode())
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
    port = 9006  # Fixed port for dark researcher
    
    def run_server():
        server = HTTPServer(('localhost', port), DarkResearcherAPIHandler)
        print(f"📡 Dark Researcher API endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the API server when this module is imported
_api_port = _start_api_server()

# Store reference to researcher instance when created
_original_init = DarkResearcher.__init__
def _wrapped_init(self, *args, **kwargs):
    global _dark_instance
    _original_init(self, *args, **kwargs)
    _dark_instance = self

DarkResearcher.__init__ = _wrapped_init
