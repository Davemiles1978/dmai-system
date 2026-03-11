#!/usr/bin/env python3
"""
DMAI Core Systems Connector
Universal interface for connecting to any core service
"""

import requests
import json
import socket
from typing import Dict, Any, Optional, List
from datetime import datetime

class ServiceConnector:
    """Connect to any DMAI core service"""
    
    # Service port mapping (consistent across all systems)
    SERVICE_PORTS = {
        "harvester_daemon": 9001,
        "harvester_api": 9002,
        "evolution_engine": 9003,
        "book_reader": 9004,
        "web_researcher": 9005,
        "dark_researcher": 9006,
        "music_learner": 9007,
        "voice_service": 9008,
        "dual_launcher": 9009
    }
    
    def __init__(self, service_name: str, host: str = "localhost"):
        if service_name not in self.SERVICE_PORTS:
            raise ValueError(f"Unknown service: {service_name}. Available: {list(self.SERVICE_PORTS.keys())}")
        
        self.service_name = service_name
        self.port = self.SERVICE_PORTS[service_name]
        self.base_url = f"http://{host}:{self.port}"
        
    def status(self) -> Dict[str, Any]:
        """Get service status"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=2)
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}", "service": self.service_name}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection refused", "service": self.service_name}
        except Exception as e:
            return {"error": str(e), "service": self.service_name}
    
    def send_command(self, command: str, params: Dict = None) -> Dict:
        """Send command to service"""
        try:
            payload = {"command": command, "params": params or {}}
            response = requests.post(
                f"{self.base_url}/command", 
                json=payload,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def is_alive(self) -> bool:
        """Check if service is responsive"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', self.port))
            sock.close()
            return result == 0
        except:
            return False

# ============================================================================
# EASY ACCESS FUNCTIONS - New systems can just import these
# ============================================================================

def get_evolution_status():
    """Quick access to evolution engine"""
    conn = ServiceConnector("evolution_engine")
    return conn.status()

def get_harvester_status():
    """Get harvester daemon status"""
    conn = ServiceConnector("harvester_daemon")
    return conn.status()

def get_harvester_api_status():
    """Get harvester API status"""
    conn = ServiceConnector("harvester_api")
    return conn.status()

def get_book_reader_status():
    """Get book reader status"""
    conn = ServiceConnector("book_reader")
    return conn.status()

def get_web_researcher_status():
    """Get web researcher status"""
    conn = ServiceConnector("web_researcher")
    return conn.status()

def get_dark_researcher_status():
    """Get dark researcher status"""
    conn = ServiceConnector("dark_researcher")
    return conn.status()

def get_music_learner_status():
    """Get music learner status"""
    conn = ServiceConnector("music_learner")
    return conn.status()

def get_voice_status():
    """Get voice service status"""
    conn = ServiceConnector("voice_service")
    return conn.status()

def get_dual_launcher_status():
    """Get dual launcher status"""
    conn = ServiceConnector("dual_launcher")
    return conn.status()

def get_all_services_status() -> Dict[str, Any]:
    """Get status of all core services"""
    results = {}
    for service in ServiceConnector.SERVICE_PORTS.keys():
        conn = ServiceConnector(service)
        results[service] = conn.status()
    return results

def trigger_harvest():
    """Trigger immediate harvest cycle"""
    conn = ServiceConnector("harvester_daemon")
    return conn.send_command("harvest_now")

def trigger_research():
    """Trigger web research cycle"""
    conn = ServiceConnector("web_researcher")
    return conn.send_command("research_now")

def trigger_dark_research():
    """Trigger dark web research cycle"""
    conn = ServiceConnector("dark_researcher")
    return conn.send_command("research_now")

def trigger_learning():
    """Trigger music learning cycle"""
    conn = ServiceConnector("music_learner")
    return conn.send_command("learn_now")

def voice_say(text: str):
    """Make voice service speak"""
    conn = ServiceConnector("voice_service")
    return conn.send_command("say", {"text": text})

def voice_start_learning():
    """Start voice ambient learning"""
    conn = ServiceConnector("voice_service")
    return conn.send_command("start_learning")

def voice_stop_learning():
    """Stop voice ambient learning"""
    conn = ServiceConnector("voice_service")
    return conn.send_command("stop_learning")

def get_vocabulary_size():
    """Get current vocabulary size from voice service"""
    conn = ServiceConnector("voice_service")
    result = conn.send_command("get_vocabulary")
    return result.get("vocabulary_size", 0)

# Example usage in a new system:
"""
from core_connector import get_all_services_status, trigger_harvest, voice_say

# Check all services
status = get_all_services_status()
print(status)

# Trigger a harvest
trigger_harvest()

# Make voice speak
voice_say("Hello, I'm DMAI")
"""
