#!/usr/bin/env python3
"""
Port Scanner - Finds available ports automatically
"""
import socket
import random
from contextlib import closing

class PortScanner:
    def __init__(self, start_port=8000, end_port=9000):
        self.start_port = start_port
        self.end_port = end_port
    
    def is_port_available(self, port):
        """Check if a port is available"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                sock.listen(1)
                return True
            except OSError:
                return False
    
    def find_free_port(self, preferred_port=8080):
        """Find a free port, trying preferred first"""
        # Try the preferred port first
        if self.is_port_available(preferred_port):
            return preferred_port
        
        print(f"⚠️  Port {preferred_port} is in use, scanning for free port...")
        
        # Try random ports in range
        available_ports = []
        for port in range(self.start_port, self.end_port + 1):
            if self.is_port_available(port):
                available_ports.append(port)
        
        if available_ports:
            chosen = random.choice(available_ports)
            print(f"✅ Found free port: {chosen}")
            return chosen
        
        raise Exception(f"No available ports in range {self.start_port}-{self.end_port}")
    
    def find_multiple_ports(self, count=2, preferred_start=8080):
        """Find multiple free ports"""
        ports = []
        current_pref = preferred_start
        
        for i in range(count):
            port = self.find_free_port(current_pref)
            ports.append(port)
            current_pref = port + 1  # Try next port for next one
        
        return ports

# Quick test
if __name__ == "__main__":
    scanner = PortScanner()
    web_port = scanner.find_free_port(8080)
    api_port = scanner.find_free_port(8889)
    print(f"📡 Web port: {web_port}")
    print(f"📡 API port: {api_port}")
