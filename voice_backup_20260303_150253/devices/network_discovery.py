"""Discover devices on local network"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import socket
import ipaddress
import subprocess
import threading
import time
import json
import os
from datetime import datetime

class NetworkDiscovery:
    """Find other devices on your network"""
    
    def __init__(self):
        self.found_devices = []
        
    def get_local_ip(self):
        """Get your local IP address"""
        try:
            # Create a temporary connection to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "192.168.1.1"
    
    def scan_port(self, ip, port, results):
        """Check if a port is open on an IP"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, port))
            if result == 0:
                # Port is open - likely a device
                try:
                    hostname = socket.gethostbyaddr(ip)[0]
                except:
                    hostname = "Unknown"
                
                device_type = self.identify_device(port)
                results.append({
                    "ip": ip,
                    "hostname": hostname,
                    "port": port,
                    "type": device_type,
                    "last_seen": datetime.now().isoformat()
                })
            sock.close()
        except:
            pass
    
    def identify_device(self, port):
        """Identify device type by common ports"""
        common_ports = {
            22: "SSH (Linux/Mac)",
            80: "Web Server",
            443: "Web Server (HTTPS)",
            139: "SMB (File Sharing)",
            445: "SMB (Windows)",
            548: "AFP (Mac File Sharing)",
            3689: "DAAP (iTunes)",
            5000: "AirPlay/HTTP",
            7000: "AirPlay",
            62078: "iOS Sync"
        }
        return common_ports.get(port, f"Unknown (port {port})")
    
    def scan_network(self):
        """Scan local network for devices"""
        local_ip = self.get_local_ip()
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        
        print(f"Scanning network: {network}")
        print("This may take a minute...")
        
        threads = []
        results = []
        
        # Common ports for mobile devices
        mobile_ports = [3689, 5000, 62078, 7000]
        
        for ip in network.hosts():
            ip_str = str(ip)
            if ip_str == local_ip:
                continue  # Skip yourself
                
            for port in mobile_ports:
                t = threading.Thread(target=self.scan_port, args=(ip_str, port, results))
                t.start()
                threads.append(t)
                
                # Limit concurrent threads
                if len(threads) > 50:
                    for t in threads:
                        t.join(timeout=1)
                    threads = []
        
        # Wait for remaining threads
        for t in threads:
            t.join(timeout=1)
        
        # Deduplicate by IP
        seen_ips = set()
        unique_results = []
        for device in results:
            if device['ip'] not in seen_ips:
                seen_ips.add(device['ip'])
                unique_results.append(device)
        
        return unique_results

# Quick test
if __name__ == "__main__":
    nd = NetworkDiscovery()
    print(f"Your IP: {nd.get_local_ip()}")
    
    devices = nd.scan_network()
    print(f"\nFound {len(devices)} potential devices:")
    for d in devices:
        print(f"  - {d['ip']} ({d['type']}) - {d['hostname']}")
