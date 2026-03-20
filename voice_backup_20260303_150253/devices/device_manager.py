"""Manage devices where DMAI can send results"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import logging
from datetime import datetime, timedelta
import platform
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceManager:
    """Track and communicate with your devices"""
    
    def __init__(self, registry_file='device_registry/devices.json'):
        self.registry_file = registry_file
        os.makedirs(os.path.dirname(registry_file), exist_ok=True)
        self.devices = self.load_devices()
        
    def load_devices(self):
        """Load known devices from registry"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            # Create default with current device
            default = {
                "devices": {
                    "current": self.detect_current_device()
                },
                "last_updated": datetime.now().isoformat()
            }
            self.save_devices(default)
            return default
    
    def save_devices(self, data=None):
        """Save device registry"""
        if data is None:
            data = self.devices
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def detect_current_device(self):
        """Detect information about current device"""
        hostname = socket.gethostname()
        system = platform.system()
        machine = platform.machine()
        
        return {
            "name": hostname,
            "type": self.classify_device(system, machine),
            "os": system,
            "last_seen": datetime.now().isoformat(),
            "capabilities": self.get_capabilities(system)
        }
    
    def classify_device(self, system, machine):
        """Classify device type"""
        if system == "Darwin":
            if "iPhone" in machine or "iPad" in machine:
                return "mobile"
            return "laptop"  # MacBook
        elif system == "Windows":
            return "laptop"
        elif system == "Linux":
            return "server"
        else:
            return "unknown"
    
    def get_capabilities(self, system):
        """What can this device do?"""
        capabilities = ["text"]
        
        if system == "Darwin":  # macOS
            capabilities.extend(["speaker", "notifications", "filesystem"])
        
        return capabilities
    
    def register_device(self, device_info=None):
        """Register a new device"""
        if device_info is None:
            device_info = self.detect_current_device()
        
        device_id = device_info.get('name', 'unknown')
        self.devices['devices'][device_id] = device_info
        self.devices['last_updated'] = datetime.now().isoformat()
        self.save_devices()
        logger.info(f"Registered device: {device_id}")
        return device_id
    
    def add_manual_device(self, name, device_type, capabilities=None):
        """Manually add a device like iPhone"""
        device_id = name.lower().replace(' ', '_')
        
        device_info = {
            "name": name,
            "type": device_type,
            "os": "iOS" if device_type == "mobile" else "Unknown",
            "last_seen": datetime.now().isoformat(),  # Use real date, not string
            "capabilities": capabilities or ["text", "speaker", "notifications"],
            "manual": True
        }
        
        self.devices['devices'][device_id] = device_info
        self.save_devices()
        logger.info(f"Added manual device: {name}")
        return device_id
    
    def get_available_devices(self):
        """Get list of devices that can receive output"""
        available = []
        for device_id, info in self.devices['devices'].items():
            # Check if device was seen recently
            last_seen_str = info.get('last_seen', '2000-01-01')
            try:
                last_seen = datetime.fromisoformat(last_seen_str)
                if datetime.now() - last_seen < timedelta(days=7):
                    available.append({
                        "id": device_id,
                        "name": info.get('name', device_id),
                        "type": info.get('type', 'unknown'),
                        "capabilities": info.get('capabilities', [])
                    })
            except (ValueError, TypeError):
                # If date parsing fails, include it anyway (for manually added)
                available.append({
                    "id": device_id,
                    "name": info.get('name', device_id),
                    "type": info.get('type', 'unknown'),
                    "capabilities": info.get('capabilities', [])
                })
        return available
    
    def update_last_seen(self, device_id):
        """Update when device was last active"""
        if device_id in self.devices['devices']:
            self.devices['devices'][device_id]['last_seen'] = datetime.now().isoformat()
            self.save_devices()
    
    def send_to_device(self, device_id, content, content_type="text"):
        """Send content to a specific device"""
        if device_id not in self.devices['devices']:
            logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices['devices'][device_id]
        
        logger.info(f"Sending to {device_id} ({device['type']}):")
        logger.info(f"Content type: {content_type}")
        logger.info(f"Content preview: {str(content)[:100]}...")
        
        print(f"\n📱 [SIMULATED] Sent to {device['name']}:")
        print(f"   {content[:60]}...")
        
        return True
    
    def ask_delivery_preference(self, content_preview=""):
        """Ask where user wants content sent"""
        devices = self.get_available_devices()
        
        if not devices:
            return None
        
        print("\n📱 Available devices:")
        for i, device in enumerate(devices, 1):
            print(f"   {i}. {device['name']} ({device['type']})")
        print(f"   {len(devices)+1}. Just tell me now")
        
        return devices

# Test the device manager
if __name__ == "__main__":
    dm = DeviceManager()
    
    print("Current device:", dm.detect_current_device())
    print("\nRegistering current device...")
    dm.register_device()
    
    print("\nAvailable devices:")
    for device in dm.get_available_devices():
        print(f"  - {device['name']} ({device['type']}): {device['capabilities']}")
    
    print("\nTesting send:")
    dm.send_to_device(dm.detect_current_device()['name'], 
                      "Your video is ready at https://dmai.io/video123")
