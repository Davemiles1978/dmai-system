#!/usr/bin/env python3
"""Manually add iPhone to device registry"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devices.device_manager import DeviceManager
import json

def add_iphone():
    dm = DeviceManager()
    
    print("\n📱 Add your iPhone to DMAI")
    print("=" * 40)
    
    # Get iPhone details
    print("\nEnter your iPhone details:")
    name = input("iPhone name (e.g., 'David's iPhone'): ").strip()
    if not name:
        name = "iPhone"
    
    # Choose how to reach it
    print("\nHow should DMAI reach your iPhone?")
    print("1. Push notifications (requires app)")
    print("2. SMS (requires email-to-SMS gateway)")
    print("3. Just remember it for now (simulated)")
    
    choice = input("\nChoice (1-3): ").strip()
    
    device_info = {
        "name": name,
        "type": "mobile",
        "os": "iOS",
        "last_seen": "just added",
        "capabilities": ["text", "speaker", "notifications", "camera"],
        "delivery_method": "simulated"
    }
    
    if choice == "1":
        print("\n📲 For real push notifications, you'll need:")
        print("  - Pushover app (pushover.net) or")
        print("  - Custom DMAI iOS app")
        device_info["delivery_method"] = "push"
        device_info["push_token"] = "PENDING"
    elif choice == "2":
        carrier = input("Carrier (att, verizon, tmobile, etc): ").strip()
        phone = input("Phone number: ").strip()
        device_info["delivery_method"] = "sms"
        device_info["carrier"] = carrier
        device_info["phone"] = phone
    else:
        device_info["delivery_method"] = "simulated"
    
    # Register the device
    device_id = f"iphone_{name.lower().replace(' ', '_')}"
    dm.devices['devices'][device_id] = device_info
    dm.save_devices()
    
    print(f"\n✅ Added {name} to your devices!")
    print("\nAvailable devices now:")
    for dev_id, info in dm.devices['devices'].items():
        print(f"  - {info.get('name', dev_id)} ({info.get('type', 'unknown')})")

if __name__ == "__main__":
    add_iphone()
