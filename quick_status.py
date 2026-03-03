#!/usr/bin/env python3
"""
Quick status check for all systems
"""
import requests
from datetime import datetime

print(f"\n{'='*60}")
print(f"🚀 DMAI SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# Check UI
try:
    r = requests.get('https://dmai-final.onrender.com', timeout=10)
    print(f"\n🌐 UI: {'✅ ONLINE' if r.status_code == 200 else '❌ OFFLINE'} (HTTP {r.status_code})")
except:
    print("\n🌐 UI: ❌ NOT REACHABLE")

# Check health endpoint
try:
    r = requests.get('https://dmai-final.onrender.com/health', timeout=10)
    if r.status_code == 200:
        data = r.json()
        print(f"📊 Health: ✅ OK - Gen {data.get('generation', '?')}")
    else:
        print(f"📊 Health: ⚠️ HTTP {r.status_code}")
except:
    print("📊 Health: ❌ Not available")

# Check local files
import os
from pathlib import Path

print(f"\n📁 Local Files:")
files = ['agi_orchestrator.py', 'capability_synthesizer.py', 'self_assessment.py']
for f in files:
    path = Path(f)
    if path.exists():
        size = path.stat().st_size
        print(f"  ✅ {f}: {size} bytes")
    else:
        print(f"  ❌ {f}: Missing")

print(f"\n{'='*60}")
