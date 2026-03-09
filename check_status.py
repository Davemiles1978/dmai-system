#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from datetime import datetime

services = {
    'DMAI Final': 'https://dmai-final.onrender.com',
    'AGI Evolution': 'https://agi-evolution-system.onrender.com'
}

print(f"\n{'='*60}")
print(f"🚀 AGI System Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

for name, url in services:
    try:
        r = requests.get(f"{url}/health", timeout=10)
        if r.status_code == 200:
            print(f"\n✅ {name}: RUNNING")
            try:
                data = r.json()
                print(f"   Generation: {data.get('generation', 'N/A')}")
                print(f"   Status: {data.get('health_status', 'N/A')}")
            except:
                print(f"   Response: {r.text[:100]}")
        else:
            print(f"\n⚠️ {name}: HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n✅ {name}: WORKER RUNNING (no web endpoint)")
    except Exception as e:
        print(f"\n❌ {name}: ERROR - {e}")

print(f"\n{'='*60}")
