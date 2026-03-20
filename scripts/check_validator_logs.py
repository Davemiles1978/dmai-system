#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Check api-validator logs and status
"""

import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv('RENDER_API_KEY')
BASE_URL = "https://api.render.com/v1"

if not API_KEY:
    print("❌ RENDER_API_KEY not set in .env")
    exit(1)

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# Validator service ID from your output
validator_id = "crn-d6ltlushg0os73avo85g"

# Get service details
print("🔍 Checking api-validator status...")
response = requests.get(
    f"{BASE_URL}/services/{validator_id}",
    headers=headers
)

if response.status_code == 200:
    data = response.json()
    service = data.get('service', {})
    print(f"📊 Service: {service.get('name')}")
    print(f"📊 Status: {'Suspended' if service.get('suspended') else 'Active'}")
    print(f"📊 Region: {service.get('region')}")
    print(f"📊 Created: {service.get('createdAt')}")
    
    # Get recent logs
    print("\n📋 Recent logs:")
    logs_response = requests.get(
        f"{BASE_URL}/services/{validator_id}/logs?limit=50",
        headers=headers
    )
    
    if logs_response.status_code == 200:
        logs = logs_response.json()
        for log in logs.get('logs', [])[-10:]:
            print(f"  {log.get('timestamp')}: {log.get('message')}")
    else:
        print("  Could not fetch logs")
else:
    print(f"❌ Error: {response.status_code}")
