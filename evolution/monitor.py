#!/usr/bin/env python3
"""Simple monitoring for cloud evolution service"""
import requests
import time
from datetime import datetime

def check_service():
    """Check if cloud service is running"""
    try:
        # You'll need to set up a simple endpoint
        response = requests.get("https://your-dmai-service.com/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {datetime.now()}: {data['stage']} - {data['evolutions']} evolutions")
            return True
    except:
        print(f"❌ {datetime.now()}: Service unreachable")
        return False

if __name__ == "__main__":
    while True:
        check_service()
        time.sleep(300)  # Check every 5 minutes
