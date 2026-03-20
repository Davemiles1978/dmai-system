#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Query Render API to get service information
"""

import os
import json
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - RENDER_API - %(message)s')
logger = logging.getLogger(__name__)

class RenderAPI:
    def __init__(self):
        self.api_key = os.getenv('RENDER_API_KEY')
        self.base_url = "https://api.render.com/v1"
        
        if not self.api_key:
            logger.error("❌ RENDER_API_KEY not set in .env")
    
    def list_services(self):
        """List all services"""
        if not self.api_key:
            return []
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/services",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            services = response.json()
            
            logger.info(f"✅ Found {len(services)} services")
            return services
            
        except Exception as e:
            logger.error(f"❌ Failed to list services: {e}")
            return []
    
    def list_databases(self):
        """List all databases"""
        if not self.api_key:
            return []
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/databases",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            dbs = response.json()
            
            logger.info(f"✅ Found {len(dbs)} databases")
            return dbs
            
        except Exception as e:
            logger.error(f"❌ Failed to list databases: {e}")
            return []

if __name__ == "__main__":
    import sys
    
    api = RenderAPI()
    
    if "--services" in sys.argv:
        services = api.list_services()
        print("\n📊 Render Services:")
        for svc in services:
            name = svc.get('service', {}).get('name', 'Unknown')
            svc_id = svc.get('service', {}).get('id', 'Unknown')
            status = 'active' if not svc.get('service', {}).get('suspended') else 'suspended'
            print(f"  • {name} (ID: {svc_id}) - {status}")
    
    elif "--databases" in sys.argv:
        databases = api.list_databases()
        print("\n📀 Render Databases:")
        for db in databases:
            name = db.get('database', {}).get('name', 'Unknown')
            region = db.get('database', {}).get('region', 'Unknown')
            version = db.get('database', {}).get('version', 'Unknown')
            print(f"  • {name} - {region} (PostgreSQL {version})")
    
    else:
        print("Usage: python render_api_query.py --services")
        print("       python render_api_query.py --databases")
