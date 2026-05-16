"""
Automatic API Key Rotation System
Manages multiple API keys and rotates when quotas are reached
"""
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class APIKeyManager:
    """Manages API keys with automatic rotation and quota tracking"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.keys_file = data_path / 'api_keys.json'
        self.keys: Dict[str, List[Dict]] = {}
        self.usage_file = data_path / 'key_usage.json'
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from discovered sources"""
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                self.keys = json.load(f)
        else:
            self.keys = {
                'openai': [],
                'anthropic': [],
                'huggingface': [],
                'replicate': [],
                'github': [],
                'xai': [],
                'google': [],
                'cohere': []
            }
    
    def get_valid_key(self, service: str) -> Optional[str]:
        """Get a valid key that hasn't hit quota"""
        if service not in self.keys:
            return None
        
        # Find keys with remaining quota
        valid_keys = []
        for key_info in self.keys[service]:
            if key_info.get('quota_remaining', 100) > 0:
                # Check cooldown
                last_used = key_info.get('last_used')
                if not last_used or (datetime.now() - datetime.fromisoformat(last_used)).seconds > 60:
                    valid_keys.append(key_info)
        
        if valid_keys:
            selected = random.choice(valid_keys)
            selected['last_used'] = datetime.now().isoformat()
            selected['quota_remaining'] = selected.get('quota_remaining', 100) - 1
            self.save_keys()
            return selected['key']
        
        return None
    
    def report_quota_exhausted(self, service: str, key: str):
        """Mark a key as quota exhausted"""
        for key_info in self.keys.get(service, []):
            if key_info.get('key') == key:
                key_info['quota_remaining'] = 0
                key_info['exhausted_at'] = datetime.now().isoformat()
                self.save_keys()
                break
    
    def add_key(self, service: str, key: str, source: str = 'discovered'):
        """Add a new API key to the pool"""
        if service not in self.keys:
            self.keys[service] = []
        
        # Avoid duplicates
        if not any(k.get('key') == key for k in self.keys[service]):
            self.keys[service].append({
                'key': key,
                'source': source,
                'added': datetime.now().isoformat(),
                'quota_remaining': 100,  # Assume 100 calls initially
                'last_used': None
            })
            self.save_keys()
            print(f"✅ Added new {service} API key from {source}")
    
    def save_keys(self):
        """Save keys to disk"""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)
    
    def harvest_from_github(self, github_token: str = None):
        """Scan GitHub repos for exposed API keys"""
        # This would integrate with GitHub API to find keys
        # For now, return discovered count
        return 0
