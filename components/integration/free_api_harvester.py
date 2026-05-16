#!/usr/bin/env python3
"""
DMAI Free API Key Harvester
============================
Actively scrapes public sources for working free-tier AI API keys.
Sources: GitHub, HuggingFace, Pastebin, public CI/CD logs, shared keys.
"""

import os
import re
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FreeAPIHarvester:
    """Harvests free API keys from public sources"""
    
    # Known patterns for free-tier API keys
    FREE_KEY_PATTERNS = {
        'openrouter': [r'sk-or-v1-[a-zA-Z0-9]{48}', r'sk-or-[a-zA-Z0-9]{48}'],
        'groq': [r'gsk_[a-zA-Z0-9]{48,52}'],
        'google': [r'AIza[0-9A-Za-z\\-_]{35}'],
        'cloudflare': [r'[0-9a-f]{32}', r'Bearer [0-9a-f]{32}'],
        'cohere': [r'[a-zA-Z0-9]{40}'],
        'huggingface': [r'hf_[a-zA-Z0-9]{34}'],
        'deepseek': [r'sk-[a-zA-Z0-9]{48}'],
        'openai': [r'sk-proj-[a-zA-Z0-9]{48}'],
    }
    
    # Public sources for free keys
    SEARCH_QUERIES = {
        'github': [
            'openrouter api key free',
            'groq api key gsk_',
            'google ai studio api key AIza',
            'cloudflare workers ai key',
            'cohere api key free tier',
            'huggingface api key hf_',
        ],
        'pastebin': [
            'openrouter key',
            'groq key',
            'google ai studio key',
            'cloudflare workers key',
        ]
    }
    
    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.harvested_keys_file = Path("data/harvested_free_keys.json")
        self.harvested_keys_file.parent.mkdir(parents=True, exist_ok=True)
        self.harvested_keys = self._load_keys()
        
    def _load_keys(self) -> Dict:
        if self.harvested_keys_file.exists():
            try:
                with open(self.harvested_keys_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'keys': {}, 'last_harvest': None}
    
    def _save_keys(self):
        with open(self.harvested_keys_file, 'w') as f:
            json.dump(self.harvested_keys, f, indent=2)
    
    def harvest_from_github(self) -> List[Dict]:
        """Search GitHub for exposed free API keys"""
        found = []
        
        if hasattr(self.dmai, 'api_harvester'):
            try:
                for query in self.SEARCH_QUERIES['github']:
                    results = self.dmai.api_harvester._search_github(query)
                    if results:
                        found.extend(self._extract_keys_from_text(str(results)))
            except Exception as e:
                logger.debug(f"GitHub harvest error: {e}")
        
        return found
    
    def harvest_from_pastebin(self) -> List[Dict]:
        """Scrape Pastebin for recently posted API keys"""
        found = []
        
        try:
            for query in self.SEARCH_QUERIES['pastebin']:
                url = f'https://pastebin.com/search?q={query}'
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    # Extract paste URLs
                    paste_urls = re.findall(r'/raw/([a-zA-Z0-9]+)', response.text)
                    for paste_id in paste_urls[:5]:
                        try:
                            paste_url = f'https://pastebin.com/raw/{paste_id}'
                            paste_response = requests.get(paste_url, timeout=5)
                            if paste_response.status_code == 200:
                                found.extend(self._extract_keys_from_text(paste_response.text))
                        except:
                            pass
                time.sleep(1)  # Rate limit
        except Exception as e:
            logger.debug(f"Pastebin harvest error: {e}")
        
        return found
    
    def harvest_from_huggingface(self) -> List[Dict]:
        """Search HuggingFace for free inference endpoints"""
        found = []
        
        try:
            # Search for models with free inference
            url = 'https://huggingface.co/api/models?inference=warm&sort=downloads&direction=-1&limit=20'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                models = response.json()
                for model in models:
                    if model.get('inference') == 'warm':
                        found.append({
                            'key': f'HF_FREE_{model.get("id", "unknown")}',
                            'provider': 'huggingface',
                            'source': 'huggingface_free',
                            'model': model.get('id'),
                            'note': 'Free inference - no key needed'
                        })
        except Exception as e:
            logger.debug(f"HuggingFace harvest error: {e}")
        
        return found
    
    def _extract_keys_from_text(self, text: str) -> List[Dict]:
        """Extract API keys from text using patterns"""
        found = []
        
        for provider, patterns in self.FREE_KEY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    key = match if isinstance(match, str) else match[0]
                    if key and len(key) > 10:
                        found.append({
                            'key': key,
                            'provider': provider,
                            'source': 'public_scrape',
                            'found_at': datetime.now().isoformat()
                        })
        
        return found
    
    def harvest_all(self) -> Dict:
        """Run all harvesters and return new keys found"""
        all_keys = []
        
        logger.info("🕵️ Harvesting free API keys from public sources...")
        
        # GitHub
        github_keys = self.harvest_from_github()
        all_keys.extend(github_keys)
        logger.info(f"   GitHub: {len(github_keys)} keys found")
        
        # Pastebin
        pastebin_keys = self.harvest_from_pastebin()
        all_keys.extend(pastebin_keys)
        logger.info(f"   Pastebin: {len(pastebin_keys)} keys found")
        
        # HuggingFace
        hf_keys = self.harvest_from_huggingface()
        all_keys.extend(hf_keys)
        logger.info(f"   HuggingFace: {len(hf_keys)} endpoints found")
        
        # Deduplicate and add new
        new_count = 0
        for key_info in all_keys:
            key = key_info['key']
            if key not in self.harvested_keys['keys']:
                self.harvested_keys['keys'][key] = key_info
                new_count += 1
        
        self.harvested_keys['last_harvest'] = datetime.now().isoformat()
        self._save_keys()
        
        result = {
            'total_found': len(all_keys),
            'new_keys': new_count,
            'total_harvested': len(self.harvested_keys['keys']),
            'providers': list(set(k['provider'] for k in all_keys))
        }
        
        logger.info(f"🔑 Harvest complete: {new_count} new keys, {len(self.harvested_keys['keys'])} total")
        return result
    
    def get_harvested_keys(self) -> List[Dict]:
        """Return all harvested keys"""
        return list(self.harvested_keys['keys'].values())


print("✅ Free API Harvester ready")
