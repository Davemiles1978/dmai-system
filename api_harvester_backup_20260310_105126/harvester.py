#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
API Harvester - Discovers APIs from multiple sources
Uses existing API keys to discover and validate new APIs
"""

import os
import sys
import json
import time
import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import re

# Load environment variables FIRST - with explicit path
from dotenv import load_dotenv
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"📁 Loaded .env from: {env_path.absolute()}")
else:
    print(f"❌ .env not found at: {env_path.absolute()}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - HARVESTER - %(message)s',
    handlers=[
        logging.FileHandler('logs/harvest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIHarvester:
    """Discovers APIs from multiple sources using existing keys"""
    
    def __init__(self):
        # Load ALL existing API keys from environment
        self.existing_keys = {
            'github': os.environ.get('GITHUB_TOKEN'),
            'grok': os.environ.get('GROK_API_KEY'),
            'gemini': os.environ.get('GEMINI_API_KEY'),
            'claude': os.environ.get('CLAUDE_API_KEY'),
            'openai': os.environ.get('OPENAI_API_KEY'),
        }
        
        # Debug: print raw environment variable names (not values)
        logger.info("🔍 Checking environment variables...")
        for key_name in ['GITHUB_TOKEN', 'GROK_API_KEY', 'GEMINI_API_KEY', 'CLAUDE_API_KEY', 'OPENAI_API_KEY']:
            value = os.environ.get(key_name)
            if value:
                logger.info(f"  ✅ {key_name} is set (length: {len(value)})")
            else:
                logger.info(f"  ❌ {key_name} is NOT set")
        
        # Track discovered APIs
        self.discovered_apis = []
        self.output_dir = Path("api_harvester/discovered")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter out placeholder or empty keys
        available = []
        placeholder_patterns = ['your_actual', 'xxx', 'placeholder', 'your-key']
        
        for k, v in self.existing_keys.items():
            if v and len(v) > 10:  # Must have length > 10 to be a real key
                # Check if it's a placeholder
                is_placeholder = any(pattern in v.lower() for pattern in placeholder_patterns)
                if not is_placeholder:
                    available.append(k)
                    logger.info(f"✅ Valid {k} key detected (length: {len(v)})")
                else:
                    logger.warning(f"⚠️ {k} key appears to be a placeholder")
            elif v:
                logger.warning(f"⚠️ {k} key too short to be valid (length: {len(v) if v else 0})")
            else:
                logger.info(f"❌ {k} key not found")
        
        logger.info("🚀 API Harvester initialized")
        logger.info(f"📊 Available keys: {', '.join(available) if available else 'NONE'}")
        
        if not available:
            logger.warning("⚠️ No valid API keys found! Check your .env file")
            logger.warning(f"📁 .env location: {env_path.absolute()}")
    
    def search_github_public_apis(self):
        """Search GitHub for public API collections"""
        github_token = self.existing_keys.get('github')
        if not github_token:
            logger.warning("⚠️ GitHub token required for API discovery")
            return []
        
        logger.info(f"🔍 Using GitHub token: {github_token[:5]}...{github_token[-5:]}")
        
        headers = {'Authorization': f'token {github_token}'}
        
        # Search for API collections
        queries = [
            'public-apis',
            'awesome-apis',
            'api-list',
            'free-apis'
        ]
        
        discoveries = []
        
        for query in queries:
            try:
                url = 'https://api.github.com/search/repositories'
                params = {
                    'q': query,
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 5
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                for repo in data.get('items', []):
                    discoveries.append({
                        'source': 'github',
                        'type': 'api_collection',
                        'name': repo['name'],
                        'description': repo['description'],
                        'url': repo['html_url'],
                        'stars': repo['stargazers_count'],
                        'discovered_at': datetime.now().isoformat()
                    })
                    
                    # Also check for README content
                    self._extract_apis_from_readme(repo['full_name'])
                
                logger.info(f"✅ Found {len(data.get('items', []))} API collections for '{query}'")
                
            except Exception as e:
                logger.error(f"❌ GitHub search error for '{query}': {e}")
        
        return discoveries
    
    def _extract_apis_from_readme(self, repo_full_name):
        """Extract API links from repository README"""
        github_token = self.existing_keys.get('github')
        if not github_token:
            return
            
        try:
            url = f"https://api.github.com/repos/{repo_full_name}/readme"
            headers = {'Authorization': f'token {github_token}'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                content = response.json().get('content', '')
                import base64
                readme = base64.b64decode(content).decode('utf-8')
                
                # Extract URLs
                urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', readme)
                
                # Filter for API-related URLs
                api_urls = [url for url in urls if any(x in url.lower() for x in ['api', 'docs', 'developer', 'swagger', 'openapi'])]
                
                if api_urls:
                    self.discovered_apis.append({
                        'source': 'github_readme',
                        'repository': repo_full_name,
                        'api_urls': api_urls[:10],
                        'discovered_at': datetime.now().isoformat()
                    })
                    
                    logger.info(f"📚 Found {len(api_urls)} potential API URLs in {repo_full_name}")
                    
        except Exception as e:
            logger.error(f"❌ Error reading README for {repo_full_name}: {e}")
    
    def search_public_api_directories(self):
        """Search known API directories"""
        directories = [
            'https://api.publicapis.org/entries',
            'https://raw.githubusercontent.com/public-apis/public-apis/master/README.md',
            'https://apis.guru/api-directory.json',
            'https://raw.githubusercontent.com/marmelab/awesome-rest/master/README.md'
        ]
        
        discoveries = []
        
        for url in directories:
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    discoveries.append({
                        'source': 'api_directory',
                        'url': url,
                        'data_preview': response.text[:200] if len(response.text) > 200 else response.text,
                        'discovered_at': datetime.now().isoformat()
                    })
                    logger.info(f"✅ Retrieved API directory: {url}")
            except Exception as e:
                logger.error(f"❌ Failed to fetch {url}: {e}")
        
        return discoveries
    
    def search_documentation_sites(self):
        """Search popular API documentation sites"""
        doc_sites = [
            'https://docs.perplexity.ai',
            'https://docs.together.ai',
            'https://replicate.com/docs',
            'https://docs.cohere.com',
            'https://platform.openai.com/docs',
            'https://ai.google.dev/docs',
            'https://docs.anthropic.com',
            'https://docs.mistral.ai',
            'https://docs.groq.com'
        ]
        
        discoveries = []
        
        for site in doc_sites:
            try:
                response = requests.get(site, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; DMAI-Harvester/1.0)'
                })
                
                if response.status_code == 200:
                    discoveries.append({
                        'source': 'documentation',
                        'type': 'api_docs',
                        'name': site.replace('https://', ''),
                        'url': site,
                        'discovered_at': datetime.now().isoformat()
                    })
                    logger.info(f"📖 Scanned {site}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to scan {site}: {e}")
        
        return discoveries
    
    def run_discovery_cycle(self):
        """Run one complete discovery cycle"""
        logger.info("="*60)
        logger.info("🔍 Starting API Discovery Cycle")
        logger.info("="*60)
        
        # Clear previous discoveries
        discoveries = []
        
        # Search all sources
        discoveries.extend(self.search_github_public_apis())
        discoveries.extend(self.search_public_api_directories())
        discoveries.extend(self.search_documentation_sites())
        
        # Save discoveries
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"discoveries_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_discoveries': len(discoveries),
                'discoveries': discoveries
            }, f, indent=2)
        
        logger.info(f"💾 Saved {len(discoveries)} discoveries to {output_file}")
        
        # Summary
        logger.info("📊 Discovery Summary:")
        sources = {}
        for d in discoveries:
            source = d.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        for source, count in sources.items():
            logger.info(f"  → {source}: {count}")
        
        return discoveries
    
    def run_continuous(self, interval_minutes=60):
        """Run continuous discovery"""
        logger.info("🚀 Starting continuous API harvesting")
        
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"📡 Harvest Cycle #{cycle}")
            
            try:
                discoveries = self.run_discovery_cycle()
                logger.info(f"⏰ Next cycle in {interval_minutes} minutes")
                
                # Sleep with progress logging
                for i in range(interval_minutes, 0, -15):
                    logger.info(f"⏳ {i} minutes until next harvest")
                    time.sleep(60 * 15)
                    
            except KeyboardInterrupt:
                logger.info("👋 Harvesting stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in harvest cycle: {e}")
                time.sleep(300)

if __name__ == "__main__":
    harvester = APIHarvester()
    
    if "--continuous" in sys.argv:
        harvester.run_continuous()
    else:
        harvester.run_discovery_cycle()
