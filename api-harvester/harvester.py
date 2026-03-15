from github_scraper_evolution import GitHubScraperEvolution
from db_hybrid import KeyEvolutionDB, process_harvested_key

#!/usr/bin/env python3
"""DMAI API Harvester - Finds and manages API keys for ALL integrated systems"""

import os
import sys
import time
import json
import logging
import signal
import threading
import traceback
import importlib.util
import inspect
import requests
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('harvester.log')
    ]
)
logger = logging.getLogger("harvester")

class Harvester:
    def __init__(self):
        self.running = True
        self.cycle_count = 0
        self.start_time = datetime.now()
        
        # Load config
        self.config = self.load_config()
        
        # Initialize components
        self.github_scraper = None
        self.storage = None
        self.db = None
        self.integrators = {}  # Store loaded integrators
        self.key_wishlist = {}  # Store all required keys from integrators
        
        # Initialize database
        self.db = KeyEvolutionDB()
        logger.info("✅ Database initialized")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("=" * 50)
        logger.info("DMAI API Harvester initializing...")
        logger.info("=" * 50)
        
        self.init_components()
        self.load_integrators()
        self.generate_key_wishlist()
        
    def load_config(self):
        """Load configuration"""
        config = {}
        config_file = Path("config.json")
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ Loaded config from {config_file.absolute()}")
                logger.info(f"📄 Config keys: {list(config.keys())}")
            except Exception as e:
                logger.error(f"❌ Error loading config: {e}")
        
        # Default config
        config.setdefault('github_token', os.environ.get('GITHUB_TOKEN'))
        config.setdefault('redis_host', 'localhost')
        config.setdefault('redis_port', 6379)
        config.setdefault('database_url', 'sqlite:///harvester.db')
        config.setdefault('check_interval', 3600)  # 1 hour
        config.setdefault('integrations_path', os.path.join(os.path.dirname(__file__), '..', 'integrations'))
        
        # Log token status (without revealing the full token)
        if config.get('github_token'):
            token = config['github_token']
            logger.info(f"🔑 GitHub token found in config: {token[:4]}...{token[-4:] if len(token) > 8 else ''} (length: {len(token)})")
            
            # Test the token with a quick API call
            try:
                test_response = requests.get(
                    "https://api.github.com/rate_limit",
                    headers={"Authorization": f"token {token}"},
                    timeout=5
                )
                if test_response.status_code == 200:
                    rate_data = test_response.json()
                    search_limit = rate_data.get('resources', {}).get('search', {}).get('limit', 30)
                    search_remaining = rate_data.get('resources', {}).get('search', {}).get('remaining', 0)
                    logger.info(f"✅ GitHub token is valid! Search rate limit: {search_remaining}/{search_limit}")
                else:
                    logger.error(f"❌ GitHub token test failed with status: {test_response.status_code}")
            except Exception as e:
                logger.error(f"❌ GitHub token test error: {e}")
        else:
            logger.warning("⚠️ No GitHub token found in config or environment")
        
        return config
    
    def init_components(self):
        """Initialize all scraper components"""
        try:
            # Get token from config
            token = self.config.get('github_token')
            
            if token:
                logger.info(f"🔑 Passing token to GitHub scraper: {token[:4]}...{token[-4:]}")
            else:
                logger.warning("⚠️ No token available for GitHub scraper")
            
            # GitHubScraper expects a config object
            github_config = {
                'github_token': self.config.get('github_token'),
                'timeout': 30,
                'retry_count': 3
            }
            
            logger.info(f"📦 Initializing GitHub scraper with config")
            
            self.github_scraper = GitHubScraperEvolution(config=github_config)
            logger.info("✅ GitHub scraper initialized")
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def load_integrators(self):
        """Load all integrator modules to discover required keys"""
        integrations_path = Path(self.config.get('integrations_path', os.path.join(os.path.dirname(__file__), '..', 'integrations')))
        
        if not integrations_path.exists():
            logger.warning(f"Integrations path not found: {integrations_path.absolute()}")
            return
        
        logger.info(f"🔍 Scanning for integrators in: {integrations_path.absolute()}")
        
        # Walk through all integrator files
        for root, dirs, files in os.walk(integrations_path):
            for file in files:
                if file.endswith('_integrator.py'):
                    module_path = os.path.join(root, file)
                    module_name = file.replace('.py', '')
                    rel_path = os.path.relpath(module_path, start=os.path.dirname(__file__))
                    
                    try:
                        # Load module dynamically
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find integrator class
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and name.endswith('Integrator'):
                                # Create instance (pass db path for key loading)
                                instance = obj(self.get_db_path())
                                self.integrators[module_name] = {
                                    'instance': instance,
                                    'path': rel_path,
                                    'name': name,
                                    'required_keys': []
                                }
                                
                                # Get required keys if method exists
                                if hasattr(instance, 'get_required_keys'):
                                    required = instance.get_required_keys()
                                    self.integrators[module_name]['required_keys'] = required
                                    logger.info(f"✅ Loaded integrator: {module_name} ({len(required)} required keys)")
                                else:
                                    logger.info(f"✅ Loaded integrator: {module_name} (no key requirements)")
                                
                                break
                                
                    except Exception as e:
                        logger.error(f"Failed to load integrator {module_name}: {e}")
        
        logger.info(f"Loaded {len(self.integrators)} integrators total")
    
    def get_db_path(self):
        """Get database path for integrators"""
        return os.path.join(os.path.dirname(__file__), 'dmai_local.db')
    
    def generate_key_wishlist(self):
        """Generate comprehensive wishlist of all keys needed by all integrators"""
        self.key_wishlist = {
            # ===== AI PROVIDERS (24/7 Learning) =====
            'ai_providers': {
                'openai': 'OpenAI API key for GPT-4 access',
                'anthropic': 'Anthropic API key for Claude models',
                'google_gemini': 'Google Gemini API key',
                'meta_llama': 'Meta Llama access token',
                'mistral': 'Mistral AI API key',
                'cohere': 'Cohere API key',
                'deepseek': 'DeepSeek API key',
                'perplexity': 'Perplexity API key',
                'together': 'Together AI API key',
                'replicate': 'Replicate API token',
                'huggingface': 'HuggingFace API token',
            },
            
            # ===== CODE GENERATION =====
            'code_generation': {
                'github_copilot': 'GitHub Copilot token',
                'amazon_codewhisperer': 'AWS CodeWhisperer credentials',
                'deepseek_coder': 'DeepSeek Coder API key',
                'codeium': 'Codeium API key',
                'tabnine': 'Tabnine API key',
                'cursor': 'Cursor IDE API key',
                'sourcegraph': 'Sourcegraph Cody token',
            },
            
            # ===== IMAGE GENERATION =====
            'image_generation': {
                'openai_dalle': 'OpenAI DALL-E API key',
                'stability_ai': 'Stability AI API key (Stable Diffusion)',
                'midjourney': 'Midjourney API key',
                'adobe_firefly': 'Adobe Firefly API key',
                'leonardo': 'Leonardo.ai API key',
                'playground': 'Playground AI API key',
                'ideogram': 'Ideogram API key',
                'clipdrop': 'Clipdrop API key',
                'getimg': 'Getimg.ai API key',
            },
            
            # ===== VIDEO GENERATION =====
            'video_generation': {
                'runway': 'RunwayML API key',
                'pika': 'Pika API key',
                'haiper': 'Haiper API key',
                'synthesia': 'Synthesia API key',
                'heygen': 'HeyGen API key',
                'd_id': 'D-ID API key',
                'kaiber': 'Kaiber API key',
            },
            
            # ===== AUDIO/VOICE =====
            'audio_voice': {
                'elevenlabs': 'ElevenLabs API key',
                'openai_whisper': 'OpenAI Whisper API key',
                'bark': 'Suno Bark API key',
                'playht': 'Play.ht API key',
                'resemble': 'Resemble AI API key',
                'wellsaid': 'WellSaid API key',
                'murf': 'Murf API key',
            },
            
            # ===== RESEARCH PAPERS =====
            'research_papers': {
                'arxiv': 'arXiv API key',
                'semantic_scholar': 'Semantic Scholar API key',
                'paperswithcode': 'PapersWithCode API key',
                'openreview': 'OpenReview API key',
                'crossref': 'Crossref API key',
                'scopus': 'Scopus API key',
                'ieee': 'IEEE Xplore API key',
                'springer': 'Springer API key',
                'acm': 'ACM Digital Library key',
            },
            
            # ===== ENGINEERING DOMAINS =====
            'engineering': {
                # Mechanical
                'onshape': 'Onshape API key',
                'fusion360': 'Fusion 360 API key',
                'solidworks': 'SolidWorks API key',
                'grabcad': 'GrabCAD API key',
                'step': 'STEP file library access',
                
                # Electrical/Electronics
                'altium': 'Altium Designer API key',
                'kicad': 'KiCad API key',
                'eagle': 'Autodesk Eagle API key',
                'easyeda': 'EasyEDA API key',
                'digikey': 'DigiKey API key',
                'mouser': 'Mouser API key',
                'octopart': 'Octopart API key',
                
                # 3D Design/Printing
                'thingiverse': 'Thingiverse API key',
                'printables': 'Printables API key',
                'cults3d': 'Cults3D API key',
                'prusaprinters': 'PrusaPrinters API key',
                
                # Project Management
                'jira': 'Jira API key',
                'asana': 'Asana API key',
                'trello': 'Trello API key',
                'monday': 'Monday.com API key',
                'clickup': 'ClickUp API key',
                'linear': 'Linear API key',
                
                # Production/Manufacturing
                'protolabs': 'ProtoLabs API key',
                'xometry': 'Xometry API key',
                'hubs': 'Hubs API key',
                'pcbway': 'PCBWay API key',
                'jlcpcb': 'JLCPCB API key',
            },
            
            # ===== SYNTHETIC INTELLIGENCE RESEARCH =====
            'synthetic_intelligence': {
                # Meta-Learning
                'meta_learning_papers': 'Access to meta-learning research',
                'few_shot_datasets': 'Few-shot learning datasets',
                'transfer_learning': 'Transfer learning resources',
                
                # Neural Architecture
                'nas_bench': 'Neural Architecture Search benchmarks',
                'model_zoo': 'Pre-trained model collections',
                'architecture_search': 'Architecture search APIs',
                
                # Self-Improvement
                'auto_ml': 'AutoML APIs',
                'hyperparameter_tuning': 'Hyperparameter optimization',
                'model_compression': 'Model compression tools',
                
                # Emergent Behavior
                'emergence_research': 'Emergent behavior datasets',
                'interpretability': 'Model interpretability tools',
                'mechanistic': 'Mechanistic interpretability',
            },
            
            # ===== KNOWLEDGE AGGREGATION =====
            'knowledge_aggregation': {
                'wikipedia': 'Wikipedia API key',
                'wikidata': 'Wikidata API key',
                'dbpedia': 'DBpedia API key',
                'wolfram': 'Wolfram Alpha API key',
                'google_knowledge': 'Google Knowledge Graph API',
                'bing_search': 'Bing Search API',
                'common_crawl': 'Common Crawl access',
                'archive': 'Internet Archive API',
            },
            
            # ===== EVOLUTION TRIGGER SYSTEMS =====
            'evolution_triggers': {
                'github_events': 'GitHub Events API',
                'arxiv_updates': 'ArXiv update feed',
                'huggingface_daily': 'HuggingFace daily papers',
                'reddit_ml': 'Reddit ML subreddit API',
                'discord_communities': 'Discord bot token',
                'slack_channels': 'Slack API token',
                'telegram_groups': 'Telegram bot token',
            },
            
            # ===== TECHNIQUE EXTRACTION =====
            'technique_extraction': {
                'github_code_search': 'GitHub Code Search API',
                'stackoverflow': 'Stack Overflow API key',
                'medium': 'Medium API key',
                'dev_to': 'Dev.to API key',
                'hackernews': 'HackerNews API',
                'lobsters': 'Lobsters API',
            }
        }
        
        # Add all keys from loaded integrators
        for integrator_name, integrator_info in self.integrators.items():
            required = integrator_info.get('required_keys', [])
            if required:
                # Determine category from integrator name
                category = integrator_name.replace('_integrator', '')
                if category not in self.key_wishlist:
                    self.key_wishlist[category] = {}
                
                for key in required:
                    self.key_wishlist[category][key] = f'Required by {integrator_name}'
        
        # Save wishlist to file
        wishlist_file = Path(os.path.join(os.path.dirname(__file__), 'key_wishlist.json'))
        with open(wishlist_file, 'w') as f:
            json.dump(self.key_wishlist, f, indent=2)
        
        # Count total keys
        total_keys = sum(len(keys) for keys in self.key_wishlist.values())
        logger.info(f"📋 Generated key wishlist with {total_keys} keys across {len(self.key_wishlist)} categories")
        
        # Log summary
        for category, keys in self.key_wishlist.items():
            logger.info(f"  • {category}: {len(keys)} keys")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🚨 SHUTDOWN SIGNAL RECEIVED: {signum}")
        logger.info(f"Stack trace at shutdown:")
        traceback.print_stack(frame)
        logger.info(f"Current cycle: #{self.cycle_count}")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down harvester...")
        self.running = False
        
        # Close connections
        if self.db:
            self.db.conn.close()
        
        logger.info("Harvester shutdown complete")
        sys.exit(0)
    
    # ==================== MISSING METHODS ADDED BELOW ====================
    
    def scrape_apis_guru(self):
        """Scrape APIs from APIS.Guru directory"""
        try:
            logger.info("Scraping APIS.Guru API directory...")
            response = requests.get("https://api.apis.guru/v2/list.json", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                saved_count = 0
                
                for api_name, api_info in list(data.items())[:100]:  # Limit to first 100
                    try:
                        # Create API data structure
                        api_data = {
                            'name': api_name,
                            'service': api_name,
                            'url': api_info.get('info', {}).get('contact', {}).get('url', ''),
                            'description': api_info.get('info', {}).get('description', '')[:200],
                            'category': 'apis.guru',
                            'source': 'apis.guru'
                        }
                        
                        # Save using compatibility method
                        self.db.save_api(api_data)
                        saved_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error saving API {api_name}: {e}")
                
                logger.info(f"Successfully saved {saved_count} APIs from APIS.Guru")
            else:
                logger.error(f"Failed to fetch APIS.Guru data: {response.status_code}")
                
        except Exception as e:
            logger.error(f"APIS.Guru scraping error: {e}")
    
    def scrape_public_apis_github(self):
        """Scrape public APIs from GitHub repository"""
        try:
            logger.info("Scraping public-apis GitHub repo...")
            response = requests.get(
                "https://raw.githubusercontent.com/public-apis/public-apis/master/README.md",
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.text
                logger.info(f"Fetched public-apis README ({len(content)} bytes)")
                
                # Simple parsing - look for API entries
                lines = content.split('\n')
                saved_count = 0
                
                for line in lines:
                    if '|' in line and 'api' in line.lower():
                        parts = line.split('|')
                        if len(parts) >= 3:
                            api_name = parts[1].strip()
                            api_desc = parts[2].strip() if len(parts) > 2 else ""
                            
                            if api_name and not api_name.startswith('---'):
                                api_data = {
                                    'name': api_name,
                                    'service': api_name.lower().replace(' ', '-'),
                                    'description': api_desc[:200],
                                    'category': 'public-apis',
                                    'source': 'public-apis-github'
                                }
                                
                                try:
                                    self.db.save_api(api_data)
                                    saved_count += 1
                                except Exception as e:
                                    logger.error(f"Public APIs scraping error: {e}")
                                    raise
                
                logger.info(f"Successfully saved {saved_count} APIs from public-apis")
            else:
                logger.error(f"Failed to fetch public-apis: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Public APIs scraping error: {e}")
            raise
    
    def update_redis_metrics(self):
        """Update Redis with current metrics (if Redis is configured)"""
        try:
            # This is a placeholder - actual Redis implementation would go here
            # For now, just log that it's not configured
            logger.debug("Redis metrics update skipped (Redis not configured)")
        except Exception as e:
            logger.error(f"Failed to update Redis metrics: {e}")
    
    # ==================== END OF MISSING METHODS ====================
    
    def run_cycle(self):
        """Run one harvesting cycle"""
        self.cycle_count += 1
        cycle_start = time.time()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"=== Starting harvesting cycle #{self.cycle_count} ===")
        logger.info(f"{'='*50}")
        
        total_keys = 0
        
        # Run GitHub scraper
        if self.github_scraper:
            logger.info("\n🔍 Starting GitHub scraping...")
            try:
                # Search for each key type
                for category, keys in self.key_wishlist.items():
                    for key_name, description in keys.items():
                        logger.debug(f"Searching for {key_name}...")
                        # Construct search queries
                        queries = [
                            f'"{key_name}" api key',
                            f'"{key_name}" token',
                            f'"{key_name}" secret',
                            f'"{key_name.upper()}_KEY"',
                            f'"{key_name.upper()}_TOKEN"'
                        ]
                        
                        for query in queries:
                            try:
                                results = self.github_scraper.search_github(query)
                                if results:
                                    logger.info(f"Found potential matches for {key_name}")
                                    # Process results
                                    for result in results:
                                        if self.extract_key_from_result(result, key_name):
                                            total_keys += 1
                            except Exception as e:
                                logger.error(f"Error searching for {key_name}: {e}")
                                
            except Exception as e:
                logger.error(f"GitHub scraper error: {e}")
                logger.error(traceback.format_exc())
        
        # Run APIS.Guru scraper
        try:
            self.scrape_apis_guru()
        except Exception as e:
            logger.error(f"APIS.Guru scraper error: {e}")
        
        # Run public-apis scraper
        try:
            self.scrape_public_apis_github()
        except Exception as e:
            logger.error(f"Public APIs scraper error: {e}")
        
        # Update Redis metrics
        try:
            self.update_redis_metrics()
        except Exception as e:
            logger.error(f"Redis metrics error: {e}")
        
        cycle_time = time.time() - cycle_start
        logger.info(f"\n{'='*50}")
        logger.info(f"=== Cycle #{self.cycle_count} complete ===")
        logger.info(f"Total keys found: {total_keys}")
        logger.info(f"Cycle time: {cycle_time:.1f} seconds")
        logger.info(f"{'='*50}\n")
        
        return total_keys
    
    def extract_key_from_result(self, result, key_name):
        """Extract potential API key from search result"""
        try:
            content = result.get('content', '')
            if not content:
                return False
            
            import re
            
            # Common key patterns
            patterns = [
                r'[a-zA-Z0-9]{32,}',  # 32+ char alphanumeric
                r'sk-[a-zA-Z0-9]{48,}',  # OpenAI style
                r'sk-ant-api03-[a-zA-Z0-9_-]{90,}',  # Anthropic style
                r'AIzaSy[a-zA-Z0-9_-]{35}',  # Google style
                r'hf_[a-zA-Z0-9]{34,}',  # HuggingFace style
                r'ghp_[a-zA-Z0-9]{36,}',  # GitHub style
                r'AKIA[0-9A-Z]{16}',  # AWS style
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        # Store the key
                        self.store_key(key_name, match, 'github', result.get('url', ''))
                        return True
                        
        except Exception as e:
            logger.error(f"Error extracting key: {e}")
        
        return False
    
    def store_key(self, service, api_key, source, url):
        """Store discovered key and create notification"""
        try:
            # Check if key exists
            existing = self.db.get_key(service)
            if existing:
                return
            
            # Store key
            self.db.add_key(service, api_key, {
                'source': source,
                'url': url,
                'discovered_at': datetime.now().isoformat()
            })
            
            logger.info(f"🔑 Found new {service} key: {api_key[:10]}... from {source}")
            
            # Create notification file for UI
            self.create_notification(service, api_key, source, url)
            
        except Exception as e:
            logger.error(f"Error storing key: {e}")
    
    def create_notification(self, service, api_key, source, url):
        """Create notification file for web UI"""
        notifications_dir = Path(os.path.join(os.path.dirname(__file__), 'notifications'))
        notifications_dir.mkdir(exist_ok=True)
        
        # Find which integrator needs this key
        integrator_name = None
        integration_code = None
        
        for name, info in self.integrators.items():
            if service in info.get('required_keys', []):
                integrator_name = name
                # Get integration code if available
                if hasattr(info['instance'], 'get_integration_code'):
                    integration_code = info['instance'].get_integration_code(service, api_key)
                break
        
        notification = {
            'service': service,
            'api_key': api_key[:20] + '...' + api_key[-10:],
            'full_key': api_key,
            'source': source,
            'url': url,
            'discovered_at': datetime.now().isoformat(),
            'integrator': integrator_name,
            'integration_code': integration_code,
            'message': f"🔑 New {service.upper()} API key discovered!"
        }
        
        filename = notifications_dir / f"{service}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(notification, f, indent=2)
        
        # Also print to console with color
        print("\n" + "="*80)
        print(f"🔑🔑🔑 NEW API KEY DISCOVERED [{service.upper()}] 🔑🔑🔑")
        print("="*80)
        print(f"Service:     {service}")
        print(f"Key:         {api_key[:20]}...{api_key[-10:]}")
        print(f"Source:      {source}")
        print(f"URL:         {url}")
        print(f"Integrator:  {integrator_name or 'Unknown'}")
        print("-"*80)
        if integration_code:
            print("Integration code available in notification file")
        print("="*80 + "\n")
    
    def run(self):
        """Main run loop"""
        logger.info("=" * 50)
        logger.info(f"DMAI API Harvester started - Continuous Mode")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Tracking {sum(len(keys) for keys in self.key_wishlist.values())} keys across {len(self.key_wishlist)} categories")
        logger.info("=" * 50)
        
        while self.running:
            try:
                self.run_cycle()
                
                # Wait before next cycle
                wait_time = self.config.get('check_interval', 3600)
                logger.info(f"⏰ Waiting {wait_time} seconds until next cycle...")
                
                # Sleep in chunks to allow for graceful shutdown
                for _ in range(wait_time):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.shutdown()
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.error(traceback.format_exc())
                logger.info("Waiting 60 seconds before retry...")
                time.sleep(60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DMAI API Harvester")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--port", type=int, default=9001, help="API port (default: 9001)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--wishlist", action="store_true", help="Generate key wishlist and exit")
    
    args = parser.parse_args()
    
    harvester = Harvester()
    
    if args.wishlist:
        print("\n📋 KEY WISHLIST SUMMARY:")
        for category, keys in harvester.key_wishlist.items():
            print(f"  • {category}: {len(keys)} keys")
        print(f"\nTotal: {sum(len(keys) for keys in harvester.key_wishlist.values())} keys")
        print(f"Wishlist saved to: {os.path.join(os.path.dirname(__file__), 'key_wishlist.json')}")
        sys.exit(0)
    
    if args.once:
        logger.info("Running single cycle")
        harvester.run_cycle()
    elif args.daemon:
        harvester.run()
    else:
        parser.print_help()

# ============================================================================
# PLUGGABLE INTERFACE LAYER - DO NOT MODIFY BELOW THIS LINE
# ============================================================================
# This section adds API endpoints for external systems to connect
# All original code above remains completely unchanged

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Memory optimization
import gc
gc.set_threshold(700, 10, 5)  # More aggressive garbage collection
import resource
try:
    # Set soft memory limit
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
except:
    pass

# Clear cache periodically
import threading
import time
from db_hybrid import KeyEvolutionDB, process_harvested_key
from github_scraper_evolution import GitHubScraperEvolution
def cache_cleaner():
    while True:
        time.sleep(300)  # Every 5 minutes
        gc.collect()  # Force garbage collection
        if hasattr(__import__('torch'), 'mps'):
            import torch
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
threading.Thread(target=cache_cleaner, daemon=True).start()


# Global reference to the harvester instance
_harvester_instance = None
_start_time = datetime.now()

class HarvesterAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query harvester status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                "name": "harvester_daemon",
                "running": True,
                "cycle": 0,
                "total_keys_found": 0,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time),
                "categories_tracked": 0,
                "keys_tracked": 0,
                "github_token_valid": False
            }
            
            # Try to get real data if harvester instance exists
            if _harvester_instance:
                try:
                    status["cycle"] = getattr(_harvester_instance, 'cycle_count', 0)
                    status["categories_tracked"] = len(getattr(_harvester_instance, 'key_wishlist', {}))
                    status["keys_tracked"] = sum(len(keys) for keys in getattr(_harvester_instance, 'key_wishlist', {}).values())
                    # Check if token is valid
                    if hasattr(_harvester_instance, 'github_scraper') and _harvester_instance.github_scraper:
                        status["github_token_valid"] = _harvester_instance.github_scraper.token is not None
                except:
                    pass
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            health_status = {
                "status": "healthy",
                "service": "api-harvester",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - _start_time)
            }
            self.wfile.write(json.dumps(health_status).encode())
            
        elif self.path == '/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            config_info = {
                "has_github_token": False,
                "check_interval": 3600,
                "integrators_loaded": 0
            }
            
            if _harvester_instance and hasattr(_harvester_instance, 'config'):
                config = _harvester_instance.config
                config_info["has_github_token"] = config.get('github_token') is not None
                config_info["check_interval"] = config.get('check_interval', 3600)
                config_info["integrators_loaded"] = len(getattr(_harvester_instance, 'integrators', {}))
            
            self.wfile.write(json.dumps(config_info).encode())
            
        elif self.path == '/wishlist':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if _harvester_instance and hasattr(_harvester_instance, 'key_wishlist'):
                self.wfile.write(json.dumps(_harvester_instance.key_wishlist).encode())
            else:
                self.wfile.write(json.dumps({}).encode())
                
        elif self.path == '/notifications':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get recent notifications
            notifications_dir = Path(os.path.join(os.path.dirname(__file__), 'notifications'))
            notifications = []
            
            if notifications_dir.exists():
                for file in sorted(notifications_dir.glob('*.json'), reverse=True)[:50]:
                    try:
                        with open(file, 'r') as f:
                            notif = json.load(f)
                            # Remove full key for security
                            if 'full_key' in notif:
                                del notif['full_key']
                            notifications.append(notif)
                    except:
                        pass
            
            self.wfile.write(json.dumps(notifications).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/command':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                command = json.loads(post_data)
                cmd = command.get('command', '')
                
                if cmd == 'harvest_now':
                    # Trigger immediate harvest cycle
                    if _harvester_instance:
                        # Run in a separate thread to not block the API
                        def run_harvest():
                            _harvester_instance.run_cycle()
                        thread = threading.Thread(target=run_harvest, daemon=True)
                        thread.start()
                        self.wfile.write(json.dumps({"status": "harvest_triggered"}).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Harvester not initialized"}).encode())
                        
                elif cmd == 'get_stats':
                    if _harvester_instance:
                        stats = {
                            "cycle": getattr(_harvester_instance, 'cycle_count', 0),
                            "running": getattr(_harvester_instance, 'running', False),
                            "uptime": str(datetime.now() - getattr(_harvester_instance, 'start_time', datetime.now())),
                            "integrators": len(getattr(_harvester_instance, 'integrators', {})),
                            "keys_tracked": sum(len(keys) for keys in getattr(_harvester_instance, 'key_wishlist', {}).values())
                        }
                        self.wfile.write(json.dumps(stats).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Harvester not initialized"}).encode())
                        
                elif cmd == 'regenerate_wishlist':
                    if _harvester_instance:
                        _harvester_instance.load_integrators()
                        _harvester_instance.generate_key_wishlist()
                        self.wfile.write(json.dumps({"status": "wishlist_regenerated"}).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Harvester not initialized"}).encode())
                        
                else:
                    self.wfile.write(json.dumps({"error": f"Unknown command: {cmd}"}).encode())
                    
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return  # Suppress HTTP logs

def _start_api_server():
    """Start API server in background thread"""
    port = 9001  # Fixed port for harvester daemon
    
    def run_server():
        server = HTTPServer(('localhost', port), HarvesterAPIHandler)
        print(f"📡 Harvester API endpoint active at http://localhost:{port}")
        print(f"   • /status - Harvester status")
        print(f"   • /health - Health check")
        print(f"   • /wishlist - Key wishlist")
        print(f"   • /notifications - Recent key discoveries")
        print(f"   • POST /command - Send commands")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the API server when this module is imported
_api_port = _start_api_server()

# Store reference to harvester instance when created
_original_init = Harvester.__init__
def _wrapped_init(self, *args, **kwargs):
    global _harvester_instance
    _original_init(self, *args, **kwargs)
    _harvester_instance = self

Harvester.__init__ = _wrapped_init
