#!/usr/bin/env python3
"""
Phase 11: Dynamic AI Discovery System
DMAI autonomously discovers, researches, and integrates new AI systems
ENHANCED: GitHub token support, HuggingFace API, ArXiv integration, AIIntegrationHub connection
"""

import os
import json
import time
import threading
import logging
import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import shutil
import feedparser

logger = logging.getLogger(__name__)


class DynamicAIDiscovery:
    """
    DMAI's autonomous AI discovery and integration system
    Constantly searches for new AI systems, analyzes them, and adds capabilities
    """
    
    def __init__(self, data_path: Path, ai_hub=None):
        self.data_path = data_path
        self.ai_hub = ai_hub  # Reference to AIIntegrationHub for adding discovered tutors
        self.discovered_path = data_path / 'phase11' / 'discoveries'
        self.learned_path = data_path / 'phase11' / 'learned_models'
        self.discovered_path.mkdir(parents=True, exist_ok=True)
        self.learned_path.mkdir(parents=True, exist_ok=True)
        
        # Load GitHub tokens
        self.github_main_token = os.getenv('GITHUB_TOKEN_MAIN')
        self.github_secondary_token = os.getenv('GITHUB_TOKEN_SECONDARY')
        self.current_github_token = self.github_main_token or self.github_secondary_token
        
        # Load HuggingFace token
        self.huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
        
        # Discovery sources with enhanced URLs
        self.discovery_sources = {
            'github_trending': {
                'url': 'https://api.github.com/search/repositories',
                'type': 'code_repo',
                'active': True,
                'requires_token': True
            },
            'huggingface': {
                'url': 'https://huggingface.co/api/models',
                'type': 'ai_model',
                'active': True,
                'requires_token': False
            },
            'papers_with_code': {
                'url': 'https://paperswithcode.com/api/v1/papers/',
                'type': 'research',
                'active': True,
                'requires_token': False
            },
            'arxiv': {
                'url': 'http://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&max_results=20',
                'type': 'research_paper',
                'active': True,
                'requires_token': False
            },
            'product_hunt': {
                'url': 'https://www.producthunt.com/topics/artificial-intelligence',
                'type': 'product',
                'active': True,
                'requires_token': False
            },
            'reddit_ml': {
                'url': 'https://www.reddit.com/r/MachineLearning/new.json',
                'type': 'discussion',
                'active': True,
                'requires_token': False
            },
            'openai_blog': {
                'url': 'https://openai.com/news/rss',
                'type': 'blog',
                'active': True,
                'requires_token': False
            },
            'google_ai_blog': {
                'url': 'https://ai.googleblog.com/atom.xml',
                'type': 'blog',
                'active': True,
                'requires_token': False
            },
            'deepmind_blog': {
                'url': 'https://www.deepmind.com/blog',
                'type': 'blog',
                'active': True,
                'requires_token': False
            },
            'anthropic_news': {
                'url': 'https://www.anthropic.com/news',
                'type': 'blog',
                'active': True,
                'requires_token': False
            }
        }
        
        # Known AI systems to research (expanded)
        self.known_ai_systems = [
            'Claude', 'ChatGPT', 'Gemini', 'DeepSeek', 'Llama', 'Mistral',
            'Midjourney', 'Stable Diffusion', 'DALL-E', 'RunwayML', 'Pika',
            'Sora', 'Kling', 'Imagen', 'Whisk', 'NotebookLM', 'Perplexity',
            'Windsurf', 'Lovable', 'Cursor', 'Copilot', 'Replit AI',
            'Grok', 'HuggingFace', 'GitHub Copilot', 'Codeium', 'Tabnine'
        ]
        
        self.discovered_ai = self._load_discovered()
        self.discovery_active = False
        self.integration_queue = []
        self.last_discovery_run = None
        
        logger.info("🔍 Dynamic AI Discovery System initialized")
        logger.info(f"   Discovery sources: {len(self.discovery_sources)}")
        logger.info(f"   Known AI systems to research: {len(self.known_ai_systems)}")
        if self.github_main_token:
            logger.info(f"   ✅ GitHub token configured (main)")
        if self.github_secondary_token:
            logger.info(f"   ✅ GitHub token configured (secondary)")
        if self.huggingface_token:
            logger.info(f"   ✅ HuggingFace token configured")
    
    def _load_discovered(self) -> Dict:
        """Load discovered AI systems"""
        discovered_file = self.discovered_path / 'discovered_ai.json'
        if discovered_file.exists():
            try:
                with open(discovered_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'discovered': [], 'integrated': [], 'researching': [], 'researched': []}
    
    def _save_discovered(self):
        """Save discovered AI systems"""
        with open(self.discovered_path / 'discovered_ai.json', 'w') as f:
            json.dump(self.discovered_ai, f, indent=2)
    
    def start_discovery_loop(self):
        """Start autonomous discovery loop"""
        self.discovery_active = True
        
        def discover():
            while self.discovery_active:
                try:
                    self.last_discovery_run = datetime.now()
                    
                    # Discover new AI systems
                    new_ai = self.discover_new_ai()
                    
                    if new_ai:
                        logger.info(f"🔍 Discovered {len(new_ai)} new AI systems")
                        for ai in new_ai:
                            if isinstance(ai, dict):
                                self.research_ai_system(ai.get('name', str(ai)))
                            else:
                                self.research_ai_system(ai)
                    
                    # Research known AI systems
                    for ai in self.known_ai_systems:
                        if ai not in self.discovered_ai.get('researched', []):
                            self.research_ai_system(ai)
                    
                    # Check trending repos
                    trending = self.check_trending_repos()
                    if trending:
                        logger.info(f"📈 Trending repos: {len(trending)}")
                        for repo in trending:
                            self.analyze_repo_for_integration(repo)
                    
                    # Process integration queue
                    self._process_integration_queue()
                    
                    # Sleep before next cycle (adaptive - 1 hour)
                    time.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"Discovery loop error: {e}")
                    time.sleep(300)
        
        logger.info("🔍 Autonomous discovery loop started")
        # Run in the caller's thread (caller already spawned a daemon thread
        # named 'dmai-ai-discovery' so self-healer can monitor it).
        discover()
    
    def _process_integration_queue(self):
        """Process systems waiting for integration"""
        for system in self.integration_queue[:]:
            if system in self.discovered_ai:
                info = self.discovered_ai[system]
                if info.get('api_available') and info.get('api_endpoint'):
                    # Add to AI Hub if available
                    if self.ai_hub:
                        self.ai_hub.integrate_discovered_model(
                            name=system,
                            endpoint=info['api_endpoint'],
                            capabilities=info.get('capabilities', [])
                        )
                        self.mark_integrated(system)
                        logger.info(f"🔌 Integrated {system} into AI Hub")
    
    def discover_new_ai(self) -> List[str]:
        """
        Discover new AI systems from various sources
        """
        discovered = []
        
        for source_name, source in self.discovery_sources.items():
            if not source.get('active'):
                continue
            
            try:
                if source_name == 'github_trending':
                    new = self._scan_github_trending()
                elif source_name == 'huggingface':
                    new = self._scan_huggingface()
                elif source_name == 'arxiv':
                    new = self._scan_arxiv()
                elif source_name == 'papers_with_code':
                    new = self._scan_papers_with_code()
                elif source_name == 'reddit_ml':
                    new = self._scan_reddit()
                else:
                    continue
                
                discovered.extend(new)
            except Exception as e:
                logger.error(f"Error scanning {source_name}: {e}")
        
        # Deduplicate and filter
        discovered = list(set(discovered))
        new_ai = [ai for ai in discovered if ai not in self.discovered_ai.get('discovered', [])]
        
        if new_ai:
            if 'discovered' not in self.discovered_ai:
                self.discovered_ai['discovered'] = []
            self.discovered_ai['discovered'].extend(new_ai)
            self._save_discovered()
        
        return new_ai
    
    def _scan_github_trending(self) -> List[str]:
        """Scan GitHub for trending AI repos using API with token"""
        try:
            headers = {}
            if self.current_github_token:
                headers['Authorization'] = f'Bearer {self.current_github_token}'
            
            # Search for AI-related repos with high stars
            params = {
                'q': 'topic:ai OR topic:llm OR topic:machine-learning',
                'sort': 'stars',
                'order': 'desc',
                'per_page': 20
            }
            
            response = requests.get(
                'https://api.github.com/search/repositories',
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                repos = []
                for item in data.get('items', []):
                    repo_name = item.get('full_name', '')
                    description = item.get('description', '')
                    # Filter for AI-related repos
                    if any(term in (repo_name + description).lower() 
                           for term in ['ai', 'ml', 'llm', 'gpt', 'agent', 'neural', 'transformer']):
                        repos.append(repo_name)
                return repos
            elif response.status_code == 403 and 'rate limit' in response.text.lower():
                logger.warning("GitHub rate limit hit - consider using token for higher limits")
                return []
                
        except Exception as e:
            logger.error(f"GitHub trending scan error: {e}")
        return []
    
    def _scan_huggingface(self) -> List[str]:
        """Scan HuggingFace for new models"""
        try:
            headers = {}
            if self.huggingface_token:
                headers['Authorization'] = f'Bearer {self.huggingface_token}'
            
            params = {
                'sort': 'downloads',
                'direction': -1,
                'limit': 30
            }
            
            response = requests.get(
                'https://huggingface.co/api/models',
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                models = response.json()
                model_names = []
                for model in models[:20]:
                    model_id = model.get('modelId', '')
                    if model_id and len(model_id) > 3:
                        # Extract name from model ID (e.g., "meta-llama/Llama-2-7b" -> "Llama")
                        name_parts = model_id.split('/')
                        if len(name_parts) > 1:
                            model_names.append(name_parts[1].split('-')[0])
                        else:
                            model_names.append(model_id.split('-')[0])
                return list(set(model_names))
                
        except Exception as e:
            logger.error(f"HuggingFace scan error: {e}")
        return []
    
    def _scan_arxiv(self) -> List[str]:
        """Scan ArXiv for recent AI papers and extract system names"""
        try:
            response = requests.get(
                self.discovery_sources['arxiv']['url'],
                timeout=30
            )
            
            if response.status_code == 200:
                feed = feedparser.parse(response.text)
                ai_mentions = []
                
                for entry in feed.entries[:15]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    # Look for capitalized AI system names
                    words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', title + ' ' + summary)
                    ai_mentions.extend(words)
                
                # Filter for potential AI system names (length > 2, not common words)
                common_words = {'The', 'And', 'For', 'With', 'From', 'This', 'That', 'These', 'Those'}
                ai_mentions = [w for w in ai_mentions if len(w) > 2 and w not in common_words]
                
                return list(set(ai_mentions[:10]))
                
        except Exception as e:
            logger.error(f"ArXiv scan error: {e}")
        return []
    
    def _scan_papers_with_code(self) -> List[str]:
        """Scan Papers with Code for new research"""
        try:
            response = requests.get(
                'https://paperswithcode.com/api/v1/papers/',
                params={'limit': 20},
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Papers with Code API returned status {response.status_code} — skipping this cycle")
                return []

            text = response.text.strip() if response.text else ""
            if not text:
                logger.warning("Papers with Code API returned empty body — skipping this cycle")
                return []

            try:
                data = response.json()
            except (ValueError, Exception) as json_err:
                logger.warning(f"Papers with Code API non-JSON response — skipping this cycle ({json_err})")
                return []

            papers = data.get('results', []) if isinstance(data, dict) else []
            ai_names = []

            for paper in papers:
                title = paper.get('title', '')
                # Extract potential AI system names
                words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', title)
                ai_names.extend(words)

            return list(set(ai_names[:10]))

        except Exception as e:
            logger.warning(f"Papers with Code scan error (non-fatal): {e}")
        return []
    
    def _scan_reddit(self) -> List[str]:
        """Scan Reddit for AI discussions"""
        try:
            response = requests.get(
                'https://www.reddit.com/r/MachineLearning/new.json',
                headers={'User-Agent': 'DMAI/1.0'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                ai_mentions = []
                
                for post in posts[:20]:
                    title = post.get('data', {}).get('title', '')
                    # Extract potential AI system names
                    words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', title)
                    ai_mentions.extend(words)
                
                return list(set(ai_mentions[:15]))
                
        except Exception as e:
            logger.error(f"Reddit scan error: {e}")
        return []
    
    def research_ai_system(self, system_name: str) -> Dict:
        """
        Research an AI system to understand its capabilities
        """
        if system_name in self.discovered_ai.get('researched', []):
            return self.discovered_ai.get(system_name, {})
        
        logger.info(f"🔬 Researching AI system: {system_name}")
        
        research = {
            'name': system_name,
            'researched_at': datetime.now().isoformat(),
            'capabilities': [],
            'api_available': False,
            'api_endpoint': None,
            'api_key_required': True,
            'documentation_url': None,
            'github_repo': None,
            'integration_status': 'pending',
            'notes': ''
        }
        
        # Search for documentation
        doc_urls = self._search_documentation(system_name)
        research['documentation_url'] = doc_urls[0] if doc_urls else None
        
        # Check if API is available
        api_info = self._check_api_availability(system_name)
        research['api_available'] = api_info['available']
        research['api_endpoint'] = api_info['endpoint']
        research['api_key_required'] = api_info['key_required']
        
        # Find GitHub repo
        repo = self._find_github_repo(system_name)
        research['github_repo'] = repo
        
        # Determine capabilities
        research['capabilities'] = self._determine_capabilities(system_name)
        
        # Store research
        if 'researched' not in self.discovered_ai:
            self.discovered_ai['researched'] = []
        self.discovered_ai['researched'].append(system_name)
        self.discovered_ai[system_name] = research
        self._save_discovered()
        
        # Add to integration queue if API available
        if research['api_available']:
            self.integration_queue.append(system_name)
            logger.info(f"🔌 {system_name} has API - added to integration queue")
        
        return research
    
    def _search_documentation(self, system_name: str) -> List[str]:
        """Search for documentation URLs"""
        # Common documentation patterns
        base_name = system_name.lower().replace(' ', '').replace('-', '')
        doc_urls = [
            f"https://docs.{base_name}.com",
            f"https://{base_name}.com/docs",
            f"https://developers.{base_name}.com",
            f"https://github.com/{base_name}/{base_name}/wiki",
            f"https://huggingface.co/docs/{base_name}"
        ]
        return doc_urls
    
    def _check_api_availability(self, system_name: str) -> Dict:
        """Check if the AI system has an API"""
        # Known APIs (expanded)
        known_apis = {
            'Claude': {'available': True, 'endpoint': 'https://api.anthropic.com', 'key_required': True},
            'ChatGPT': {'available': True, 'endpoint': 'https://api.openai.com', 'key_required': True},
            'Gemini': {'available': True, 'endpoint': 'https://generativelanguage.googleapis.com', 'key_required': True},
            'DeepSeek': {'available': True, 'endpoint': 'https://api.deepseek.com', 'key_required': True},
            'Perplexity': {'available': True, 'endpoint': 'https://api.perplexity.ai', 'key_required': True},
            'Grok': {'available': True, 'endpoint': 'https://api.x.ai', 'key_required': True},
            'HuggingFace': {'available': True, 'endpoint': 'https://api-inference.huggingface.co', 'key_required': True},
            'GitHub Copilot': {'available': True, 'endpoint': 'https://api.github.com/copilot', 'key_required': True},
            'Midjourney': {'available': False, 'endpoint': None, 'key_required': True},
            'Stable Diffusion': {'available': True, 'endpoint': 'https://api.stability.ai', 'key_required': True},
            'RunwayML': {'available': True, 'endpoint': 'https://api.runwayml.com', 'key_required': True},
            'Llama': {'available': True, 'endpoint': 'https://api.llama.com', 'key_required': False},
            'Mistral': {'available': True, 'endpoint': 'https://api.mistral.ai', 'key_required': True},
        }
        
        # Case-insensitive lookup
        for key, value in known_apis.items():
            if key.lower() in system_name.lower():
                return value
        
        return {'available': False, 'endpoint': None, 'key_required': True}
    
    def _find_github_repo(self, system_name: str) -> Optional[str]:
        """Find GitHub repository for the AI system"""
        try:
            headers = {}
            if self.current_github_token:
                headers['Authorization'] = f'Bearer {self.current_github_token}'
            
            response = requests.get(
                "https://api.github.com/search/repositories",
                headers=headers,
                params={'q': f"{system_name} AI", 'sort': 'stars', 'per_page': 1},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    return data['items'][0]['html_url']
                    
        except Exception as e:
            logger.error(f"GitHub search error for {system_name}: {e}")
        return None
    
    def _determine_capabilities(self, system_name: str) -> List[str]:
        """Determine what the AI system can do"""
        capabilities = []
        name_lower = system_name.lower()
        
        # Text AI
        text_ai = ['claude', 'chatgpt', 'gemini', 'deepseek', 'perplexity', 'llama', 'mistral', 'grok']
        if any(term in name_lower for term in text_ai):
            capabilities.append('text_generation')
            capabilities.append('conversation')
        
        # Image AI
        image_ai = ['midjourney', 'stable diffusion', 'dall-e', 'imagen', 'flux']
        if any(term in name_lower for term in image_ai):
            capabilities.append('image_generation')
        
        # Video AI
        video_ai = ['runwayml', 'pika', 'sora', 'kling', 'opal']
        if any(term in name_lower for term in video_ai):
            capabilities.append('video_generation')
        
        # Code AI
        code_ai = ['windsurf', 'lovable', 'cursor', 'copilot', 'replit', 'codeium', 'tabnine']
        if any(term in name_lower for term in code_ai):
            capabilities.append('code_generation')
        
        # Research/Knowledge AI
        research_ai = ['perplexity', 'notebooklm']
        if any(term in name_lower for term in research_ai):
            capabilities.append('research')
            capabilities.append('web_search')
        
        # Model Hub
        if 'huggingface' in name_lower:
            capabilities.append('model_hub')
            capabilities.append('model_inference')
        
        # GitHub
        if 'github' in name_lower:
            capabilities.append('code_repository')
            capabilities.append('version_control')
        
        return capabilities if capabilities else ['general']
    
    def analyze_repo_for_integration(self, repo_name: str) -> Dict:
        """
        Analyze a GitHub repo for integration potential
        """
        logger.info(f"📦 Analyzing repo: {repo_name}")
        
        analysis = {
            'repo': repo_name,
            'analyzed_at': datetime.now().isoformat(),
            'language': 'unknown',
            'has_api': False,
            'has_python_bindings': False,
            'integration_difficulty': 'unknown',
            'useful_for': [],
            'status': 'pending'
        }
        
        # Get repo details
        try:
            headers = {}
            if self.current_github_token:
                headers['Authorization'] = f'Bearer {self.current_github_token}'
            
            response = requests.get(
                f"https://api.github.com/repos/{repo_name}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis['language'] = data.get('language', 'unknown')
                analysis['description'] = data.get('description', '')
                analysis['stars'] = data.get('stargazers_count', 0)
                analysis['forks'] = data.get('forks_count', 0)
                analysis['last_updated'] = data.get('updated_at', '')
                
                # Determine usefulness
                desc_lower = analysis.get('description', '').lower()
                
                if 'python' in desc_lower:
                    analysis['has_python_bindings'] = True
                    analysis['integration_difficulty'] = 'medium'
                    analysis['useful_for'].append('python_integration')
                
                if 'api' in desc_lower:
                    analysis['has_api'] = True
                    analysis['integration_difficulty'] = 'easy'
                
                if 'llm' in desc_lower or 'gpt' in desc_lower or 'transformer' in desc_lower:
                    analysis['useful_for'].append('language_model')
                
                if 'training' in desc_lower or 'fine-tune' in desc_lower:
                    analysis['useful_for'].append('model_training')
                
                # Check for Python package
                analysis['has_pypi'] = self._check_pypi_exists(repo_name.split('/')[-1])
                
        except Exception as e:
            logger.error(f"Repo analysis error for {repo_name}: {e}")
        
        # Store analysis
        safe_name = repo_name.replace('/', '_')
        analysis_file = self.discovered_path / f"repo_{safe_name}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _check_pypi_exists(self, package_name: str) -> bool:
        """Check if a package exists on PyPI"""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def check_trending_repos(self) -> List[str]:
        """Get trending AI repos from GitHub"""
        try:
            headers = {}
            if self.current_github_token:
                headers['Authorization'] = f'Bearer {self.current_github_token}'
            
            params = {
                'q': 'topic:ai',
                'sort': 'stars',
                'order': 'desc',
                'per_page': 10
            }
            
            response = requests.get(
                'https://api.github.com/search/repositories',
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return [item['full_name'] for item in data.get('items', [])[:10]]
            elif response.status_code == 403 and 'rate limit' in response.text.lower():
                logger.warning("GitHub rate limit hit for trending repos")
                return []
                
        except Exception as e:
            logger.error(f"Trending repos error: {e}")
        return []
    
    def get_integration_queue(self) -> List[Dict]:
        """Get systems ready for integration"""
        queue = []
        for system in self.integration_queue:
            if system in self.discovered_ai:
                queue.append({
                    'name': system,
                    'capabilities': self.discovered_ai[system].get('capabilities', []),
                    'api_endpoint': self.discovered_ai[system].get('api_endpoint'),
                    'integration_status': self.discovered_ai[system].get('integration_status', 'pending')
                })
        return queue
    
    def mark_integrated(self, system_name: str):
        """Mark a system as integrated"""
        if system_name in self.discovered_ai:
            self.discovered_ai[system_name]['integration_status'] = 'integrated'
            if 'integrated' not in self.discovered_ai:
                self.discovered_ai['integrated'] = []
            if system_name not in self.discovered_ai['integrated']:
                self.discovered_ai['integrated'].append(system_name)
            if system_name in self.integration_queue:
                self.integration_queue.remove(system_name)
            self._save_discovered()
            logger.info(f"✅ Integrated {system_name} into DMAI")
    
    def get_status(self) -> Dict:
        """Get discovery system status"""
        return {
            'discovery_active': self.discovery_active,
            'discovered_count': len(self.discovered_ai.get('discovered', [])),
            'researched_count': len(self.discovered_ai.get('researched', [])),
            'integrated_count': len(self.discovered_ai.get('integrated', [])),
            'queue_size': len(self.integration_queue),
            'active_sources': sum(1 for s in self.discovery_sources.values() if s.get('active')),
            'github_token_configured': bool(self.current_github_token),
            'huggingface_token_configured': bool(self.huggingface_token),
            'last_discovery_run': self.last_discovery_run.isoformat() if self.last_discovery_run else None,
            'last_update': datetime.now().isoformat()
        }


# For testing
if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    
    print("=" * 60)
    print("Dynamic AI Discovery Test")
    print("=" * 60)
    
    discovery = DynamicAIDiscovery(Path("./data"))
    
    print("\nStatus:")
    print(json.dumps(discovery.get_status(), indent=2))
    
    print("\nTesting discovery...")
    new_ai = discovery.discover_new_ai()
    print(f"Discovered: {new_ai[:10]}...")
    
    print("\nTesting repo analysis...")
    if new_ai:
        analysis = discovery.analyze_repo_for_integration(new_ai[0])
        print(json.dumps(analysis, indent=2))
    
    print("\n✅ Dynamic AI Discovery ready")
