#!/usr/bin/env python3
"""
DMAI GitHub Starred Repos Scanner
Fetches all starred repositories and feeds them to the Repo Integration Engine.
Monitors for new stars and auto-queues them.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class GitHubStarredScanner:
    """Scans starred GitHub repos and integrates them into DMAI"""
    
    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.scan_history_file = Path("data/github_starred_history.json")
        self.scan_history_file.parent.mkdir(parents=True, exist_ok=True)
        self.scan_history = self._load_scan_history()
        self.tokens = self._load_tokens()
    
    def _load_tokens(self) -> List[str]:
        """Load GitHub tokens from environment and harvester"""
        tokens = []
        # Check environment
        for key in ['GITHUB_TOKEN', 'GITHUB_ACCESS_TOKEN']:
            token = os.getenv(key)
            if token:
                tokens.append(token)
        # Check harvester
        if hasattr(self.dmai, 'api_harvester') and hasattr(self.dmai.api_harvester, 'github_tokens'):
            tokens.extend(self.dmai.api_harvester.github_tokens)
        # Check AI hub
        if hasattr(self.dmai, 'ai_hub') and hasattr(self.dmai.ai_hub, 'api_keys'):
            for key_name in ['github_main', 'github_secondary', 'github']:
                token = self.dmai.ai_hub.api_keys.get(key_name)
                if token and token != 'pending':
                    tokens.append(token)
        logger.info(f"🔑 Loaded {len(set(tokens))} unique GitHub tokens")
        return list(set(tokens))
    
    def _load_scan_history(self) -> Dict:
        if self.scan_history_file.exists():
            try:
                with open(self.scan_history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'scanned_repos': {}, 'last_scan': None}
    
    def _save_scan_history(self):
        with open(self.scan_history_file, 'w') as f:
            json.dump(self.scan_history, f, indent=2, default=str)
    
    def _get_headers(self) -> Dict:
        """Get auth headers using available token"""
        if self.tokens:
            return {
                'Authorization': f'token {self.tokens[0]}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'DMAI-Starred-Repo-Scanner'
            }
        return {'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'DMAI-Starred-Repo-Scanner'}
    
    def scan_starred_repos(self, username: str = None, max_pages: int = 10) -> Dict:
        """
        Fetch all starred repositories for a GitHub user.
        If no username provided, tries to determine from token.
        """
        result = {
            'scanned_at': datetime.now().isoformat(),
            'repos_found': 0,
            'repos_queued': 0,
            'repos_already_processed': 0,
            'errors': []
        }
        
        headers = self._get_headers()
        
        # Determine username
        if not username:
            username = self._get_authenticated_user(headers)
        
        if not username:
            result['errors'].append("No GitHub username provided and couldn't determine from token")
            return result
        
        logger.info(f"⭐ Scanning starred repos for: {username}")
        
        try:
            page = 1
            while page <= max_pages:
                url = f'https://api.github.com/users/{username}/starred?per_page=100&page={page}'
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 403:
                    logger.warning("GitHub rate limit hit")
                    break
                if response.status_code == 404:
                    result['errors'].append(f"User not found: {username}")
                    break
                if response.status_code != 200:
                    result['errors'].append(f"GitHub API error: {response.status_code}")
                    break
                
                repos = response.json()
                if not repos:
                    break
                
                result['repos_found'] += len(repos)
                
                for repo in repos:
                    self._process_starred_repo(repo, result)
                
                page += 1
            
            self.scan_history['last_scan'] = datetime.now().isoformat()
            self._save_scan_history()
            
            logger.info(f"⭐ Starred scan complete: {result['repos_found']} found, {result['repos_queued']} queued")
            
        except Exception as e:
            logger.error(f"Starred repo scan failed: {e}")
            result['errors'].append(str(e))
        
        return result
    
    def _get_authenticated_user(self, headers: Dict) -> Optional[str]:
        """Get the authenticated user's login from GitHub API"""
        try:
            response = requests.get('https://api.github.com/user', headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json().get('login')
        except Exception:
            pass
        return None
    
    def _process_starred_repo(self, repo: Dict, result: Dict):
        """Process a single starred repo — queue it if new or updated"""
        repo_name = repo.get('full_name', '')
        repo_url = repo.get('html_url', '')
        repo_id = str(repo.get('id', ''))
        pushed_at = repo.get('pushed_at', '')
        stars = repo.get('stargazers_count', 0)
        language = repo.get('language', '')
        description = repo.get('description', '')
        
        # Check if already scanned
        if repo_id in self.scan_history['scanned_repos']:
            prev = self.scan_history['scanned_repos'][repo_id]
            if prev.get('pushed_at') == pushed_at:
                result['repos_already_processed'] += 1
                return
        
        # Determine priority
        priority = self._determine_priority(repo)
        
        # Queue in integration engine
        if hasattr(self.dmai, 'integration_engine') and self.dmai.integration_engine:
            try:
                queue_result = self.dmai.integration_engine.add_to_queue(
                    repo_url,
                    priority=priority,
                    repo_name=repo_name
                )
                if queue_result.get('status') in ['queued', 'already_queued']:
                    result['repos_queued'] += 1
                    logger.info(f"⭐ Queued: {repo_name} (P{priority}, {stars}★, {language})")
            except Exception as e:
                logger.warning(f"Failed to queue {repo_name}: {e}")
        
        # Record in history
        self.scan_history['scanned_repos'][repo_id] = {
            'full_name': repo_name,
            'url': repo_url,
            'pushed_at': pushed_at,
            'stars': stars,
            'language': language,
            'description': description[:200] if description else '',
            'scanned_at': datetime.now().isoformat()
        }
    
    def _determine_priority(self, repo: Dict) -> int:
        """Determine integration priority based on repo characteristics"""
        name = (repo.get('full_name', '') + ' ' + repo.get('description', '')).lower()
        stars = repo.get('stargazers_count', 0)
        language = (repo.get('language') or '').lower()
        
        # P0: Critical AI infrastructure with high stars
        if any(kw in name for kw in ['claude', 'gpt', 'llama', 'deepseek', 'transformer', 'mlx']):
            return 0 if stars > 100 else 1
        
        # P0: AI models and inference
        if any(kw in name for kw in ['ai-model', 'inference', 'neural', 'fine-tun', 'rag']):
            return 0 if stars > 500 else 1
        
        # P1: Funding, trading, automation
        if any(kw in name for kw in ['trading', 'funding', 'finance', 'automaton', 'arbitrage']):
            return 1
        
        # P1: Security, safety
        if any(kw in name for kw in ['security', 'safety', 'jailbreak', 'guardrail', 'alignment']):
            return 1
        
        # P2: Tools, frameworks
        if any(kw in name for kw in ['tool', 'framework', 'sdk', 'api', 'cli', 'agent']):
            return 2
        
        # Default: based on stars
        if stars > 1000:
            return 1
        elif stars > 100:
            return 2
        
        return 2
    
    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            'tokens_available': len(self.tokens),
            'repos_scanned': len(self.scan_history.get('scanned_repos', {})),
            'last_scan': self.scan_history.get('last_scan')
        }


print("✅ GitHub Starred Repos Scanner ready")
