# components/phase10/GitHubStarMonitor.py

import os
import json
import requests
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class GitHubStarMonitor:
    """
    Monitors GitHub stars and automatically processes new repositories
    """
    
    def __init__(self, data_path: Path, github_username: str, github_token: str = None):
        self.data_path = Path(data_path)  # Ensure it's a Path object
        self.github_username = github_username
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.processed_file = self.data_path / 'github_processed_stars.json'
        self.processed = self._load_processed()
        self.monitoring = False
        self.check_interval = 86400  # Check every 24 hours
        self.headers = {}
        
        if self.github_token:
            self.headers['Authorization'] = f'token {self.github_token}'
        
        logger.info(f"⭐ GitHub Star Monitor initialized for @{github_username}")
    
    def _load_processed(self):
        """Load list of already processed repos"""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load processed stars: {e}")
        return {"processed": [], "last_check": None}
    
    def _save_processed(self):
        """Save processed repos list"""
        try:
            with open(self.processed_file, 'w') as f:
                json.dump(self.processed, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processed stars: {e}")
    
    def fetch_starred_repos(self):
        """Fetch all starred repositories"""
        url = f"https://api.github.com/users/{self.github_username}/starred"
        repos = []
        page = 1
        
        while True:
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params={'page': page, 'per_page': 100},
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch stars: {response.status_code}")
                    return []  # Return empty list on error
                
                data = response.json()
                if data is None:
                    logger.warning("GitHub API returned None")
                    return []
                
                if not data:
                    break
                
                repos.extend(data)
                page += 1
                
            except requests.exceptions.Timeout:
                logger.error("GitHub API timeout")
                return []
            except Exception as e:
                logger.error(f"Error fetching stars: {e}")
                return []
        
        return repos
    
    def analyze_repo(self, repo):
        """
        Analyze a repository and determine how to use it
        """
        name = repo.get('full_name')
        description = repo.get('description', '')
        language = repo.get('language', 'Unknown')
        stars = repo.get('stargazers_count', 0)
        url = repo.get('html_url')
        clone_url = repo.get('clone_url')
        
        analysis = {
            "name": name,
            "url": url,
            "clone_url": clone_url,
            "language": language,
            "stars": stars,
            "description": description,
            "type": "unknown",
            "integration_plan": None
        }
        
        # Determine repo type based on language and description
        if language == "Python":
            analysis["type"] = "python_library"
            analysis["integration_plan"] = {
                "action": "analyze_and_integrate",
                "steps": [
                    "clone_repo",
                    "analyze_code_structure",
                    "extract_useful_modules",
                    "integrate_into_core_or_agent",
                    "create_api_wrapper_if_needed"
                ]
            }
        elif language in ["JavaScript", "TypeScript"]:
            analysis["type"] = "javascript_library"
            analysis["integration_plan"] = {
                "action": "extract_concepts",
                "steps": [
                    "clone_repo",
                    "analyze_patterns",
                    "extract_algorithms",
                    "port_to_python_if_useful"
                ]
            }
        elif "AI" in description or "machine learning" in description.lower():
            analysis["type"] = "ai_tool"
            analysis["integration_plan"] = {
                "action": "research_and_integrate",
                "steps": [
                    "clone_repo",
                    "research_capabilities",
                    "test_in_isolation",
                    "create_agent_connector"
                ]
            }
        else:
            analysis["type"] = "reference"
            analysis["integration_plan"] = {
                "action": "learn_from",
                "steps": [
                    "clone_repo",
                    "extract_documentation",
                    "learn_patterns",
                    "store_in_knowledge_base"
                ]
            }
        
        return analysis
    
    def process_repo(self, repo):
        """
        Process a new repository
        """
        name = repo.get('full_name')
        analysis = self.analyze_repo(repo)
        
        logger.info(f"📦 Processing new starred repo: {name}")
        logger.info(f"   Type: {analysis['type']}")
        logger.info(f"   Language: {analysis['language']}")
        logger.info(f"   Stars: {analysis['stars']}")
        
        # Store the analysis for DMAI to work on
        task_file = self.data_path / 'github_task.json'
        task_data = {
            "type": "github_repo",
            "repo": name,
            "analysis": analysis,
            "assigned_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        try:
            with open(task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save github task: {e}")
        
        # Also log to master task
        self._add_to_master_task(name, analysis)
        
        # Mark as processed
        self.processed["processed"].append({
            "name": name,
            "processed_at": datetime.now().isoformat(),
            "type": analysis["type"]
        })
        self.processed["last_check"] = datetime.now().isoformat()
        self._save_processed()
    
    def _add_to_master_task(self, repo_name, analysis):
        """Add to master task queue"""
        task_file = 'data/master_task.json'
        if os.path.exists(task_file):
            try:
                with open(task_file, 'r') as f:
                    master_task = json.load(f)
                
                # Append repo task to existing or create new
                if "subtasks" not in master_task:
                    master_task["subtasks"] = []
                
                master_task["subtasks"].append({
                    "type": "github_repo",
                    "repo": repo_name,
                    "analysis": analysis,
                    "status": "pending"
                })
                
                with open(task_file, 'w') as f:
                    json.dump(master_task, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed to add to master task: {e}")
    
    def run_monitor(self):
        """Main monitoring loop"""
        self.monitoring = True
        
        while self.monitoring:
            try:
                # Fetch current stars
                repos = self.fetch_starred_repos()
                
                # Check if repos is valid
                if repos is None:
                    logger.warning("GitHub API returned None, waiting before retry")
                    time.sleep(300)  # Wait 5 minutes on error
                    continue
                
                if repos:
                    # Check for new repos
                    processed_names = [p["name"] for p in self.processed["processed"]]
                    
                    for repo in repos:
                        name = repo.get('full_name')
                        if name not in processed_names:
                            self.process_repo(repo)
                else:
                    logger.debug("No repos fetched from GitHub")
                
                # Wait for next check
                for _ in range(self.check_interval):
                    if not self.monitoring:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Star monitor error: {e}")
                time.sleep(60)
    
    def start(self):
        """Start monitoring in background thread"""
        thread = threading.Thread(target=self.run_monitor, daemon=True)
        thread.start()
        logger.info("⭐ GitHub Star Monitor started")
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("⭐ GitHub Star Monitor stopped")
