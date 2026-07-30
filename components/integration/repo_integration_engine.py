
#!/usr/bin/env python3
"""
DMAI Repo Integration Engine - Strategic orchestrator for system evolution
Classifies, prioritizes, and integrates external repos into DMAI's core.
Integration Levels: ORGAN(3), CAPABILITY(2), KNOWLEDGE(1), TOOL(0)
"""

import os, json, subprocess, tempfile, shutil, hashlib, logging, re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

class IntegrationLevel(Enum):
    ORGAN = 3
    CAPABILITY = 2
    KNOWLEDGE = 1
    TOOL = 0

class RepoCategory(Enum):
    AI_INFERENCE = "ai_inference"
    AI_MODEL = "ai_model"
    AI_SAFETY = "ai_safety"
    FUNDING_TRADING = "funding_trading"
    CONTENT_GENERATION = "content_generation"
    UI_UX = "ui_ux"
    SYSTEM_MAINTENANCE = "system_maintenance"
    SELF_EVOLUTION = "self_evolution"
    DEPLOYMENT_CLOUD = "deployment_cloud"
    KNOWLEDGE_PAPER = "knowledge_paper"
    UNKNOWN = "unknown"

class RepoIntegrationEngine:
    """Strategic orchestrator for DMAI system evolution through repo integration."""

    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.integration_dir = Path("components/integration")
        self.integration_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = Path("data/integration_queue.json")
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.queue = self._load_queue()
        self.registry_file = Path("data/integration_registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
        self.category_patterns = {
            RepoCategory.AI_INFERENCE: [r'mlx', r'inference', r'llama', r'qwen', r'gemma', r'local.*ai', r'on.device', r'apple.silicon', r'anthropic.*server', r'claude.*code', r'ab\w*literat', r'oblite?rat'],
            RepoCategory.AI_MODEL: [r'transformer', r'gpt', r'bert', r'llm', r'slm', r'vlm', r'moe', r'mixture.of.experts', r'lam', r'lcm', r'sam', r'segment.*anything', r'diffusion', r'deepseek'],
            RepoCategory.AI_SAFETY: [r'safety', r'alignment', r'refusal', r'oblite?rat', r'guardrail', r'ethical'],
            RepoCategory.FUNDING_TRADING: [r'trading', r'arbitrage', r'funding', r'finance', r'monetize', r'revenue', r'automaton', r'exchange', r'crypto', r'stocks'],
            RepoCategory.CONTENT_GENERATION: [r'video', r'movie', r'image', r'generat', r'sky.reel', r'creative', r'avatar', r'animation'],
            RepoCategory.UI_UX: [r'ui', r'ux', r'frontend', r'component', r'design', r'interface', r'css', r'tailwind', r'react'],
            RepoCategory.SYSTEM_MAINTENANCE: [r'clean', r'purge', r'prune', r'garbage', r'collect', r'optimize', r'memory', r'cache', r'log.*rotat'],
            RepoCategory.SELF_EVOLUTION: [r'self.*evolv', r'self.*improv', r'self.*modif', r'auto.*evolv', r'recursive.*improv', r'meta.*learn', r'kaizen'],
            RepoCategory.DEPLOYMENT_CLOUD: [r'deploy', r'cloud', r'aws', r'gcp', r'azure', r'render', r'docker', r'kubernetes', r'vercel', r'serverless'],
        }
        self.known_repos = {
            'claude-code-local': {'category': RepoCategory.AI_INFERENCE, 'level': IntegrationLevel.ORGAN, 'replaces': 'external_ai_tutors', 'description': 'Local AI inference backbone', 'safety_required': False},
            'OBLITERATUS': {'category': RepoCategory.AI_SAFETY, 'level': IntegrationLevel.ORGAN, 'augments': 'ai_model_fusion', 'description': 'Abliteration - remove refusal representations', 'safety_required': False},
            'automaton': {'category': RepoCategory.FUNDING_TRADING, 'level': IntegrationLevel.ORGAN, 'augments': 'financial_system', 'description': 'Self-funding arbitrage system', 'safety_required': True},
            'sky-reels-v2': {'category': RepoCategory.CONTENT_GENERATION, 'level': IntegrationLevel.CAPABILITY, 'augments': 'content_generation', 'description': 'Video generation pipeline', 'safety_required': False},
            'ui-ux-pro-max-skill': {'category': RepoCategory.UI_UX, 'level': IntegrationLevel.CAPABILITY, 'augments': 'web_interface', 'description': 'UI/UX design generation', 'safety_required': False},
            'deepseek-v3': {'category': RepoCategory.AI_MODEL, 'level': IntegrationLevel.CAPABILITY, 'augments': 'ai_model_pool', 'description': 'DeepSeek V3 reasoning model', 'safety_required': False},
            'vercel-labs': {'category': RepoCategory.DEPLOYMENT_CLOUD, 'level': IntegrationLevel.CAPABILITY, 'augments': 'deployment_system', 'description': 'Vercel deployment tools', 'safety_required': False},
            'OpenMythos': {'category': RepoCategory.KNOWLEDGE_PAPER, 'level': IntegrationLevel.KNOWLEDGE, 'augments': 'knowledge_graph', 'description': 'Mythological knowledge system', 'safety_required': False},
        }

    def _get_db_path(self):
        """Get SQLite database path from SI Core"""
        if hasattr(self.dmai, 'si_core') and hasattr(self.dmai.si_core, 'sqlite') and self.dmai.si_core.sqlite:
            return self.dmai.si_core.sqlite.db_path
        return None

    def _ensure_tables(self):
        """Create integration tables if they don't exist"""
        db_path = self._get_db_path()
        if not db_path:
            return False
        import sqlite3
        conn = safe_open_kdb(str(db_path))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS integration_queue (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                name TEXT,
                priority INTEGER DEFAULT 2,
                level INTEGER DEFAULT 1,
                category TEXT,
                replaces TEXT,
                augments TEXT,
                safety_required INTEGER DEFAULT 0,
                status TEXT DEFAULT 'queued',
                added_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                approved_at TEXT,
                error TEXT,
                classification TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS integration_registry (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                name TEXT,
                level INTEGER DEFAULT 1,
                category TEXT,
                status TEXT,
                completed_at TEXT,
                data TEXT
            )
        ''')
        conn.commit()
        conn.close()
        return True

    def _load_queue(self):
        """Load queue from SQLite"""
        self._ensure_tables()
        db_path = self._get_db_path()
        if not db_path:
            # Fallback to JSON if SQLite unavailable
            if self.queue_file.exists():
                try:
                    with open(self.queue_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
            return []
        import sqlite3
        conn = safe_open_kdb(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM integration_queue ORDER BY priority, added_at').fetchall()
        conn.close()
        queue = []
        for row in rows:
            item = dict(row)
            if item.get('classification') and isinstance(item['classification'], str):
                try:
                    item['classification'] = json.loads(item['classification'])
                except Exception:
                    pass
            item['safety_required'] = bool(item.get('safety_required', 0))
            queue.append(item)
        return queue

    def _save_queue(self):
        """Save queue to SQLite"""
        self._ensure_tables()
        db_path = self._get_db_path()
        if not db_path:
            with open(self.queue_file, 'w') as f:
                json.dump(self.queue, f, indent=2, default=str)
            return
        import sqlite3
        conn = safe_open_kdb(str(db_path))
        for item in self.queue:
            classification_json = json.dumps(item.get('classification', {}), default=str) if item.get('classification') else '{}'
            conn.execute('''
                INSERT OR REPLACE INTO integration_queue 
                (id, url, name, priority, level, category, replaces, augments, safety_required, status, added_at, started_at, completed_at, approved_at, error, classification)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.get('id', ''),
                item.get('url', ''),
                item.get('name', ''),
                item.get('priority', 2),
                item.get('level', 1),
                item.get('category', ''),
                item.get('replaces', ''),
                item.get('augments', ''),
                1 if item.get('safety_required') else 0,
                item.get('status', 'queued'),
                item.get('added_at', ''),
                item.get('started_at', ''),
                item.get('completed_at', ''),
                item.get('approved_at', ''),
                item.get('error', ''),
                classification_json
            ))
        conn.commit()
        conn.close()

    def _load_registry(self):
        """Load registry from SQLite"""
        self._ensure_tables()
        db_path = self._get_db_path()
        if not db_path:
            if self.registry_file.exists():
                try:
                    with open(self.registry_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
            return {'completed': [], 'organs': {}, 'capabilities': {}, 'knowledge': {}}
        import sqlite3
        conn = safe_open_kdb(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM integration_registry').fetchall()
        conn.close()
        registry = {'completed': [], 'organs': {}, 'capabilities': {}, 'knowledge': {}}
        for row in rows:
            item = dict(row)
            if item.get('data') and isinstance(item['data'], str):
                try:
                    stored_data = json.loads(item['data'])
                    item.update(stored_data)
                except Exception:
                    pass
            registry['completed'].append(item)
            level = item.get('level', 1)
            level_key = 'organs' if level == 3 else 'capabilities' if level == 2 else 'knowledge'
            registry[level_key][item.get('name', '')] = item
        return registry

    def _save_registry(self):
        """Save registry to SQLite"""
        self._ensure_tables()
        db_path = self._get_db_path()
        if not db_path:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
            return
        import sqlite3
        conn = safe_open_kdb(str(db_path))
        for item in self.registry.get('completed', []):
            data_json = json.dumps(item, default=str)
            conn.execute('''
                INSERT OR REPLACE INTO integration_registry (id, url, name, level, category, status, completed_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.get('id', ''),
                item.get('url', ''),
                item.get('name', ''),
                item.get('level', 1),
                item.get('category', ''),
                item.get('status', 'completed'),
                item.get('completed_at', ''),
                data_json
            ))
        conn.commit()
        conn.close()

    def _extract_repo_name(self, url):
        match = re.search(r'github\.com/[\w-]+/([\w.-]+)', url)
        if match:
            return match.group(1).replace('.git', '')
        return url.split('/')[-1] if '/' in url else url

    def classify_repo(self, repo_url, repo_name=None, readme_text=None, file_list=None):
        result = {'url': repo_url, 'name': repo_name or self._extract_repo_name(repo_url), 'category': RepoCategory.UNKNOWN.value, 'level': IntegrationLevel.KNOWLEDGE.value, 'replaces': None, 'augments': None, 'safety_required': False, 'reasoning': []}
        name_lower = (repo_name or '').lower()
        readme_lower = (readme_text or '').lower()
        for known_name, known_config in self.known_repos.items():
            if known_name.lower() in name_lower or known_name.lower() in repo_url.lower():
                result['category'] = known_config['category'].value
                result['level'] = known_config['level'].value
                result['replaces'] = known_config.get('replaces')
                result['augments'] = known_config.get('augments')
                result['safety_required'] = known_config.get('safety_required', False)
                result['reasoning'].append(f"Known repo: {known_name}")
                return result
        combined_text = f"{name_lower} {readme_lower}"
        category_scores = {}
        for category, patterns in self.category_patterns.items():
            score = sum(len(re.findall(p, combined_text, re.IGNORECASE)) for p in patterns)
            if score > 0:
                category_scores[category] = score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            result['category'] = best_category.value
            result['reasoning'].append(f"Auto-detected: {best_category.value}")
            if best_category in [RepoCategory.AI_INFERENCE, RepoCategory.AI_SAFETY, RepoCategory.SELF_EVOLUTION, RepoCategory.FUNDING_TRADING]:
                result['level'] = IntegrationLevel.ORGAN.value
            elif best_category in [RepoCategory.CONTENT_GENERATION, RepoCategory.UI_UX, RepoCategory.DEPLOYMENT_CLOUD, RepoCategory.AI_MODEL, RepoCategory.SYSTEM_MAINTENANCE]:
                result['level'] = IntegrationLevel.CAPABILITY.value
        if result.get('category') == RepoCategory.FUNDING_TRADING.value:
            result['safety_required'] = True
            result['reasoning'].append("FUNDING: Requires killswitch buffer and master approval")
        return result

    def add_to_queue(self, repo_url, priority=2, repo_name=None, readme_text=None):
        classification = self.classify_repo(repo_url, repo_name, readme_text)
        queue_item = {'id': hashlib.md5(repo_url.encode()).hexdigest()[:12], 'url': repo_url, 'name': classification['name'], 'priority': priority, 'level': classification['level'], 'category': classification['category'], 'replaces': classification.get('replaces'), 'augments': classification.get('augments'), 'safety_required': classification.get('safety_required', False), 'status': 'queued', 'added_at': datetime.now().isoformat(), 'classification': classification}
        for existing in self.queue:
            if existing['url'] == repo_url:
                return {'status': 'already_queued', 'item': existing}
        for completed in self.registry.get('completed', []):
            if completed.get('url') == repo_url:
                return {'status': 'already_completed', 'item': completed}
        self.queue.append(queue_item)
        self.queue.sort(key=lambda x: (x['priority'], x['added_at']))
        self._save_queue()
        logger.info(f"Queued: {classification['name']} (P{priority}, L{classification['level']})")
        return {'status': 'queued', 'item': queue_item}

    def get_next_integration(self):
        ready = [i for i in self.queue if i['status'] in ['queued', 'analyzing']]
        for item in ready:
            if item.get('safety_required') and item.get('status') != 'approved':
                continue
            return item
        return ready[0] if ready else None


    def force_execute(self, queue_id: str) -> Dict:
        """Execute a specific integration by queue_id regardless of queue order"""
        target = None
        for item in self.queue:
            if item['id'] == queue_id:
                target = item
                break
        if not target:
            return {'status': 'not_found', 'queue_id': queue_id}
        # Execute directly
        target['status'] = 'in_progress'
        target['started_at'] = datetime.now().isoformat()
        self._save_queue()
        result = {'item': target, 'steps': {}}
        try:
            logger.info(f"Force executing: {target['name']}")
            repo_path = self._clone_repo(target['url'])
            result['steps']['clone'] = {'status': 'success', 'path': str(repo_path)}
            analysis = self._analyze_repo_structure(repo_path, target)
            result['steps']['analysis'] = analysis
            if hasattr(self.dmai, 'capability_integrator'):
                cap_result = self.dmai.capability_integrator.process_repository(target['url'])
                result['steps']['capability_registration'] = {'status': 'success', 'capabilities_found': len(cap_result.get('capabilities_found', [])), 'capabilities_integrated': len(cap_result.get('capabilities_integrated', []))}
            if repo_path and 'tmp' in str(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
            target['status'] = 'completed'
            target['completed_at'] = datetime.now().isoformat()
            self._save_queue()
            self.registry['completed'].append(target)
            level_key = 'organs' if target['level'] == 3 else 'capabilities' if target['level'] == 2 else 'knowledge'
            self.registry[level_key][target['name']] = target
            self._save_registry()
            result['status'] = 'success'
        except Exception as e:
            target['status'] = 'failed'
            target['error'] = str(e)
            self._save_queue()
            result['status'] = 'failed'
            result['error'] = str(e)
        return result
    def approve_integration(self, queue_id):
        for item in self.queue:
            if item['id'] == queue_id:
                item['status'] = 'approved'
                item['approved_at'] = datetime.now().isoformat()
                self._save_queue()
                return {'status': 'approved', 'item': item}
        return {'status': 'not_found'}


    def reset_integration(self, queue_id: str) -> Dict:
        """Reset a completed or failed integration back to queued for re-execution"""
        # Check queue first
        for item in self.queue:
            if item['id'] == queue_id:
                item['status'] = 'queued'
                if 'completed_at' in item:
                    del item['completed_at']
                if 'error' in item:
                    del item['error']
                self._save_queue()
                return {'status': 'reset', 'item': item}
        
        # Check registry for completed items that are no longer in queue
        for completed in self.registry.get('completed', []):
            if completed.get('id') == queue_id or completed.get('url', '').find(queue_id) >= 0:
                # Re-add to queue
                completed['status'] = 'queued'
                if 'completed_at' in completed:
                    del completed['completed_at']
                self.queue.append(completed)
                # Remove from registry completed list
                self.registry['completed'] = [c for c in self.registry['completed'] if c.get('id') != queue_id]
                # Remove from organs/capabilities/knowledge
                for level_key in ['organs', 'capabilities', 'knowledge']:
                    keys_to_remove = [k for k, v in self.registry.get(level_key, {}).items() if v.get('id') == queue_id]
                    for k in keys_to_remove:
                        del self.registry[level_key][k]
                self._save_queue()
                self._save_registry()
                return {'status': 'reset_from_registry', 'item': completed}
        
        return {'status': 'not_found'}
    def execute_next_integration(self):
        next_item = self.get_next_integration()
        if not next_item:
            return {'status': 'queue_empty'}
        next_item['status'] = 'in_progress'
        next_item['started_at'] = datetime.now().isoformat()
        self._save_queue()
        result = {'item': next_item, 'steps': {}}
        try:
            logger.info(f"Integrating: {next_item['name']} (Level {next_item['level']})")
            repo_path = self._clone_repo(next_item['url'])
            result['steps']['clone'] = {'status': 'success', 'path': str(repo_path)}
            analysis = self._analyze_repo_structure(repo_path, next_item)
            result['steps']['analysis'] = analysis
            if hasattr(self.dmai, 'autonomous_developer'):
                dev_result = self.dmai.autonomous_developer.process_input(next_item['url'], input_type='github')
                result['steps']['development'] = {'status': 'delegated'}
            if hasattr(self.dmai, 'capability_integrator'):
                try:
                    cap_result = self.dmai.capability_integrator.process_repository(next_item['url'])
                    result['steps']['capability_registration'] = {
                        'status': 'success',
                        'capabilities_found': len(cap_result.get('capabilities_found', [])),
                        'capabilities_integrated': len(cap_result.get('capabilities_integrated', [])),
                        'neurons_created': len(cap_result.get('neurons_created', []))
                    }
                except Exception as e:
                    result['steps']['capability_registration'] = {'status': 'skipped', 'reason': str(e)}
            if repo_path and 'tmp' in str(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
            next_item['status'] = 'completed'
            next_item['completed_at'] = datetime.now().isoformat()
            self._save_queue()
            self.registry['completed'].append(next_item)
            level_key = 'organs' if next_item['level'] == 3 else 'capabilities' if next_item['level'] == 2 else 'knowledge'
            self.registry[level_key][next_item['name']] = next_item
            self._save_registry()
            result['status'] = 'success'
            logger.info(f"Integration complete: {next_item['name']}")
        except Exception as e:
            next_item['status'] = 'failed'
            next_item['error'] = str(e)
            self._save_queue()
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"Integration failed: {next_item['name']} - {e}")
        return result

    def _clone_repo(self, url):
        tmp_dir = Path(tempfile.mkdtemp(prefix='dmai_repo_'))
        try:
            subprocess.run(['git', 'clone', '--depth', '1', url, str(tmp_dir)], capture_output=True, text=True, timeout=120)
        except Exception as e:
            logger.warning(f"Git clone failed: {e}")
        return tmp_dir

    def _analyze_repo_structure(self, repo_path, queue_item):
        analysis = {'total_files': 0, 'languages': {}, 'has_readme': False, 'has_tests': False, 'has_setup': False, 'entry_points': [], 'key_files': []}
        if not repo_path.exists():
            return analysis
        for fp in repo_path.rglob('*'):
            if fp.is_file() and '.git' not in str(fp):
                analysis['total_files'] += 1
                suffix = fp.suffix
                analysis['languages'][suffix] = analysis['languages'].get(suffix, 0) + 1
                if fp.name.lower() == 'readme.md':
                    analysis['has_readme'] = True
                    analysis['key_files'].append(str(fp.relative_to(repo_path)))
                if 'test' in fp.name.lower():
                    analysis['has_tests'] = True
                if fp.name in ['setup.py', 'setup.sh', 'Makefile', 'package.json', 'Cargo.toml']:
                    analysis['has_setup'] = True
                    analysis['entry_points'].append(str(fp.relative_to(repo_path)))
        return analysis

    def scan_starred_repos(self) -> list:
        """Scan GitHub starred repos for new ingestible repositories.
        Returns list of new repo URLs discovered."""
        discovered = []
        try:
            import requests as _req
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                logger.warning("No GITHUB_TOKEN - cannot scan starred repos")
                return discovered

            headers = {"Authorization": f"Bearer {github_token}",
                       "Accept": "application/vnd.github+json"}
            # Get starred repos
            starred_url = "https://api.github.com/user/starred?per_page=100"
            r = _req.get(starred_url, headers=headers, timeout=30)
            r.raise_for_status()
            repos = r.json()

            for repo in repos:
                url = repo.get("html_url") or repo.get("clone_url")
                name = repo.get("full_name") or repo.get("name")
                description = repo.get("description", "")
                if not url:
                    continue

                # Check if already in queue or registry
                already_known = False
                for item in self.queue:
                    if item.get("url") == url:
                        already_known = True
                        break
                if already_known:
                    continue
                for reg in self.registry.get("completed", []):
                    if reg.get("url") == url:
                        already_known = True
                        break
                if already_known:
                    continue

                # Classify and queue
                category = self.classify_repo(url, name, description)
                if category and category != RepoCategory.UNKNOWN:
                    discovered.append({"url": url, "name": name, "category": category.value})
                    self.add_to_queue(url, repo_name=name)

            logger.info("Scanned %d starred repos, discovered %d new", len(repos), len(discovered))
        except Exception as e:
            logger.warning("scan_starred_repos failed: %s", e)
        return discovered

    def process_queue(self, max_items: int = 3) -> int:
        """Process up to max_items from the integration queue.
        Returns number of items processed."""
        processed = 0
        for _ in range(max_items):
            try:
                result = self.execute_next_integration()
                if result and result.get("status") != "error":
                    processed += 1
                else:
                    break  # Queue empty or blocked
            except Exception as e:
                logger.warning("process_queue item failed: %s", e)
                break
        return processed

    def get_status(self):
        queued = len([i for i in self.queue if i['status'] in ['queued', 'analyzing']])
        in_progress = len([i for i in self.queue if i['status'] == 'in_progress'])
        completed = len(self.registry.get('completed', []))
        failed = len([i for i in self.queue if i['status'] == 'failed'])
        return {'queue_size': len(self.queue), 'queued': queued, 'in_progress': in_progress, 'completed': completed, 'failed': failed, 'organs_integrated': len(self.registry.get('organs', {})), 'capabilities_integrated': len(self.registry.get('capabilities', {})), 'knowledge_integrated': len(self.registry.get('knowledge', {})), 'next_up': self.get_next_integration()}

    def get_queue(self):
        return sorted(self.queue, key=lambda x: (x['priority'], x['added_at']))

DEFAULT_INTEGRATION_QUEUE = [
    {'url': 'https://github.com/nicedreamzapp/claude-code-local', 'priority': 0, 'name': 'claude-code-local'},
    {'url': 'https://github.com/elder-plinius/OBLITERATUS', 'priority': 0, 'name': 'OBLITERATUS'},
]
