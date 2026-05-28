"""Repository processor - uses controlled threads"""
import logging
import sqlite3
from datetime import datetime
from components.thread_manager import thread_manager

logger = logging.getLogger(__name__)

class RepoProcessor:
    def __init__(self, db_path="data/dmai_knowledge.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_repos (
                repo_name TEXT PRIMARY KEY,
                processed_at TIMESTAMP,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def process_repo(self, repo_name):
        """Process a single repository"""
        try:
            logger.info(f"Processing repo: {repo_name}")
            # Simulate processing - in production, this would clone and analyze
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO processed_repos (repo_name, processed_at, status)
                VALUES (?, ?, ?)
            ''', (repo_name, datetime.now().isoformat(), 'completed'))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to process {repo_name}: {e}")
            return False
    
    def process_queue(self, repos):
        """Process repos using controlled thread pool"""
        for repo in repos:
            thread_manager.submit(lambda r=repo: self.process_repo(r))
        logger.info(f"Submitted {len(repos)} repos to processing queue")

# Global instance
repo_processor = RepoProcessor()
