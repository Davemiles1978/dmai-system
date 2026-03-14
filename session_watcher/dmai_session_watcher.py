#!/usr/bin/env python3
"""
DMAI Session Watcher - Monitors terminal sessions and file changes
Captures all activity to feed DMAI's learning
"""
import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
import glob
import hashlib

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SESSION_WATCHER")

class DMAISessionWatcher:
    """
    Watches terminal sessions and file changes
    Every command and edit feeds into DMAI's learning
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.learning_dir = os.path.join(self.base_dir, 'learning')
        self.commands_dir = os.path.join(self.learning_dir, 'commands')
        self.files_dir = os.path.join(self.learning_dir, 'files_changed')
        self.evolution_dir = os.path.join(self.base_dir, 'evolution')
        
        # Create directories
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.evolution_dir, exist_ok=True)
        
        # Track files for changes
        self.file_hashes = {}
        self.command_history = []
        
        logger.info("=" * 60)
        logger.info("📝 DMAI SESSION WATCHER INITIALIZED")
        logger.info(f"Monitoring sessions in: {self.base_dir}")
        logger.info("=" * 60)
    
    def start(self):
        """Start all monitoring threads"""
        threads = [
            threading.Thread(target=self._watch_file_changes, daemon=True),
            threading.Thread(target=self._feed_evolution_engine, daemon=True)
        ]
        
        for t in threads:
            t.start()
            logger.info(f"Started thread: {t.name}")
        
        logger.info("✅ Session watcher running 24/7")
        
        try:
            while True:
                time.sleep(60)
                self._log_stats()
        except KeyboardInterrupt:
            logger.info("Stopping session watcher")
    
    def _watch_file_changes(self):
        """Watch for file changes in the project"""
        while True:
            try:
                for root, dirs, files in os.walk(self.base_dir):
                    # Skip virtual environments and git
                    if 'venv' in root or '__pycache__' in root or '.git' in root:
                        continue
                    
                    for file in files:
                        if file.endswith(('.py', '.json', '.txt', '.md', '.yml', '.yaml', '.sh')):
                            filepath = os.path.join(root, file)
                            try:
                                # Get file hash
                                with open(filepath, 'rb') as f:
                                    file_hash = hashlib.md5(f.read()).hexdigest()
                                
                                # Check if changed
                                if filepath in self.file_hashes:
                                    if self.file_hashes[filepath] != file_hash:
                                        self._process_file_change(filepath)
                                        self.file_hashes[filepath] = file_hash
                                else:
                                    self.file_hashes[filepath] = file_hash
                                    
                            except:
                                continue
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error watching files: {e}")
                time.sleep(30)
    
    def _process_file_change(self, filepath):
        """Process a file change"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Save snapshot
            timestamp = int(time.time())
            filename = os.path.basename(filepath)
            snapshot_file = os.path.join(self.files_dir, f"file_{timestamp}_{filename}.txt")
            
            with open(snapshot_file, 'w') as f:
                f.write(f"# File: {filepath}\n")
                f.write(f"# Time: {datetime.now().isoformat()}\n")
                f.write("#" * 80 + "\n")
                f.write(content[:2000])  # First 2000 chars
            
            logger.info(f"📄 File changed: {filename}")
            
        except Exception as e:
            logger.error(f"Error processing file change: {e}")
    
    def _feed_evolution_engine(self):
        """Feed session data to evolution engine"""
        while True:
            try:
                # Count learning items
                cmd_count = len(glob.glob(f"{self.commands_dir}/*.json")) if os.path.exists(self.commands_dir) else 0
                file_count = len(glob.glob(f"{self.files_dir}/*.txt")) if os.path.exists(self.files_dir) else 0
                
                # Create summary
                summary = {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'session_watcher',
                    'files_changed': file_count,
                    'commands_recorded': cmd_count,
                    'total_events': cmd_count + file_count
                }
                
                # Save for evolution
                evolution_file = os.path.join(self.evolution_dir, 'session_learning.json')
                with open(evolution_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # Trigger evolution if significant activity
                if cmd_count + file_count > 0:
                    trigger_file = os.path.join(self.evolution_dir, 'evolve_now.signal')
                    with open(trigger_file, 'w') as f:
                        f.write(f"Session activity at {datetime.now().isoformat()}")
                
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error feeding evolution: {e}")
                time.sleep(60)
    
    def _log_stats(self):
        """Log session watcher statistics"""
        try:
            cmd_count = len(glob.glob(f"{self.commands_dir}/*.json")) if os.path.exists(self.commands_dir) else 0
            file_count = len(glob.glob(f"{self.files_dir}/*.txt")) if os.path.exists(self.files_dir) else 0
            
            logger.info("=" * 50)
            logger.info("📊 SESSION WATCHER STATISTICS")
            logger.info(f"Files tracked: {len(self.file_hashes)}")
            logger.info(f"Changes recorded: {file_count}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error logging stats: {e}")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     DMAI SESSION WATCHER - Monitoring All Activity          ║
    ║     Every file change feeds DMAI's learning                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    watcher = DMAISessionWatcher()
    watcher.start()
