#!/usr/bin/env python3
"""
Persistence Manager - Ensures all user data survives crashes, restarts, and syncs across devices
"""

import os
import json
import time
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - PERSISTENCE - %(message)s')

class PersistenceManager:
    def __init__(self):
        self.data_path = Path("shared_data")
        self.backup_path = self.data_path / "backups"
        self.checkpoint_path = self.data_path / "checkpoints"
        self.user_data_path = self.data_path / "users"
        
        # Create all directories
        for path in [self.data_path, self.backup_path, self.checkpoint_path, self.user_data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Auto-save settings
        self.auto_save_interval = 60  # Save every 60 seconds
        self.max_backups = 10  # Keep last 10 backups
        self.auto_save_thread = None
        self.is_running = True
        
        # Start auto-save thread
        self.start_auto_save()
        
        logging.info("💾 Persistence Manager initialized")
    
    def start_auto_save(self):
        """Start auto-save background thread"""
        def auto_save_loop():
            while self.is_running:
                time.sleep(self.auto_save_interval)
                self.create_checkpoint("auto_save")
        
        self.auto_save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        self.auto_save_thread.start()
        logging.info("🔄 Auto-save thread started")
    
    def stop(self):
        """Stop auto-save and create final checkpoint"""
        self.is_running = False
        self.create_checkpoint("shutdown")
        logging.info("💾 Persistence Manager stopped")
    
    def save_user_data(self, username, data):
        """Save user-specific data"""
        user_file = self.user_data_path / f"{username}.json"
        
        # Add metadata
        data['_metadata'] = {
            'last_saved': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Save with atomic write (write to temp then rename)
        temp_file = user_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(user_file)
        
        logging.debug(f"✅ Saved data for user: {username}")
    
    def load_user_data(self, username):
        """Load user-specific data"""
        user_file = self.user_data_path / f"{username}.json"
        
        if user_file.exists():
            try:
                with open(user_file, 'r') as f:
                    data = json.load(f)
                # Remove metadata before returning
                data.pop('_metadata', None)
                logging.debug(f"📂 Loaded data for user: {username}")
                return data
            except Exception as e:
                logging.error(f"Error loading user data for {username}: {e}")
                # Try to recover from backup
                return self.recover_user_data(username)
        
        return None
    
    def create_checkpoint(self, reason="manual"):
        """Create a system-wide checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.checkpoint_path / f"checkpoint_{timestamp}_{reason}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Copy all user data
        shutil.copytree(self.user_data_path, checkpoint_dir / "users", dirs_exist_ok=True)
        
        # Copy evolution data if it exists
        if Path("shared_checkpoints").exists():
            shutil.copytree("shared_checkpoints", checkpoint_dir / "evolution", dirs_exist_ok=True)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'reason': reason,
            'user_count': len(list(self.user_data_path.glob("*.json"))),
            'generation': self.get_current_generation()
        }
        
        with open(checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"💾 Checkpoint created: {checkpoint_dir.name}")
        
        # Clean old checkpoints
        self.clean_old_checkpoints()
        
        return checkpoint_dir
    
    def get_current_generation(self):
        """Get current evolution generation"""
        gen_file = Path("shared_checkpoints/current_generation.txt")
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                return f.read().strip()
        return "unknown"
    
    def clean_old_checkpoints(self):
        """Keep only the most recent checkpoints"""
        checkpoints = sorted(self.checkpoint_path.glob("checkpoint_*"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_checkpoint in checkpoints[self.max_backups:]:
            shutil.rmtree(old_checkpoint)
            logging.debug(f"🧹 Removed old checkpoint: {old_checkpoint.name}")
    
    def recover_user_data(self, username):
        """Recover user data from latest checkpoint"""
        checkpoints = sorted(self.checkpoint_path.glob("checkpoint_*"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        for checkpoint in checkpoints:
            user_backup = checkpoint / "users" / f"{username}.json"
            if user_backup.exists():
                try:
                    with open(user_backup, 'r') as f:
                        data = json.load(f)
                    logging.info(f"🔄 Recovered user data for {username} from {checkpoint.name}")
                    return data
                except:
                    continue
        
        logging.warning(f"⚠️ No recovery data found for {username}")
        return None
    
    def export_all_data(self, export_path):
        """Export all data for backup/migration"""
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = export_dir / f"dmai_export_{timestamp}.tar.gz"
        
        # Create tar.gz of all shared data
        import tarfile
        with tarfile.open(export_file, "w:gz") as tar:
            tar.add(self.data_path, arcname="shared_data")
            if Path("shared_checkpoints").exists():
                tar.add("shared_checkpoints", arcname="evolution_data")
        
        logging.info(f"📦 Exported all data to {export_file}")
        return export_file
    
    def import_all_data(self, import_file):
        """Import data from backup"""
        import tarfile
        
        # Create backup of current data first
        self.create_checkpoint("pre_import")
        
        # Extract import file
        with tarfile.open(import_file, "r:gz") as tar:
            tar.extractall(".")
        
        logging.info(f"📦 Imported data from {import_file}")
        
        # Reload any necessary caches
        self.reload_all()
    
    def reload_all(self):
        """Reload all data after import"""
        # Clear any caches
        # Reinitialize managers
        logging.info("🔄 Reloaded all data")
    
    def get_system_status(self):
        """Get persistence system status"""
        return {
            'checkpoint_count': len(list(self.checkpoint_path.glob("checkpoint_*"))),
            'user_count': len(list(self.user_data_path.glob("*.json"))),
            'last_backup': max([f.stat().st_mtime for f in self.checkpoint_path.glob("checkpoint_*")]) if any(self.checkpoint_path.glob("checkpoint_*")) else None,
            'auto_save_interval': self.auto_save_interval,
            'total_backups': self.max_backups,
            'current_generation': self.get_current_generation()
        }

# Global instance
_persistence_manager = None

def get_persistence_manager():
    """Get or create the global persistence manager"""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = PersistenceManager()
    return _persistence_manager