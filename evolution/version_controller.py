#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Version Controller for DMAI Evolution
Manages rollbacks and version history
"""

import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

class VersionController:
    def __init__(self):
        self.versions_dir = Path("/Users/davidmiles/Desktop/dmai-system/evolution/versions")
        self.history_file = Path("/Users/davidmiles/Desktop/dmai-system/data/version_history.json")
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.load_history()
    
    def load_history(self):
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {}
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def create_version(self, filepath, metadata=None):
        """Create a new version of a file"""
        if not Path(filepath).exists():
            return None
        
        # Calculate hash
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Create version record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(filepath).name
        version_name = f"{filename}_{timestamp}"
        version_path = self.versions_dir / version_name
        
        # Copy file
        shutil.copy2(filepath, version_path)
        
        version_info = {
            'version': version_name,
            'timestamp': timestamp,
            'hash': file_hash,
            'path': str(version_path),
            'metadata': metadata or {}
        }
        
        # Store in history
        if filename not in self.history:
            self.history[filename] = []
        
        self.history[filename].append(version_info)
        
        # Keep last 10 versions
        if len(self.history[filename]) > 10:
            old = self.history[filename].pop(0)
            Path(old['path']).unlink(missing_ok=True)
        
        self.save_history()
        return version_info
    
    def rollback(self, filename, version=None):
        """Rollback to a specific version"""
        if filename not in self.history:
            return False
        
        versions = self.history[filename]
        if not versions:
            return False
        
        if version is None:
            # Rollback to previous version (if exists)
            if len(versions) < 2:
                return False
            target = versions[-2]
        else:
            # Find specific version
            target = None
            for v in versions:
                if v['version'] == version or v['timestamp'] == version:
                    target = v
                    break
            if not target:
                return False
        
        return target['path']

vc = VersionController()
