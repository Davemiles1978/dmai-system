#!/usr/bin/env python3
"""
Version Controller for DMAI Evolution System
Manages version history and rollback capability
"""

import json
import os
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

class VersionController:
    def __init__(self):
        self.evolution_dir = "/Users/davidmiles/Desktop/AI-Evolution-System/evolution"
        self.versions_dir = f"{self.evolution_dir}/history/versions"
        self.history_file = f"{self.evolution_dir}/history/evolution_log.json"
        self.max_versions_per_target = 10
        self.load_history()
    
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except:
                self.history = {}
        else:
            self.history = {}
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def calculate_hash(self, filepath):
        """Calculate MD5 hash of a file"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def create_version(self, target_name, filepath, metadata=None):
        """Create a new version of a target file"""
        if not os.path.exists(filepath):
            return None
        
        # Ensure target exists in history
        if target_name not in self.history:
            self.history[target_name] = []
        
        # Calculate hash
        file_hash = self.calculate_hash(filepath)
        
        # Check if this exact version already exists
        for version in self.history[target_name]:
            if version.get('hash') == file_hash:
                return version  # Already have this version
        
        # Create version record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{target_name}_v{timestamp}"
        backup_path = f"{self.versions_dir}/{version_id}.py"
        
        # Copy file to versions directory
        shutil.copy2(filepath, backup_path)
        
        version_info = {
            'version_id': version_id,
            'timestamp': timestamp,
            'hash': file_hash,
            'backup_path': backup_path,
            'metadata': metadata or {},
            'improvement_score': metadata.get('improvement_score', 0) if metadata else 0,
            'evaluator': metadata.get('evaluator') if metadata else None
        }
        
        # Add to history
        self.history[target_name].append(version_info)
        
        # Trim to max versions
        if len(self.history[target_name]) > self.max_versions_per_target:
            oldest = self.history[target_name].pop(0)
            if os.path.exists(oldest['backup_path']):
                os.remove(oldest['backup_path'])
        
        self.save_history()
        return version_info
    
    def get_latest_version(self, target_name):
        """Get the latest version of a target"""
        if target_name not in self.history or not self.history[target_name]:
            return None
        return self.history[target_name][-1]
    
    def get_version_history(self, target_name):
        """Get full version history for a target"""
        return self.history.get(target_name, [])
    
    def rollback(self, target_name, version_id=None):
        """Rollback to a specific version"""
        if target_name not in self.history:
            return False
        
        versions = self.history[target_name]
        
        if version_id:
            # Find specific version
            target_version = None
            for v in versions:
                if v['version_id'] == version_id:
                    target_version = v
                    break
        else:
            # Rollback to previous version (second last)
            if len(versions) < 2:
                return False
            target_version = versions[-2]
        
        if not target_version:
            return False
        
        # Get the actual file path from metadata
        # This needs to be mapped to the actual system file
        # For now, we'll return the backup path
        return target_version['backup_path']
    
    def validate_improvement(self, old_version, new_version):
        """Validate that new version actually improves on old"""
        if not old_version or not new_version:
            return False
        
        old_score = old_version.get('improvement_score', 0)
        new_score = new_version.get('improvement_score', 0)
        
        # Require at least 5% improvement
        return new_score > old_score * 1.05
    
    def get_stagnation_report(self):
        """Generate report on which targets are stagnating"""
        report = {}
        for target_name, versions in self.history.items():
            if len(versions) >= 5:
                recent = versions[-5:]
                scores = [v.get('improvement_score', 0) for v in recent]
                
                # Check if scores are plateauing
                if len(set(scores)) == 1 or all(s <= scores[0] for s in scores[1:]):
                    report[target_name] = {
                        'status': 'STAGNANT',
                        'last_improvement': versions[-1]['timestamp'],
                        'scores': scores
                    }
                else:
                    report[target_name] = {
                        'status': 'EVOLVING',
                        'improvement_trend': scores[-1] - scores[0],
                        'scores': scores
                    }
        
        return report

if __name__ == "__main__":
    vc = VersionController()
    print("✅ Version Controller initialized")
    print(f"📚 Tracking {len(vc.history)} targets")
