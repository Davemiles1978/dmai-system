#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Protected vocabulary manager - APPEND ONLY
This ensures vocabulary can only grow, never be overwritten
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

class ProtectedVocabulary:
    """Vocabulary that can only be appended to, never overwritten"""
    
    def __init__(self):
        self.secure_dir = Path(__file__).parent
        self.master_file = self.secure_dir / 'vocabulary_master.json'
        self.public_file = self.secure_dir.parent / 'vocabulary.json'
        self.backup_dir = self.secure_dir.parent / 'backups'
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load master vocabulary
        self.vocab = self._load_master()
        
    def _load_master(self):
        """Load master vocabulary (read-only source of truth)"""
        if self.master_file.exists():
            with open(self.master_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_master(self):
        """Save to master (only called by protected methods)"""
        # Create timestamped backup first
        backup_file = self.backup_dir / f'vocabulary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        shutil.copy2(self.master_file, backup_file)
        
        # Save master
        with open(self.master_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        # Update public copy (read-only)
        with open(self.public_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        # Make public file read-only
        os.chmod(self.public_file, 0o444)
        
        # Keep only last 10 backups
        backups = sorted(self.backup_dir.glob('vocabulary_*.json'))
        for old_backup in backups[:-10]:
            old_backup.unlink()
    
    def add_words(self, new_words):
        """APPEND ONLY - add new words, never remove"""
        added = 0
        for word in new_words:
            if word not in self.vocab:
                self.vocab[word] = {
                    'added': datetime.now().isoformat(),
                    'source': 'ambient',
                    'count': 1
                }
                added += 1
            else:
                self.vocab[word]['count'] += 1
        
        if added > 0:
            self._save_master()
        
        return added
    
    def get_size(self):
        return len(self.vocab)
    
    def get_words(self):
        return list(self.vocab.keys())

# Usage example
if __name__ == "__main__":
    pv = ProtectedVocabulary()
    print(f"Current vocabulary size: {pv.get_size()}")
