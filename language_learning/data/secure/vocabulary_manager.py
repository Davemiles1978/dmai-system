#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Vocabulary Manager - Enforces append-only behavior
DMAI can add new words but never modify or delete existing ones
"""

import json
import sys
import time
from pathlib import Path
import hashlib

MASTER_PATH = "/Users/davidmiles/Desktop/dmai-system/language_learning/data/secure/vocabulary_master.json"
BACKUP_DIR = "/Users/davidmiles/Desktop/dmai-system/language_learning/data/secure/backups/"
LOG_PATH = "/Users/davidmiles/Desktop/dmai-system/logs/vocabulary_changes.log"

class VocabularyManager:
    def __init__(self):
        self.master_path = MASTER_PATH
        self.backup_dir = Path(BACKUP_DIR)
        self.backup_dir.mkdir(exist_ok=True)
        
    def load_vocabulary(self):
        """Load current vocabulary"""
        with open(self.master_path, 'r') as f:
            return json.load(f)
    
    def save_vocabulary(self, vocab, create_backup=True):
        """Save vocabulary with backup"""
        if create_backup:
            # Create backup before saving
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"vocabulary_{timestamp}.json"
            with open(self.master_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
        
        # Save new version
        with open(self.master_path, 'w') as f:
            json.dump(vocab, f, indent=2)
    
    def add_words(self, new_words):
        """
        Add new words - only appends, never modifies existing
        Returns: (added_count, existing_count, failed_count)
        """
        vocab = self.load_vocabulary()
        
        added = []
        existing = []
        failed = []
        
        for word_data in new_words:
            try:
                # Handle both string and dict inputs
                if isinstance(word_data, str):
                    word = word_data
                    word_dict = {
                        "word": word,
                        "added": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "voice",
                        "confidence": 1.0
                    }
                else:
                    word = word_data.get('word', '')
                    word_dict = word_data
                
                if not word:
                    continue
                
                # Check if word exists (case-insensitive)
                exists = False
                for existing_word in vocab.keys():
                    if existing_word.lower() == word.lower():
                        exists = True
                        break
                
                if not exists:
                    # Add new word
                    vocab[word] = word_dict
                    added.append(word)
                else:
                    existing.append(word)
                    
            except Exception as e:
                failed.append(str(word_data))
                print(f"Error adding {word_data}: {e}")
        
        # Save if we added anything
        if added:
            self.save_vocabulary(vocab)
            
            # Log additions
            with open(LOG_PATH, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ADDED: {', '.join(added)}\n")
        
        return added, existing, failed
    
    def get_word(self, word):
        """Retrieve a word (read-only operation)"""
        vocab = self.load_vocabulary()
        # Case-insensitive lookup
        for key in vocab:
            if key.lower() == word.lower():
                return vocab[key]
        return None
    
    def word_exists(self, word):
        """Check if word exists (read-only)"""
        vocab = self.load_vocabulary()
        for key in vocab:
            if key.lower() == word.lower():
                return True
        return False
    
    def get_stats(self):
        """Get vocabulary statistics"""
        vocab = self.load_vocabulary()
        return {
            "total_words": len(vocab),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backup_count": len(list(self.backup_dir.glob("*.json")))
        }
    
    def rollback(self, backup_file=None):
        """
        Rollback to previous version (YOU only, not DMAI)
        """
        if backup_file:
            backup_path = self.backup_dir / backup_file
        else:
            # Get latest backup
            backups = sorted(self.backup_dir.glob("*.json"))
            if not backups:
                print("No backups found")
                return False
            backup_path = backups[-1]
        
        # Restore from backup
        with open(backup_path, 'r') as f:
            backup_vocab = json.load(f)
        
        # Create backup of current before rollback
        self.save_vocabulary(backup_vocab, create_backup=True)
        
        print(f"✅ Rolled back to {backup_path.name}")
        return True

# Command-line interface
if __name__ == "__main__":
    manager = VocabularyManager()
    
    if len(sys.argv) < 2:
        # Show stats
        stats = manager.get_stats()
        print(f"📚 Vocabulary Statistics:")
        print(f"   Total words: {stats['total_words']}")
        print(f"   Last updated: {stats['last_updated']}")
        print(f"   Backups available: {stats['backup_count']}")
        
    elif sys.argv[1] == "add":
        # Add words
        words = sys.argv[2:]
        if words:
            added, existing, failed = manager.add_words(words)
            print(f"✅ Added: {len(added)} words")
            if added:
                print(f"   New: {', '.join(added)}")
            if existing:
                print(f"⏭️  Already existed: {len(existing)}")
            if failed:
                print(f"❌ Failed: {len(failed)}")
        else:
            print("Usage: vocabulary_manager.py add word1 word2 ...")
            
    elif sys.argv[1] == "check":
        # Check if word exists
        if len(sys.argv) > 2:
            word = sys.argv[2]
            exists = manager.word_exists(word)
            print(f"Word '{word}': {'✅ Exists' if exists else '❌ Not found'}")
            
    elif sys.argv[1] == "backups":
        # List backups
        backups = sorted(manager.backup_dir.glob("*.json"))
        print(f"📦 Available backups ({len(backups)}):")
        for i, backup in enumerate(backups[-10:], 1):  # Show last 10
            size = backup.stat().st_size
            print(f"  {i}. {backup.name} ({size} bytes)")
            
    elif sys.argv[1] == "rollback" and len(sys.argv) > 2:
        # Rollback to specific backup (YOU only)
        print("⚠️  This will revert vocabulary to a previous version")
        confirm = input("Type 'ROLLBACK' to confirm: ")
        if confirm == "ROLLBACK":
            manager.rollback(sys.argv[2])
        else:
            print("Rollback cancelled")

# Add this to ensure backups are created on every change
import shutil
from pathlib import Path

def create_backup():
    """Manual backup function"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = Path("/Users/davidmiles/Desktop/dmai-system/language_learning/data/secure/backups/") / f"vocabulary_{timestamp}.json"
    shutil.copy2(MASTER_PATH, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

# Run this to create an initial backup
create_backup()
