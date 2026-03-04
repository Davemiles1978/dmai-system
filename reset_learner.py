#!/usr/bin/env python3
import json
import os
import shutil
from datetime import datetime

print("💣 COMPLETE LEARNER RESET")
print("="*50)

# Backup current files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = f"language_learning/backups/reset_{timestamp}"
os.makedirs(backup_dir, exist_ok=True)

files_to_backup = [
    'language_learning/data/vocabulary.json',
    'language_learning/data/stats.json',
    'language_learning/data/phrases.json',
    'language_learning/data/rejected_non_english.json'
]

for file in files_to_backup:
    if os.path.exists(file):
        shutil.copy(file, backup_dir)
        print(f"✅ Backed up {file} to {backup_dir}")

# Create fresh vocabulary from phrases only
print("\n🔄 Rebuilding vocabulary from phrases...")

with open('language_learning/data/phrases.json', 'r') as f:
    phrases = json.load(f)

# Extract ALL words from phrases
word_set = set()
for phrase in phrases:
    text = phrase.get('text', '').lower()
    import re
    words = re.findall(r'\b[a-z]+\b', text)
    word_set.update(words)

# Create brand new vocabulary with zero memory of previous counts
fresh_vocab = {}
for word in sorted(word_set):
    fresh_vocab[word] = {
        "first_heard": "post_reset",
        "count": 1,
        "sources": ["reset_rebuild"]
    }

# Save fresh vocabulary
with open('language_learning/data/vocabulary.json', 'w') as f:
    json.dump(fresh_vocab, f, indent=2)

print(f"✅ Created fresh vocabulary with {len(fresh_vocab)} words")

# Reset stats
fresh_stats = {
    "total_phrases_heard": len(phrases),
    "english_phrases": len(phrases),
    "non_english_phrases": 0,
    "unique_words": len(fresh_vocab),
    "common_words": {},
    "learning_rate": len(fresh_vocab) / max(1, len(phrases))
}

with open('language_learning/data/stats.json', 'w') as f:
    json.dump(fresh_stats, f, indent=2)

print("✅ Reset stats file")

print("\n" + "="*50)
print(f"🎯 Final vocabulary: {len(fresh_vocab)} words")
print("="*50)
