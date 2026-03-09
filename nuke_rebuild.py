#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import os

print("💣 NUKE REBUILD - Starting from scratch")

# Load phrases
phrases_file = 'language_learning/data/phrases.json'
if not os.path.exists(phrases_file):
    print("❌ phrases.json not found")
    exit(1)

with open(phrases_file, 'r') as f:
    phrases = json.load(f)

print(f"📝 Loaded {len(phrases)} phrases")

# Extract ALL words from phrases
word_set = set()
word_count = {}

for i, phrase in enumerate(phrases):
    text = phrase.get('text', '').lower()
    # Split into words and clean
    words = re.findall(r'\b[a-z]+\b', text)
    for word in words:
        if len(word) > 1:  # Skip single letters
            word_set.add(word)
            word_count[word] = word_count.get(word, 0) + 1
    
    if i % 100 == 0:
        print(f"  Processed {i} phrases, found {len(word_set)} words so far")

print(f"📚 Found {len(word_set)} unique words")

# Create brand new vocabulary with clean structure
new_vocab = {}
for word in sorted(word_set):
    new_vocab[word] = {
        "first_heard": "nuke_rebuild",
        "count": word_count.get(word, 1),
        "sources": ["phrase_rebuild"]
    }

# Write with simple, clean formatting
with open('language_learning/data/vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(new_vocab, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(new_vocab)} clean words to vocabulary.json")

# Verify it's readable
try:
    with open('language_learning/data/vocabulary.json', 'r', encoding='utf-8') as f:
        test = json.load(f)
    print(f"✅ Verification passed: {len(test)} words readable")
except Exception as e:
    print(f"❌ Verification failed: {e}")
