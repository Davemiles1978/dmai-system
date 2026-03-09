#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re

print("Creating ultra-simple vocabulary...")

# Load phrases
with open('language_learning/data/phrases.json', 'r') as f:
    phrases = json.load(f)

# Just collect words - no metadata
words = set()
for phrase in phrases:
    text = phrase.get('text', '').lower()
    words.update(re.findall(r'\b[a-z]+\b', text))

# Create minimal vocabulary (just word list)
vocab = {word: {"first_heard": "simple"} for word in sorted(words)}

# Write as simple JSON
with open('language_learning/data/vocabulary.json', 'w') as f:
    json.dump(vocab, f)

print(f"✅ Created simple vocabulary with {len(vocab)} words")
