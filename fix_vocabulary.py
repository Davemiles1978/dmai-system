#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re

# Read the file content
with open('language_learning/data/vocabulary.json', 'r') as f:
    content = f.read()

# Rebuild from phrases
with open('language_learning/data/phrases.json', 'r') as pf:
    phrases = json.load(pf)

# Extract all unique words from phrases
word_set = set()
for phrase in phrases:
    text = phrase.get('text', '')
    words = re.findall(r'\b[a-z]+\b', text.lower())
    word_set.update(words)

# Also try to extract words from the corrupted vocab
try:
    word_matches = re.findall(r'"([a-zA-Z]+)":\s*\{', content)
    word_set.update(word_matches)
except:
    pass

# Create clean vocabulary
new_vocab = {}
for word in sorted(word_set):
    new_vocab[word] = {
        'first_heard': 'recovered',
        'count': 1,
        'sources': ['repair']
    }

# Save with proper JSON formatting
with open('language_learning/data/vocabulary.json', 'w') as f:
    json.dump(new_vocab, f, indent=2)

print(f'✅ Rebuilt vocabulary with {len(new_vocab)} clean words')
