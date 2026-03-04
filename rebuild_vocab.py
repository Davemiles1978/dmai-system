#!/usr/bin/env python3
import json
import re
import os

print("🔧 Rebuilding vocabulary from scratch...")

# Load phrases
phrases_file = 'language_learning/data/phrases.json'
if not os.path.exists(phrases_file):
    print("❌ phrases.json not found")
    exit(1)

with open(phrases_file, 'r') as f:
    phrases = json.load(f)

print(f"📝 Loaded {len(phrases)} phrases")

# Extract all unique words
word_set = set()
word_count = {}

for phrase in phrases:
    text = phrase.get('text', '').lower()
    # Simple word extraction
    words = re.findall(r'\b[a-z]+\b', text)
    for word in words:
        word_set.add(word)
        word_count[word] = word_count.get(word, 0) + 1

print(f"📚 Found {len(word_set)} unique words")

# Create new vocabulary
new_vocab = {}
for word in sorted(word_set):
    new_vocab[word] = {
        'first_heard': 'rebuild',
        'count': word_count.get(word, 1),
        'sources': ['phrase_rebuild']
    }

# Write with proper formatting
with open('language_learning/data/vocabulary.json', 'w') as f:
    json.dump(new_vocab, f, indent=2)

print(f"✅ Saved {len(new_vocab)} words to vocabulary.json")
