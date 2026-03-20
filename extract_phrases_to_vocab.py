#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Extract all words from phrases.json and add to vocabulary"""
import json
import re
from pathlib import Path
import sys

def extract_words(text):
    """Extract unique words from text"""
    # Convert to lowercase and split by non-alphabetic characters
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return set(words)

def main():
    print("📚 DMAI Vocabulary Extractor")
    print("="*60)
    
    # Load phrases
    phrases_path = Path("language_learning/data/phrases.json")
    if not phrases_path.exists():
        print(f"❌ Phrases file not found: {phrases_path}")
        return
    
    with open(phrases_path, 'r') as f:
        phrases = json.load(f)
    
    print(f"✅ Loaded {len(phrases)} phrases")
    
    # Extract all words from phrases
    all_words = set()
    word_sources = {}
    
    for i, phrase in enumerate(phrases):
        text = phrase.get('text', '')
        source = phrase.get('source', f'phrase_{i}')
        
        words = extract_words(text)
        all_words.update(words)
        
        # Track source for each word (optional)
        for word in words:
            if word not in word_sources:
                word_sources[word] = []
            word_sources[word].append(source)
    
    print(f"✅ Extracted {len(all_words):,} unique words from phrases")
    
    # Load existing vocabulary
    vocab_path = Path("language_learning/data/secure/vocabulary_master.json")
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            existing_vocab = json.load(f)
        print(f"✅ Loaded existing vocabulary with {len(existing_vocab):,} words")
    else:
        existing_vocab = {}
        print("⚠️ No existing vocabulary found, creating new one")
    
    # Add new words to vocabulary
    new_words = 0
    for word in all_words:
        if word not in existing_vocab:
            existing_vocab[word] = {
                'added': '2026-03-08',
                'source': word_sources.get(word, ['unknown'])[0],
                'count': 1
            }
            new_words += 1
    
    print(f"✅ Added {new_words:,} new words to vocabulary")
    
    # Save updated vocabulary
    with open(vocab_path, 'w') as f:
        json.dump(existing_vocab, f, indent=2)
    
    print(f"📊 Total vocabulary now: {len(existing_vocab):,} words")
    
    # Also update the symlink
    symlink_path = Path("language_learning/data/vocabulary.json")
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    
    # Create symlink
    os.symlink("secure/vocabulary_master.json", symlink_path)
    print("✅ Updated vocabulary symlink")
    
    print("\n" + "="*60)
    print("✅ Vocabulary extraction complete!")

if __name__ == "__main__":
    main()
