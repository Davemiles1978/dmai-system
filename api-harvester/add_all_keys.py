#!/usr/bin/env python3
"""
Add all your API keys to the database using KeyEvolutionDB
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import db_hybrid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Import KeyEvolutionDB
from db_hybrid import KeyEvolutionDB

def add_key(service, key_value, source):
    """Add a key to the database"""
    if not key_value or key_value.startswith('your_'):
        print(f"⚠️  Skipping {service}: No valid key found")
        return False
    
    db = KeyEvolutionDB()
    metadata = {
        'source': source,
        'added_by': 'add_all_keys.py'
    }
    
    result = db.add_key(service, key_value, metadata)
    
    if result:
        print(f"✅ Added {service} key")
        return True
    else:
        print(f"ℹ️  {service} key already in database or failed")
        return False

if __name__ == "__main__":
    # Get keys from environment
    keys = [
        ('github', os.getenv('GITHUB_TOKEN'), 'environment'),
        ('openai', os.getenv('OPENAI_API_KEY'), 'environment'),
        ('anthropic', os.getenv('ANTHROPIC_API_KEY'), 'environment'),
        ('gemini', os.getenv('GEMINI_API_KEY'), 'environment'),
        ('groq', os.getenv('GROK_API_KEY'), 'environment'),
    ]
    
    print("🔑 Adding keys to database...")
    success_count = 0
    
    for service, key, source in keys:
        if add_key(service, key, source):
            success_count += 1
    
    print(f"\n✅ Done! Added {success_count} keys to database.")
