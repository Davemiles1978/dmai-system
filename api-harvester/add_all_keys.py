#!/usr/bin/env python3
"""
Add all your API keys to the validation queue
"""
import os
import hashlib
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def hash_key(key):
    """Create a SHA-256 hash of the key (don't store raw keys)"""
    return hashlib.sha256(key.encode()).hexdigest()

def add_key(service, key_value, source):
    """Add a key to pending_keys"""
    if not key_value or key_value.startswith('your_'):
        print(f"⚠️  Skipping {service}: No valid key found")
        return
    
    key_hash = hash_key(key_value)
    
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    
    try:
        cur.execute("""
            INSERT INTO pending_keys (service, key_hash, source, status)
            VALUES (%s, %s, %s, 'pending')
            ON CONFLICT (key_hash) DO NOTHING
            RETURNING id
        """, (service, key_hash, source))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            print(f"✅ Added {service} key (ID: {result[0]})")
        else:
            print(f"ℹ️  {service} key already in queue")
            
    except Exception as e:
        print(f"❌ Error adding {service}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # Get keys from environment
    keys = [
        ('github', os.getenv('GITHUB_TOKEN'), 'environment'),
        ('openai', os.getenv('OPENAI_API_KEY'), 'environment'),
        ('anthropic', os.getenv('ANTHROPIC_API_KEY'), 'environment'),
        ('gemini', os.getenv('GEMINI_API_KEY'), 'environment'),
        ('groq', os.getenv('GROK_API_KEY'), 'environment'),
    ]
    
    print("🔑 Adding keys to validation queue...")
    for service, key, source in keys:
        add_key(service, key, source)
    
    print("\n✅ Done! Run 'python3 validator.py' to validate them")
