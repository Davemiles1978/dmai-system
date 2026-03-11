#!/usr/bin/env python3
"""
Add API keys to validation queue
"""
import os
import hashlib
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from {env_path.absolute()}")
else:
    print(f"❌ .env not found at {env_path.absolute()}")
    sys.exit(1)

# Get database URL directly from environment (not from config)
db_url = os.getenv('DATABASE_URL')
if not db_url:
    print("❌ DATABASE_URL not found in environment")
    sys.exit(1)

print(f"🔌 Database URL: {db_url[:50]}...")

# Import DatabaseManager after env is loaded
from storage.db_manager import DatabaseManager

# Connect to database
try:
    db = DatabaseManager(db_url)
    print("✅ Connected to database successfully")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    sys.exit(1)

# Your API keys from environment
services = {
    'github': os.getenv('GITHUB_TOKEN'),
    'openai': os.getenv('OPENAI_API_KEY'),
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'gemini': os.getenv('GEMINI_API_KEY'),
    'groq': os.getenv('GROK_API_KEY')
}

print("\n📊 Adding keys to validation queue:")
for service, token in services.items():
    if token and not token.startswith('your_') and len(token) > 20:
        # Create hash of the token
        key_hash = hashlib.sha256(token.encode()).hexdigest()
        print(f"  {service}: {token[:8]}...{token[-8:]} (hash: {key_hash[:16]}...)")
        
        # Add to queue
        result = db.add_pending_key(service, key_hash, 'environment')
        if result:
            print(f"  ✅ Added {service} to queue")
        else:
            print(f"  ⚠️ {service} already in queue or failed")
    else:
        print(f"  ❌ {service}: No valid token found")

print("\n✅ Done! Run 'python3 validator.py' to validate the keys")
