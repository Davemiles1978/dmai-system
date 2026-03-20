#!/usr/bin/env python3
"""
Add API keys to validation queue - Updated to use KeyEvolutionDB
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import db_hybrid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from current directory
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from {env_path.absolute()}")
else:
    print(f"❌ .env not found at {env_path.absolute()}")
    # Try to load from parent
    parent_env = Path('../.env')
    if parent_env.exists():
        load_dotenv(dotenv_path=parent_env)
        print(f"✅ Loaded .env from {parent_env.absolute()}")
    else:
        print("❌ No .env file found")
        sys.exit(1)

# Import KeyEvolutionDB instead of DatabaseManager
from db_hybrid import KeyEvolutionDB

# Initialize database
try:
    db = KeyEvolutionDB()
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

print("\n📊 Adding keys to database:")
for service, token in services.items():
    if token and not token.startswith('your_') and len(token) > 20:
        print(f"  {service}: {token[:8]}...{token[-8:]}")
        
        # Add to database using KeyEvolutionDB
        metadata = {
            'source': 'environment',
            'added_by': 'add_keys.py'
        }
        
        result = db.add_key(service, token, metadata)
        if result:
            print(f"  ✅ Added {service} to database")
        else:
            print(f"  ⚠️ {service} already in database or failed")
    else:
        print(f"  ❌ {service}: No valid token found")

print("\n✅ Done! Keys added to database.")
