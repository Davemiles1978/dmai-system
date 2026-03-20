import os
import json
import sys
from pathlib import Path

# Load the token from .env
env_file = Path('.env')
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            if line.startswith('GITHUB_TOKEN'):
                token = line.strip().split('=')[1]
                print(f"Found token: {token[:10]}...")
                
                # Update config.json
                config_file = Path('config.json')
                if config_file.exists():
                    with open(config_file, 'r') as cf:
                        config = json.load(cf)
                    
                    config['github_token'] = token
                    
                    with open(config_file, 'w') as cf:
                        json.dump(config, cf, indent=2)
                    
                    print("✅ Updated config.json with GitHub token")
                else:
                    print("❌ config.json not found")
                
                # Also update environment
                os.environ['GITHUB_TOKEN'] = token
                print("✅ Set GITHUB_TOKEN in environment")
                
                # Test the token
                import requests
                response = requests.get(
                    'https://api.github.com/user',
                    headers={'Authorization': f'token {token}'}
                )
                if response.ok:
                    print(f"✅ Token valid for user: {response.json().get('login')}")
                else:
                    print(f"❌ Token invalid: {response.status_code}")
