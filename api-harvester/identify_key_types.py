#!/usr/bin/env python3
"""Identify unknown API key types"""
import re

unknown_keys = [
    "0aYAx2eY37dkfjqsrrZ53SSCkY1yY2kRYGvY27rv",
    "E2T436336DVTJ676A15WURXT2PHM5E8QX8"
]

def check_key_pattern(key):
    """Check key against known patterns"""
    patterns = {
        'Stripe Live': r'^sk_live_[0-9a-zA-Z]+$',
        'Stripe Test': r'^sk_test_[0-9a-zA-Z]+$',
        'GitHub': r'^ghp_[0-9a-zA-Z]+$',
        'Google': r'^AIza[0-9A-Za-z_-]{35}$',
        'AWS': r'^AKIA[0-9A-Z]{16}$',
        'Slack': r'^xox[baprs]-[0-9a-zA-Z]+$',
        'Discord': r'^[MN][0-9a-zA-Z_-]{23,25}$',
        'SendGrid': r'^SG\.[0-9a-zA-Z_-]+$',
        'Mailgun': r'^key-[0-9a-zA-Z]{32}$',
        'Twilio': r'^SK[0-9a-f]{32}$',
    }
    
    for service, pattern in patterns.items():
        if re.match(pattern, key):
            return service
    return "Unknown"

for key in unknown_keys:
    print(f"\nKey: {key[:20]}...")
    print(f"Length: {len(key)}")
    print(f"Starts with: {key[:10]}")
    print(f"Possible type: {check_key_pattern(key)}")
    
    # Try to determine based on length and prefix
    if key.startswith('0a'):
        print("  → Looks like a random string or hash (MD5? SHA?)")
    elif key.startswith('E2'):
        print("  → Could be an API key from a service like Etherscan or similar")
