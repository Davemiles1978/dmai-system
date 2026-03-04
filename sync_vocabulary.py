#!/usr/bin/env python3
"""Sync DMAI's vocabulary across all systems"""
import json
import os
import requests
from datetime import datetime

# Load real vocabulary
with open('language_learning/data/vocabulary.json', 'r') as f:
    vocab = json.load(f)
real_size = len(vocab)

print(f"📚 True vocabulary: {real_size} words")

# Update local stats file
with open('language_learning/data/stats.json', 'r') as f:
    stats = json.load(f)

stats['unique_words'] = real_size
stats['last_sync'] = datetime.now().isoformat()

with open('language_learning/data/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("✅ Local stats synced")

# Update cloud service (if available)
try:
    # This would update your cloud evolution service
    # You'll need to add a /sync/vocabulary endpoint
    response = requests.post(
        'https://dmai-cloud-evolution.onrender.com/sync/vocabulary',
        json={'size': real_size}
    )
    if response.status_code == 200:
        print("✅ Cloud stats synced")
except:
    print("⚠️ Cloud sync unavailable")

print(f"\n📊 DMAI now knows {real_size} words consistently")
