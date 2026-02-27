#!/usr/bin/env python3
"""
Auto-updates the project tracker with real-time data
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path

def get_evolution_stats():
    """Get current evolution stats"""
    stats = {
        'generation': 1,
        'best_score': 0,
        'files_improved': 0
    }
    
    checkpoints = Path('checkpoints')
    if checkpoints.exists():
        gen_file = checkpoints / 'current_generation.txt'
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                stats['generation'] = int(f.read().strip())
        
        scores_file = checkpoints / 'best_scores.json'
        if scores_file.exists():
            with open(scores_file, 'r') as f:
                scores = json.load(f)
                if scores:
                    stats['best_score'] = max([s.get('score', 0) for s in scores.values()])
    
    # Count improved files
    log_file = Path('evolution.log')
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()
            stats['files_improved'] = content.count('✅ Improved')
    
    return stats

def update_tracker():
    """Update the PROJECT_TRACKER.md file"""
    stats = get_evolution_stats()
    
    tracker = Path('PROJECT_TRACKER.md')
    if not tracker.exists():
        print("❌ Tracker not found - creating it")
        # Create default tracker
        with open(tracker, 'w') as f:
            f.write("""# 🧬 DMAI PROJECT MASTER DASHBOARD
**Last Updated:** 2026-02-26 08:00:00
**Current Generation:** 1
**Overall Progress:** ████░░░░░░ 40%

## 🚨 **CRITICAL ISSUES (MUST FIX NOW)**
| Status | Issue | Priority | Notes |
|--------|-------|----------|-------|
| ❌ | **API 404 Error** | 🔴 CRITICAL | `/api/evolution-stats` not working |
| ❌ | **Login Persistence** | 🔴 CRITICAL | Need to stay logged in |
| ❌ | **Dashboard Non-Functional** | 🔴 CRITICAL | Generation shows "?" |
| ❌ | **Buttons Lead Nowhere** | 🔴 CRITICAL | Tools don't work |
""")
    
    with open(tracker, 'r') as f:
        content = f.read()
    
    # Update timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    import re
    content = re.sub(r'\*\*Last Updated:\*\* .*', f'**Last Updated:** {now}', content)
    content = re.sub(r'\*\*Current Generation:\*\* \d+', f'**Current Generation:** {stats["generation"]}', content)
    
    with open(tracker, 'w') as f:
        f.write(content)
    
    print(f"✅ Tracker updated - Generation {stats['generation']}")

if __name__ == "__main__":
    update_tracker()
