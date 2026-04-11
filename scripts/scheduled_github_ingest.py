#!/usr/bin/env python3
"""
Scheduled task for DMAI to check GitHub starred repos weekly
Runs every Monday at 00:00 UTC
"""

import requests
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Configuration
GITHUB_USER = "Davemiles1978"
DMAI_URL = "https://dmai-web.onrender.com"
STATE_FILE = Path("/tmp/github_starred_state.json")

def get_starred_repos(page=1):
    """Fetch starred repositories from GitHub"""
    url = f"https://api.github.com/users/{GITHUB_USER}/starred?page={page}&per_page=100"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    # Add GitHub token if available (for higher rate limit)
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'token {token}'
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return []

def get_all_starred():
    """Get all starred repos across pages"""
    all_repos = []
    page = 1
    while True:
        repos = get_starred_repos(page)
        if not repos:
            break
        all_repos.extend(repos)
        page += 1
        time.sleep(0.5)
    return all_repos

def load_previous_state():
    """Load previously tracked starred repos"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return set(json.load(f).get('starred_repos', []))
    return set()

def save_state(starred_repos):
    """Save current starred repos state"""
    with open(STATE_FILE, 'w') as f:
        json.dump({
            'starred_repos': list(starred_repos),
            'last_check': datetime.now().isoformat()
        }, f, indent=2)

def ingest_repo(repo_url, repo_name, description):
    """Send repo to DMAI for ingestion"""
    message = f"/ingest {repo_url} --recursive --extract-code --extract-apis --implement-if-beneficial"
    
    payload = {"message": message, "user": "system"}
    try:
        response = requests.post(f"{DMAI_URL}/api/chat", json=payload, timeout=30)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to ingest {repo_name}: {e}")
        return False

def categorize_repo(repo):
    """Categorize repository by type"""
    name = repo['full_name'].lower()
    description = (repo.get('description') or '').lower()
    
    if any(kw in name or kw in description for kw in ['video', 'movie', 'animate', 'avatar', 'talking-head', 'deepfake', 'comfyui', 'stable-diffusion']):
        return 'video'
    elif any(kw in name or kw in description for kw in ['ai', 'llm', 'gpt', 'model', 'neural', 'machine-learning', 'deep-learning', 'transformer']):
        return 'ai'
    elif any(kw in name or kw in description for kw in ['api', 'sdk', 'client', 'library', 'rest']):
        return 'api'
    elif any(kw in name or kw in description for kw in ['funding', 'trading', 'finance', 'crypto', 'binance']):
        return 'funding'
    else:
        return 'other'

def main():
    print(f"🔍 DMAI Weekly GitHub Star Check - {datetime.now().isoformat()}")
    
    # Get current starred repos
    current_starred = get_all_starred()
    current_ids = {repo['full_name'] for repo in current_starred}
    
    # Get previously tracked repos
    previous_ids = load_previous_state()
    
    # Find new repos
    new_repos = current_ids - previous_ids
    
    if not new_repos:
        print("✅ No new starred repositories found")
        save_state(current_ids)
        return
    
    print(f"📦 Found {len(new_repos)} new starred repositories")
    
    # Categorize new repos
    repo_map = {repo['full_name']: repo for repo in current_starred}
    
    new_by_category = {'video': [], 'ai': [], 'api': [], 'funding': [], 'other': []}
    for repo_id in new_repos:
        repo = repo_map.get(repo_id)
        if repo:
            category = categorize_repo(repo)
            new_by_category[category].append(repo)
    
    print(f"\n🎬 Video/Avatar repos: {len(new_by_category['video'])}")
    print(f"🤖 AI/ML repos: {len(new_by_category['ai'])}")
    print(f"🔌 API/SDK repos: {len(new_by_category['api'])}")
    print(f"💰 Funding/Trading repos: {len(new_by_category['funding'])}")
    print(f"📦 Other repos: {len(new_by_category['other'])}")
    
    # Send summary to DMAI
    summary = f"""Weekly GitHub Star Update - {datetime.now().strftime('%Y-%m-%d')}

New starred repositories found: {len(new_repos)}

Categories:
- 🎬 Video/Avatar: {len(new_by_category['video'])}
- 🤖 AI/ML: {len(new_by_category['ai'])}
- 🔌 API/SDK: {len(new_by_category['api'])}
- 💰 Funding/Trading: {len(new_by_category['funding'])}
- 📦 Other: {len(new_by_category['other'])}

Ingesting all new repositories for capability enhancement.
"""
    
    # Send summary to DMAI
    requests.post(f"{DMAI_URL}/api/chat", json={
        "message": f"/develop {summary}",
        "user": "system"
    })
    
    # Ingest new repos by priority
    for category in ['video', 'ai', 'funding', 'api', 'other']:
        for repo in new_by_category[category]:
            print(f"📥 Ingesting: {repo['full_name']}")
            success = ingest_repo(
                repo['html_url'],
                repo['full_name'],
                repo.get('description', '')
            )
            if success:
                print(f"   ✅ Queued for ingestion")
            else:
                print(f"   ❌ Failed to queue")
            time.sleep(2)  # Rate limiting
    
    # Save updated state
    save_state(current_ids)
    
    print(f"\n✅ Weekly check complete. {len(new_repos)} new repositories ingested.")

if __name__ == "__main__":
    main()
