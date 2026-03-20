#!/usr/bin/env python3
"""
Export keys from PostgreSQL to JSON for the API server
Using correct schema with encrypted keys
"""

import json
import psycopg2
from datetime import datetime
from pathlib import Path

DB_CONFIG = {
    "host": "dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com",
    "database": "harvester_u9ni",
    "user": "dmai",
    "password": "xQjt0tbhmT0vRExNv9wTSbe3t7n34J85",
    "port": 5432
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    

    cur.execute("""
        SELECT 
            key_type,
            key_preview,
            source_repo,
            created_at,
            weight,
            is_valid
        FROM api_keys 
        WHERE is_valid = true
        ORDER BY created_at DESC
    """)
    
    rows = cur.fetchall()
    
    keys = []
    for row in rows:
        key_type, key_preview, source_repo, created_at, weight, is_valid = row
        
        keys.append({
            'service': key_type or 'unknown',
            'key': key_preview or '***',
            'source': source_repo or 'database',
            'found_at': str(created_at) if created_at else str(datetime.now()),
            'status': 'valid' if is_valid else 'unknown',
            'weight': weight or 0
        })
    

    keys_dir = Path('keys')
    keys_dir.mkdir(exist_ok=True)
    keys_file = keys_dir / 'found_keys.json'
    
    output = {
        'keys': keys,
        'total': len(keys),
        'last_updated': str(datetime.now())
    }
    
    with open(keys_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'✅ Exported {len(keys)} keys to {keys_file}')
    
  
    if keys:
        print('\n📊 Keys found:')
        for k in keys:
            print(f'  • {k["service"]}: {k["key"]} (from {k["source"]})')
    else:
        print('⚠️ No valid keys found in database')
    
    # Also show raw data from database for debugging
    print('\n🔍 Raw data from database:')
    cur.execute("SELECT key_type, key_preview, source_repo FROM api_keys WHERE is_valid = true")
    for row in cur.fetchall():
        print(f'  • {row[0]}: {row[1]} (from {row[2]})')
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f'❌ Error: {e}')
