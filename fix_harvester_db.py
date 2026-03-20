#!/usr/bin/env python3
"""
Fix harvester database issues
"""
import sqlite3
import os

db_path = "api-harvester/dmai_local.db"

# Connect to database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service TEXT NOT NULL,
        api_key TEXT NOT NULL,
        source TEXT,
        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )
''')

conn.commit()
conn.close()
print("✅ Database fixed!")
