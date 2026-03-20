#!/usr/bin/env python3
"""
P0T0_Test_Database_Connections_v2.py
Test database connections with SSL parameters
"""

import psycopg2
import ssl

# URLs
SOURCE_URL = "postgresql://dmai:yCAKCSRzNWv4t4URrZRWB31sJAbXA1Pr@dpg-d6ltllshg0os73avo390-a.oregon-postgres.render.com/harvester_63nu"
TARGET_URL = "postgresql://dmai:xQjt0tbhmT0vRExNv9wTSbe3t7n34J85@dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com/harvester_u9ni"

def test_connection(url, name, sslmode='require'):
    print(f"\n🔌 Testing connection to {name} (sslmode={sslmode})...")
    try:
        # Add SSL parameters to connection
        conn = psycopg2.connect(
            url,
            sslmode=sslmode,
            sslcompression=0,
            target_session_attrs='read-write'
        )
        cur = conn.cursor()
        cur.execute("SELECT current_database(), version();")
        db, version = cur.fetchone()
        print(f"✅ SUCCESS! Connected to: {db}")
        print(f"   Version: {version[:50]}...")
        
        # Get table count
        cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")
        tables = cur.fetchone()[0]
        print(f"   Tables: {tables}")
        
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

print("=" * 60)
print("🔍 DATABASE CONNECTION TEST WITH SSL")
print("=" * 60)

# Try different SSL modes for SOURCE
print("\n📤 SOURCE DATABASE (My Workspace - 1 day old)")
print("-" * 40)
test_connection(SOURCE_URL, "SOURCE", "require")
test_connection(SOURCE_URL, "SOURCE", "prefer")
test_connection(SOURCE_URL, "SOURCE", "allow")
test_connection(SOURCE_URL, "SOURCE", "disable")

print("\n📥 TARGET DATABASE (dmai-production - 10 days old)")
print("-" * 40)
test_connection(TARGET_URL, "TARGET", "require")
