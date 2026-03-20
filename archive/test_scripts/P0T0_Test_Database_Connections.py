#!/usr/bin/env python3
"""
P0T0_Test_Database_Connections.py
Test database connections before migration
"""

import psycopg2

# Corrected URLs with full domain
SOURCE_URL = "postgresql://dmai:yCAKCSRzNWv4t4URrZRWB31sJAbXA1Pr@dpg-d6ltllshg0os73avo390-a.oregon-postgres.render.com/harvester_63nu"
TARGET_URL = "postgresql://dmai:xQjt0tbhmT0vRExNv9wTSbe3t7n34J85@dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com/harvester_u9ni"

def test_connection(url, name):
    print(f"\n🔌 Testing connection to {name}...")
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute("SELECT current_database(), version();")
        db, version = cur.fetchone()
        print(f"✅ SUCCESS! Connected to: {db}")
        print(f"   Version: {version[:50]}...")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

print("=" * 60)
print("🔍 DATABASE CONNECTION TEST")
print("=" * 60)

test_connection(SOURCE_URL, "SOURCE (My Workspace)")
test_connection(TARGET_URL, "TARGET (dmai-production)")
