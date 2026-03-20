#!/usr/bin/env python3
"""
P0T0_Check_Render_Internal_Connection.py
Check if we need to use internal Render connection
"""

import os
import psycopg2

print("=" * 60)
print("🔍 CHECKING RENDER INTERNAL CONNECTION")
print("=" * 60)

# Check if we're running on Render
is_render = os.environ.get('RENDER') == 'true'
print(f"Running on Render: {is_render}")

if is_render:
    print("\n📌 On Render - should use internal connection:")
    print("   DATABASE_URL environment variable should work")
    
    internal_url = os.environ.get('DATABASE_URL')
    if internal_url:
        print(f"\n🔌 Testing internal DATABASE_URL...")
        try:
            conn = psycopg2.connect(internal_url)
            cur = conn.cursor()
            cur.execute("SELECT current_database();")
            db = cur.fetchone()[0]
            print(f"✅ Connected to: {db}")
            cur.close()
            conn.close()
        except Exception as e:
            print(f"❌ Failed: {e}")
else:
    print("\n📌 Not on Render - need external connection")
    print("   May need to enable 'Public Network' access in Render dashboard")

