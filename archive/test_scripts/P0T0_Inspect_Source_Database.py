#!/usr/bin/env python3
"""
P0T0_Inspect_Source_Database.py
Inspect what's in the SOURCE database
"""

import psycopg2

SOURCE_URL = "postgresql://dmai:yCAKCSRzNWv4t4URrZRWB31sJAbXA1Pr@dpg-d6ltllshg0os73avo390-a.oregon-postgres.render.com/harvester_63nu"
TARGET_URL = "postgresql://dmai:xQjt0tbhmT0vRExNv9wTSbe3t7n34J85@dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com/harvester_u9ni"

def inspect_database(url, name):
    print(f"\n🔍 Inspecting {name}...")
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        
        # Check all schemas
        cur.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        """)
        schemas = cur.fetchall()
        print(f"\n📊 Schemas: {[s[0] for s in schemas]}")
        
        # Check tables in each schema
        for schema in schemas:
            schema_name = schema[0]
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
            """, (schema_name,))
            tables = cur.fetchall()
            
            if tables:
                print(f"\n📋 Tables in schema '{schema_name}':")
                for table in tables:
                    table_name = table[0]
                    # Get row count
                    cur.execute(f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"')
                    count = cur.fetchone()[0]
                    print(f"   - {table_name}: {count} rows")
        
        # Check if there's any data at all
        cur.execute("""
            SELECT datname, pg_database_size(datname)/1024/1024 as size_mb
            FROM pg_database
            WHERE datname = current_database()
        """)
        db_name, size_mb = cur.fetchone()
        print(f"\n💾 Database size: {size_mb:.2f} MB")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

print("=" * 60)
print("🔍 DATABASE INSPECTION")
print("=" * 60)

inspect_database(SOURCE_URL, "SOURCE (My Workspace)")
inspect_database(TARGET_URL, "TARGET (dmai-production)")
