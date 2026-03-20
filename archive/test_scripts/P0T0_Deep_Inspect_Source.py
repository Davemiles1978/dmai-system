#!/usr/bin/env python3
"""
P0T0_Deep_Inspect_Source.py
Deep inspection of SOURCE database to find where the data is
"""

import psycopg2

SOURCE_URL = "postgresql://dmai:yCAKCSRzNWv4t4URrZRWB31sJAbXA1Pr@dpg-d6ltllshg0os73avo390-a.oregon-postgres.render.com/harvester_63nu"

def deep_inspect():
    print("=" * 60)
    print("🔍 DEEP INSPECTION OF SOURCE DATABASE")
    print("=" * 60)
    
    try:
        conn = psycopg2.connect(SOURCE_URL)
        cur = conn.cursor()
        
        # 1. Check ALL schemas including hidden ones
        print("\n📊 ALL SCHEMAS:")
        cur.execute("""
            SELECT nspname, 
                   nspowner::regrole::text,
                   pg_namespace_size(nspname)/1024 as size_kb
            FROM pg_namespace 
            WHERE nspname NOT LIKE 'pg_%' 
            AND nspname != 'information_schema'
            ORDER BY nspname
        """)
        schemas = cur.fetchall()
        for schema, owner, size_kb in schemas:
            print(f"   - {schema} (owner: {owner}, size: {size_kb} KB)")
        
        # 2. For each schema, list tables and row counts
        for schema, _, _ in schemas:
            print(f"\n📋 TABLES IN SCHEMA '{schema}':")
            
            # Get tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = %s
                ORDER BY tablename
            """, (schema,))
            tables = cur.fetchall()
            
            if tables:
                for table in tables:
                    table_name = table[0]
                    try:
                        # Get row count
                        cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{table_name}"')
                        count = cur.fetchone()[0]
                        
                        # Get table size
                        cur.execute(f"""
                            SELECT pg_total_relation_size('"{schema}"."{table_name}"')/1024 as size_kb
                        """)
                        size_kb = cur.fetchone()[0]
                        
                        print(f"   - {table_name}: {count} rows, {size_kb} KB")
                    except Exception as e:
                        print(f"   - {table_name}: ERROR - {e}")
            else:
                print("   (no tables)")
        
        # 3. Check for any data in any table
        print("\n📈 TOTAL DATABASE STATISTICS:")
        cur.execute("""
            SELECT sum(pg_total_relation_size(relid))/1024/1024 as total_mb
            FROM pg_statio_user_tables
        """)
        total_mb = cur.fetchone()[0]
        print(f"   Total user table data: {total_mb:.2f} MB")
        
        # 4. Check if there's a search_path set
        print("\n🛣️  SEARCH PATH:")
        cur.execute("SHOW search_path")
        search_path = cur.fetchone()[0]
        print(f"   Current search_path: {search_path}")
        
        # 5. Check if there are any foreign tables
        print("\n🌍 FOREIGN TABLES:")
        cur.execute("""
            SELECT foreign_table_schema, foreign_table_name
            FROM information_schema.foreign_tables
        """)
        foreign_tables = cur.fetchall()
        if foreign_tables:
            for schema, table in foreign_tables:
                print(f"   - {schema}.{table}")
        else:
            print("   No foreign tables")
        
        # 6. Check database configuration
        print("\n⚙️  DATABASE CONFIG:")
        cur.execute("""
            SELECT name, setting, unit
            FROM pg_settings
            WHERE name IN ('server_version', 'shared_buffers', 'work_mem', 'maintenance_work_mem')
        """)
        config = cur.fetchall()
        for name, setting, unit in config:
            print(f"   - {name}: {setting} {unit or ''}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    deep_inspect()
