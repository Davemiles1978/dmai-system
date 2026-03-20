#!/usr/bin/env python3
"""Test connection to dmai-harvester-db"""
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test PostgreSQL connection"""
    database_url = os.getenv('DATABASE_URL')
    print(f"🔌 Connecting to: {database_url.split('@')[1] if '@' in database_url else database_url}")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Test query
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"✅ Connected successfully!")
        print(f"📊 PostgreSQL version: {version[0]}")
        
        # Check if our tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()
        
        if tables:
            print(f"📋 Existing tables: {', '.join([t[0] for t in tables])}")
        else:
            print("📋 No tables found yet - will be created on first run")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
