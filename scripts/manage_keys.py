#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
DMAI API Key Manager
List and activate discovered API keys
"""

import os
import sys
import psycopg2
from psycopg2 import extras
from datetime import datetime
import hashlib

class KeyManager:
    def __init__(self, db_url=None):
        self.db_url = db_url or os.getenv('PRODUCTION_DB_URL')
        if not self.db_url:
            raise ValueError("Please set PRODUCTION_DB_URL environment variable")
        self.conn = psycopg2.connect(self.db_url)
    
    def get_keys(self, limit=50):
        """Get all keys from database"""
        with self.conn.cursor(cursor_factory=extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, key_hash, service, source_url, source_type, 
                       discovered_at, is_valid, last_validated
                FROM discovered_keys 
                ORDER BY discovered_at DESC 
                LIMIT %s
            """, (limit,))
            return cur.fetchall()
    
    def get_valid_keys(self, limit=50):
        """Get validated keys"""
        with self.conn.cursor(cursor_factory=extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, key_hash, service, source_url, discovered_at
                FROM discovered_keys 
                WHERE is_valid = true
                ORDER BY discovered_at DESC 
                LIMIT %s
            """, (limit,))
            return cur.fetchall()
    
    def get_stats(self):
        """Get key statistics"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM discovered_keys")
            total = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM discovered_keys WHERE is_valid = true")
            valid = cur.fetchone()[0]
            
            return total, valid
    
    def add_key_to_env(self, key_id):
        """Add a validated key to .env file"""
        with self.conn.cursor() as cur:
            # Get key details - note we can't get actual key, only hash
            cur.execute("""
                SELECT key_hash, service, is_valid 
                FROM discovered_keys 
                WHERE id = %s
            """, (key_id,))
            key = cur.fetchone()
            if not key:
                print(f"❌ Key ID {key_id} not found")
                return False
            
            key_hash, service, is_valid = key
            
            if not is_valid:
                print(f"❌ Key {key_id} is not validated yet")
                return False
            
            # Since we only have hash, we need to get the actual key from somewhere
            # This would need to be stored separately or retrieved from secure storage
            print(f"\n⚠️  Note: Database stores only key hashes for security")
            print(f"   To use this key, you need to:")
            print(f"   1. Find the original key (from logs or secure storage)")
            print(f"   2. Add it manually to .env as:")
            print(f"   {service.upper()}_API_KEY=actual_key_here")
            
            # Create .env if it doesn't exist
            env_file = os.path.expanduser("~/Desktop/dmai-system/.env")
            if not os.path.exists(env_file):
                open(env_file, 'a').close()
            
            with open(env_file, 'a') as f:
                f.write(f"\n# Placeholder for key ID {key_id} (hash: {key_hash[:8]}...) added on {datetime.now()}\n")
                f.write(f"# {service.upper()}_API_KEY=insert_actual_key_here\n")
            
            print(f"\n✅ Placeholder added to {env_file}")
            print("⚠️  Edit the file to add the actual key, then restart services")
            return True
    
    def interactive_menu(self):
        """Run interactive menu"""
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("\n" + "="*60)
            print("🔑 DMAI API KEY MANAGER")
            print("="*60)
            
            # Show stats
            total, valid = self.get_stats()
            print(f"\n📊 DATABASE STATISTICS")
            print(f"   Total Keys: {total}")
            print(f"   Valid Keys: {valid}")
            
            # Show valid keys (ready to use)
            if valid > 0:
                print(f"\n✅ VALIDATED KEYS (ready to add to .env)")
                keys = self.get_valid_keys(10)
                for i, key in enumerate(keys, 1):
                    masked = key['key_hash'][:8] + "..." if key['key_hash'] else "unknown"
                    print(f"{i}. [{key['id']}] {key['service']}: {masked}")
                    print(f"   Found: {key['discovered_at'][:16]} | Source: {key['source_url'][:30]}...")
            else:
                print(f"\n⏳ No validated keys yet. Harvester is still searching...")
                if total > 0:
                    print(f"   (Found {total} unvalidated keys - validator needs to check them)")
            
            print("\n" + "-"*60)
            print("Commands:")
            if valid > 0:
                print("  [number] - Add key placeholder to .env")
            print("  [q] - Quit")
            print("  [r] - Refresh")
            
            choice = input("\n> ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'r':
                continue
            
            # Try to parse as number
            try:
                if valid > 0:
                    idx = int(choice) - 1
                    keys = self.get_valid_keys(10)
                    if 0 <= idx < len(keys):
                        key = keys[idx]
                        print(f"\nAdding placeholder for key {key['id']}...")
                        self.add_key_to_env(key['id'])
                        input("\nPress Enter to continue...")
                    else:
                        print("❌ Invalid selection")
                        input("\nPress Enter to continue...")
            except ValueError:
                print("❌ Invalid command")
                input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Check if DB URL is set
    if not os.getenv('PRODUCTION_DB_URL'):
        print("❌ PRODUCTION_DB_URL environment variable not set")
        print("\nRun this command first:")
        print('export PRODUCTION_DB_URL="postgresql://dmai:xQjt0tbhmT0vRExNv9wTSbe3t7n34J85@dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com/harvester_u9ni?sslmode=require"')
        sys.exit(1)
    
    manager = KeyManager()
    try:
        manager.interactive_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        manager.conn.close()
