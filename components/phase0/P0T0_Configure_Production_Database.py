#!/usr/bin/env python3
"""
P0T0_Configure_Production_Database.py
Configure DMAI to use the correct production database
"""

import os
import sys

# The CORRECT database URL (dmai-production - 8 MB with data)
PRODUCTION_DB_URL = "postgresql://dmai:xQjt0tbhmT0vRExNv9wTSbe3t7n34J85@dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com/harvester_u9ni"

def main():
    print("\n" + "="*60)
    print("🔧 CONFIGURING DMAI TO USE PRODUCTION DATABASE")
    print("="*60)
    
    # Update .env file
    env_files = ['.env', '.env.production', 'config/.env', 'components/.env']
    
    for env_file in env_files:
        try:
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    content = f.read()
                
                # Update or add DATABASE_URL
                lines = content.split('\n')
                new_lines = []
                updated = False
                
                for line in lines:
                    if line.startswith('DATABASE_URL='):
                        new_lines.append(f'DATABASE_URL="{PRODUCTION_DB_URL}"')
                        updated = True
                    else:
                        new_lines.append(line)
                
                if not updated:
                    new_lines.append(f'\n# Production database - contains evolution data\nDATABASE_URL="{PRODUCTION_DB_URL}"')
                
                with open(env_file, 'w') as f:
                    f.write('\n'.join(new_lines))
                
                print(f"✅ Updated {env_file}")
            else:
                # Create file with database URL
                with open(env_file, 'w') as f:
                    f.write(f'# DMAI Production Configuration\nDATABASE_URL="{PRODUCTION_DB_URL}"\n')
                print(f"✅ Created {env_file} with production database")
                
        except Exception as e:
            print(f"⚠️  Could not update {env_file}: {e}")
    
    # Create a marker file
    with open('production_db_configured.txt', 'w') as f:
        f.write(f"""DMAI PRODUCTION DATABASE CONFIGURED
================================
Database: harvester_u9ni
Workspace: dmai-production
Size: 8 MB
Tables: 9 tables with data (including 69 system_weaknesses)
Configured at: {__import__('datetime').datetime.now()}

This database is now the primary data store for DMAI.
""")
    
    print("\n✅ Production database configuration complete!")
    print("\n📊 DATABASE STATUS:")
    print("   - Database: harvester_u9ni")
    print("   - Workspace: dmai-production")
    print("   - Size: 8 MB")
    print("   - Contains: 69 system_weaknesses, API keys, validation logs")
    print("\n🚀 NEXT STEP: Restart DMAI services")
    print("   pkill -f python; python3 dmai_core.py & python3 dmai_web.py &")
    print("\n📝 After restart, verify data is being read correctly")

if __name__ == "__main__":
    main()
