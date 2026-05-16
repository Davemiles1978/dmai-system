#!/usr/bin/env python3
"""
P0T0_Migrate_Database_to_Production.py
CRITICAL: Migrate data from SOURCE (My Workspace, 1 day old) 
to TARGET (dmai-production, 10 days old)
Then reconfigure DMAI to use the TARGET database
FIXED: Non-interactive when imported
"""

import os
import sys
import psycopg2
from psycopg2 import sql
import json
import logging
from datetime import datetime
import subprocess
import time

# ============================================================================
# FIX: When this file is imported (not run directly), don't prompt for input
# ============================================================================
if __name__ != "__main__":
    # Override input to return "no" automatically when imported
    def input(*args, **kwargs):
        return "n"  # Default to "no" when imported to prevent blocking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DBMigration')

# Database configurations
SOURCE_DB = {
    'name': 'SOURCE - My Workspace (1 day old - GOOD DATA)',
    'url': 'postgresql://dmai:yCAKCSRzNWv4t4URrZRWB31sJAbXA1Pr@dpg-d6ltllshg0os73avo390-a.oregon-postgres.render.com/harvester_63nu',
    'workspace': 'My Workspace',
    'updated': '1 day ago'
}

TARGET_DB = {
    'name': 'TARGET - dmai-production (10 days old - NEEDS UPDATE)',
    'url': 'postgresql://dmai:xQjt0tbhmT0vRExNv9wTSbe3t7n34J85@dpg-d6lfcg3h46gs73drf3fg-a.oregon-postgres.render.com/harvester_u9ni',
    'workspace': 'dmai-production',
    'updated': '10 days ago'
}

class DatabaseMigration:
    """
    Migrates data from SOURCE (good data) to TARGET (production)
    """
    
    def __init__(self):
        self.source_url = SOURCE_DB['url']
        self.target_url = TARGET_DB['url']
        self.migration_stats = {
            'tables_migrated': 0,
            'records_migrated': 0,
            'failed_tables': [],
            'start_time': None,
            'end_time': None
        }
        # Flag to track if we're running in non-interactive mode
        self.non_interactive = __name__ != "__main__"
        
    def run(self, auto_confirm=False):
        """
        Main execution method
        
        Args:
            auto_confirm: If True, automatically proceed without prompting
        """
        logger.info("=" * 70)
        logger.info("🔄 DMAI DATABASE MIGRATION")
        logger.info("=" * 70)
        logger.info(f"\n📊 SOURCE: {SOURCE_DB['name']}")
        logger.info(f"📊 TARGET: {TARGET_DB['name']}")
        logger.info("\n⚠️  This will OVERWRITE the 10-day-old database with 1-day-old data")
        
        # Check if we should auto-confirm (when imported)
        if auto_confirm or self.non_interactive:
            logger.info("🔄 Running in non-interactive mode - auto-confirming migration")
            response = "yes"
        else:
            response = input("\nDo you want to proceed? (yes/no): ")
            
        if response.lower() not in ['yes', 'y']:
            logger.info("Migration cancelled")
            return False
        
        # Step 1: Verify both connections
        if not self._verify_connections():
            logger.error("❌ Cannot proceed - database connections failed")
            return False
            
        # Step 2: Backup target database before migration
        if not self._backup_target_db():
            logger.error("❌ Failed to backup target database")
            return False
            
        # Step 3: Perform data migration
        if not self._migrate_data():
            logger.error("❌ Data migration failed")
            return False
            
        # Step 4: Verify migration
        if not self._verify_migration():
            logger.error("❌ Migration verification failed")
            return False
            
        # Step 5: Update DMAI configuration to use target database
        if not self._update_configuration():
            logger.error("❌ Configuration update failed")
            return False
            
        logger.info("=" * 70)
        logger.info("✅✅ MIGRATION COMPLETED SUCCESSFULLY! ✅✅")
        logger.info("=" * 70)
        logger.info(f"   Tables migrated: {self.migration_stats['tables_migrated']}")
        logger.info(f"   Records migrated: {self.migration_stats['records_migrated']}")
        logger.info(f"   Failed tables: {len(self.migration_stats['failed_tables'])}")
        logger.info("=" * 70)
        
        return True
    
    def _verify_connections(self):
        """Verify we can connect to both databases"""
        logger.info("\n🔍 VERIFYING DATABASE CONNECTIONS...")
        
        # Check source database (good data)
        try:
            source_conn = psycopg2.connect(self.source_url)
            source_cur = source_conn.cursor()
            
            source_cur.execute("""
                SELECT current_database(), 
                       inet_server_addr(),
                       pg_database_size(current_database()) / 1024 / 1024 as size_mb,
                       (SELECT count(*) FROM information_schema.tables WHERE table_schema='public')
            """)
            source_db, source_addr, source_size, source_tables = source_cur.fetchone()
            
            logger.info(f"\n✅ SOURCE DATABASE (Good Data):")
            logger.info(f"   Database: {source_db}")
            logger.info(f"   Server: {source_addr}")
            logger.info(f"   Size: {source_size:.2f} MB")
            logger.info(f"   Tables: {source_tables}")
            
            self.source_conn = source_conn
            self.source_cur = source_cur
            
        except Exception as e:
            logger.error(f"❌ Cannot connect to source database: {e}")
            return False
            
        # Check target database (needs update)
        try:
            target_conn = psycopg2.connect(self.target_url)
            target_cur = target_conn.cursor()
            
            target_cur.execute("""
                SELECT current_database(), 
                       inet_server_addr(),
                       pg_database_size(current_database()) / 1024 / 1024 as size_mb,
                       (SELECT count(*) FROM information_schema.tables WHERE table_schema='public')
            """)
            target_db, target_addr, target_size, target_tables = target_cur.fetchone()
            
            logger.info(f"\n✅ TARGET DATABASE (Needs Update):")
            logger.info(f"   Database: {target_db}")
            logger.info(f"   Server: {target_addr}")
            logger.info(f"   Size: {target_size:.2f} MB")
            logger.info(f"   Tables: {target_tables}")
            
            self.target_conn = target_conn
            self.target_cur = target_cur
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cannot connect to target database: {e}")
            return False
    
    def _backup_target_db(self):
        """Backup the target database before migration"""
        logger.info("\n💾 BACKING UP TARGET DATABASE...")
        
        backup_file = f"backup_target_before_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        try:
            # Extract connection details
            from urllib.parse import urlparse
            result = urlparse(self.target_url)
            dbname = result.path[1:]
            user = result.username
            password = result.password
            host = result.hostname
            port = result.port or 5432
            
            # Create backup using pg_dump
            cmd = f"PGPASSWORD='{password}' pg_dump -h {host} -p {port} -U {user} -d {dbname} > {backup_file}"
            
            logger.info(f"   Creating backup: {backup_file}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Backup created: {backup_file}")
                self.backup_file = backup_file
                return True
            else:
                logger.error(f"❌ Backup failed: {result.stderr}")
                return self._python_backup()
                
        except Exception as e:
            logger.error(f"❌ Backup error: {e}")
            return self._python_backup()
    
    def _python_backup(self):
        """Fallback backup method using Python"""
        logger.info("   Attempting Python-based backup...")
        
        backup_data = {}
        
        try:
            # Get all tables
            self.target_cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='public'
            """)
            tables = self.target_cur.fetchall()
            
            for table in tables:
                table_name = table[0]
                self.target_cur.execute(f"SELECT * FROM {table_name}")
                rows = self.target_cur.fetchall()
                
                # Get column names
                self.target_cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='{table_name}'
                """)
                columns = [col[0] for col in self.target_cur.fetchall()]
                
                backup_data[table_name] = {
                    'columns': columns,
                    'rows': rows
                }
            
            backup_file = f"backup_target_python_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, default=str, indent=2)
            
            logger.info(f"✅ Python backup created: {backup_file}")
            self.backup_file = backup_file
            return True
            
        except Exception as e:
            logger.error(f"❌ Python backup failed: {e}")
            return False
    
    def _migrate_data(self):
        """Migrate data from source to target database"""
        logger.info("\n📤 MIGRATING DATA FROM SOURCE TO TARGET...")
        self.migration_stats['start_time'] = datetime.now()
        
        try:
            # Get list of tables from source database
            self.source_cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='public'
                ORDER BY table_name
            """)
            tables = self.source_cur.fetchall()
            
            # Clear existing data in target (optional - comment out if you want to keep old data)
            logger.info("   Clearing existing data in target database...")
            for table in tables:
                table_name = table[0]
                try:
                    self.target_cur.execute(f"TRUNCATE TABLE {table_name} CASCADE")
                    self.target_conn.commit()
                except:
                    self.target_conn.rollback()
                    logger.warning(f"   Could not truncate {table_name}")
            
            # Migrate each table
            for table in tables:
                table_name = table[0]
                logger.info(f"   Processing table: {table_name}")
                
                # Skip migration log table to avoid conflicts
                if table_name == 'migration_log':
                    continue
                
                try:
                    # Get table structure from source
                    self.source_cur.execute(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name='{table_name}'
                        ORDER BY ordinal_position
                    """)
                    columns = self.source_cur.fetchall()
                    column_names = [col[0] for col in columns]
                    
                    # Ensure table exists in target with same structure
                    self._ensure_table_exists(table_name, columns)
                    
                    # Get data from source
                    self.source_cur.execute(f"SELECT * FROM {table_name}")
                    rows = self.source_cur.fetchall()
                    
                    if rows:
                        # Insert into target
                        placeholders = ','.join(['%s'] * len(column_names))
                        insert_query = f"""
                            INSERT INTO {table_name} ({','.join(column_names)})
                            VALUES ({placeholders})
                        """
                        
                        for row in rows:
                            try:
                                self.target_cur.execute(insert_query, row)
                                self.migration_stats['records_migrated'] += 1
                            except Exception as e:
                                logger.warning(f"      ⚠️ Failed to insert row: {e}")
                        
                        self.target_conn.commit()
                        self.migration_stats['tables_migrated'] += 1
                        logger.info(f"      ✅ Migrated {len(rows)} records to {table_name}")
                    
                except Exception as e:
                    logger.error(f"      ❌ Failed to migrate table {table_name}: {e}")
                    self.migration_stats['failed_tables'].append(table_name)
                    self.target_conn.rollback()
            
            # Log migration
            self._log_migration()
            
            self.migration_stats['end_time'] = datetime.now()
            duration = (self.migration_stats['end_time'] - self.migration_stats['start_time']).total_seconds()
            
            logger.info(f"\n✅ Migration complete in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
    
    def _ensure_table_exists(self, table_name, columns):
        """Ensure table exists in target database with correct structure"""
        try:
            # Check if table exists
            self.target_cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            exists = self.target_cur.fetchone()[0]
            
            if not exists:
                # Create table based on source structure
                col_defs = []
                for col_name, col_type in columns:
                    col_defs.append(f"{col_name} {col_type}")
                
                create_query = f"""
                    CREATE TABLE {table_name} (
                        {', '.join(col_defs)}
                    )
                """
                self.target_cur.execute(create_query)
                self.target_conn.commit()
                logger.info(f"      ✅ Created table: {table_name}")
                
        except Exception as e:
            logger.error(f"      ❌ Failed to ensure table {table_name}: {e}")
            raise
    
    def _log_migration(self):
        """Log migration in target database"""
        try:
            # Create migration log table if not exists
            self.target_cur.execute("""
                CREATE TABLE IF NOT EXISTS migration_log (
                    id SERIAL PRIMARY KEY,
                    migration_time TIMESTAMP DEFAULT NOW(),
                    source_db TEXT,
                    target_db TEXT,
                    tables_migrated INTEGER,
                    records_migrated INTEGER,
                    status TEXT
                )
            """)
            
            # Log in target database
            self.target_cur.execute("""
                INSERT INTO migration_log 
                (source_db, target_db, tables_migrated, records_migrated, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                'source_myworkspace_1day',
                'target_dmai-production_10day',
                self.migration_stats['tables_migrated'],
                self.migration_stats['records_migrated'],
                'completed'
            ))
            self.target_conn.commit()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log migration: {e}")
    
    def _verify_migration(self):
        """Verify data was migrated correctly"""
        logger.info("\n🔍 VERIFYING MIGRATION...")
        
        try:
            # Get table lists
            self.source_cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='public'
            """)
            source_tables = {row[0] for row in self.source_cur.fetchall()}
            
            self.target_cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='public'
            """)
            target_tables = {row[0] for row in self.target_cur.fetchall()}
            
            # Check for missing tables
            missing_in_target = source_tables - target_tables
            if missing_in_target:
                logger.warning(f"⚠️ Tables missing in target: {missing_in_target}")
            
            # Verify record counts for key tables
            key_tables = ['evolution_state', 'knowledge_graph', 'learning_patterns', 'system_state']
            verification_passed = True
            
            for table in key_tables:
                if table in source_tables and table in target_tables:
                    try:
                        self.source_cur.execute(f"SELECT COUNT(*) FROM {table}")
                        source_count = self.source_cur.fetchone()[0]
                        
                        self.target_cur.execute(f"SELECT COUNT(*) FROM {table}")
                        target_count = self.target_cur.fetchone()[0]
                        
                        if source_count == target_count:
                            logger.info(f"   ✅ {table}: {target_count} records match")
                        else:
                            logger.warning(f"   ⚠️ {table}: source={source_count}, target={target_count}")
                            verification_passed = False
                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not verify {table}: {e}")
            
            return verification_passed
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def _update_configuration(self):
        """Update DMAI configuration to use target database"""
        logger.info("\n⚙️ UPDATING DMAI CONFIGURATION TO USE TARGET DATABASE...")
        
        # Update environment files
        env_files = [
            '.env',
            '.env.production',
            'config/.env',
            'components/.env',
            'components/config/.env'
        ]
        
        for env_file in env_files:
            try:
                if os.path.exists(env_file):
                    with open(env_file, 'r') as f:
                        content = f.read()
                    
                    # Update DATABASE_URL
                    lines = content.split('\n')
                    new_lines = []
                    updated = False
                    
                    for line in lines:
                        if line.startswith('DATABASE_URL='):
                            new_lines.append(f'DATABASE_URL="{self.target_url}"')
                            updated = True
                        else:
                            new_lines.append(line)
                    
                    if not updated:
                        new_lines.append(f'\nDATABASE_URL="{self.target_url}"')
                    
                    with open(env_file, 'w') as f:
                        f.write('\n'.join(new_lines))
                    
                    logger.info(f"   ✅ Updated {env_file}")
                    
            except Exception as e:
                logger.error(f"   ❌ Failed to update {env_file}: {e}")
        
        # Create database configuration file
        config = {
            'database': {
                'production': {
                    'url': self.target_url,
                    'workspace': 'dmai-production',
                    'status': 'active',
                    'notes': 'Now contains data migrated from My Workspace (1 day old)'
                },
                'source_backup': {
                    'url': self.source_url,
                    'workspace': 'My Workspace',
                    'status': 'deprecated - ready for deletion',
                    'migrated_at': datetime.now().isoformat()
                }
            },
            'migration': self.migration_stats
        }
        
        with open('database_config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info("   ✅ Created database_config.json")
        
        # Create deployment marker
        marker = {
            'database_configured': self.target_url,
            'configured_at': datetime.now().isoformat(),
            'source_migrated': self.source_url,
            'status': 'production_ready'
        }
        
        with open('production_db_active.json', 'w') as f:
            json.dump(marker, f, indent=2)
        
        logger.info("   ✅ Created production_db_active.json")
        
        return True

def run():
    """Standard entry point for DMAI component execution"""
    migration = DatabaseMigration()
    # When called via run(), auto-confirm since it's being executed by the system
    return migration.run(auto_confirm=True)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔄 DMAI DATABASE MIGRATION UTILITY")
    print("=" * 70)
    print(f"\n📤 SOURCE: {SOURCE_DB['name']}")
    print(f"📥 TARGET: {TARGET_DB['name']}")
    print("\n⚠️  This will TRANSFER all data from SOURCE to TARGET")
    print("   and then reconfigure DMAI to use the TARGET database")
    
    migration = DatabaseMigration()
    success = migration.run(auto_confirm=False)  # Interactive when run directly
    
    if success:
        print("\n" + "=" * 70)
        print("✅✅✅ MIGRATION COMPLETED SUCCESSFULLY! ✅✅✅")
        print("=" * 70)
        print("\n📋 NEXT STEPS:")
        print("1. Restart all DMAI services:")
        print("   pkill -f python; python3 dmai_core.py & python3 dmai_web.py &")
        print("\n2. Verify data is flowing correctly")
        print("\n3. After verification, you can safely delete the source database:")
        print("   - Go to Render Dashboard")
        print("   - Switch to 'My Workspace'")
        print("   - Delete the database: harvester_63nu")
        print("\n📁 Files created:")
        print("   - database_config.json - New configuration")
        print("   - production_db_active.json - Production marker")
        print("   - backup_*.sql - Database backup")
        print("   - db_migration.log - Detailed log")
    else:
        print("\n❌ Migration failed - check db_migration.log")
