"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

Database Manager for API Harvester
Handles all database operations for discovered keys, sources, and validation logs
"""

import psycopg2
from psycopg2 import sql, extras
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations for the harvester"""
    
    def __init__(self, connection_string=None):
        """Initialize database connection"""
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("DATABASE_URL environment variable not set")
        self.conn = None
        self.connect()
        self.init_tables()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.autocommit = False
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def init_tables(self):
        """Initialize database tables if they don't exist"""
        try:
            with self.conn.cursor() as cur:
                # Create discovered_keys table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS discovered_keys (
                        id SERIAL PRIMARY KEY,
                        key_value TEXT NOT NULL,
                        source TEXT,
                        service TEXT,
                        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(50) DEFAULT 'pending',
                        metadata JSONB,
                        UNIQUE(key_value)
                    )
                """)
                
                # Create scraped_sources table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scraped_sources (
                        id SERIAL PRIMARY KEY,
                        source_url TEXT NOT NULL,
                        source_type VARCHAR(50),
                        last_scraped TIMESTAMP,
                        items_found INTEGER DEFAULT 0,
                        status VARCHAR(50) DEFAULT 'active',
                        UNIQUE(source_url)
                    )
                """)
                
                # Create validation_logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS validation_logs (
                        id SERIAL PRIMARY KEY,
                        key_id INTEGER REFERENCES discovered_keys(id),
                        validator_name VARCHAR(100),
                        checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_valid BOOLEAN,
                        response_time_ms INTEGER,
                        error_message TEXT,
                        metadata JSONB
                    )
                """)
                
                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_keys_status ON discovered_keys(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_keys_discovered ON discovered_keys(discovered_at)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_validation_key ON validation_logs(key_id)")
                
                self.conn.commit()
                logger.info("Database tables initialized successfully")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to initialize tables: {e}")
            raise
    
    def save_api(self, api_data: Dict[str, Any]) -> Optional[int]:
        """Save API/key information to database"""
        try:
            with self.conn.cursor() as cur:
                key_value = api_data.get('key_value')
                if not key_value:
                    logger.error("Cannot save API: missing key_value")
                    return None
                
                source = api_data.get('source', 'unknown')
                service = api_data.get('service', 'unknown')
                metadata = api_data.get('metadata', {})
                status = api_data.get('status', 'pending')
                
                cur.execute("""
                    INSERT INTO discovered_keys 
                    (key_value, source, service, discovered_at, status, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (key_value) DO UPDATE SET
                        source = EXCLUDED.source,
                        service = EXCLUDED.service,
                        status = EXCLUDED.status,
                        metadata = EXCLUDED.metadata
                    RETURNING id
                """, (
                    key_value, 
                    source, 
                    service, 
                    api_data.get('discovered_at', datetime.now()),
                    status,
                    extras.Json(metadata)
                ))
                
                record_id = cur.fetchone()[0]
                self.conn.commit()
                logger.debug(f"Saved API {service}:{key_value[:10]}... (ID: {record_id})")
                return record_id
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving API {api_data.get('service')}: {e}")
            return None
    
    def save_apikey(self, key_data: Dict[str, Any]) -> Optional[int]:
        """Alias for save_api"""
        return self.save_api(key_data)
    
    def save_discovered_key(self, key_data: Dict[str, Any]) -> Optional[int]:
        """Alias for save_api"""
        return self.save_api(key_data)
    
    def insert_api(self, api_data: Dict[str, Any]) -> Optional[int]:
        """Alias for save_api"""
        return self.save_api(api_data)
    
    def get_pending_keys(self, limit: int = 100) -> List[Dict]:
        """Get keys pending validation"""
        try:
            with self.conn.cursor(cursor_factory=extras.DictCursor) as cur:
                cur.execute("""
                    SELECT * FROM discovered_keys 
                    WHERE status = 'pending' 
                    ORDER BY discovered_at 
                    LIMIT %s
                """, (limit,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching pending keys: {e}")
            return []
    
    def update_key_status(self, key_id: int, status: str, metadata: Optional[Dict] = None):
        """Update the status of a key"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE discovered_keys 
                    SET status = %s, metadata = metadata || %s
                    WHERE id = %s
                """, (status, extras.Json(metadata or {}), key_id))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating key status: {e}")
    
    def log_validation(self, key_id: int, validator_name: str, is_valid: bool, 
                      response_time_ms: int = None, error_message: str = None, 
                      metadata: Optional[Dict] = None):
        """Log a validation attempt"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO validation_logs 
                    (key_id, validator_name, is_valid, response_time_ms, error_message, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    key_id, validator_name, is_valid, response_time_ms, 
                    error_message, extras.Json(metadata or {})
                ))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error logging validation: {e}")
    
    def record_source_scrape(self, source_url: str, source_type: str, items_found: int):
        """Record that a source was scraped"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO scraped_sources (source_url, source_type, last_scraped, items_found)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_url) DO UPDATE SET
                        last_scraped = EXCLUDED.last_scraped,
                        items_found = scraped_sources.items_found + EXCLUDED.items_found
                """, (source_url, source_type, datetime.now(), items_found))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error recording source scrape: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
