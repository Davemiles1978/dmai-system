#!/usr/bin/env python3
"""
Database configuration for Render PostgreSQL
"""
import os
import json
import psycopg2
from psycopg2.extras import Json
import logging

logger = logging.getLogger('database')

class DMAIDatabase:
    def __init__(self):
        # Get database URL from Render environment
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            # Fallback to local SQLite for development
            self.use_sqlite = True
            import sqlite3
            self.sqlite_conn = sqlite3.connect('dmai_local.db', check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            logger.info("Using SQLite local database")
        else:
            self.use_sqlite = False
            self.conn = None
            self._connect()
    
    def _connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(self.database_url)
            self._init_tables()
            logger.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.use_sqlite = True
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        with self.conn.cursor() as cur:
            # API Keys table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id SERIAL PRIMARY KEY,
                    service TEXT NOT NULL,
                    key TEXT NOT NULL,
                    source TEXT,
                    url TEXT,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # Knowledge graph table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    id SERIAL PRIMARY KEY,
                    node_type TEXT,
                    node_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Evolution history
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evolution_history (
                    id SERIAL PRIMARY KEY,
                    generation INTEGER,
                    component TEXT,
                    improvement JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.commit()
    
    def store_api_key(self, service, key, source=None, url=None, metadata=None):
        """Store a discovered API key"""
        if self.use_sqlite:
            cur = self.sqlite_conn.cursor()
            cur.execute(
                "INSERT INTO api_keys (service, key, source, url, metadata) VALUES (?, ?, ?, ?, ?)",
                (service, key, source, url, json.dumps(metadata) if metadata else None)
            )
            self.sqlite_conn.commit()
        else:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO api_keys (service, key, source, url, metadata) VALUES (%s, %s, %s, %s, %s)",
                    (service, key, source, url, Json(metadata) if metadata else None)
                )
                self.conn.commit()
        logger.info(f"💾 Stored API key for {service}")
    
    def get_all_api_keys(self):
        """Retrieve all API keys"""
        if self.use_sqlite:
            cur = self.sqlite_conn.cursor()
            cur.execute("SELECT * FROM api_keys ORDER BY discovered_at DESC")
            return [dict(row) for row in cur.fetchall()]
        else:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM api_keys ORDER BY discovered_at DESC")
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def get_keys_by_service(self, service):
        """Get keys for specific service"""
        if self.use_sqlite:
            cur = self.sqlite_conn.cursor()
            cur.execute("SELECT * FROM api_keys WHERE service = ?", (service,))
            return [dict(row) for row in cur.fetchall()]
        else:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM api_keys WHERE service = %s", (service,))
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
