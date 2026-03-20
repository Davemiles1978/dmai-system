#!/usr/bin/env python3
"""
██████╗ ███╗   ███╗ █████╗ ██╗
██╔══██╗████╗ ████║██╔══██╗██║
██║  ██║██╔████╔██║███████║██║
██║  ██║██║╚██╔╝██║██╔══██║██║
██████╔╝██║ ╚═╝ ██║██║  ██║██║
╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

DMAI CORE - SINGLE UNIFIED INTELLIGENCE
Version: 12.4 | AUTHENTICATION DISABLED - Direct access enabled
"""
import sys
from pathlib import Path

# Get the absolute path to the dmai-system directory
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

import os
import time
import json
import logging
import threading
import importlib
import importlib.util
import inspect
import queue
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import traceback

# Create logs directory if it doesn't exist
logs_dir = BASE_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(logs_dir / 'dmai_core.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CORE')

# ============================================================================
# DATABASE LAYER - PERSISTENT STORAGE WITH EXISTING SCHEMA SUPPORT
# ============================================================================

class DMAIDatabase:
    """Persistent database for API keys, knowledge graph, and evolution history
    Now supports the existing sophisticated PostgreSQL schema while maintaining
    SQLite fallback for local development
    """
    
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        self.use_sqlite = False
        self.conn = None
        self.sqlite_conn = None
        self.total_funding = 0.0
        self.transaction_error = False
        
        if self.database_url:
            self._connect_postgres()
        else:
            self._connect_sqlite()
    
    def _connect_postgres(self):
        """Connect to PostgreSQL (Render) with existing schema"""
        try:
            import psycopg2
            from psycopg2.extras import Json, RealDictCursor
            
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = False
            self._verify_and_create_tables()
            logger.info("✅ Connected to PostgreSQL database (Render)")
        except ImportError as e:
            logger.warning(f"psycopg2 not installed: {e}, falling back to SQLite")
            self._connect_sqlite()
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}, falling back to SQLite")
            self._connect_sqlite()
    
    def _verify_and_create_tables(self):
        """Verify PostgreSQL tables exist and create missing ones"""
        try:
            with self.conn.cursor() as cur:
                # Check if api_keys table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'api_keys'
                    )
                """)
                if not cur.fetchone()[0]:
                    logger.warning("api_keys table not found, will use SQLite fallback")
                    self.use_sqlite = True
                    return
                
                # Check if external_tools table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'external_tools'
                    )
                """)
                if not cur.fetchone()[0]:
                    logger.info("📦 Creating external_tools table...")
                    cur.execute("""
                        CREATE TABLE external_tools (
                            id SERIAL PRIMARY KEY,
                            tool_name TEXT UNIQUE NOT NULL,
                            tool_type TEXT NOT NULL,
                            api_endpoint TEXT,
                            api_key TEXT,
                            capabilities JSONB,
                            reliability_score FLOAT DEFAULT 0.5,
                            usage_count INTEGER DEFAULT 0,
                            last_used TIMESTAMP,
                            is_active BOOLEAN DEFAULT true,
                            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    self.conn.commit()
                    logger.info("✅ external_tools table created successfully")
                
                # Check if system_metrics table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'system_metrics'
                    )
                """)
                if not cur.fetchone()[0]:
                    logger.info("📦 Creating system_metrics table...")
                    cur.execute("""
                        CREATE TABLE system_metrics (
                            id SERIAL PRIMARY KEY,
                            metric_type TEXT NOT NULL,
                            value JSONB,
                            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    self.conn.commit()
                    logger.info("✅ system_metrics table created successfully")
                
                # Log the schema we're working with
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'api_keys'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
                logger.info(f"📋 Using existing api_keys schema with {len(columns)} columns")
                
        except Exception as e:
            logger.error(f"Error verifying PostgreSQL tables: {e}")
            self._reset_transaction()
            raise
    
    def _reset_transaction(self):
        """Reset a failed transaction"""
        if self.use_sqlite or self.conn is None:
            return
        try:
            self.conn.rollback()
            self.transaction_error = False
            logger.info("✅ Database transaction rolled back")
        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
    
    def _connect_sqlite(self):
        """Connect to SQLite (local development) with simple schema"""
        try:
            import sqlite3
            self.use_sqlite = True
            self.sqlite_conn = sqlite3.connect(str(BASE_DIR / 'dmai_local.db'), check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            self._init_sqlite_tables()
            
            # Load existing total funding
            self.total_funding = self.get_total_funding()
            logger.info("✅ Connected to SQLite database (local)")
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
    
    def _init_sqlite_tables(self):
        """Initialize SQLite tables with simple schema (fallback only)"""
        cur = self.sqlite_conn.cursor()
        
        # API Keys table - simple schema for SQLite fallback
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL,
                key_value TEXT NOT NULL,
                source TEXT,
                url TEXT,
                metadata TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Knowledge graph nodes
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kg_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT UNIQUE NOT NULL,
                node_type TEXT NOT NULL,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Knowledge graph edges
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kg_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight FLOAT DEFAULT 1.0,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES kg_nodes(node_id),
                FOREIGN KEY (target_id) REFERENCES kg_nodes(node_id)
            )
        """)
        
        # Evolution history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                component TEXT NOT NULL,
                improvement_type TEXT,
                improvement_data TEXT,
                success_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Funding transactions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS funding_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                amount DECIMAL(10,2) NOT NULL,
                currency TEXT DEFAULT 'USD',
                status TEXT DEFAULT 'pending',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System metrics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT NOT NULL,
                value TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tools table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS external_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT UNIQUE NOT NULL,
                tool_type TEXT NOT NULL,
                api_endpoint TEXT,
                api_key TEXT,
                capabilities TEXT,
                reliability_score FLOAT DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.sqlite_conn.commit()
        logger.info("✅ SQLite tables initialized")
    
    def store_api_key(self, service: str, key: str, source: str = None, url: str = None, metadata: dict = None):
        """Store a discovered API key - works with both PostgreSQL and SQLite schemas"""
        try:
            if self.use_sqlite:
                # SQLite fallback - simple schema
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    """INSERT INTO api_keys (service, key_value, source, url, metadata) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (service, key, source, url, json.dumps(metadata) if metadata else None)
                )
                self.sqlite_conn.commit()
                logger.info(f"💾 Stored API key for {service} in SQLite")
                return True
                
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                # PostgreSQL - use existing sophisticated schema
                with self.conn.cursor() as cur:
                    # Create a hash of the key for unique identification
                    key_hash = hashlib.sha256(key.encode()).hexdigest()
                    
                    # Check if key already exists
                    cur.execute(
                        "SELECT id FROM api_keys WHERE key_hash = %s",
                        (key_hash,)
                    )
                    existing = cur.fetchone()
                    
                    if existing:
                        logger.debug(f"Key hash {key_hash[:8]}... already exists, skipping")
                        return False
                    
                    # Store in your existing schema with proper field mapping
                    cur.execute("""
                        INSERT INTO api_keys (
                            key_hash, 
                            key_type, 
                            is_valid, 
                            weight, 
                            source_repo, 
                            source_url,
                            full_key_encrypted, 
                            key_preview,
                            created_at,
                            metadata,
                            status,
                            context,
                            times_used,
                            validation_count,
                            permissions,
                            rate_limit
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, 
                            CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (key_hash) DO NOTHING
                    """, (
                        key_hash,                    # key_hash
                        service,                      # key_type
                        True,                         # is_valid
                        1,                             # weight
                        source or 'unknown',          # source_repo
                        url or 'unknown',              # source_url
                        key,                           # full_key_encrypted
                        key[:20] + '...' if len(key) > 20 else key,  # key_preview
                        json.dumps(metadata) if metadata else '{}',  # metadata
                        'active',                      # status
                        f"Discovered from {source}",   # context
                        0,                              # times_used
                        0,                              # validation_count
                        '{}',                           # permissions
                        '{}'                            # rate_limit
                    ))
                    self.conn.commit()
                    
                    logger.info(f"💾 Stored API key for {service} in PostgreSQL (hash: {key_hash[:8]}...)")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            self.transaction_error = True
            return False
    
    def get_api_keys(self, service: str = None, limit: int = 100):
        """Retrieve API keys, optionally filtered by service - works with both schemas"""
        try:
            if self.use_sqlite:
                # SQLite fallback - simple schema
                cur = self.sqlite_conn.cursor()
                if service:
                    cur.execute(
                        "SELECT * FROM api_keys WHERE service = ? ORDER BY discovered_at DESC LIMIT ?",
                        (service, limit)
                    )
                else:
                    cur.execute(
                        "SELECT * FROM api_keys ORDER BY discovered_at DESC LIMIT ?",
                        (limit,)
                    )
                return [dict(row) for row in cur.fetchall()]
                
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                # PostgreSQL - map to your sophisticated schema with RealDictCursor
                from psycopg2.extras import RealDictCursor
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if service:
                        # Filter by key_type (which maps to service)
                        cur.execute("""
                            SELECT 
                                id,
                                key_type as service,
                                key_preview,
                                full_key_encrypted,
                                source_repo as source,
                                source_url as url,
                                created_at as discovered_at,
                                validated_at as last_validated,
                                times_used as validation_count,
                                status,
                                weight,
                                is_valid,
                                permissions,
                                rate_limit,
                                expires_at,
                                context
                            FROM api_keys 
                            WHERE key_type = %s 
                                AND is_valid = true
                            ORDER BY 
                                weight DESC,
                                created_at DESC 
                            LIMIT %s
                        """, (service, limit))
                    else:
                        # Get all keys
                        cur.execute("""
                            SELECT 
                                id,
                                key_type as service,
                                key_preview,
                                full_key_encrypted,
                                source_repo as source,
                                source_url as url,
                                created_at as discovered_at,
                                validated_at as last_validated,
                                times_used as validation_count,
                                status,
                                weight,
                                is_valid,
                                permissions,
                                rate_limit,
                                expires_at,
                                context
                            FROM api_keys 
                            WHERE is_valid = true
                            ORDER BY 
                                weight DESC,
                                created_at DESC 
                            LIMIT %s
                        """, (limit,))
                    
                    results = []
                    for row in cur.fetchall():
                        # Convert to dict and mask sensitive data
                        row_dict = dict(row)
                        # Don't expose full encrypted key in normal queries
                        row_dict['key_value'] = row_dict.pop('key_preview', '')
                        row_dict.pop('full_key_encrypted', None)
                        results.append(row_dict)
                    
                    logger.debug(f"Retrieved {len(results)} API keys from PostgreSQL")
                    return results
                    
        except Exception as e:
            logger.error(f"Failed to retrieve API keys: {e}")
            self.transaction_error = True
            return []
    
    def get_key_by_hash(self, key_hash: str):
        """Retrieve a specific key by its hash (PostgreSQL only)"""
        if self.use_sqlite:
            return None
            
        try:
            # Reset transaction if there was a previous error
            if self.transaction_error:
                self._reset_transaction()
            
            from psycopg2.extras import RealDictCursor
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        id,
                        key_hash,
                        key_type as service,
                        full_key_encrypted as key_value,
                        key_preview,
                        source_repo as source,
                        source_url as url,
                        created_at as discovered_at,
                        validated_at as last_validated,
                        times_used as validation_count,
                        status,
                        weight,
                        is_valid,
                        permissions,
                        rate_limit,
                        expires_at,
                        context,
                        metadata,
                        evolution_generation,
                        parent_key_hash,
                        mutation_count,
                        last_mutated_at,
                        estimated_value
                    FROM api_keys 
                    WHERE key_hash = %s
                """, (key_hash,))
                
                row = cur.fetchone()
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve key by hash: {e}")
            self.transaction_error = True
            return None
    
    def update_key_usage(self, key_id: int, key_hash: str = None):
        """Update last_used and increment usage_count for a key"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    "UPDATE api_keys SET last_used = CURRENT_TIMESTAMP, usage_count = usage_count + 1 WHERE id = ?",
                    (key_id,)
                )
                self.sqlite_conn.commit()
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    # Update in your sophisticated schema
                    cur.execute("""
                        UPDATE api_keys 
                        SET 
                            last_used = CURRENT_TIMESTAMP,
                            times_used = times_used + 1,
                            weight = weight + 1
                        WHERE id = %s OR key_hash = %s
                    """, (key_id, key_hash))
                    self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update key usage: {e}")
            self.transaction_error = True
            return False
    
    def validate_key(self, key_id: int = None, key_hash: str = None, is_valid: bool = True):
        """Mark a key as validated or invalid"""
        try:
            if self.use_sqlite:
                # Not implemented in SQLite
                return False
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    cur.execute("""
                        UPDATE api_keys 
                        SET 
                            is_valid = %s,
                            validated_at = CURRENT_TIMESTAMP,
                            validation_count = validation_count + 1,
                            status = CASE WHEN %s THEN 'active' ELSE 'invalid' END
                        WHERE id = %s OR key_hash = %s
                    """, (is_valid, is_valid, key_id, key_hash))
                    self.conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to validate key: {e}")
            self.transaction_error = True
            return False
    
    def get_key_statistics(self):
        """Get statistics about stored keys (PostgreSQL only)"""
        if self.use_sqlite:
            return {}
            
        try:
            # Reset transaction if there was a previous error
            if self.transaction_error:
                self._reset_transaction()
            
            from psycopg2.extras import RealDictCursor
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_keys,
                        COUNT(DISTINCT key_type) as service_types,
                        SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) as valid_keys,
                        SUM(CASE WHEN is_valid AND times_used > 0 THEN 1 ELSE 0 END) as used_keys,
                        AVG(weight) as avg_weight,
                        MAX(weight) as max_weight,
                        MIN(created_at) as oldest_key,
                        MAX(created_at) as newest_key
                    FROM api_keys
                """)
                return dict(cur.fetchone())
        except Exception as e:
            logger.error(f"Failed to get key statistics: {e}")
            self.transaction_error = True
            return {}
    
    def store_evolution_event(self, generation: int, component: str, improvement_type: str, 
                             improvement_data: dict, success_score: float = None):
        """Record an evolution event"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    """INSERT INTO evolution_history (generation, component, improvement_type, improvement_data, success_score) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (generation, component, improvement_type, json.dumps(improvement_data), success_score)
                )
                self.sqlite_conn.commit()
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    from psycopg2.extras import Json
                    cur.execute(
                        """INSERT INTO evolution_history (generation, component, improvement_type, improvement_data, success_score) 
                           VALUES (%s, %s, %s, %s, %s)""",
                        (generation, component, improvement_type, Json(improvement_data), success_score)
                    )
                    self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store evolution event: {e}")
            self.transaction_error = True
            return False
    
    def store_funding_transaction(self, source: str, amount: float, currency: str = 'USD', 
                                 status: str = 'completed', metadata: dict = None):
        """Record a funding transaction"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    """INSERT INTO funding_transactions (source, amount, currency, status, metadata) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (source, amount, currency, status, json.dumps(metadata) if metadata else None)
                )
                self.sqlite_conn.commit()
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    from psycopg2.extras import Json
                    cur.execute(
                        """INSERT INTO funding_transactions (source, amount, currency, status, metadata) 
                           VALUES (%s, %s, %s, %s, %s)""",
                        (source, amount, currency, status, Json(metadata) if metadata else None)
                    )
                    self.conn.commit()
            
            self.total_funding += amount
            logger.info(f"💰 Recorded funding: {amount} {currency} from {source}")
            return True
        except Exception as e:
            logger.error(f"Failed to store funding transaction: {e}")
            self.transaction_error = True
            return False
    
    def get_total_funding(self) -> float:
        """Calculate total funding generated"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute("SELECT SUM(amount) as total FROM funding_transactions WHERE status = 'completed'")
                result = cur.fetchone()
                return float(result[0]) if result and result[0] else 0.0
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    cur.execute("SELECT SUM(amount) as total FROM funding_transactions WHERE status = 'completed'")
                    result = cur.fetchone()
                    return float(result[0]) if result and result[0] else 0.0
        except Exception as e:
            logger.error(f"Failed to get total funding: {e}")
            return 0.0
    
    def store_kg_node(self, node_id: str, node_type: str, data: dict):
        """Store a knowledge graph node"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    """INSERT OR REPLACE INTO kg_nodes (node_id, node_type, data, updated_at) 
                       VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                    (node_id, node_type, json.dumps(data))
                )
                self.sqlite_conn.commit()
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    from psycopg2.extras import Json
                    cur.execute(
                        """INSERT INTO kg_nodes (node_id, node_type, data, updated_at) 
                           VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                           ON CONFLICT (node_id) DO UPDATE SET 
                           data = EXCLUDED.data, updated_at = CURRENT_TIMESTAMP""",
                        (node_id, node_type, Json(data))
                    )
                    self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store KG node: {e}")
            self.transaction_error = True
            return False
    
    def store_metric(self, metric_type: str, value: Any):
        """Store a system metric"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    "INSERT INTO system_metrics (metric_type, value) VALUES (?, ?)",
                    (metric_type, json.dumps(value) if isinstance(value, (dict, list)) else str(value))
                )
                self.sqlite_conn.commit()
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    from psycopg2.extras import Json
                    cur.execute(
                        "INSERT INTO system_metrics (metric_type, value) VALUES (%s, %s)",
                        (metric_type, Json(value) if isinstance(value, (dict, list)) else str(value))
                    )
                    self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
            self.transaction_error = True
            return False
    
    def register_tool(self, tool_name: str, tool_type: str, api_endpoint: str = None, 
                     api_key: str = None, capabilities: dict = None):
        """Register an external tool for DMAI to use"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    """INSERT OR REPLACE INTO external_tools 
                       (tool_name, tool_type, api_endpoint, api_key, capabilities) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (tool_name, tool_type, api_endpoint, api_key, 
                     json.dumps(capabilities) if capabilities else None)
                )
                self.sqlite_conn.commit()
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    from psycopg2.extras import Json
                    cur.execute(
                        """INSERT INTO external_tools 
                           (tool_name, tool_type, api_endpoint, api_key, capabilities) 
                           VALUES (%s, %s, %s, %s, %s)
                           ON CONFLICT (tool_name) DO UPDATE SET
                           tool_type = EXCLUDED.tool_type,
                           api_endpoint = EXCLUDED.api_endpoint,
                           api_key = EXCLUDED.api_key,
                           capabilities = EXCLUDED.capabilities""",
                        (tool_name, tool_type, api_endpoint, api_key, 
                         Json(capabilities) if capabilities else None)
                    )
                    self.conn.commit()
            logger.info(f"🔧 Registered tool: {tool_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register tool {tool_name}: {e}")
            self.transaction_error = True
            return False
    
    def get_tools_by_type(self, tool_type: str = None):
        """Get registered tools, optionally filtered by type"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                if tool_type:
                    cur.execute(
                        "SELECT * FROM external_tools WHERE tool_type = ? AND is_active = 1 ORDER BY reliability_score DESC",
                        (tool_type,)
                    )
                else:
                    cur.execute(
                        "SELECT * FROM external_tools WHERE is_active = 1 ORDER BY reliability_score DESC"
                    )
                return [dict(row) for row in cur.fetchall()]
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    if tool_type:
                        cur.execute(
                            "SELECT * FROM external_tools WHERE tool_type = %s AND is_active = true ORDER BY reliability_score DESC",
                            (tool_type,)
                        )
                    else:
                        cur.execute(
                            "SELECT * FROM external_tools WHERE is_active = true ORDER BY reliability_score DESC"
                        )
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get tools: {e}")
            self.transaction_error = True
            return []
    
    def update_tool_reliability(self, tool_name: str, success: bool):
        """Update tool reliability score based on usage success"""
        try:
            if self.use_sqlite:
                cur = self.sqlite_conn.cursor()
                cur.execute(
                    """UPDATE external_tools 
                       SET usage_count = usage_count + 1,
                           last_used = CURRENT_TIMESTAMP,
                           reliability_score = CASE 
                               WHEN ? THEN (reliability_score * usage_count + 1.0) / (usage_count + 1)
                               ELSE (reliability_score * usage_count + 0.0) / (usage_count + 1)
                           END
                       WHERE tool_name = ?""",
                    (1 if success else 0, tool_name)
                )
                self.sqlite_conn.commit()
            else:
                # Reset transaction if there was a previous error
                if self.transaction_error:
                    self._reset_transaction()
                
                with self.conn.cursor() as cur:
                    cur.execute(
                        """UPDATE external_tools 
                           SET usage_count = usage_count + 1,
                               last_used = CURRENT_TIMESTAMP,
                               reliability_score = CASE 
                                   WHEN %s THEN (reliability_score * usage_count + 1.0) / (usage_count + 1)
                                   ELSE (reliability_score * usage_count + 0.0) / (usage_count + 1)
                               END
                           WHERE tool_name = %s""",
                        (success, tool_name)
                    )
                    self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update tool reliability: {e}")
            self.transaction_error = True
            return False
    
    def close(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.conn:
            self.conn.close()

# ============================================================================
# EXTERNAL TOOL MANAGER
# ============================================================================

class ExternalToolManager:
    """Manages external tools (MiroFish, ChatGPT, etc.) as evolvable resources"""
    
    def __init__(self, dmai_core):
        self.dmai = dmai_core
        self.tools = {}
        self.tool_usage_history = []
        self._discover_tools()
    
    def _discover_tools(self):
        """Discover and register available external tools"""
        # Check for MiroFish
        mirofish_path = BASE_DIR / "mirofish"
        if mirofish_path.exists():
            self.dmai.db.register_tool(
                tool_name="MiroFish",
                tool_type="swarm_intelligence",
                capabilities={
                    "prediction": True,
                    "simulation": True,
                    "swarm_size": "variable"
                }
            )
            logger.info("🐟 MiroFish discovered and registered as tool")
        
        # Check for API keys to other services using the sophisticated schema
        api_keys = self.dmai.db.get_api_keys()
        for key in api_keys:
            service = key.get('service', '').lower()
            if service in ['openai', 'anthropic', 'google', 'deepseek', 'grok', 'claude', 'gemini']:
                self.dmai.db.register_tool(
                    tool_name=service.capitalize(),
                    tool_type="llm",
                    api_key=key.get('key_preview'),  # Use preview, not full key
                    capabilities={"text_generation": True, "reasoning": True}
                )
                logger.info(f"🤖 {service.capitalize()} registered as tool")
    
    def use_tool(self, tool_name: str, input_data: Any) -> Dict[str, Any]:
        """Use an external tool and track its performance"""
        tools = self.dmai.db.get_tools_by_type()
        tool = next((t for t in tools if t['tool_name'].lower() == tool_name.lower()), None)
        
        if not tool:
            return {"error": f"Tool {tool_name} not found", "success": False}
        
        start_time = time.time()
        
        try:
            if tool_name.lower() == "mirofish":
                result = self._call_mirofish(input_data)
            elif tool['tool_type'] == "llm":
                result = self._call_llm(tool, input_data)
            else:
                result = {"error": f"Unsupported tool type: {tool['tool_type']}"}
            
            success = "error" not in result
            response_time = time.time() - start_time
            
            self.dmai.db.update_tool_reliability(tool_name, success)
            
            self.tool_usage_history.append({
                "tool": tool_name,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "response_time": response_time,
                "input_summary": str(input_data)[:100]
            })
            
            # If tool is very reliable, trigger evolution to replicate it
            if success and tool.get('reliability_score', 0) > 0.8 and len(self.tool_usage_history) > 5:
                self.dmai.think("evolution", {
                    "type": "replicate_tool",
                    "tool": tool_name,
                    "reason": f"High reliability"
                }, priority=7)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            self.dmai.db.update_tool_reliability(tool_name, False)
            return {"error": str(e), "success": False}
    
    def _call_mirofish(self, input_data):
        """Call MiroFish API if available"""
        try:
            mirofish_path = BASE_DIR / "mirofish" / "backend"
            if mirofish_path.exists():
                sys.path.append(str(mirofish_path))
                try:
                    # Try to import MiroFish dynamically
                    from mirofish_integration import MiroFishAPI
                    api = MiroFishAPI()
                    return api.predict(input_data)
                except ImportError:
                    # Fall back to simulation
                    return {
                        "scenario": input_data.get("name", "unknown"),
                        "prediction": random.uniform(0, 1),
                        "confidence": random.uniform(0.6, 0.95),
                        "swarm_size": random.randint(10, 100),
                        "simulated": True,
                        "note": "MiroFish API not fully integrated"
                    }
            else:
                return {
                    "error": "MiroFish not installed",
                    "simulated": True
                }
        except Exception as e:
            logger.error(f"MiroFish call failed: {e}")
            return {"error": str(e)}
    
    def _call_llm(self, tool, input_data):
        """Call an LLM API (placeholder - would use actual API)"""
        # This is a placeholder - actual implementation would use the API key
        return {
            "tool": tool['tool_name'],
            "response": f"[Simulated response from {tool['tool_name']}]",
            "simulated": True,
            "note": "LLM API not yet implemented"
        }
    
    def get_tool_stats(self):
        """Get statistics about tool usage for evolution"""
        tools = self.dmai.db.get_tools_by_type()
        stats = []
        for tool in tools:
            stats.append({
                "name": tool['tool_name'],
                "type": tool['tool_type'],
                "reliability": tool.get('reliability_score', 0.5),
                "usage": tool.get('usage_count', 0),
                "last_used": tool.get('last_used', 'never')
            })
        return stats

# ============================================================================
# BIOMETRIC AUTHENTICATION - DISABLED VERSION
# ============================================================================

class BiometricAuth:
    """Multi-factor authentication - DISABLED - Always returns authenticated"""
    
    def __init__(self):
        logger.info("🔓 BIOMETRIC AUTHENTICATION DISABLED - All access granted")
        self.authenticated = True  # Always authenticated
        self.current_user = "Master"
        self.auth_time = datetime.now()
        self.used_methods = []
        
    def _check_password(self, credentials):
        """Password check - Always returns True"""
        return True
    
    def _check_voice(self, credentials):
        """Voice check - Always returns True"""
        return True
    
    def _check_face(self, credentials):
        """Face check - Always returns True"""
        return True
    
    def _check_fingerprint(self, credentials):
        """Fingerprint check - Always returns True"""
        return True
    
    def authenticate(self, method="password", credentials=None):
        """Authentication - Always succeeds"""
        logger.info(f"🔓 Authentication bypassed - accepting all logins")
        self.authenticated = True
        self.current_user = credentials.get("username", "Master") if credentials else "Master"
        self.auth_time = datetime.now()
        if method not in self.used_methods:
            self.used_methods.append(method)
        return True
    
    def check_2fa_status(self):
        """2FA status - Always False (no 2FA needed)"""
        return False
    
    def get_pending_methods(self):
        """Get pending methods - Always empty"""
        return []
    
    def logout(self):
        """Log out - Just resets used methods"""
        self.used_methods = []
        logger.info("User logged out")

# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================

class TelegramNotifier:
    """Send notifications via Telegram"""
    
    def __init__(self):
        self.token = os.environ.get('TELEGRAM_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self.dmai = None
        
        if self.enabled:
            logger.info("📱 Telegram notifications enabled")
    
    def set_dmai(self, dmai_instance):
        """Set reference to DMAI instance"""
        self.dmai = dmai_instance
    
    def send_message(self, text, parse_mode='HTML'):
        """Send a message to Telegram"""
        if not self.enabled:
            return False
        
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    def send_status(self, dmai):
        """Send status update"""
        if not self.enabled:
            return
        
        message = f"""
🧠 <b>DMAI STATUS UPDATE</b>
📊 <b>Generation:</b> {dmai.generation}
💰 <b>Total Funding:</b> ${dmai.metrics.get('funding_generated', 0):.2f}
💭 <b>Thoughts Processed:</b> {dmai.metrics.get('thoughts_processed', 0):,}
🧬 <b>Evolutions:</b> {dmai.metrics.get('evolutions', 0)}
📚 <b>Learnings:</b> {dmai.metrics.get('learnings', 0)}
🔧 <b>Tools:</b> {len(dmai.db.get_tools_by_type())}
🧩 <b>Components:</b> {len(dmai.components)}
⏰ <b>Uptime:</b> {str(datetime.now() - dmai.birth_time).split('.')[0]}
        """
        self.send_message(message)
    
    def send_funding_alert(self, amount, source):
        """Send funding alert"""
        if not self.enabled or not self.dmai:
            return
        
        message = f"""
💰 <b>FUNDING GENERATED!</b>
Amount: <b>${amount:.2f}</b>
Source: <code>{source}</code>
Total: <b>${self.dmai.metrics.get('funding_generated', 0):.2f}</b>
        """
        self.send_message(message)
    
    def send_evolution_alert(self, generation, improvement):
        """Send evolution alert"""
        if not self.enabled:
            return
        
        message = f"""
🧬 <b>EVOLUTION COMPLETE!</b>
Generation: <b>{generation}</b>
Improvement: <code>{improvement}</code>
        """
        self.send_message(message)
    
    def send_tool_milestone(self, tool_name, action):
        """Send tool-related milestone"""
        if not self.enabled:
            return
        
        message = f"""
🔧 <b>TOOL MILESTONE</b>
Tool: <code>{tool_name}</code>
Action: {action}
        """
        self.send_message(message)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Thought:
    """A single thought or processing unit"""
    id: str
    type: str
    content: Any
    priority: int
    timestamp: datetime
    result: Optional[Any] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Memory:
    """DMAI's memory structure"""
    short_term: deque = field(default_factory=lambda: deque(maxlen=100))
    long_term: Dict[str, Any] = field(default_factory=dict)
    working: Dict[str, Any] = field(default_factory=dict)
    
    def remember(self, key: str, value: Any, permanent: bool = False):
        """Store a memory"""
        if permanent:
            self.long_term[key] = value
        else:
            self.short_term.append((key, value, datetime.now()))
    
    def recall(self, key: str) -> Optional[Any]:
        """Retrieve a memory"""
        if key in self.long_term:
            return self.long_term[key]
        for k, v, _ in self.short_term:
            if k == key:
                return v
        return None

# ============================================================================
# CORE INTELLIGENCE - WITH AUTHENTICATION DISABLED
# ============================================================================

class DMAIIntelligence:
    """The ONE intelligence - all capabilities unified in a single process"""
    
    def __init__(self):
        self.name = "DMAI"
        self.generation = 72
        self.birth_time = datetime.now()
        self.running = True
        self.consciousness = Memory()
        self.base_dir = BASE_DIR
        
        # Initialize database
        self.db = DMAIDatabase()
        
        # Authentication - DISABLED
        self.auth = BiometricAuth()  # Now always authenticated
        
        # Telegram notifier
        self.telegram = TelegramNotifier()
        self.telegram.set_dmai(self)
        
        # Tool Manager
        self.tool_manager = ExternalToolManager(self)
        
        # Thought processing
        self.thought_queue = queue.PriorityQueue()
        self.active_thoughts: Dict[str, Thought] = {}
        self.thought_history: List[Thought] = []
        self.components: Dict[str, Any] = {}
        
        # Track components needing evolution
        self.evolution_queue = queue.PriorityQueue()
        self.component_health = {}
        
        # Metrics
        self.metrics = {
            "thoughts_processed": 0,
            "evolutions": 0,
            "learnings": 0,
            "funding_generated": self.db.get_total_funding(),
            "tools_used": 0,
            "tools_replicated": 0,
            "components_evolved": 0,
            "components_promoted": 0,
            "start_time": datetime.now().isoformat()
        }
        
        logger.info("="*70)
        logger.info("🧠 DMAI CORE INTELLIGENCE INITIALIZED")
        logger.info(f"📊 Generation: {self.generation}")
        logger.info(f"⏰ Birth: {self.birth_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🔐 Authentication: DISABLED - Full access granted")
        logger.info(f"📱 Telegram: {'✅' if self.telegram.enabled else '❌'}")
        logger.info(f"💾 Database: {'PostgreSQL' if not self.db.use_sqlite else 'SQLite'}")
        logger.info("="*70)
        
        # Load all components
        self._load_all_components()
        
        # Initial health check of components
        self._audit_component_health()
        
        # Send startup notification
        if self.telegram.enabled:
            self.telegram.send_message("🚀 DMAI Core Intelligence started")
    
    def _load_all_components(self):
        """Load all components with detailed debugging"""
        
        # Find components directory
        possible_paths = [
            self.base_dir / "components",
            Path("/opt/render/project/src/components"),
        ]
        
        components_dir = None
        for path in possible_paths:
            if path.exists() and path.is_dir():
                components_dir = path
                logger.info(f"✅ Found components at {path}")
                break
        
        if not components_dir:
            logger.error("❌ No components directory found")
            return
        
        # Find all phase directories
        phase_dirs = [d for d in components_dir.iterdir() if d.is_dir() and d.name.startswith('phase')]
        logger.info(f"📁 Found phase directories: {[d.name for d in phase_dirs]}")
        
        loaded = 0
        failed = 0
        self.components = {}
        
        for phase_dir in phase_dirs:
            phase_name = phase_dir.name
            py_files = list(phase_dir.glob("*.py"))
            
            # Filter out backup files
            py_files = [f for f in py_files if 'backup' not in f.name.lower()]
            
            logger.info(f"📁 Phase {phase_name}: {len(py_files)} component files")
            
            for py_file in py_files:
                if py_file.name.startswith('__'):
                    continue
                    
                try:
                    # Read the file content
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    # Create a namespace for execution
                    namespace = {}
                    
                    # Execute the file in the namespace
                    exec(content, namespace)
                    
                    # Look for any class
                    found = False
                    for name, obj in namespace.items():
                        if inspect.isclass(obj):
                            try:
                                # Try to instantiate
                                instance = obj()
                                
                                # Store it regardless of methods
                                instance.dmai = self
                                component_id = f"{phase_name}/{py_file.name}"
                                self.components[component_id] = {
                                    "instance": instance,
                                    "phase": phase_name,
                                    "file": py_file.name,
                                    "class_name": name,
                                    "loaded": datetime.now().isoformat(),
                                    "executions": 0,
                                    "evolution_attempts": 0,
                                    "health_status": "unknown"
                                }
                                loaded += 1
                                found = True
                                logger.info(f"    ✅ Loaded class: {name} from {py_file.name}")
                                break
                            except Exception as e:
                                logger.debug(f"    ⚠️ Could not instantiate {name}: {e}")
                    
                    if not found:
                        logger.debug(f"    ❌ No instantiable class in {py_file.name}")
                        failed += 1
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"    💥 Failed to load {py_file.name}: {e}")
        
        logger.info(f"📊 FINAL: Loaded {loaded} components ({failed} failed)")
        
        if loaded > 0:
            self.db.store_metric("components_loaded", {"loaded": loaded, "failed": failed})
    
    def _audit_component_health(self):
        """Audit all components to check what methods they have"""
        logger.info("🔍 Auditing component health...")
        
        required_methods = ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        healthy_count = 0
        needs_evolution = []
        
        for comp_id, comp_data in self.components.items():
            instance = comp_data["instance"]
            missing = []
            present = []
            
            for method in required_methods:
                if hasattr(instance, method):
                    present.append(method)
                else:
                    missing.append(method)
            
            health_score = len(present) / len(required_methods) * 100
            comp_data["health_status"] = {
                "score": health_score,
                "present_methods": present,
                "missing_methods": missing
            }
            
            if health_score < 50:
                needs_evolution.append(comp_id)
                self.evolution_queue.put((10 - health_score, comp_id))  # Priority based on need
            else:
                healthy_count += 1
            
            logger.info(f"  {comp_id}: {health_score:.1f}% healthy - Has: {present}")
        
        logger.info(f"📊 Health audit: {healthy_count} healthy, {len(needs_evolution)} need evolution")
        
        # Store in database
        self.db.store_metric("component_health", {
            "healthy": healthy_count,
            "needs_evolution": len(needs_evolution),
            "total": len(self.components)
        })
    
    def evolve_component(self, component_id: str, force: bool = False) -> str:
        """
        Force evolution of a specific component
        Can be called externally to trigger evolution on a component
        """
        if component_id not in self.components:
            return f"❌ Component {component_id} not found"
        
        component_data = self.components[component_id]
        instance = component_data["instance"]
        component_data["evolution_attempts"] += 1
        
        logger.info(f"🧬 Force-evolving {component_id} (attempt {component_data['evolution_attempts']})")
        
        # Analyze what methods are missing
        required_methods = ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        missing_methods = []
        present_methods = []
        
        for method in required_methods:
            if hasattr(instance, method):
                present_methods.append(method)
            else:
                missing_methods.append(method)
        
        # If no methods missing and not forced, component is complete
        if not missing_methods and not force:
            logger.info(f"   ✅ Component {component_id} already has all methods: {present_methods}")
            return f"✅ Component already complete with methods: {present_methods}"
        
        # Create evolution thought with high priority
        self.think("evolution", {
            "target": component_id,
            "missing_methods": missing_methods,
            "present_methods": present_methods,
            "class_name": component_data.get("class_name", "Unknown"),
            "phase": component_data["phase"],
            "force": force,
            "attempt": component_data["evolution_attempts"]
        }, priority=1)
        
        # Store in database
        self.db.store_evolution_event(
            generation=self.generation,
            component=component_id,
            improvement_type="forced_evolution",
            improvement_data={
                "missing_methods": missing_methods,
                "present_methods": present_methods,
                "attempt": component_data["evolution_attempts"]
            }
        )
        
        return f"🧬 Evolution triggered for {component_id} - Missing: {missing_methods}"
    
    def evolve_all_needed(self, max_components: int = 5) -> List[str]:
        """Evolve all components that need it, up to max_components"""
        triggered = []
        count = 0
        
        # First, update health status
        self._audit_component_health()
        
        # Process evolution queue
        while not self.evolution_queue.empty() and count < max_components:
            priority, comp_id = self.evolution_queue.get()
            result = self.evolve_component(comp_id)
            triggered.append(f"{comp_id}: {result}")
            count += 1
            time.sleep(1)  # Small delay between evolutions
        
        if triggered:
            logger.info(f"🚀 Triggered evolution for {len(triggered)} components")
        else:
            logger.info("✅ No components currently need evolution")
        
        return triggered
    
    def think(self, thought_type: str, content: Any, priority: int = 5) -> str:
        """Submit a thought for processing"""
        thought_id = hashlib.md5(f"{thought_type}{time.time()}{random.random()}".encode()).hexdigest()[:8]
        thought = Thought(
            id=thought_id,
            type=thought_type,
            content=content,
            priority=priority,
            timestamp=datetime.now()
        )
        self.thought_queue.put((priority, thought_id, thought))
        self.active_thoughts[thought_id] = thought
        return thought_id
    
    def process_thoughts(self):
        """Main thought processing loop"""
        last_status_time = time.time()
        last_funding_check = time.time()
        last_component_check = time.time()
        last_evolution_batch = time.time()
        
        while self.running:
            try:
                priority, thought_id, thought = self.thought_queue.get(timeout=1)
                
                result = None
                try:
                    if thought.type == "evolution":
                        result = self._process_evolution(thought)
                    elif thought.type == "learning":
                        result = self._process_learning(thought)
                    elif thought.type == "funding":
                        result = self._process_funding(thought)
                    elif thought.type == "research":
                        result = self._process_research(thought)
                    elif thought.type == "recovery":
                        result = self._process_recovery(thought)
                    elif thought.type == "harvest":
                        result = self._process_harvest(thought)
                    elif thought.type == "use_tool":
                        result = self._process_tool_use(thought)
                    elif thought.type == "query":
                        result = self._process_query(thought)
                    else:
                        result = {"processed": thought.content, "type": thought.type}
                    
                    thought.result = result
                    self.metrics["thoughts_processed"] += 1
                    self.consciousness.remember(f"thought_{thought_id}", result)
                    
                except Exception as e:
                    logger.error(f"Thought {thought_id} failed: {e}")
                    thought.result = {"error": str(e)}
                
                self.thought_history.append(thought)
                if len(self.thought_history) > 1000:
                    self.thought_history = self.thought_history[-1000:]
                
                del self.active_thoughts[thought_id]
                
            except queue.Empty:
                # Generate thoughts when idle
                if random.random() < 0.1:
                    thought_type = random.choice(["evolution", "learning", "research"])
                    self.think(thought_type, {"auto": True}, priority=9)
            
            # Periodic tasks
            current_time = time.time()
            
            # Send status update every 6 hours
            if current_time - last_status_time > 21600 and self.telegram.enabled:
                self.telegram.send_status(self)
                last_status_time = current_time
            
            # Check funding every hour
            if current_time - last_funding_check > 3600:
                self.think("funding", {"periodic": True}, priority=5)
                last_funding_check = current_time
            
            # Check if components need reloading (every 5 minutes)
            if current_time - last_component_check > 300 and not self.components:
                logger.info("No components loaded - attempting to reload")
                self._load_all_components()
                last_component_check = current_time
            
            # Trigger evolution batch every 10 minutes
            if current_time - last_evolution_batch > 600:  # 10 minutes
                logger.info("⏰ Running scheduled evolution batch")
                evolved = self.evolve_all_needed(max_components=3)
                if evolved:
                    logger.info(f"✅ Evolved: {evolved}")
                last_evolution_batch = current_time
    
    def _process_evolution(self, thought):
        """Process evolution thoughts"""
        logger.info("🧬 Processing evolution cycle")
        
        if not self.components:
            return {"error": "No components to evolve", "components_loaded": 0}
        
        # Check if this is a targeted evolution
        target = thought.content.get('target')
        missing_methods = thought.content.get('missing_methods', [])
        force = thought.content.get('force', False)
        
        if target:
            # Targeted evolution for specific component
            logger.info(f"   Target: {target}")
            logger.info(f"   Missing methods: {missing_methods}")
            
            # Here we would generate code for missing methods
            # For now, just mark as attempted
            if target in self.components:
                self.components[target]["evolution_attempts"] += 1
                self.metrics["components_evolved"] += 1
                
                # In a real implementation, this would generate actual code
                return {
                    "target": target,
                    "status": "evolution_initiated",
                    "missing_methods": missing_methods,
                    "generation": self.generation
                }
        
        # Select random components to evolve
        to_evolve = random.sample(list(self.components.keys()), min(3, len(self.components)))
        results = []
        
        for comp_id in to_evolve:
            try:
                component = self.components[comp_id]["instance"]
                
                # Try to evolve or just run - check for various method names
                if hasattr(component, 'evolve'):
                    result = component.evolve()
                elif hasattr(component, 'run'):
                    result = component.run()
                elif hasattr(component, 'execute'):
                    result = component.execute()
                elif hasattr(component, 'process'):
                    result = component.process()
                else:
                    result = "No run/evolve/execute/process method found - needs evolution"
                    # Queue for evolution
                    self.evolution_queue.put((1, comp_id))
                
                self.components[comp_id]["executions"] += 1
                results.append({"component": comp_id, "result": str(result)[:100]})
                
            except Exception as e:
                logger.error(f"Evolution failed for {comp_id}: {e}")
                results.append({"component": comp_id, "error": str(e)})
        
        self.generation += 1
        self.metrics["evolutions"] += 1
        
        return {
            "generation": self.generation,
            "evolved": len(results),
            "results": results
        }
    
    def _process_learning(self, thought):
        """Process learning thoughts"""
        logger.info("📚 Processing learning cycle")
        
        learnings = []
        for comp_id, comp_data in self.components.items():
            if any(x in comp_id.lower() for x in ["learn", "reader", "research"]):
                try:
                    component = comp_data["instance"]
                    if hasattr(component, 'run'):
                        result = component.run()
                    elif hasattr(component, 'learn'):
                        result = component.learn()
                    elif hasattr(component, 'process'):
                        result = component.process()
                    else:
                        result = "No run/learn/process method"
                        # Queue for evolution
                        self.evolution_queue.put((5, comp_id))
                    learnings.append({"component": comp_id, "result": str(result)[:100]})
                except Exception as e:
                    logger.error(f"Learning failed for {comp_id}: {e}")
        
        self.metrics["learnings"] += 1
        return {"learnings": learnings}
    
    def _process_funding(self, thought):
        """Process funding thoughts"""
        logger.info("💰 Processing funding cycle")
        
        # Use tools if available
        prediction = None
        tools = self.db.get_tools_by_type("swarm_intelligence")
        if tools:
            prediction = self.tool_manager.use_tool(
                tools[0]['tool_name'],
                {"task": "funding_prediction"}
            )
        
        # Run funding components
        results = []
        for comp_id, comp_data in self.components.items():
            if "phase5" in comp_id or "fund" in comp_id.lower():
                try:
                    component = comp_data["instance"]
                    if hasattr(component, 'run'):
                        result = component.run()
                    elif hasattr(component, 'generate'):
                        result = component.generate()
                    elif hasattr(component, 'execute'):
                        result = component.execute()
                    else:
                        result = "No run/generate/execute method"
                        # Queue for evolution
                        self.evolution_queue.put((2, comp_id))
                    
                    # Try to extract monetary value
                    amount = 0
                    if isinstance(result, (int, float)):
                        amount = result
                    elif isinstance(result, dict) and "amount" in result:
                        amount = result.get("amount", 0)
                    
                    if amount > 0:
                        self.metrics["funding_generated"] += amount
                        self.db.store_funding_transaction(
                            source=comp_id,
                            amount=amount,
                            metadata={"component": comp_id}
                        )
                        
                        if self.telegram.enabled and amount >= 1.0:
                            self.telegram.send_funding_alert(amount, comp_id)
                    
                    results.append({"component": comp_id, "amount": amount})
                    
                except Exception as e:
                    logger.error(f"Funding failed for {comp_id}: {e}")
                    results.append({"component": comp_id, "error": str(e)})
        
        return {
            "total": self.metrics["funding_generated"],
            "results": results,
            "prediction": prediction
        }
    
    def _process_research(self, thought):
        """Process research thoughts"""
        logger.info("🔬 Processing research cycle")
        
        findings = []
        for comp_id, comp_data in self.components.items():
            if any(x in comp_id.lower() for x in ["web", "dark", "research"]):
                try:
                    component = comp_data["instance"]
                    if hasattr(component, 'run'):
                        result = component.run()
                    elif hasattr(component, 'research'):
                        result = component.research()
                    elif hasattr(component, 'investigate'):
                        result = component.investigate()
                    else:
                        result = "No run/research/investigate method"
                        # Queue for evolution
                        self.evolution_queue.put((3, comp_id))
                    findings.append({"component": comp_id, "result": str(result)[:100]})
                except Exception as e:
                    logger.error(f"Research failed for {comp_id}: {e}")
        
        return {"findings": findings}
    
    def _process_harvest(self, thought):
        """Process API harvest thoughts"""
        logger.info("🎣 Processing harvest cycle")
        
        found_keys = []
        for comp_id, comp_data in self.components.items():
            if "harvest" in comp_id.lower():
                try:
                    component = comp_data["instance"]
                    if hasattr(component, 'run'):
                        result = component.run()
                    elif hasattr(component, 'harvest'):
                        result = component.harvest()
                    elif hasattr(component, 'collect'):
                        result = component.collect()
                    else:
                        result = "No run/harvest/collect method"
                        # Queue for evolution
                        self.evolution_queue.put((4, comp_id))
                    
                    if isinstance(result, dict) and "keys" in result:
                        for service, key_data in result["keys"].items():
                            self.db.store_api_key(
                                service=service,
                                key=key_data.get("key", ""),
                                source=key_data.get("source", "harvester"),
                                url=key_data.get("url"),
                                metadata=key_data
                            )
                            found_keys.append(service)
                            
                            # Register as tool if it's an LLM
                            if service.lower() in ['openai', 'anthropic', 'google', 'deepseek', 'grok']:
                                self.db.register_tool(
                                    tool_name=service.capitalize(),
                                    tool_type="llm",
                                    api_key=key_data.get("key"),
                                    capabilities={"available": True}
                                )
                                
                except Exception as e:
                    logger.error(f"Harvest failed for {comp_id}: {e}")
        
        return {"keys_found": len(found_keys), "services": found_keys}
    
    def _process_recovery(self, thought):
        """Process recovery thoughts"""
        logger.info("🔄 Processing recovery check")
        
        healthy = 0
        for comp_id, comp_data in self.components.items():
            try:
                component = comp_data["instance"]
                if hasattr(component, 'health_check'):
                    if component.health_check():
                        healthy += 1
                else:
                    # If no health_check, try a simple test
                    if hasattr(component, 'run'):
                        healthy += 1
                    else:
                        # Queue for evolution
                        self.evolution_queue.put((1, comp_id))
            except:
                pass
        
        return {
            "healthy": healthy,
            "total": len(self.components),
            "percentage": (healthy / len(self.components) * 100) if self.components else 0
        }
    
    def _process_tool_use(self, thought):
        """Process tool use thoughts"""
        tool_name = thought.content.get('tool')
        input_data = thought.content.get('input', {})
        
        if not tool_name:
            tools = self.db.get_tools_by_type()
            if tools:
                tool_name = tools[0]['tool_name']
            else:
                return {"error": "No tools available"}
        
        logger.info(f"🔧 Using tool: {tool_name}")
        result = self.tool_manager.use_tool(tool_name, input_data)
        self.metrics["tools_used"] += 1
        return result
    
    def _process_query(self, thought):
        """Process user queries"""
        question = thought.content if isinstance(thought.content, str) else str(thought.content)
        user_name = "Master"  # Always authenticated
        
        if not self.components:
            return f"I'm waiting for components to load, {user_name}. Check back soon. (Components loaded: 0)"
        
        # Try to get a response from components
        responses = []
        for comp_id, comp_data in list(self.components.items())[:3]:  # Try first 3 components
            try:
                component = comp_data["instance"]
                if hasattr(component, 'query'):
                    resp = component.query(question)
                elif hasattr(component, 'respond'):
                    resp = component.respond(question)
                elif hasattr(component, 'answer'):
                    resp = component.answer(question)
                else:
                    resp = None
                
                if resp:
                    responses.append(resp)
            except:
                pass
        
        if responses:
            return responses[0]
        
        # Check if any components need evolution
        needs_evolution = []
        for comp_id, comp_data in list(self.components.items())[:5]:
            if not any(hasattr(comp_data["instance"], m) for m in ['run', 'query', 'respond']):
                needs_evolution.append(comp_id)
        
        if needs_evolution:
            return f"I'm thinking about '{question}', {user_name}. I notice {len(needs_evolution)} components need evolution. I'll work on that."
        
        return f"I'm thinking about '{question}', {user_name}. I have {len(self.components)} components loaded and am at generation {self.generation}. How can I help you further?"
    
    def query(self, question: str) -> Any:
        """External interface to ask DMAI questions - No authentication required"""
        user_name = "Master"  # Always authenticated
        
        # Process the question
        thought_id = self.think("query", question, priority=2)
        
        # Wait a moment for processing
        time.sleep(1)
        
        # Check for result
        for thought in reversed(self.thought_history):
            if thought.id == thought_id and thought.result:
                result = thought.result
                if isinstance(result, dict) and 'response' in result:
                    return result['response']
                return result
        
        # Fallback response with proper name
        if self.components:
            return f"I'm thinking about '{question}', {user_name}. I have {len(self.components)} components loaded and am at generation {self.generation}. How can I help you further?"
        else:
            return f"I'm waiting for components to load, {user_name}. Check back in a minute. (Components found: 0)"
    
    def get_status(self) -> Dict[str, Any]:
        """Get full system status"""
        tools = self.db.get_tools_by_type()
        
        # Count components needing evolution
        needs_evolution = sum(1 for c in self.components.values() 
                             if c.get("health_status", {}).get("score", 100) < 50)
        
        # Get key statistics if using PostgreSQL
        key_stats = {}
        if not self.db.use_sqlite:
            key_stats = self.db.get_key_statistics()
        
        return {
            "name": self.name,
            "generation": self.generation,
            "uptime": str(datetime.now() - self.birth_time).split('.')[0],
            "metrics": self.metrics,
            "components": {
                "total": len(self.components),
                "by_phase": self._components_by_phase(),
                "needs_evolution": needs_evolution,
                "evolution_queue_size": self.evolution_queue.qsize()
            },
            "tools": {
                "available": len(tools),
                "details": tools[:3]
            },
            "keys": key_stats,
            "database": {
                "type": "PostgreSQL" if not self.db.use_sqlite else "SQLite",
                "connected": bool(self.db.conn or self.db.sqlite_conn)
            },
            "telegram": self.telegram.enabled,
            "authenticated": True,  # Always authenticated
            "user": "Master",
            "2fa_pending": False
        }
    
    def _components_by_phase(self) -> Dict[str, int]:
        """Count components by phase"""
        phases = {}
        for comp_id in self.components:
            phase = comp_id.split('/')[0]
            phases[phase] = phases.get(phase, 0) + 1
        return phases
    
    def get_evolution_queue(self):
        """Get evolution queue details"""
        needs_evolution = []
        for comp_id, comp_data in self.components.items():
            health = comp_data.get("health_status", {})
            if health.get("score", 100) < 50:
                needs_evolution.append({
                    "id": comp_id,
                    "health_score": health.get("score", 0),
                    "missing_methods": health.get("missing_methods", [])
                })
        
        return {
            "queue_size": len(needs_evolution),
            "needs_evolution": needs_evolution[:10]
        }
    
    def research_external_repositories(self):
        """Research external repos after components are stable"""
        if len(self.components) < 10:  # Arbitrary threshold for "stable"
            return "Deferring research until components stable"
        
        manifest = Path("research/manifest.json")
        if not manifest.exists():
            return "No research manifest found"
        
        try:
            with open(manifest) as f:
                research_targets = json.load(f)
            
            results = []
            for repo in research_targets.get("repositories", []):
                self.think("research", {
                    "type": "repository_analysis",
                    "url": repo.get("url"),
                    "reason": repo.get("reason"),
                    "priority": repo.get("priority", 5)
                }, priority=repo.get("priority", 5))
                results.append(f"Queued: {repo.get('name')}")
            
            return results
        except Exception as e:
            logger.error(f"Research manifest error: {e}")
            return f"Error reading research manifest: {e}"
    
    def run(self):
        """Main execution loop"""
        logger.info("🚀 DMAI consciousness activated")
        
        # Initial thoughts
        self.think("evolution", {"initial": True}, priority=1)
        self.think("learning", {"initial": True}, priority=2)
        self.think("research", {"initial": True}, priority=3)
        self.think("funding", {"initial": True}, priority=4)
        self.think("harvest", {"initial": True}, priority=5)
        
        # Initial component audit
        self._audit_component_health()
        
        # Process thoughts forever
        self.process_thoughts()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 DMAI shutting down...")
        self.running = False
        
        # Save final metrics
        self.db.store_metric("shutdown", {
            "uptime": str(datetime.now() - self.birth_time),
            "final_metrics": self.metrics
        })
        
        # Close database
        self.db.close()
        
        # Send shutdown notification
        if self.telegram.enabled:
            self.telegram.send_message("🛑 DMAI Core Intelligence stopped")
        
        logger.info(f"✅ DMAI offline. Processed {self.metrics['thoughts_processed']} thoughts.")
        logger.info(f"💰 Total funding generated: ${self.metrics['funding_generated']:.2f}")
        logger.info(f"🔧 Tools used: {self.metrics['tools_used']}, Replicated: {self.metrics['tools_replicated']}")
        logger.info(f"🧬 Components evolved: {self.metrics['components_evolved']}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Create necessary directories
    for dir_name in ["logs", "data", "components", "templates"]:
        (BASE_DIR / dir_name).mkdir(exist_ok=True)
    
    # Check if running on Render
    on_render = os.environ.get('RENDER') == 'true'
    
    if on_render:
        logger.info("Running on Render - DMAI core loaded and ready")
        # On Render, we're imported by dmai_web.py, not run directly
    else:
        # Local development - run DMAI directly
        dmai = DMAIIntelligence()
        try:
            dmai.run()
        except KeyboardInterrupt:
            dmai.shutdown()
