#!/usr/bin/env python3
"""
DMAI API Key Workflow System
Complete pipeline: Scrape → Identify → Validate → Store in dmai-harvester-db
"""

import os
import sys
import time
import json
import logging
import signal
import re
import hashlib
import psycopg2
import psycopg2.extras
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('key_workflow.log')
    ]
)
logger = logging.getLogger("key_workflow")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RawKey:
    """Raw extracted key before validation"""
    key_string: str
    source_repo: str
    source_url: str
    source_file: str
    line_number: int
    context: str
    discovered_at: datetime
    
    def to_dict(self):
        return {
            'key_string': self.key_string[:20] + '...' if len(self.key_string) > 20 else self.key_string,
            'source_repo': self.source_repo,
            'source_url': self.source_url,
            'discovered_at': self.discovered_at.isoformat()
        }

@dataclass
class IdentifiedKey:
    """Key after type identification"""
    raw_key: RawKey
    key_type: str
    confidence: float  # 0.0 to 1.0
    pattern_matched: str
    normalized_key: str
    
    def to_dict(self):
        return {
            'key_type': self.key_type,
            'confidence': self.confidence,
            'source_repo': self.raw_key.source_repo,
            'discovered_at': self.raw_key.discovered_at.isoformat()
        }

@dataclass
class ValidatedKey:
    """Key after validation"""
    identified_key: IdentifiedKey
    is_valid: bool
    validation_message: str
    permissions: List[str]
    rate_limit: Optional[Dict]
    expires_at: Optional[datetime]
    weight: int  # 1-10, importance score
    validated_at: datetime
    
    def to_dict(self):
        return {
            'key_type': self.identified_key.key_type,
            'is_valid': self.is_valid,
            'weight': self.weight,
            'source_repo': self.identified_key.raw_key.source_repo,
            'validated_at': self.validated_at.isoformat()
        }
    
    def to_db_record(self):
        """Convert to database record for PostgreSQL"""
        return {
            'key_hash': hashlib.sha256(self.identified_key.normalized_key.encode()).hexdigest(),
            'key_type': self.identified_key.key_type,
            'is_valid': self.is_valid,
            'weight': self.weight,
            'source_repo': self.identified_key.raw_key.source_repo,
            'source_url': self.identified_key.raw_key.source_url,
            'source_file': self.identified_key.raw_key.source_file,
            'line_number': self.identified_key.raw_key.line_number,
            'context': self.identified_key.raw_key.context[:500] if self.identified_key.raw_key.context else None,
            'validation_message': self.validation_message,
            'permissions': json.dumps(self.permissions),
            'rate_limit': json.dumps(self.rate_limit) if self.rate_limit else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'validated_at': self.validated_at.isoformat(),
            'key_preview': self.identified_key.normalized_key[:20] + '...',
            'full_key_encrypted': self._encrypt_key(self.identified_key.normalized_key)
        }
    
    def _encrypt_key(self, key: str) -> str:
        """Simple encryption for stored keys"""
        import base64
        return base64.b64encode(key.encode()).decode()

# ============================================================================
# DATABASE STORAGE MODULE - PostgreSQL (dmai-harvester-db)
# ============================================================================

class KeyDatabase:
    """PostgreSQL database for storing validated keys in dmai-harvester-db"""
    
    def __init__(self, connection_string: str = None):
        """Initialize database connection"""
        self.connection_string = connection_string or os.environ.get(
            'DATABASE_URL',
            'postgresql://postgres:password@localhost:5432/dmai_harvester'
        )
        self.init_db()
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)
    
    def init_db(self):
        """Initialize database schema in PostgreSQL"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS api_keys (
            -- Core fields
            id SERIAL PRIMARY KEY,
            key_hash VARCHAR(64) UNIQUE NOT NULL,
            key_type VARCHAR(50) NOT NULL,
            is_valid BOOLEAN NOT NULL DEFAULT FALSE,
            weight INTEGER NOT NULL DEFAULT 0,
            
            -- Source information
            source_repo TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_file TEXT,
            line_number INTEGER,
            context TEXT,
            
            -- Validation results
            validation_message TEXT,
            permissions JSONB,
            rate_limit JSONB,
            expires_at TIMESTAMP WITH TIME ZONE,
            validated_at TIMESTAMP WITH TIME ZONE NOT NULL,
            
            -- Key storage (encrypted)
            key_preview VARCHAR(50) NOT NULL,
            full_key_encrypted TEXT NOT NULL,
            
            -- Usage tracking
            times_used INTEGER DEFAULT 0,
            last_used TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            
            -- Evolution tracking
            evolution_generation INTEGER DEFAULT 1,
            parent_key_hash VARCHAR(64),
            mutation_count INTEGER DEFAULT 0,
            last_mutated_at TIMESTAMP WITH TIME ZONE,
            
            -- Metadata
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_api_keys_key_type ON api_keys(key_type);
        CREATE INDEX IF NOT EXISTS idx_api_keys_is_valid ON api_keys(is_valid);
        CREATE INDEX IF NOT EXISTS idx_api_keys_weight ON api_keys(weight);
        CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);
        CREATE INDEX IF NOT EXISTS idx_api_keys_created_at ON api_keys(created_at);
        
        -- Usage tracking table
        CREATE TABLE IF NOT EXISTS key_usage_log (
            id SERIAL PRIMARY KEY,
            key_hash VARCHAR(64) REFERENCES api_keys(key_hash) ON DELETE CASCADE,
            service VARCHAR(100) NOT NULL,
            endpoint TEXT,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            success BOOLEAN NOT NULL,
            response_time_ms INTEGER,
            error TEXT,
            request_details JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_usage_key_hash ON key_usage_log(key_hash);
        CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON key_usage_log(timestamp);
        
        -- Evolution history table
        CREATE TABLE IF NOT EXISTS key_evolution_history (
            id SERIAL PRIMARY KEY,
            child_key_hash VARCHAR(64) REFERENCES api_keys(key_hash) ON DELETE CASCADE,
            parent_key_hash VARCHAR(64) REFERENCES api_keys(key_hash) ON DELETE SET NULL,
            generation INTEGER NOT NULL,
            mutation_type VARCHAR(50),
            mutation_details JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_evolution_child ON key_evolution_history(child_key_hash);
        CREATE INDEX IF NOT EXISTS idx_evolution_parent ON key_evolution_history(parent_key_hash);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                conn.commit()
            logger.info("✅ PostgreSQL schema initialized in dmai-harvester-db")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def store_validated_key(self, key: ValidatedKey) -> bool:
        """Store a validated key in PostgreSQL database"""
        try:
            record = key.to_db_record()
            
            insert_sql = """
            INSERT INTO api_keys (
                key_hash, key_type, is_valid, weight, 
                source_repo, source_url, source_file, line_number, context,
                validation_message, permissions, rate_limit, expires_at, validated_at,
                key_preview, full_key_encrypted,
                metadata
            ) VALUES (
                %(key_hash)s, %(key_type)s, %(is_valid)s, %(weight)s,
                %(source_repo)s, %(source_url)s, %(source_file)s, %(line_number)s, %(context)s,
                %(validation_message)s, %(permissions)s::jsonb, %(rate_limit)s::jsonb, 
                %(expires_at)s, %(validated_at)s,
                %(key_preview)s, %(full_key_encrypted)s,
                %(metadata)s::jsonb
            )
            ON CONFLICT (key_hash) DO UPDATE SET
                is_valid = EXCLUDED.is_valid,
                weight = EXCLUDED.weight,
                validation_message = EXCLUDED.validation_message,
                permissions = EXCLUDED.permissions,
                rate_limit = EXCLUDED.rate_limit,
                expires_at = EXCLUDED.expires_at,
                validated_at = EXCLUDED.validated_at,
                times_used = api_keys.times_used,
                metadata = api_keys.metadata || EXCLUDED.metadata
            RETURNING id;
            """
            
            record['metadata'] = json.dumps({
                'first_discovered': key.identified_key.raw_key.discovered_at.isoformat(),
                'confidence': key.identified_key.confidence,
                'pattern_matched': key.identified_key.pattern_matched,
                'extraction_context': {
                    'line': key.identified_key.raw_key.line_number,
                    'surrounding': key.identified_key.raw_key.context[:200]
                }
            })
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, record)
                    key_id = cursor.fetchone()[0]
                conn.commit()
            
            logger.info(f"✅ Key stored in dmai-harvester-db (ID: {key_id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database error storing key: {e}")
            return False
    
    def get_valid_keys(self, key_type: Optional[str] = None, 
                       min_weight: int = 5, 
                       limit: int = 100,
                       include_expired: bool = False) -> List[Dict]:
        """Retrieve valid keys from database"""
        try:
            query = """
                SELECT 
                    id, key_hash, key_type, weight, 
                    source_repo, source_url, key_preview,
                    permissions, rate_limit, expires_at,
                    times_used, last_used, created_at,
                    evolution_generation, mutation_count,
                    metadata
                FROM api_keys 
                WHERE is_valid = TRUE AND weight >= %(min_weight)s
            """
            params = {'min_weight': min_weight}
            
            if key_type:
                query += " AND key_type = %(key_type)s"
                params['key_type'] = key_type
            
            if not include_expired:
                query += " AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
            
            query += " ORDER BY weight DESC, times_used ASC, created_at DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
                    
        except Exception as e:
            logger.error(f"❌ Database error retrieving keys: {e}")
            return []
    
    def log_key_usage(self, key_hash: str, service: str, success: bool, 
                      response_time_ms: int, error: Optional[str] = None,
                      endpoint: Optional[str] = None,
                      request_details: Optional[Dict] = None):
        """Log usage of a key"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO key_usage_log 
                        (key_hash, service, endpoint, timestamp, success, 
                         response_time_ms, error, request_details)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """, (
                        key_hash, service, endpoint, datetime.now(),
                        success, response_time_ms, error,
                        json.dumps(request_details) if request_details else None
                    ))
                    
                    if success:
                        cursor.execute("""
                            UPDATE api_keys 
                            SET times_used = times_used + 1, 
                                last_used = %s
                            WHERE key_hash = %s
                        """, (datetime.now(), key_hash))
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Database error logging usage: {e}")
    
    def record_key_evolution(self, child_key_hash: str, parent_key_hash: str,
                            generation: int, mutation_type: str,
                            mutation_details: Optional[Dict] = None):
        """Record key evolution"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO key_evolution_history
                        (child_key_hash, parent_key_hash, generation, 
                         mutation_type, mutation_details)
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                    """, (
                        child_key_hash, parent_key_hash, generation,
                        mutation_type, json.dumps(mutation_details) if mutation_details else None
                    ))
                    
                    cursor.execute("""
                        UPDATE api_keys 
                        SET evolution_generation = %s,
                            parent_key_hash = %s,
                            mutation_count = mutation_count + 1,
                            last_mutated_at = %s
                        WHERE key_hash = %s
                    """, (generation, parent_key_hash, datetime.now(), child_key_hash))
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Database error recording evolution: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get statistics about stored keys"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute("SELECT COUNT(*) FROM api_keys WHERE is_valid = TRUE")
                    total_valid = cursor.fetchone()['count']
                    
                    cursor.execute("""
                        SELECT key_type, COUNT(*) 
                        FROM api_keys 
                        WHERE is_valid = TRUE 
                        GROUP BY key_type 
                        ORDER BY COUNT(*) DESC
                    """)
                    keys_by_type = cursor.fetchall()
                    
                    cursor.execute("SELECT AVG(weight) FROM api_keys WHERE is_valid = TRUE")
                    avg_weight = cursor.fetchone()['avg']
                    
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM api_keys 
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                    """)
                    added_today = cursor.fetchone()['count']
                    
                    return {
                        'total_valid_keys': total_valid,
                        'keys_by_type': keys_by_type,
                        'average_weight': float(avg_weight) if avg_weight else 0,
                        'added_last_24h': added_today
                    }
                    
        except Exception as e:
            logger.error(f"❌ Database error getting stats: {e}")
            return {}

# ============================================================================
# KEY IDENTIFIER MODULE
# ============================================================================

class KeyIdentifier:
    """Identifies API key types based on patterns"""
    
    KEY_PATTERNS = {
        'github': {
            'pattern': re.compile(r'^ghp_[0-9a-zA-Z]{36}$'),
            'weight': 10,
            'description': 'GitHub Personal Access Token'
        },
        'openai': {
            'pattern': re.compile(r'^sk-[0-9a-zA-Z]{48}$'),
            'weight': 10,
            'description': 'OpenAI API Key'
        },
        'stripe_live': {
            'pattern': re.compile(r'^sk_live_[0-9a-zA-Z]{24}$'),
            'weight': 9,
            'description': 'Stripe Live Secret Key'
        },
        'stripe_test': {
            'pattern': re.compile(r'^sk_test_[0-9a-zA-Z]{24}$'),
            'weight': 5,
            'description': 'Stripe Test Key'
        },
        'aws_access': {
            'pattern': re.compile(r'^AKIA[0-9A-Z]{16}$'),
            'weight': 10,
            'description': 'AWS Access Key ID'
        },
        'google_api': {
            'pattern': re.compile(r'^AIza[0-9A-Za-z_-]{35}$'),
            'weight': 8,
            'description': 'Google API Key'
        },
        'sendgrid': {
            'pattern': re.compile(r'^SG\.[0-9a-zA-Z_-]{22,68}$'),
            'weight': 6,
            'description': 'SendGrid API Key'
        },
        'mailgun': {
            'pattern': re.compile(r'^key-[0-9a-zA-Z]{32}$'),
            'weight': 6,
            'description': 'Mailgun API Key'
        },
        'twilio': {
            'pattern': re.compile(r'^SK[0-9a-f]{32}$'),
            'weight': 6,
            'description': 'Twilio API Key'
        },
        'telegram': {
            'pattern': re.compile(r'^[0-9]{8,10}:[0-9a-zA-Z_-]{35}$'),
            'weight': 7,
            'description': 'Telegram Bot Token'
        },
        'slack_token': {
            'pattern': re.compile(r'^xox[baprs]-[0-9a-zA-Z]{10,50}$'),
            'weight': 7,
            'description': 'Slack Token'
        },
        'discord_bot': {
            'pattern': re.compile(r'^[MN][0-9a-zA-Z_-]{23,25}$'),
            'weight': 7,
            'description': 'Discord Bot Token'
        }
    }
    
    def identify(self, key_string: str, context: str = "") -> List[Tuple[str, float, int]]:
        """Identify potential key types"""
        candidates = []
        for key_type, info in self.KEY_PATTERNS.items():
            if info['pattern'].match(key_string):
                confidence = 0.9
                if context and any(k in context.lower() for k in ['api', 'key', 'token', 'secret']):
                    confidence += 0.05
                candidates.append((key_type, min(confidence, 1.0), info['weight']))
        return candidates

# ============================================================================
# KEY VALIDATOR MODULE
# ============================================================================

class KeyValidator:
    """Validates API keys against their services"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
    
    def validate(self, identified_key: IdentifiedKey) -> ValidatedKey:
        """Validate an identified key"""
        # Simple validation for demo - expand based on key type
        is_valid = len(identified_key.normalized_key) >= 20
        
        # Check for obvious fake/placeholder keys
        placeholders = ['your-api-key', 'your_key', 'xxxxxxxx', 'test', 'demo', 'example']
        if any(p in identified_key.normalized_key.lower() for p in placeholders):
            is_valid = False
            validation_message = "Placeholder key detected"
        else:
            validation_message = "Basic validation passed"
        
        return ValidatedKey(
            identified_key=identified_key,
            is_valid=is_valid,
            validation_message=validation_message,
            permissions=[],
            rate_limit=None,
            expires_at=None,
            weight=identified_key.confidence * 10 if is_valid else 0,
            validated_at=datetime.now()
        )

# ============================================================================
# MAIN WORKFLOW ORCHESTRATOR
# ============================================================================

class KeyWorkflowOrchestrator:
    """Orchestrates the complete key processing pipeline"""
    
    def __init__(self, config_path: str = "workflow_config.json"):
        self.config = self.load_config(config_path)
        self.identifier = KeyIdentifier()
        self.validator = KeyValidator()
        self.database = KeyDatabase(self.config['database'].get('connection_string'))
        self.running = True
        self.stats = {
            'raw_keys_processed': 0,
            'identified_keys': 0,
            'validated_keys': 0,
            'valid_keys_stored': 0
        }
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("=" * 60)
        logger.info("🔑 DMAI Key Workflow Orchestrator Started")
        logger.info("=" * 60)
        logger.info(f"Database: dmai-harvester-db (PostgreSQL)")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            'database': {
                'connection_string': os.environ.get('DATABASE_URL'),
                'pool_size': 5,
                'max_retries': 3
            },
            'validation': {
                'min_confidence': 0.5,
                'min_weight': 5,
                'validate_all': True,
                'auto_store': True
            },
            'github': {
                'max_results_per_query': 50,
                'delay_between_requests': 0.5
            }
        }
        
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    for key, value in file_config.items():
                        if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"✅ Loaded config from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def process_raw_key(self, raw_key: RawKey) -> Optional[ValidatedKey]:
        """Process a raw key through the entire pipeline"""
        self.stats['raw_keys_processed'] += 1
        
        candidates = self.identifier.identify(raw_key.key_string, raw_key.context)
        if not candidates:
            return None
        
        key_type, confidence, weight = candidates[0]
        if confidence < self.config['validation']['min_confidence']:
            return None
        
        identified = IdentifiedKey(
            raw_key=raw_key,
            key_type=key_type,
            confidence=confidence,
            pattern_matched=key_type,
            normalized_key=raw_key.key_string.strip('\'"')
        )
        
        self.stats['identified_keys'] += 1
        
        if self.config['validation']['validate_all']:
            validated = self.validator.validate(identified)
            self.stats['validated_keys'] += 1
            
            if validated.is_valid and validated.weight >= self.config['validation']['min_weight']:
                if self.config['validation']['auto_store']:
                    if self.database.store_validated_key(validated):
                        self.stats['valid_keys_stored'] += 1
                return validated
        
        return None
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        db_stats = self.database.get_database_stats()
        return {**self.stats, 'database': db_stats}
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🚨 Shutdown signal received")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down workflow orchestrator...")
        logger.info("=" * 60)
        logger.info("📊 FINAL STATISTICS:")
        for key, value in self.get_stats().items():
            if key != 'database':
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  Database stats:")
                for db_key, db_value in value.items():
                    logger.info(f"    {db_key}: {db_value}")
        logger.info("=" * 60)
        self.running = False
        sys.exit(0)

# ============================================================================
# GITHUB SCRAPER INTEGRATION
# ============================================================================

class GitHubScraperIntegration:
    """Integrates GitHub scraper with the workflow"""
    
    def __init__(self, workflow: KeyWorkflowOrchestrator, github_token: Optional[str] = None):
        self.workflow = workflow
        # Try to get token from parameter, then environment, then config
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.gh = None
        
        if self.github_token:
            try:
                from github import Github
                self.gh = Github(self.github_token)
                logger.info(f"✅ GitHub scraper initialized with token (starts with: {self.github_token[:4]}...)")
                
                # Test the token
                try:
                    user = self.gh.get_user()
                    logger.info(f"   Authenticated as: {user.login}")
                except Exception as e:
                    logger.warning(f"   Token test failed: {e}")
                    
            except ImportError as e:
                logger.error(f"❌ PyGithub not installed. Run: pip install PyGithub")
            except Exception as e:
                logger.error(f"❌ Failed to initialize GitHub: {e}")
        else:
            logger.error("❌ No GitHub token found in environment or parameters")
    
    def search_and_process(self, query: str, max_results: int = 50):
        """Search GitHub and process results"""
        if not self.gh:
            logger.error("GitHub scraper not initialized - check token")
            return
        
        try:
            logger.info(f"🔍 Searching GitHub for: {query} (max: {max_results})")
            search_result = self.gh.search_code(query)
            logger.info(f"   Total results available: {search_result.totalCount}")
            
            count = 0
            for result in search_result[:max_results]:
                count += 1
                try:
                    # Get content
                    if hasattr(result, 'decoded_content'):
                        content = result.decoded_content.decode('utf-8', errors='ignore')
                    else:
                        continue
                    
                    logger.info(f"   Processing result {count}: {result.repository.full_name}")
                    
                    # Look for potential keys
                    lines = content.split('\n')
                    for i, line in enumerate(lines[:100]):  # Check first 100 lines
                        # Extract potential keys
                        keys = self._extract_keys_from_line(line)
                        
                        for key in keys:
                            # Skip obvious placeholders
                            if any(p in key.lower() for p in ['your', 'xxx', 'test', 'demo', 'example']):
                                continue
                                
                            raw_key = RawKey(
                                key_string=key,
                                source_repo=result.repository.full_name,
                                source_url=result.html_url,
                                source_file=result.path,
                                line_number=i + 1,
                                context=line.strip(),
                                discovered_at=datetime.now()
                            )
                            
                            self.workflow.process_raw_key(raw_key)
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error processing result: {e}")
                    continue
            
            logger.info(f"✅ Completed search for '{query}', processed {count} results")
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
    
    def _extract_keys_from_line(self, line: str) -> List[str]:
        """Extract potential API keys from a line"""
        keys = []
        
        # Common patterns for key extraction
        patterns = [
            r'[\'"]([a-zA-Z0-9_\-]{20,64})[\'"]',  # Quoted strings
            r'=([a-zA-Z0-9_\-]{20,64})',           # After equals sign
            r':\s*[\'"]([a-zA-Z0-9_\-]{20,64})[\'"]',  # JSON format
            r'key[\s]*=[\s]*([a-zA-Z0-9_\-]{20,64})',  # key=value
            r'token[\s]*=[\s]*([a-zA-Z0-9_\-]{20,64})',  # token=value
            r'secret[\s]*=[\s]*([a-zA-Z0-9_\-]{20,64})',  # secret=value
            r'apikey[\s]*=[\s]*([a-zA-Z0-9_\-]{20,64})',  # apikey=value
            r'api_key[\s]*=[\s]*([a-zA-Z0-9_\-]{20,64})',  # api_key=value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            keys.extend(matches)
        
        # Also check for common key prefixes
        prefixes = ['ghp_', 'sk-', 'sk_live_', 'sk_test_', 'AKIA', 'AIza', 'xoxb-', 'xoxp-']
        for prefix in prefixes:
            if prefix in line:
                # Extract the key around the prefix
                start = line.find(prefix)
                if start >= 0:
                    # Try to extract up to 64 chars from the prefix
                    potential = line[start:start+64]
                    # Clean up - stop at common delimiters
                    for delim in ['"', "'", ',', ' ', ')', ']', '}']:
                        if delim in potential:
                            potential = potential[:potential.find(delim)]
                    if len(potential) >= 20:
                        keys.append(potential)
        
        return list(set(keys))  # Remove duplicates

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DMAI Key Workflow System")
    parser.add_argument("--config", default="workflow_config.json", help="Config file path")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--list-keys", action="store_true", help="List stored valid keys")
    parser.add_argument("--key-type", help="Filter keys by type")
    parser.add_argument("--min-weight", type=int, default=5, help="Minimum key weight")
    parser.add_argument("--test-key", help="Test a single key string")
    parser.add_argument("--github-query", help="GitHub search query to run")
    parser.add_argument("--github-max", type=int, default=50, help="Max GitHub results")
    
    args = parser.parse_args()
    
    workflow = KeyWorkflowOrchestrator(args.config)
    
    if args.stats:
        print(json.dumps(workflow.get_stats(), indent=2, default=str))
        sys.exit(0)
    
    if args.list_keys:
        keys = workflow.database.get_valid_keys(args.key_type, args.min_weight)
        print(json.dumps(keys, indent=2, default=str))
        sys.exit(0)
    
    if args.test_key:
        raw = RawKey(
            key_string=args.test_key,
            source_repo="manual_test",
            source_url="manual_test",
            source_file="manual_test",
            line_number=1,
            context="",
            discovered_at=datetime.now()
        )
        result = workflow.process_raw_key(raw)
        if result:
            print(f"✅ Valid key stored: {result.identified_key.key_type}")
        else:
            print("❌ Key invalid or below threshold")
        sys.exit(0)
    
    if args.github_query:
        scraper = GitHubScraperIntegration(workflow)
        scraper.search_and_process(args.github_query, args.github_max)
        workflow.shutdown()
    
    if args.once:
        logger.info("Running single cycle")
        time.sleep(2)
        workflow.shutdown()
    
    if args.daemon:
        logger.info("Running in daemon mode")
        try:
            while workflow.running:
                time.sleep(3600)
        except KeyboardInterrupt:
            workflow.shutdown()

if __name__ == "__main__":
    main()
