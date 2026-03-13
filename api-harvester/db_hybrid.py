import sqlite3
import json
import hashlib
import logging
logger = logging.getLogger(__name__)

class KeyEvolutionDB:
    def __init__(self, db_path='dmai_local.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                key_hash TEXT,
                source_url TEXT,
                key_type TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'active',
                estimated_value REAL DEFAULT 0,
                evolution_generation INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
        
    def store_key(self, key_data):
        cursor = self.conn.cursor()
        key_hash = hashlib.sha256(key_data['key'].encode()).hexdigest()
        cursor.execute('''
            INSERT OR REPLACE INTO api_keys 
            (key, key_hash, source_url, key_type, metadata, status, estimated_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            key_data['key'],
            key_hash,
            key_data.get('source_url', ''),
            key_data.get('key_type', 'unknown'),
            json.dumps(key_data.get('metadata', {})),
            key_data.get('status', 'active'),
            key_data.get('estimated_value', 0)
        ))
        self.conn.commit()
        return cursor.lastrowid

def process_harvested_key(key_data):
    """Process and store a harvested key"""
    db = KeyEvolutionDB()
    return db.store_key(key_data)
