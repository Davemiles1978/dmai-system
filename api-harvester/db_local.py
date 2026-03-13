import sqlite3
import json
import hashlib
from datetime import datetime

class KeyEvolutionDB:
    def __init__(self, db_path='dmai_local.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return True
        
    def store_key(self, key_data):
        if not self.conn:
            self.connect()
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
