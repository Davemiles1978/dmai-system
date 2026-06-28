#!/usr/bin/env python3
"""
DMAI Centralized API Key Store
===============================
Single source of truth for ALL API keys — dynamic, extensible, zero hardcoded keys.
Backed by SQLite (same dmai_knowledge.db).
Auto-imports keys from environment variables on first run.
Safe to commit to GitHub — contains no secrets.
"""

import os
import sqlite3
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

# Known key prefixes for auto-detection of provider when not specified.
# The system can learn new prefixes dynamically via add_key(provider, key).
KEY_PREFIX_HINTS = {
    'sk-or-':   'openrouter',
    'sk-proj-': 'openai',
    'sk-ant-':  'anthropic',
    'sk-':      'openai',          # catch-all after more specific ones
    'gsk_':     'groq',
    'AIza':     'google',
    'cf-':      'cloudflare',
    'hf_':      'huggingface',
    'cohere-':  'cohere',
    'COHERE-':  'cohere',
    'pplx-':    'perplexity',
    'perplexity-': 'perplexity',
}

# Environment variable -> provider mapping (for auto-import on startup)
ENV_TO_PROVIDER = {
    'GROQ_API_KEY':             'groq',
    'OPENROUTER_API_KEY':       'openrouter',
    'CLOUDFLARE_API_KEY':       'cloudflare',
    'COHERE_API_KEY':           'cohere',
    'HUGGINGFACE_API_KEY':      'huggingface',
    'OPENAI_API_KEY':           'openai',
    'ANTHROPIC_API_KEY':        'anthropic',
    'GEMINI_API_KEY':           'google',
    'DEEPSEEK_API_KEY':         'deepseek',
    'GOOGLE_AI_STUDIO_KEY':     'google',
    'PERPLEXITY_API_KEY':       'perplexity',
}


class APIKeyStore:
    """Central dynamic key registry. Accepts any provider, stores only metadata."""

    def __init__(self, sqlite_persistence=None, db_path: Optional[str] = None):
        self.sqlite = sqlite_persistence
        if db_path:
            self.db_path = Path(db_path)
        elif self.sqlite and hasattr(self.sqlite, 'db_path'):
            self.db_path = self.sqlite.db_path
        else:
            self.db_path = Path("data/dmai_knowledge.db")

        self._init_table()
        self._migrate_env_keys()
        logger.info(f"🔑 APIKeyStore ready (dynamic, {self._count_active()} active keys)")

    def _get_conn(self):
        conn = safe_open_kdb(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    source TEXT DEFAULT 'manual',
                    added_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_validated TEXT,
                    last_used TEXT,
                    health_status TEXT DEFAULT 'unknown',
                    call_count INTEGER DEFAULT 0,
                    notes TEXT,
                    is_active INTEGER DEFAULT 1
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_keys_provider ON api_keys(provider)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)')

    def _count_active(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute('SELECT COUNT(*) as c FROM api_keys WHERE is_active = 1').fetchone()
        return row['c'] if row else 0

    def _migrate_env_keys(self):
        """Import all known environment keys into registry (metadata only, no secret stored in DB)."""
        imported = 0
        for env_var, provider in ENV_TO_PROVIDER.items():
            key = os.getenv(env_var)
            if key and len(key) > 10:
                if self.add_key(provider, key, source=f'env:{env_var}', skip_reconfig=True):
                    imported += 1
        if imported:
            logger.info(f"🔑 Auto-registered {imported} environment keys")

    # ------------------------------------------------------------------
    # Dynamic provider detection
    # ------------------------------------------------------------------
    def _guess_provider(self, key: str) -> str:
        """Return provider name based on key prefix.
        Unknown prefixes yield 'unknown'. System can still store them.
        """
        for prefix, provider in KEY_PREFIX_HINTS.items():
            if key.startswith(prefix):
                return provider
        return 'unknown'

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def add_key(self, provider: str, key: str, source: str = 'manual',
                skip_reconfig: bool = False) -> bool:
        """
        Register an API key.
        :param provider: any provider name (dynamic; 'perplexity', 'new_ai', etc.)
        :param key: the actual API key (never stored in DB, used only for dedup hash)
        :param source: origin label ('manual', 'harvester', 'env:...')
        :returns: True if new, False if already known
        """
        key = key.strip()
        if not key or len(key) < 10:
            logger.warning(f"Rejected short/empty key for provider '{provider}'")
            return False

        # Auto-detect provider if unknown
        if not provider or provider.lower() in ('unknown', 'auto'):
            provider = self._guess_provider(key)

        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_prefix = key[:10] + '...'

        with self._get_conn() as conn:
            existing = conn.execute(
                'SELECT id FROM api_keys WHERE key_hash = ?', (key_hash,)
            ).fetchone()
            if existing:
                # Update last_validated and ensure active
                conn.execute(
                    'UPDATE api_keys SET last_validated = datetime("now"), is_active = 1 '
                    'WHERE id = ?', (existing['id'],)
                )
                return False

            conn.execute('''
                INSERT INTO api_keys (provider, key_hash, key_prefix, source)
                VALUES (?, ?, ?, ?)
            ''', (provider, key_hash, key_prefix, source))
            logger.info(f"🔑 Registered {provider} key ({source}) prefix: {key_prefix}")
            return True

    def get_active_keys(self, provider: Optional[str] = None) -> List[Dict]:
        """Return active key metadata (no secrets)."""
        with self._get_conn() as conn:
            if provider:
                rows = conn.execute(
                    'SELECT * FROM api_keys WHERE provider = ? AND is_active = 1 '
                    'ORDER BY last_validated DESC', (provider,)
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT * FROM api_keys WHERE is_active = 1 '
                    'ORDER BY provider, last_validated DESC'
                ).fetchall()
        return [dict(r) for r in rows]

    def get_known_providers(self) -> List[str]:
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT DISTINCT provider FROM api_keys WHERE is_active = 1'
            ).fetchall()
        return sorted([r['provider'] for r in rows])

    def mark_used(self, provider: str, success: bool = True):
        with self._get_conn() as conn:
            if success:
                conn.execute(
                    'UPDATE api_keys SET call_count = call_count + 1, '
                    'last_used = datetime("now"), health_status = "healthy", '
                    'last_validated = datetime("now") '
                    'WHERE provider = ? AND is_active = 1', (provider,)
                )
            else:
                conn.execute(
                    'UPDATE api_keys SET health_status = "failing" '
                    'WHERE provider = ? AND is_active = 1', (provider,)
                )

    def deactivate_key(self, key_hash: str):
        with self._get_conn() as conn:
            conn.execute('UPDATE api_keys SET is_active = 0 WHERE key_hash = ?', (key_hash,))

    def resolve_key(self, provider: str) -> Optional[str]:
        """
        Get the actual key value for a provider.
        Resolves via environment variable matching the provider name.
        """
        # Try direct environment variable
        env_var = f'{provider.upper()}_API_KEY'
        key = os.getenv(env_var)
        if key:
            return key
        # Try common aliases
        aliases = {
            'google': 'GEMINI_API_KEY',
            'cloudflare': 'CLOUDFLARE_API_KEY',
            'perplexity': 'PERPLEXITY_API_KEY',
        }
        alias = aliases.get(provider)
        if alias:
            key = os.getenv(alias)
            if key:
                return key
        # Fallback: check registry for any env-based source that matches provider
        # (no key exposure, just logs warning)
        logger.debug(f"No environment key found for provider '{provider}'")
        return None

    def status_summary(self) -> Dict:
        """Return human-readable status for /api/tutors/status."""
        with self._get_conn() as conn:
            total = conn.execute('SELECT COUNT(*) as c FROM api_keys WHERE is_active = 1').fetchone()['c']
            healthy = conn.execute(
                "SELECT COUNT(*) as c FROM api_keys WHERE is_active = 1 AND health_status = 'healthy'"
            ).fetchone()['c']
            providers = [dict(r) for r in conn.execute(
                'SELECT provider, COUNT(*) as key_count, MAX(last_validated) as last_check '
                'FROM api_keys WHERE is_active = 1 GROUP BY provider'
            ).fetchall()]
        return {
            'total_active_keys': total,
            'healthy_keys': healthy,
            'providers': providers
        }
