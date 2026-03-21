#!/usr/bin/env python3
"""
autonomous_token_manager.py
DMAI's Autonomous Token Management System
Creates, manages, and rotates authentication tokens for all services
"""

import os
import sys
import json
import time
import hashlib
import base64
import secrets
import random
import requests
import threading
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[TokenManager] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TokenManager')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class TokenType(Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    CUSTOM = "custom"


class TokenStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    EXPIRED = "expired"
    REVOKED = "revoked"
    TESTING = "testing"
    PENDING = "pending"


@dataclass
class Token:
    """Token with full metadata"""
    id: str
    token_type: TokenType
    token_value: str
    token_hash: str
    account_id: str
    account_name: str
    created_at: datetime
    expires_at: datetime
    last_used: datetime
    usage_count: int = 0
    rate_limit_remaining: int = 5000
    rate_limit_reset: datetime = None
    status: TokenStatus = TokenStatus.PENDING
    permissions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Account:
    """Account that can generate tokens"""
    id: str
    service: TokenType
    username: str
    email: str
    password_encrypted: str
    created_at: datetime
    last_login: datetime
    tokens: List[str] = field(default_factory=list)
    is_active: bool = True
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# CRYPTOGRAPHY UTILITIES
# ============================================================================

class CryptoManager:
    """Manages encryption for sensitive data"""
    
    def __init__(self, master_key: str = None):
        self.master_key = master_key or os.environ.get('DMAI_MASTER_KEY')
        self.cipher = None
        if self.master_key:
            self._init_cipher()
    
    def _init_cipher(self):
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'dmai_salt_2026',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self.cipher = Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize cipher: {e}")
    
    def encrypt(self, data: str) -> str:
        if not self.cipher:
            return data
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except:
            return data
    
    def decrypt(self, encrypted_data: str) -> str:
        if not self.cipher:
            return encrypted_data
        try:
            decoded = base64.b64decode(encrypted_data)
            return self.cipher.decrypt(decoded).decode()
        except:
            return encrypted_data
    
    def hash_token(self, token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()


# ============================================================================
# GITHUB ACCOUNT CREATOR
# ============================================================================

class GitHubAccountCreator:
    """Creates GitHub accounts using Playwright"""
    
    def __init__(self, crypto: CryptoManager, identity_manager):
        self.crypto = crypto
        self.identity = identity_manager
        self.playwright_available = False
        
        try:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright
            self.playwright_available = True
            logger.info("✅ Playwright available")
        except ImportError:
            logger.warning("Playwright not installed")
    
    def generate_credentials(self) -> Dict[str, str]:
        random_suffix = secrets.token_hex(4)
        base_name = self.identity.public['name'].lower().replace(' ', '')
        return {
            'username': f"{base_name}_{random_suffix}",
            'email': f"{base_name}.{random_suffix}@guerrillamail.com",
            'password': secrets.token_urlsafe(16)
        }
    
    def create_account(self, headless: bool = False) -> Optional[Account]:
        if not self.playwright_available:
            return self._simulate_account()
        
        creds = self.generate_credentials()
        logger.info(f"🚀 Creating GitHub account: {creds['username']}")
        
        try:
            with self.playwright() as p:
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    viewport={'width': 1280, 'height': 720}
                )
                page = context.new_page()
                page.goto('https://github.com/signup')
                page.wait_for_load_state('networkidle')
                
                # Fill email
                page.locator('#email').fill(creds['email'])
                page.wait_for_timeout(1000)
                page.locator('button[type="submit"]').click()
                page.wait_for_timeout(2000)
                
                # Fill password
                page.locator('#password').fill(creds['password'])
                page.wait_for_timeout(1000)
                page.locator('button[type="submit"]').click()
                page.wait_for_timeout(2000)
                
                # Fill username
                page.locator('#login').fill(creds['username'])
                page.wait_for_timeout(1000)
                page.locator('button[type="submit"]').click()
                page.wait_for_timeout(2000)
                
                input("✅ Complete verification in browser, then press Enter...")
                
                browser.close()
                
                account = Account(
                    id=f"github_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    service=TokenType.GITHUB,
                    username=creds['username'],
                    email=creds['email'],
                    password_encrypted=self.crypto.encrypt(creds['password']),
                    created_at=datetime.now(),
                    last_login=datetime.now()
                )
                logger.info(f"✅ Account created: {creds['username']}")
                return account
                
        except Exception as e:
            logger.error(f"Failed to create account: {e}")
            return self._simulate_account()
    
    def _simulate_account(self) -> Optional[Account]:
        creds = self.generate_credentials()
        logger.warning(f"⚠️ SIMULATION: Account created")
        return Account(
            id=f"github_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            service=TokenType.GITHUB,
            username=creds['username'],
            email=creds['email'],
            password_encrypted=self.crypto.encrypt(creds['password']),
            created_at=datetime.now(),
            last_login=datetime.now()
        )


# ============================================================================
# TOKEN POOL MANAGER
# ============================================================================

class TokenPool:
    def __init__(self, token_type: TokenType):
        self.token_type = token_type
        self.tokens: List[Token] = []
        self.lock = threading.Lock()
    
    def add_token(self, token: Token):
        with self.lock:
            self.tokens.append(token)
    
    def remove_token(self, token_id: str):
        with self.lock:
            self.tokens = [t for t in self.tokens if t.id != token_id]
    
    def get_token(self, strategy: str = 'round_robin') -> Optional[Token]:
        with self.lock:
            active_tokens = [t for t in self.tokens if t.status == TokenStatus.ACTIVE]
            if not active_tokens:
                return None
            return active_tokens[0]
    
    def update_token_usage(self, token_id: str, rate_limit_remaining: int = None):
        with self.lock:
            for token in self.tokens:
                if token.id == token_id:
                    token.usage_count += 1
                    token.last_used = datetime.now()
                    if rate_limit_remaining is not None:
                        token.rate_limit_remaining = rate_limit_remaining
                    break
    
    def get_status(self) -> Dict:
        with self.lock:
            return {
                'token_type': self.token_type.value,
                'total_tokens': len(self.tokens),
                'active_tokens': len([t for t in self.tokens if t.status == TokenStatus.ACTIVE]),
                'rate_limited': len([t for t in self.tokens if t.status == TokenStatus.RATE_LIMITED]),
                'total_usage': sum(t.usage_count for t in self.tokens)
            }


# ============================================================================
# MAIN TOKEN MANAGER
# ============================================================================

class AutonomousTokenManager:
    def __init__(self, data_path: Path, identity_manager):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        self.identity = identity_manager
        
        self.crypto = CryptoManager()
        self.github_creator = GitHubAccountCreator(self.crypto, identity_manager)
        
        self.pools: Dict[TokenType, TokenPool] = {}
        self.accounts_file = self.data_path / 'accounts.json'
        self.tokens_file = self.data_path / 'tokens.json'
        self.accounts: List[Account] = []
        self.tokens: List[Token] = []
        
        self._load()
        self._init_pools()
        logger.info("Autonomous Token Manager initialized")
    
    def _load(self):
        if self.accounts_file.exists():
            try:
                with open(self.accounts_file, 'r') as f:
                    data = json.load(f)
                    for acc_data in data.get('accounts', []):
                        acc = Account(
                            id=acc_data['id'],
                            service=TokenType(acc_data['service']),
                            username=acc_data['username'],
                            email=acc_data['email'],
                            password_encrypted=acc_data['password_encrypted'],
                            created_at=datetime.fromisoformat(acc_data['created_at']),
                            last_login=datetime.fromisoformat(acc_data['last_login']),
                            tokens=acc_data.get('tokens', []),
                            is_active=acc_data.get('is_active', True)
                        )
                        self.accounts.append(acc)
            except Exception as e:
                logger.error(f"Failed to load accounts: {e}")
        
        if self.tokens_file.exists():
            try:
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
                    for tok_data in data.get('tokens', []):
                        token = Token(
                            id=tok_data['id'],
                            token_type=TokenType(tok_data['token_type']),
                            token_value=tok_data['token_value'],
                            token_hash=tok_data['token_hash'],
                            account_id=tok_data['account_id'],
                            account_name=tok_data['account_name'],
                            created_at=datetime.fromisoformat(tok_data['created_at']),
                            expires_at=datetime.fromisoformat(tok_data['expires_at']) if tok_data.get('expires_at') else None,
                            last_used=datetime.fromisoformat(tok_data['last_used']),
                            usage_count=tok_data.get('usage_count', 0),
                            rate_limit_remaining=tok_data.get('rate_limit_remaining', 5000),
                            status=TokenStatus(tok_data.get('status', 'pending')),
                            permissions=tok_data.get('permissions', [])
                        )
                        self.tokens.append(token)
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")
    
    def _save(self):
        try:
            with open(self.accounts_file, 'w') as f:
                json.dump({
                    'accounts': [
                        {
                            'id': a.id,
                            'service': a.service.value,
                            'username': a.username,
                            'email': a.email,
                            'password_encrypted': a.password_encrypted,
                            'created_at': a.created_at.isoformat(),
                            'last_login': a.last_login.isoformat(),
                            'tokens': a.tokens,
                            'is_active': a.is_active
                        }
                        for a in self.accounts
                    ]
                }, f, indent=2)
            
            with open(self.tokens_file, 'w') as f:
                json.dump({
                    'tokens': [
                        {
                            'id': t.id,
                            'token_type': t.token_type.value,
                            'token_value': t.token_value,
                            'token_hash': t.token_hash,
                            'account_id': t.account_id,
                            'account_name': t.account_name,
                            'created_at': t.created_at.isoformat(),
                            'expires_at': t.expires_at.isoformat() if t.expires_at else None,
                            'last_used': t.last_used.isoformat(),
                            'usage_count': t.usage_count,
                            'rate_limit_remaining': t.rate_limit_remaining,
                            'status': t.status.value,
                            'permissions': t.permissions
                        }
                        for t in self.tokens
                    ]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save: {e}")
    
    def _init_pools(self):
        for token_type in TokenType:
            self.pools[token_type] = TokenPool(token_type)
        for token in self.tokens:
            if token.status == TokenStatus.ACTIVE:
                self.pools[token.token_type].add_token(token)
    
    def create_account_manual(self, username: str, email: str, password: str, 
                               service: TokenType = TokenType.GITHUB) -> Account:
        """Manually create an account record"""
        account = Account(
            id=f"{service.value}_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            service=service,
            username=username,
            email=email,
            password_encrypted=self.crypto.encrypt(password),
            created_at=datetime.now(),
            last_login=datetime.now(),
            is_active=True
        )
        self.accounts.append(account)
        self._save()
        logger.info(f"✅ Account added: {username}")
        return account
    
    def add_token(self, token_value: str, account_id: str = None) -> Optional[Token]:
        if not token_value:
            return None
        
        token_type = TokenType.GITHUB
        if token_value.startswith('ghp_') or token_value.startswith('gho_'):
            token_type = TokenType.GITHUB
        elif token_value.startswith('sk-'):
            token_type = TokenType.OPENAI
        
        if account_id:
            account = next((a for a in self.accounts if a.id == account_id), None)
        else:
            accounts = [a for a in self.accounts if a.service == token_type]
            account = accounts[0] if accounts else None
        
        if not account:
            account = self.create_account_manual(
                username=f"auto_{token_type.value}",
                email=f"auto@{token_type.value}.com",
                password="auto",
                service=token_type
            )
        
        token = Token(
            id=f"token_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            token_type=token_type,
            token_value=self.crypto.encrypt(token_value),
            token_hash=self.crypto.hash_token(token_value),
            account_id=account.id,
            account_name=account.username,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90),
            last_used=datetime.now(),
            status=TokenStatus.ACTIVE,
            permissions=['public_repo', 'read:user']
        )
        
        self.tokens.append(token)
        self.pools[token.token_type].add_token(token)
        account.tokens.append(token.id)
        self._save()
        
        logger.info(f"✅ Token added for {account.username}")
        return token
    
    def get_token(self, service: TokenType, strategy: str = 'round_robin') -> Optional[str]:
        pool = self.pools.get(service)
        if not pool:
            return None
        
        token = pool.get_token(strategy)
        if not token:
            return None
        
        token_value = self.crypto.decrypt(token.token_value)
        pool.update_token_usage(token.id)
        return token_value
    
    def run_maintenance(self):
        logger.info("🔧 Running token maintenance")
        now = datetime.now()
        for token in self.tokens:
            if token.expires_at and token.expires_at < now:
                if token.status == TokenStatus.ACTIVE:
                    token.status = TokenStatus.EXPIRED
                    self.pools[token.token_type].remove_token(token.id)
        self._save()
    
    def get_status(self) -> Dict:
        return {
            'accounts': {
                'total': len(self.accounts),
                'by_service': {
                    service.value: len([a for a in self.accounts if a.service == service])
                    for service in TokenType
                }
            },
            'tokens': {
                'total': len(self.tokens),
                'active': len([t for t in self.tokens if t.status == TokenStatus.ACTIVE]),
                'by_service': {
                    service.value: self.pools.get(service, TokenPool(service)).get_status()
                    for service in TokenType
                }
            }
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔐 DMAI AUTONOMOUS TOKEN MANAGER")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--add-token', help='Add a GitHub token')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--maintenance', action='store_true', help='Run maintenance')
    
    args = parser.parse_args()
    
    class MockIdentity:
        def __init__(self):
            self.public = {'name': 'Alex Riviera', 'email': 'alex@riviera.com'}
    
    identity = MockIdentity()
    data_path = Path(__file__).parent / 'data'
    manager = AutonomousTokenManager(data_path, identity)
    
    if args.add_token:
        token = manager.add_token(args.add_token)
        if token:
            print("✅ Token added")
    elif args.status:
        print(json.dumps(manager.get_status(), indent=2))
    elif args.maintenance:
        manager.run_maintenance()
        print("Maintenance completed")
    else:
        print(json.dumps(manager.get_status(), indent=2))
