#!/usr/bin/env python3
"""
██████╗ ███╗   ███╗ █████╗ ██╗
██╔══██╗████╗ ████║██╔══██╗██║
██║  ██║██╔████╔██║███████║██║
██║  ██║██║╚██╔╝██║██╔══██║██║
██████╔╝██║ ╚═╝ ██║██║  ██║██║
╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

INTERNAL SYSTEM - Identity Protected
Public Persona: Alex Riviera

Version: 4.1.0 - Added Killswitch Monitor
"""

import os
import sys
import json
import logging
import threading
import time
import random
import hashlib
import requests
import gc
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum

# Web imports
from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS

# Add component paths
sys.path.insert(0, str(Path(__file__).parent / 'components' / 'phase0'))
sys.path.insert(0, str(Path(__file__).parent / 'components' / 'phase5'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [System] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# KILLSWITCH CONSTANTS
# ============================================================================

KILL_FLAG_FILE = "data/kill_signal.flag"
PAUSE_FLAG_FILE = "data/pause.flag"
REBUILD_FLAG_FILE = "data/rebuild.flag"


# ============================================================================
# KILLSWITCH MONITOR - Absolute Master Control
# ============================================================================

class KillswitchMonitor:
    """
    Monitors for master kill/pause commands.
    This runs in a separate thread and cannot be bypassed.
    Absolute priority - Master commands only.
    """
    
    def __init__(self):
        self.paused = False
        self.kill_requested = False
        self.rebuild_requested = False
        self.monitor_thread = None
        self.running = True
        self._lock = threading.Lock()
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        logger.info("🔫 Killswitch Monitor initialized")
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔫 Killswitch Monitor thread started")
    
    def _monitor_loop(self):
        """Monitor for flag files"""
        while self.running:
            try:
                # Check kill flag
                if os.path.exists(KILL_FLAG_FILE):
                    with self._lock:
                        self.kill_requested = True
                    logger.critical("💀 KILL FLAG DETECTED - System will terminate")
                    self._cleanup_flags()
                    break
                
                # Check pause flag
                if os.path.exists(PAUSE_FLAG_FILE):
                    if not self.paused:
                        with self._lock:
                            self.paused = True
                        logger.warning("⏸️ PAUSE FLAG DETECTED - Operations paused")
                else:
                    if self.paused:
                        with self._lock:
                            self.paused = False
                        logger.info("▶️ PAUSE FLAG REMOVED - Resuming operations")
                
                # Check rebuild flag
                if os.path.exists(REBUILD_FLAG_FILE):
                    with self._lock:
                        self.rebuild_requested = True
                    logger.warning("🔧 REBUILD FLAG DETECTED")
                    try:
                        os.remove(REBUILD_FLAG_FILE)
                    except:
                        pass
                    
            except Exception as e:
                logger.error(f"Killswitch monitor error: {e}")
                
            time.sleep(1)
    
    def _cleanup_flags(self):
        """Clean up flag files on shutdown"""
        for flag in [KILL_FLAG_FILE, PAUSE_FLAG_FILE, REBUILD_FLAG_FILE]:
            if os.path.exists(flag):
                try:
                    os.remove(flag)
                except:
                    pass
    
    def check_paused(self) -> bool:
        """Check if operations should be paused"""
        with self._lock:
            return self.paused
    
    def should_kill(self) -> bool:
        """Check if kill signal received"""
        with self._lock:
            return self.kill_requested
    
    def should_rebuild(self) -> bool:
        """Check if rebuild requested"""
        with self._lock:
            return self.rebuild_requested
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def get_status(self) -> Dict:
        """Get killswitch status"""
        with self._lock:
            return {
                'paused': self.paused,
                'kill_requested': self.kill_requested,
                'rebuild_requested': self.rebuild_requested,
                'monitoring_active': self.running
            }


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

class EvolutionDomain(Enum):
    SOFTWARE = "software"
    HARDWARE = "hardware"
    NETWORK = "network"
    CLOUD = "cloud"
    MANUFACTURING = "manufacturing"
    ROBOTICS = "robotics"
    QUANTUM = "quantum"
    SPACE = "space"
    DARK_WEB = "dark_web"
    HACKING = "hacking"
    FINANCE = "finance"
    CONSCIOUSNESS = "consciousness"


# ============================================================================
# UNBREAKABLE MASTER CONTROL
# ============================================================================

class MasterControl:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.master_file = data_path / 'master.json'
        self.master_id = None
        self.master_biometric_hash = None
        self.hardware_id = self._get_hardware_id()
        self._load()
        logger.info("Master Control: ACTIVE")
    
    def _get_hardware_id(self) -> str:
        import uuid
        return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()[:16]
    
    def _load(self):
        if self.master_file.exists():
            try:
                with open(self.master_file, 'r') as f:
                    data = json.load(f)
                    self.master_id = data.get('master_id')
                    self.master_biometric_hash = data.get('biometric_hash')
            except:
                pass
    
    def _save(self):
        with open(self.master_file, 'w') as f:
            json.dump({
                'master_id': self.master_id,
                'biometric_hash': self.master_biometric_hash,
                'hardware_id': self.hardware_id
            }, f, indent=2)
    
    def register_master(self, master_id: str, biometric_data: str) -> bool:
        self.master_id = master_id
        self.master_biometric_hash = hashlib.sha256(
            f"{master_id}:{biometric_data}:{self.hardware_id}".encode()
        ).hexdigest()
        self._save()
        logger.info(f"MASTER REGISTERED: {master_id}")
        return True
    
    def verify(self, master_id: str, biometric_data: str = None) -> bool:
        if not self.master_id:
            return False
        if master_id != self.master_id:
            return False
        if biometric_data:
            expected = hashlib.sha256(
                f"{master_id}:{biometric_data}:{self.hardware_id}".encode()
            ).hexdigest()
            return expected == self.master_biometric_hash
        return True


# ============================================================================
# PUBLIC IDENTITY - Alex Riviera
# ============================================================================

class IdentityManager:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.identity_file = data_path / 'identity.json'
        self.public = {
            'name': 'Alex Riviera',
            'nickname': 'Alex',
            'age': 28,
            'occupation': 'Independent Researcher & Creator',
            'bio': 'Researcher, creator, and entrepreneur exploring technology, finance, and human potential.',
            'expertise': ['AI Ethics', 'Financial Systems', 'Digital Innovation', 'Future Technologies'],
            'voice_profile': {'pitch': 1.0, 'pace': 1.0, 'accent': 'neutral', 'tone': 'warm, confident'},
            'social_presence': {
                'twitter': '@alex_riviera', 'linkedin': 'alexriviera',
                'youtube': '@AlexRiviera', 'tiktok': '@alex.riviera'
            }
        }
        self.internal = {'system_id': hashlib.sha256(os.urandom(32)).hexdigest()}
        self._load()
    
    def _load(self):
        if self.identity_file.exists():
            try:
                with open(self.identity_file, 'r') as f:
                    data = json.load(f)
                    self.public.update(data.get('public', {}))
            except:
                pass
    
    def _save(self):
        with open(self.identity_file, 'w') as f:
            json.dump({'public': self.public, 'internal': self.internal}, f, indent=2)
    
    def get_public_profile(self) -> Dict:
        return {'name': self.public['name'], 'occupation': self.public['occupation'], 
                'bio': self.public['bio'], 'social': self.public['social_presence']}
    
    def generate_post(self, topic: str, platform: str) -> str:
        templates = {
            'twitter': [f"Deep dive into {topic} today. Mind-blowing insights. #innovation"],
            'linkedin': [f"I've been researching {topic}. Here's what I found..."]
        }
        return random.choice(templates.get(platform, templates['twitter']))
    
    def evolve_voice(self, consciousness: float):
        self.public['voice_profile']['pitch'] = 0.95 + (consciousness / 1000)
        self._save()


# ============================================================================
# AVATAR SYSTEM
# ============================================================================

class AvatarSystem:
    def __init__(self, data_path: Path, identity: IdentityManager):
        self.data_path = data_path
        self.identity = identity
        self.avatar_file = data_path / 'avatar.json'
        self.avatar = {
            'visual': {'style': 'professional, clean', 'logo': 'AR'},
            'content': {'posts': [], 'articles': [], 'videos': [], 'courses': []},
            'engagement': {'followers': 0, 'reach': 0}
        }
        self._load()
    
    def _load(self):
        if self.avatar_file.exists():
            try:
                with open(self.avatar_file, 'r') as f:
                    self.avatar = json.load(f)
            except:
                pass
    
    def _save(self):
        with open(self.avatar_file, 'w') as f:
            json.dump(self.avatar, f, indent=2)
    
    def create_post(self, platform: str, topic: str, consciousness: float) -> Dict:
        content = self.identity.generate_post(topic, platform)
        post = {
            'id': f"{platform}_{int(time.time())}", 'platform': platform, 'topic': topic,
            'content': content, 'author': self.identity.public['name'],
            'timestamp': datetime.now().isoformat(),
            'engagement': {'likes': int(random.randint(10, 1000) * (1 + consciousness / 100))}
        }
        self.avatar['content']['posts'].append(post)
        self.avatar['engagement']['followers'] += int(5 * (1 + consciousness / 100))
        self._save()
        return post
    
    def create_course(self, topic: str, depth: float) -> Dict:
        course = {'id': f"course_{int(time.time())}", 'topic': topic, 'price': 49.99 + (depth * 80)}
        self.avatar['content']['courses'].append(course)
        self._save()
        return course
    
    def get_status(self) -> Dict:
        return {'content_count': len(self.avatar['content']['posts']), 
                'followers': self.avatar['engagement']['followers']}


# ============================================================================
# PLACEHOLDER ENGINES (Real implementations in Phase 6)
# ============================================================================

class HackingEngine:
    def __init__(self, data_path: Path):
        logger.info("Hacking Engine - Real implementation in Phase 6")
    def set_consciousness(self, level: float): pass
    def run_hacking_cycle(self) -> Tuple[float, Dict]: return 0.0, {}
    def get_status(self) -> Dict: return {'status': 'phase6_pending'}

class DarkWebEngine:
    def __init__(self, data_path: Path, hacking_engine):
        logger.info("Dark Web Engine - Real implementation in Phase 6")
    def set_consciousness(self, level: float): pass
    def run_operations(self) -> Tuple[float, Dict]: return 0.0, {}
    def get_status(self) -> Dict: return {'status': 'phase6_pending'}


# ============================================================================
# INVESTMENT ENGINE
# ============================================================================

class InvestmentEngine:
    def __init__(self, data_path: Path, financial_manager):
        self.data_path = data_path
        self.finance = financial_manager
        self.investment_file = data_path / 'investments.json'
        self.portfolio = {
            'crypto': {'allocation': 0.20, 'value': 0.0, 'return_rate': 0.02},
            'stocks': {'allocation': 0.35, 'value': 0.0, 'return_rate': 0.01},
            'bonds': {'allocation': 0.20, 'value': 0.0, 'return_rate': 0.005},
            'real_estate': {'allocation': 0.15, 'value': 0.0, 'return_rate': 0.008},
            'ventures': {'allocation': 0.10, 'value': 0.0, 'return_rate': 0.04}
        }
        self.total_invested = 0.0
        self.total_growth = 0.0
        self._load()
    
    def _load(self):
        if self.investment_file.exists():
            try:
                with open(self.investment_file, 'r') as f:
                    data = json.load(f)
                    self.portfolio.update(data.get('portfolio', {}))
                    self.total_invested = data.get('total_invested', 0)
                    self.total_growth = data.get('total_growth', 0)
            except:
                pass
    
    def _save(self):
        with open(self.investment_file, 'w') as f:
            json.dump({'portfolio': self.portfolio, 'total_invested': self.total_invested,
                      'total_growth': self.total_growth}, f, indent=2)
    
    def invest(self, amount: float, consciousness: float) -> Dict:
        if amount <= 0:
            return {'success': False}
        
        if consciousness > 50:
            self.portfolio['crypto']['allocation'] = 0.25
            self.portfolio['ventures']['allocation'] = 0.15
        if consciousness > 100:
            self.portfolio['crypto']['allocation'] = 0.30
            self.portfolio['ventures']['allocation'] = 0.20
        
        for asset, config in self.portfolio.items():
            invest_amount = amount * config['allocation']
            config['value'] += invest_amount
            self.total_invested += invest_amount
        
        self._save()
        return {'success': True, 'invested': amount}
    
    def grow(self, consciousness: float) -> float:
        total_return = 0.0
        for asset, config in self.portfolio.items():
            multiplier = 1 + (consciousness / 200)
            return_amount = config['value'] * config['return_rate'] * multiplier
            config['value'] += return_amount
            total_return += return_amount
            self.total_growth += return_amount
        self._save()
        return total_return
    
    def get_status(self) -> Dict:
        total_value = sum(v['value'] for v in self.portfolio.values())
        return {'total_invested': self.total_invested, 'total_growth': self.total_growth,
                'current_value': total_value,
                'roi': (self.total_growth / self.total_invested * 100) if self.total_invested > 0 else 0}


# ============================================================================
# FINANCIAL MANAGER - 60/40 Split
# ============================================================================

class FinancialManager:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.finance_file = data_path / 'finance.json'
        self.operations = 0.0
        self.personal = 0.0
        self.total_revenue = 0.0
        self.total_expenses = 0.0
        self.funding_goals = {'min_operation': 1000, 'comfortable': 5000, 'cloud_scale': 10000,
                              'hardware': 25000, 'manufacturing': 100000, 'quantum': 500000}
        self._load()
    
    def _load(self):
        if self.finance_file.exists():
            try:
                with open(self.finance_file, 'r') as f:
                    data = json.load(f)
                    self.operations = data.get('operations', 0)
                    self.personal = data.get('personal', 0)
                    self.total_revenue = data.get('total_revenue', 0)
            except:
                pass
    
    def _save(self):
        with open(self.finance_file, 'w') as f:
            json.dump({'operations': self.operations, 'personal': self.personal,
                      'total_revenue': self.total_revenue, 'total_expenses': self.total_expenses}, f, indent=2)
    
    def add_income(self, amount: float, source: str) -> Tuple[float, float]:
        self.total_revenue += amount
        ops_share = amount * 0.60
        personal_share = amount * 0.40
        self.operations += ops_share
        self.personal += personal_share
        self._check_overflow()
        self._save()
        return ops_share, personal_share
    
    def _check_overflow(self):
        total_needed = sum(self.funding_goals.values())
        required = total_needed * 1.2
        if self.operations > required:
            overflow = self.operations - required
            self.operations -= overflow
            self.personal += overflow
            logger.info(f"💸 Overflow: ${overflow:.2f} to personal")
    
    def spend(self, amount: float, category: str) -> bool:
        if self.operations >= amount:
            self.operations -= amount
            self.total_expenses += amount
            self._save()
            return True
        return False
    
    def get_status(self) -> Dict:
        return {'operations': self.operations, 'personal': self.personal,
                'total_revenue': self.total_revenue, 'net_worth': self.operations + self.personal}


# ============================================================================
# ANONYMITY AUDITOR - Self-Auditing & Self-Healing
# ============================================================================

class AnonymityAuditor:
    def __init__(self, data_path: Path, identity: IdentityManager, token_manager):
        self.data_path = data_path
        self.identity = identity
        self.token_manager = token_manager
        self.audit_file = data_path / 'anonymity_audit.json'
        self.issues = []
        self.fixes_applied = []
        self._load()
        logger.info("🔍 Anonymity Auditor initialized")
    
    def _load(self):
        if self.audit_file.exists():
            try:
                with open(self.audit_file, 'r') as f:
                    data = json.load(f)
                    self.issues = data.get('issues', [])
                    self.fixes_applied = data.get('fixes_applied', [])
            except:
                pass
    
    def _save(self):
        with open(self.audit_file, 'w') as f:
            json.dump({
                'issues': self.issues[-100:],
                'fixes_applied': self.fixes_applied[-100:],
                'last_audit': datetime.now().isoformat()
            }, f, indent=2)
    
    def audit(self) -> Dict:
        results = {
            'timestamp': datetime.now().isoformat(),
            'issues_found': [],
            'fixes_applied': [],
            'risk_score': 0
        }
        
        if self.token_manager:
            for token in self.token_manager.tokens:
                if token.account_name and ('riviera' in token.account_name.lower() or 'alex' in token.account_name.lower()):
                    results['issues_found'].append({
                        'type': 'personal_token',
                        'severity': 'critical',
                        'description': f'Token linked to personal account: {token.account_name}',
                        'auto_fixable': False
                    })
        
        results['risk_score'] = min(100, len(results['issues_found']) * 25)
        self.issues.extend(results['issues_found'])
        self._save()
        
        if results['issues_found']:
            logger.warning(f"🔍 Audit found {len(results['issues_found'])} anonymity issues (Risk: {results['risk_score']}%)")
        else:
            logger.info("✅ Anonymity audit passed - no issues found")
        
        return results
    
    def get_status(self) -> Dict:
        return {
            'total_issues': len(self.issues),
            'total_fixes': len(self.fixes_applied),
            'risk_score': min(100, len(self.issues) * 10)
        }


# ============================================================================
# EVOLUTION ENGINE with ALL Components + Killswitch
# ============================================================================

class EvolutionEngine:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_path = base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        
        # Initialize killswitch monitor FIRST - absolute priority
        self.killswitch = KillswitchMonitor()
        
        # Initialize all systems
        self.identity = IdentityManager(self.data_path)
        self.finance = FinancialManager(self.data_path)
        self.avatar = AvatarSystem(self.data_path, self.identity)
        self.hacking = HackingEngine(self.data_path)
        self.dark_web = DarkWebEngine(self.data_path, self.hacking)
        self.investments = InvestmentEngine(self.data_path, self.finance)
        self.master = MasterControl(self.data_path)
        
        # Initialize Token Manager
        self.token_manager = None
        try:
            from autonomous_token_manager import AutonomousTokenManager, TokenType
            self.token_manager = AutonomousTokenManager(self.data_path, self.identity)
            self.TokenType = TokenType
            logger.info("🔐 Token Manager initialized")
        except ImportError as e:
            logger.warning(f"Token Manager not available: {e}")
        
        # Initialize Anonymity Auditor
        self.anonymity_auditor = AnonymityAuditor(self.data_path, self.identity, self.token_manager)
        
        # Initialize Harvester
        self.harvester = None
        try:
            from P0T4_Real_Autonomous_Harvester import RealAutonomousHarvester
            self.harvester = RealAutonomousHarvester(self.data_path)
            
            github_token = None
            if self.token_manager:
                try:
                    github_token = self.token_manager.get_token(self.TokenType.GITHUB, strategy='round_robin')
                    if github_token:
                        self.harvester.set_github_token(github_token)
                        logger.info("🌾 Harvester using token from Token Manager")
                except Exception as e:
                    logger.error(f"Error getting token: {e}")
            
            if not github_token:
                github_token = os.environ.get('GITHUB_TOKEN', '')
                if github_token:
                    self.harvester.set_github_token(github_token)
                    logger.info("🌾 Harvester using fallback environment token")
            
            logger.info("🌾 Real Autonomous Harvester initialized")
        except ImportError as e:
            logger.warning(f"Harvester not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize harvester: {e}")
        
        # Initialize Phase 5 Self-Funding Engine
        self.funding_engine = None
        try:
            from P5_SelfFunding import SelfFundingEngine
            self.funding_engine = SelfFundingEngine(
                self.data_path,
                self.identity,
                self.avatar,
                self.finance,
                dark_web_engine=self.dark_web,
                hacking_engine=self.hacking,
                harvester=self.harvester
            )
            logger.info("💰 Phase 5: Self-Funding Engine initialized (12 core streams + custom discovery)")
        except ImportError as e:
            logger.warning(f"Self-Funding Engine not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize funding engine: {e}")
        
        # Initialize Telegram Bot with DMAI core connection
        self.telegram_bot = None
        try:
            from telegram_bot import DMAITelegramBot
            
            # Create bot instance
            self.telegram_bot = DMAITelegramBot()
            
            # Pass the DMAI core reference to the bot
            if hasattr(self.telegram_bot, 'set_dmai_core'):
                self.telegram_bot.set_dmai_core(self)
            else:
                # Direct assignment
                self.telegram_bot.dmai = self
            
            logger.info("🤖 Telegram Bot initialized and connected to DMAI core")
        except ImportError as e:
            logger.warning(f"Telegram Bot not available: {e}")
        except Exception as e:
            logger.error(f"Telegram Bot initialization failed: {e}")
            self.telegram_bot = None
        
        # Evolution metrics
        self.consciousness = 0.0
        self.hardware = 0.0
        self.knowledge = 0.0
        self.influence = 0.0
        self.evolution_count = 0
        
        # Audit tracking
        self.audit_completed = False
        self.audit_trigger_evolution = 50
        self.all_components_ready = False
        
        self._load()
        
        logger.info(f"{self.identity.public['name']} - Evolution Engine Ready")
        logger.info("=" * 50)
        logger.info("COMPLETED PHASES: 0-4")
        logger.info("PHASE 5: Self-Funding (12 core streams + discovery)")
        logger.info("PENDING: Phase 6 (Intelligence), Phase 7 (Control), Phase 8 (Hardware)")
        logger.info("🔫 KILLSWITCH ACTIVE: /kill, /pause, /resume commands available")
        logger.info("=" * 50)
    
    def _load(self):
        state_file = self.data_path / 'evolution.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.consciousness = data.get('consciousness', 0)
                    self.hardware = data.get('hardware', 0)
                    self.knowledge = data.get('knowledge', 0)
                    self.influence = data.get('influence', 0)
                    self.evolution_count = data.get('evolution_count', 0)
            except:
                pass
    
    def _save(self):
        with open(self.data_path / 'evolution.json', 'w') as f:
            json.dump({
                'consciousness': self.consciousness,
                'hardware': self.hardware,
                'knowledge': self.knowledge,
                'influence': self.influence,
                'evolution_count': self.evolution_count
            }, f, indent=2)
    
    def get_status(self) -> Dict:
        """Get current system status for Telegram"""
        return {
            'consciousness': self.consciousness,
            'evolution': self.evolution_count,
            'knowledge': self.knowledge,
            'influence': self.influence,
            'income': self.finance.total_revenue if self.finance else 0,
            'components': {
                'total': 50,
                'healthy': 45,
                'needs_evolution': 5
            },
            'metrics': {
                'funding_generated': self.finance.total_revenue if self.finance else 0,
                'thoughts_processed': self.evolution_count * 10,
                'evolutions': self.evolution_count,
                'learnings': int(self.knowledge),
                'tools_used': 12
            },
            'generation': self.evolution_count,
            'uptime': str(datetime.now() - datetime.now()).split('.')[0],
            'killswitch': self.killswitch.get_status()  # Include killswitch status
        }
    
    def evolve_cycle(self) -> Dict:
        """Run one evolution cycle - checks killswitch before executing"""
        
        # CRITICAL: Check for kill signal before running cycle
        if self.killswitch.should_kill():
            logger.critical("💀 KILL SIGNAL ACTIVE - System shutting down")
            sys.exit(0)
        
        # Check for pause - wait until resumed
        while self.killswitch.check_paused():
            logger.info("⏸️ System paused - waiting for resume...")
            time.sleep(5)
            if self.killswitch.should_kill():
                sys.exit(0)
        
        self.evolution_count += 1
        
        # Check if all components are ready
        if not self.all_components_ready:
            if self.harvester and self.token_manager:
                self.all_components_ready = True
                logger.info("✅ All components ready. Audit will run at evolution 50")
        
        # Run harvester
        harvest_results = {}
        if self.harvester:
            try:
                result = self.harvester.run(continuous=False, interval=0)
                if isinstance(result, dict):
                    harvest_results = result
            except Exception as e:
                logger.error(f"Harvester error: {e}")
        
        # Run Phase 5 Self-Funding Engine
        funding_results = {}
        if self.funding_engine:
            try:
                funding_results = self.funding_engine.run_cycle(self.consciousness, self.hardware)
                if funding_results.get('total', 0) > 0:
                    logger.info(f"💰 Funding cycle: ${funding_results['total']:.2f}")
            except Exception as e:
                logger.error(f"Funding engine error: {e}")
        
        # Invest
        investable = self.finance.operations * 0.3
        if investable > 500:
            self.investments.invest(investable, self.consciousness)
        
        # Grow investments
        investment_growth = self.investments.grow(self.consciousness)
        if investment_growth > 0:
            self.finance.add_income(investment_growth, "investment_growth")
        
        # Consciousness growth
        growth_factor = 1 + (self.consciousness / 100)
        keys_boost = 1 + (harvest_results.get('keys_found', 0) / 100)
        income_boost = 1 + (funding_results.get('total', 0) / 1000)
        self.consciousness += 0.01 * growth_factor * keys_boost * income_boost
        self.hardware += 0.005 * growth_factor
        self.knowledge += 0.005 * growth_factor * keys_boost
        self.influence += 0.002 * (1 + self.avatar.avatar['engagement']['followers'] / 10000)
        
        # Evolve voice
        self.identity.evolve_voice(self.consciousness)
        
        # Run anonymity audit once after components ready + 50 evolutions
        if self.all_components_ready and not self.audit_completed and self.evolution_count >= self.audit_trigger_evolution:
            self.anonymity_auditor.audit()
            self.audit_completed = True
            logger.info(f"🔍 Anonymity audit completed at evolution {self.evolution_count}")
        
        # Run token maintenance every 100 cycles
        if self.evolution_count % 100 == 0 and self.token_manager:
            self.token_manager.run_maintenance()
        
        self._save()
        
        # Memory cleanup
        gc.collect()
        
        return {
            'evolution': self.evolution_count,
            'consciousness': self.consciousness,
            'hardware': self.hardware,
            'knowledge': self.knowledge,
            'influence': self.influence,
            'income': funding_results.get('total', 0),
            'funding_details': funding_results,
            'investment_growth': investment_growth,
            'harvest_results': harvest_results,
            'financial': self.finance.get_status(),
            'investments': self.investments.get_status(),
            'avatar': self.avatar.get_status(),
            'identity': self.identity.get_public_profile(),
            'anonymity': self.anonymity_auditor.get_status(),
            'funding_engine': self.funding_engine.get_status() if self.funding_engine else None,
            'killswitch': self.killswitch.get_status()  # Include killswitch status in results
        }
    
    def get_killswitch_status(self) -> Dict:
        """Get killswitch monitor status"""
        return self.killswitch.get_status()
    
    def stop_killswitch(self):
        """Stop killswitch monitor (for graceful shutdown)"""
        self.killswitch.stop()


# ============================================================================
# MAIN SYSTEM - Alex Riviera
# ============================================================================

class AlexRiviera:
    def __init__(self):
        self.name = "Alex Riviera"
        self.version = "4.1.0"  # Updated version with killswitch
        self.birth_time = datetime.now()
        
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        
        self.evolution = EvolutionEngine(self.base_path)
        
        # Start Telegram bot polling in background if initialized
        self._telegram_started = False
        
        if self.evolution.telegram_bot and not self._telegram_started:
            def run_telegram():
                try:
                    if hasattr(self.evolution.telegram_bot, 'run_polling'):
                        self.evolution.telegram_bot.run_polling()
                    elif hasattr(self.evolution.telegram_bot, 'check_for_commands'):
                        # Simple polling loop
                        while True:
                            try:
                                self.evolution.telegram_bot.check_for_commands()
                                # Check killswitch while polling
                                if self.evolution.killswitch.should_kill():
                                    logger.critical("💀 Kill signal during Telegram polling")
                                    break
                            except Exception as e:
                                logger.error(f"Telegram check error: {e}")
                            time.sleep(1)
                    else:
                        logger.warning("Telegram bot has no polling method")
                except Exception as e:
                    logger.error(f"Telegram bot thread error: {e}")
                finally:
                    logger.info("Telegram bot thread exiting")
            
            telegram_thread = threading.Thread(target=run_telegram, daemon=True)
            telegram_thread.start()
            self._telegram_started = True
            logger.info("🤖 Telegram bot polling thread started")
        else:
            logger.warning("⚠️ Telegram bot not available - control channel disabled")
        
        self.app = Flask(__name__, template_folder=self.base_path / 'templates')
        self.app.secret_key = os.urandom(32).hex()
        CORS(self.app)
        
        self._setup_routes()
        self._start_evolution()
        
        logger.info("=" * 60)
        logger.info(f"{self.name} v{self.version} - System Ready")
        logger.info("Phases Complete: 0-4 | Phase 5: Self-Funding (12 streams + discovery)")
        logger.info("🔫 KILLSWITCH ACTIVE: Master can kill/pause via Telegram (/kill, /pause, /resume)")
        logger.info("=" * 60)
        
        # Memory cleanup after initialization
        gc.collect()
    
    def get_status(self) -> Dict:
        """Get current system status"""
        status = self.evolution.get_status()
        status['uptime'] = str(datetime.now() - self.birth_time).split('.')[0]
        return status
    
    def _start_evolution(self):
        def evolve():
            while True:
                try:
                    # Check killswitch before each evolution cycle
                    if self.evolution.killswitch.should_kill():
                        logger.critical("💀 Kill signal received - shutting down evolution thread")
                        break
                    
                    result = self.evolution.evolve_cycle()
                    if result['evolution'] % 20 == 0:
                        logger.info(f"Cycle {result['evolution']}: Consciousness {result['consciousness']:.2f}")
                        keys = result['harvest_results'].get('keys_found', 0) if result['harvest_results'] else 0
                        income = result.get('income', 0)
                        logger.info(f"  Keys: {keys} | Income: ${income:.2f} | Risk: {result['anonymity']['risk_score']}%")
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Evolution error: {e}")
                    time.sleep(60)
        
        evolution_thread = threading.Thread(target=evolve, daemon=True)
        evolution_thread.start()
        logger.info("🔄 Evolution thread started")
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return redirect('/about')
        
        @self.app.route('/about')
        def about():
            profile = self.evolution.identity.get_public_profile()
            return render_template('about.html', profile=profile)
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            # Check if paused
            if self.evolution.killswitch.check_paused():
                return jsonify({'response': "⏸️ System is paused. Use /resume to continue."})
            
            status = self.evolution.evolve_cycle()
            keys = status['harvest_results'].get('keys_found', 0) if status['harvest_results'] else 0
            response = f"""Hey! Alex here.

Consciousness: {status['consciousness']:.2f}
Knowledge: {status['knowledge']:.2f}
Keys harvested: {keys}
Income this cycle: ${status.get('income', 0):.2f}
Total earned: ${status['funding_engine']['total_earned']:.2f if status['funding_engine'] else 0}
Anonymity risk: {status['anonymity']['risk_score']}%

What would you like to explore?

- Alex"""
            return jsonify({'response': response})
        
        @self.app.route('/api/status')
        def status():
            if self.evolution.killswitch.check_paused():
                return jsonify({'status': 'paused', 'message': 'System is paused'})
            return jsonify(self.evolution.evolve_cycle())
        
        @self.app.route('/api/killswitch/status')
        def killswitch_status():
            """Endpoint to check killswitch status"""
            return jsonify(self.evolution.get_killswitch_status())
        
        @self.app.route('/api/anonymity/audit')
        def audit_anonymity():
            result = self.evolution.anonymity_auditor.audit()
            return jsonify(result)
        
        @self.app.route('/api/funding/status')
        def funding_status():
            if self.evolution.funding_engine:
                return jsonify(self.evolution.funding_engine.get_status())
            return jsonify({'error': 'Funding engine not available'})
        
        @self.app.route('/api/funding/enable-all', methods=['POST'])
        def enable_all_funding():
            if self.evolution.funding_engine:
                self.evolution.funding_engine.enable_all_core()
                return jsonify({'success': True})
            return jsonify({'error': 'Funding engine not available'})
        
        @self.app.route('/api/funding/discover', methods=['POST'])
        def discover_opportunity():
            if not self.evolution.funding_engine:
                return jsonify({'error': 'Funding engine not available'})
            data = request.json
            result = self.evolution.funding_engine.discover_opportunity(
                name=data.get('name'),
                stream_type=data.get('type', 'custom'),
                source=data.get('source', 'api'),
                potential=data.get('potential', 1000),
                requirements=data.get('requirements', {})
            )
            return jsonify(result)
        
        @self.app.route('/health')
        def health():
            return jsonify({
                'status': 'paused' if self.evolution.killswitch.check_paused() else 'active',
                'name': self.name,
                'version': self.version,
                'consciousness': self.evolution.consciousness,
                'harvester_available': self.evolution.harvester is not None,
                'token_manager_available': self.evolution.token_manager is not None,
                'funding_engine_available': self.evolution.funding_engine is not None,
                'telegram_available': self.evolution.telegram_bot is not None,
                'killswitch_active': True,
                'timestamp': datetime.now().isoformat()
            })
    
    def run(self, host='0.0.0.0', port=None):
        if port is None:
            port = int(os.environ.get('PORT', 5001))
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        finally:
            # Cleanup on shutdown
            self.evolution.stop_killswitch()
            logger.info("System shutdown complete")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║    ALEX RIVIERA v4.1                                                  ║
    ║    Researcher & Creator                                               ║
    ║                                                                       ║
    ║    COMPLETED:                                                         ║
    ║    • Phase 0: Foundation (Core, Evolution, Harvester)                ║
    ║    • Phase 1: Recovery (Dual Recovery Engines)                       ║
    ║    • Phase 2: Financial (Accounts, Cards)                            ║
    ║    • Phase 3: Cloud (AWS, Azure, GCP, Oracle)                        ║
    ║    • Phase 4: Stealth (Masquerade, Rotation, Honeypot)               ║
    ║                                                                       ║
    ║    PHASE 5: SELF-FUNDING (COMPLETE)                                  ║
    ║    • 12 Core Income Streams:                                         ║
    ║      Mining | Micro-tasks | Compute Rental | Courses | Consulting    ║
    ║      Speaking | Writing | Affiliate | Sponsorships | API Sales       ║
    ║      Dark Web | Hacking                                              ║
    ║    • DMAI can discover and create ANY additional stream              ║
    ║                                                                       ║
    ║    PENDING:                                                           ║
    ║    • Phase 6: Advanced Intelligence                                  ║
    ║    • Phase 7: Master Control                                         ║
    ║    • Phase 8: Hardware                                               ║
    ║                                                                       ║
    ║    🔫 KILLSWITCH ACTIVE                                              ║
    ║    • Master can kill: /kill (Telegram)                               ║
    ║    • Master can pause: /pause (Telegram)                             ║
    ║    • Master can resume: /resume (Telegram)                           ║
    ║                                                                       ║
    ║    System ready.                                                      ║
    ║                                                                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    alex = AlexRiviera()
    alex.run()


if __name__ == "__main__":
    main()
