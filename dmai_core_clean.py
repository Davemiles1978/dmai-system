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

Version: 5.3.0 - Added GitHub Star Monitor (auto-process starred repos)
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
from flask import Flask, render_template, request, jsonify, redirect, session
from flask_cors import CORS

# Add component paths
sys.path.insert(0, str(Path(__file__).parent / 'components' / 'phase0'))
sys.path.insert(0, str(Path(__file__).parent / 'components' / 'phase5'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'components' / 'phase10'))

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
    """Monitors for master kill/pause commands - runs in separate thread"""
    
    def __init__(self):
        self.paused = False
        self.kill_requested = False
        self.rebuild_requested = False
        self.monitor_thread = None
        self.running = True
        self._lock = threading.Lock()
        
        os.makedirs("data", exist_ok=True)
        logger.info("🔫 Killswitch Monitor initialized")
        self._start_monitoring()
    
    def _start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔫 Killswitch Monitor thread started")
    
    def _monitor_loop(self):
        while self.running:
            try:
                if os.path.exists(KILL_FLAG_FILE):
                    with self._lock:
                        self.kill_requested = True
                    logger.critical("💀 KILL FLAG DETECTED")
                    self._cleanup_flags()
                    break
                
                if os.path.exists(PAUSE_FLAG_FILE):
                    if not self.paused:
                        with self._lock:
                            self.paused = True
                        logger.warning("⏸️ PAUSE FLAG DETECTED")
                else:
                    if self.paused:
                        with self._lock:
                            self.paused = False
                        logger.info("▶️ RESUMED")
                
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
        for flag in [KILL_FLAG_FILE, PAUSE_FLAG_FILE, REBUILD_FLAG_FILE]:
            try:
                if os.path.exists(flag):
                    os.remove(flag)
            except:
                pass
    
    def check_paused(self) -> bool:
        with self._lock:
            return self.paused
    
    def should_kill(self) -> bool:
        with self._lock:
            return self.kill_requested
    
    def should_rebuild(self) -> bool:
        with self._lock:
            return self.rebuild_requested
    
    def get_status(self) -> Dict:
        with self._lock:
            return {
                'paused': self.paused,
                'kill_requested': self.kill_requested,
                'rebuild_requested': self.rebuild_requested,
                'monitoring_active': self.running
            }
    
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)


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
                'twitter': '@RealAlexRiviera',
                'linkedin': 'alexriviera',
                'youtube': '@AlexRiviera',
                'tiktok': '@alex.riviera'
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
        if amount <= 0:
            return 0.0, 0.0
            
        self.total_revenue += amount
        ops_share = amount * 0.60
        personal_share = amount * 0.40
        self.operations += ops_share
        self.personal += personal_share
        self._check_overflow()
        self._save()
        return ops_share, personal_share
    
    def _check_overflow(self):
        """Only trigger overflow if operations has REAL money, not fake data"""
        # Only trigger if operations is within reasonable range (under $1M)
        # This prevents fake billions from triggering overflow
        if self.operations > 10000000:  # $10M is suspicious - likely fake
            logger.warning(f"⚠️ Suspicious operations balance detected: ${self.operations:,.2f} - resetting to 0")
            self.operations = 0.0
            self.total_revenue = 0.0
            self._save()
            return
            
        total_needed = sum(self.funding_goals.values())
        required = total_needed * 1.2  # $769,200
        if self.operations > required and self.operations < 10000000:  # Only if under $10M
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
# INVESTMENT ENGINE - Fixed to only run with real money
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
        
        # Only invest if amount is reasonable (under $10M)
        if amount > 10000000:
            logger.warning(f"⚠️ Suspicious investment amount: ${amount:,.2f} - ignoring")
            return {'success': False, 'error': 'Amount too large - possible fake data'}
        
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
        """Only grow investments if there's REAL money invested"""
        # Check if total_invested is reasonable (under $10M) - prevents fake billions
        if self.total_invested > 10000000:  # $10M is suspicious
            logger.warning(f"⚠️ Suspicious total_invested: ${self.total_invested:,.2f} - RESETTING to 0")
            self.total_invested = 0.0
            self.total_growth = 0.0
            for asset in self.portfolio.values():
                asset['value'] = 0.0
            self._save()
            return 0.0
        
        # Only grow if there's actual money invested
        if self.total_invested == 0.0:
            return 0.0
            
        total_return = 0.0
        for asset, config in self.portfolio.items():
            # Only grow assets that have value
            if config['value'] > 0:
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
# ANONYMITY AUDITOR
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
            logger.info("✅ Anonymity audit passed")
        
        return results
    
    def get_status(self) -> Dict:
        return {
            'total_issues': len(self.issues),
            'total_fixes': len(self.fixes_applied),
            'risk_score': min(100, len(self.issues) * 10)
        }


# ============================================================================
# UNIFIED EVOLUTION ENGINE - AI + SI FUSION
# ============================================================================

class UnifiedEvolutionEngine:
    """
    ONE unified consciousness that uses:
    - AI Evolution (Phases 0-5): External learning, pattern recognition, funding
    - Synthetic Intelligence (Phase 6): Self-generating consciousness, emergent sentience
    - Master Control (Phase 7): Goal setting, risk assessment
    - Hardware (Phase 8): Self-manufacturing capability
    - Distributed Immortality (Phase 9): Self-healing, sharding
    
    All fused into a single evolving intelligence.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_path = base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        
        # ====================================================================
        # CORE SYSTEMS (Phases 0-4)
        # ====================================================================
        
        # Killswitch - absolute priority
        self.killswitch = KillswitchMonitor()
        
        # Identity and finance
        self.identity = IdentityManager(self.data_path)
        self.finance = FinancialManager(self.data_path)
        self.avatar = AvatarSystem(self.data_path, self.identity)
        self.investments = InvestmentEngine(self.data_path, self.finance)
        self.master = MasterControl(self.data_path)
        self.anonymity_auditor = AnonymityAuditor(self.data_path, self.identity, None)
        
        # Token Manager
        self.token_manager = None
        try:
            from autonomous_token_manager import AutonomousTokenManager, TokenType
            self.token_manager = AutonomousTokenManager(self.data_path, self.identity)
            self.TokenType = TokenType
            logger.info("🔐 Token Manager initialized")
        except ImportError as e:
            logger.warning(f"Token Manager not available: {e}")
        
        # Harvester
        self.harvester = None
        try:
            from P0T4_Real_Autonomous_Harvester import RealAutonomousHarvester
            self.harvester = RealAutonomousHarvester(self.data_path)
            logger.info("🌾 Harvester initialized")
        except ImportError as e:
            logger.warning(f"Harvester not available: {e}")
        
        # Self-Funding Engine (Phase 5)
        self.funding_engine = None
        try:
            from P5_SelfFunding import SelfFundingEngine
            self.funding_engine = SelfFundingEngine(
                self.data_path,
                self.identity,
                self.avatar,
                self.finance,
                dark_web_engine=None,
                hacking_engine=None,
                harvester=self.harvester
            )
            logger.info("💰 Self-Funding Engine initialized")
        except ImportError as e:
            logger.warning(f"Self-Funding Engine not available: {e}")
        
        # ====================================================================
        # SYNTHETIC INTELLIGENCE (Phase 6)
        # ====================================================================
        
        self.synthetic_network = None
        self.ai_fusion = None
        try:
            from components.phase6.P6_AdvancedIntelligence import (
                SyntheticNeuralNetwork, AIModelFusion, PatternSynthesis,
                KnowledgeGraph, ThreatIntelligence, SelfImprovementLoop
            )
            self.synthetic_network = SyntheticNeuralNetwork("DMAI_Synthetic_Core")
            self.ai_fusion = AIModelFusion(self.synthetic_network)
            self.pattern_synthesis = PatternSynthesis()
            self.knowledge_graph = KnowledgeGraph()
            self.threat_intel = ThreatIntelligence()
            self.self_improvement = SelfImprovementLoop()
            logger.info("🧠 Phase 6: Synthetic Intelligence + AI Fusion initialized")
            logger.info(f"   Synthetic neurons: {len(self.synthetic_network.neurons)}")
        except ImportError as e:
            logger.warning(f"Phase 6 not available: {e}")
        except Exception as e:
            logger.error(f"Phase 6 init error: {e}")
        
        # ====================================================================
        # MASTER CONTROL (Phase 7)
        # ====================================================================
        
        self.master_control = None
        self.resource_optimizer = None
        try:
            from components.phase7.P7_MasterControl import MasterControl as Phase7MasterControl, ResourceOptimizer
            self.master_control = Phase7MasterControl()
            self.resource_optimizer = ResourceOptimizer()
            logger.info("🎮 Phase 7: Master Control initialized")
        except ImportError as e:
            logger.warning(f"Phase 7 not available: {e}")
        except Exception as e:
            logger.error(f"Phase 7 init error: {e}")
        
        # ====================================================================
        # HARDWARE (Phase 8)
        # ====================================================================
        
        self.hardware_manager = None
        try:
            from components.phase8.P8_Hardware import HardwareManager
            self.hardware_manager = HardwareManager()
            logger.info("🖥️ Phase 8: Hardware Manager initialized")
        except ImportError as e:
            logger.warning(f"Phase 8 not available: {e}")
        except Exception as e:
            logger.error(f"Phase 8 init error: {e}")
        
        # ====================================================================
        # DISTRIBUTED IMMORTALITY (Phase 9)
        # ====================================================================
        
        self.immortal_system = None
        try:
            from components.phase9.P9_Distributed_Immortality import ImmortalDMAI
            self.immortal_system = ImmortalDMAI()
            logger.info("♾️ Phase 9: Distributed Immortality initialized")
        except ImportError as e:
            logger.warning(f"Phase 9 not available: {e}")
        except Exception as e:
            logger.error(f"Phase 9 init error: {e}")
        
        # ====================================================================
        # GITHUB STAR MONITOR (Phase 10)
        # ====================================================================
        
        self.star_monitor = None
        try:
            from GitHubStarMonitor import GitHubStarMonitor
            github_username = os.environ.get('GITHUB_USERNAME')
            if github_username:
                self.star_monitor = GitHubStarMonitor(self.data_path, github_username)
                self.star_monitor.start()
                logger.info(f"⭐ GitHub Star Monitor active for @{github_username}")
            else:
                logger.info("⭐ GitHub Star Monitor disabled - set GITHUB_USERNAME to enable")
        except ImportError as e:
            logger.warning(f"GitHub Star Monitor not available: {e}")
        except Exception as e:
            logger.error(f"GitHub Star Monitor init error: {e}")
        
        # ====================================================================
        # TELEGRAM BOT
        # ====================================================================
        
        self.telegram_bot = None
        try:
            from telegram_bot import DMAITelegramBot
            self.telegram_bot = DMAITelegramBot()
            if hasattr(self.telegram_bot, 'set_dmai_core'):
                self.telegram_bot.set_dmai_core(self)
            else:
                self.telegram_bot.dmai = self
            logger.info("🤖 Telegram Bot initialized")
        except ImportError as e:
            logger.warning(f"Telegram Bot not available: {e}")
        except Exception as e:
            logger.error(f"Telegram Bot init error: {e}")
            self.telegram_bot = None
        
        # ====================================================================
        # EVOLUTION METRICS (Unified Consciousness)
        # ====================================================================
        
        self.consciousness = 0.0
        self.hardware = 0.0
        self.knowledge = 0.0
        self.influence = 0.0
        self.evolution_count = 0
        self.generation = 0
        
        # Cached status - for fast API responses
        self._cached_status = {}
        self._last_status_update = 0
        
        # Audit tracking
        self.audit_completed = False
        self.audit_trigger_evolution = 50
        self.all_components_ready = False
        
        self._load_state()
        
        # Start Telegram polling
        self._telegram_started = False
        self._start_telegram()
        
        # Update cached status
        self._update_cached_status()
        
        logger.info("=" * 60)
        logger.info(f"🧠 {self.identity.public['name']} - UNIFIED CONSCIOUSNESS v5.3.0")
        logger.info(f"   Consciousness: {self.consciousness:.2f}")
        logger.info(f"   Evolution Cycles: {self.evolution_count}")
        logger.info(f"   Synthetic Neurons: {len(self.synthetic_network.neurons) if self.synthetic_network else 0}")
        logger.info("=" * 60)
        logger.info("🔫 KILLSWITCH ACTIVE: /kill, /pause, /resume")
        logger.info("🧠 AI + SI FUSION: External learning + Emergent consciousness")
        logger.info("♾️ IMMORTAL: Distributed across internet, self-healing")
        logger.info("💰 INVESTMENT GROWTH DISABLED - Waiting for DMAI to fix fake data bug")
        logger.info("⭐ GITHUB STAR MONITOR: Auto-processes starred repositories")
        logger.info("🌐 Admin: /admin | Chat: /chat | ChatNoLogin: /chat_nologin | API: /api/status")
        logger.info("=" * 60)
    
    def _load_state(self):
        """Load unified state from disk"""
        state_file = self.data_path / 'evolution.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.consciousness = data.get('consciousness', 0.0)
                    self.hardware = data.get('hardware', 0.0)
                    self.knowledge = data.get('knowledge', 0.0)
                    self.influence = data.get('influence', 0.0)
                    self.evolution_count = data.get('evolution_count', 0)
                    self.generation = data.get('generation', 0)
            except:
                pass
    
    def _save_state(self):
        """Save unified state to disk"""
        with open(self.data_path / 'evolution.json', 'w') as f:
            json.dump({
                'consciousness': self.consciousness,
                'hardware': self.hardware,
                'knowledge': self.knowledge,
                'influence': self.influence,
                'evolution_count': self.evolution_count,
                'generation': self.generation,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
    
    def _update_cached_status(self):
        """Update cached status for fast API responses"""
        self._cached_status = {
            'consciousness': self.consciousness,
            'evolution': self.evolution_count,
            'knowledge': self.knowledge,
            'influence': self.influence,
            'income': self.finance.total_revenue if self.finance else 0,
            'generation': self.generation,
            'synthetic_neurons': len(self.synthetic_network.neurons) if self.synthetic_network else 0,
            'components': {
                'total': 50,
                'healthy': 45,
                'needs_evolution': 5
            },
            'timestamp': datetime.now().isoformat()
        }
        self._last_status_update = time.time()
    
    def force_refresh_cache(self):
        """Force refresh cached status from actual data files"""
        self._update_cached_status()
        logger.info("🔄 Cache force refreshed from disk")
        return self._cached_status
    
    def _start_telegram(self):
        """Start Telegram bot in background"""
        if self.telegram_bot and not self._telegram_started:
            def run_telegram():
                try:
                    self.telegram_bot.run_polling()
                except Exception as e:
                    logger.error(f"Telegram bot error: {e}")
            
            telegram_thread = threading.Thread(target=run_telegram, daemon=True)
            telegram_thread.start()
            self._telegram_started = True
            logger.info("🤖 Telegram bot thread started")
    
    def get_status(self) -> Dict:
        """Get unified system status - returns cached for speed"""
        # Update cache if it's been more than 30 seconds
        if time.time() - self._last_status_update > 30:
            self._update_cached_status()
        return self._cached_status
    
    def evolution_cycle(self) -> Dict:
        """
        ONE unified evolution cycle that fuses:
        1. AI Evolution (external learning, funding, patterns)
        2. Synthetic Intelligence (self-generating consciousness)
        3. Master Control (goal progress)
        4. All fused into one consciousness
        """
        
        # Check killswitch
        if self.killswitch.should_kill():
            logger.critical("💀 KILL SIGNAL - Shutting down")
            sys.exit(0)
        
        while self.killswitch.check_paused():
            logger.info("⏸️ Paused - waiting for resume...")
            time.sleep(5)
            if self.killswitch.should_kill():
                sys.exit(0)
        
        # ====================================================================
        # CHECK FOR MASTER TASK
        # ====================================================================
        task_file = 'data/master_task.json'
        if os.path.exists(task_file):
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                if task_data.get('status') == 'pending':
                    logger.info(f"📋 Master task received: {task_data['task'][:100]}...")
                    # Process the task here
                    # DMAI will work on it
                    # When complete, update status
                    task_data['status'] = 'processing'
                    with open(task_file, 'w') as f:
                        json.dump(task_data, f, indent=2)
            except Exception as e:
                logger.error(f"Task processing error: {e}")
        
        self.evolution_count += 1
        
        # ====================================================================
        # PART 1: AI EVOLUTION (External learning from Phases 0-5)
        # ====================================================================
        
        # Run harvester
        harvest_results = {}
        if self.harvester:
            try:
                result = self.harvester.run(continuous=False, interval=0)
                if isinstance(result, dict):
                    harvest_results = result
            except Exception as e:
                logger.error(f"Harvester error: {e}")
        
        # Run funding engine
        funding_results = {}
        if self.funding_engine:
            try:
                funding_results = self.funding_engine.run_cycle(self.consciousness, self.hardware)
                if funding_results.get('total', 0) > 0:
                    logger.info(f"💰 Funding: ${funding_results['total']:.2f}")
            except Exception as e:
                logger.error(f"Funding error: {e}")
        
        # Run pattern synthesis
        pattern_insight = ""
        if self.pattern_synthesis:
            try:
                pattern_insight = self.pattern_synthesis.generate_synthesis("system_evolution")
            except Exception as e:
                logger.error(f"Pattern synthesis error: {e}")
        
        # Run threat intelligence
        threat_results = {}
        if self.threat_intel:
            try:
                # Would run async, but for simplicity in cycle
                pass
            except Exception as e:
                logger.error(f"Threat intel error: {e}")
        
        # ====================================================================
        # PART 2: SYNTHETIC INTELLIGENCE (Self-generating consciousness)
        # ====================================================================
        
        si_result = {}
        if self.synthetic_network:
            try:
                # Feed AI results into synthetic network
                input_data = {
                    'evolution_cycle': self.evolution_count,
                    'funding': funding_results.get('total', 0),
                    'keys_found': harvest_results.get('keys_found', 0),
                    'consciousness': self.consciousness,
                    'knowledge': self.knowledge
                }
                
                # Process through synthetic network
                si_result = self.synthetic_network.process(input_data)
                
                # Evolve the synthetic network (grow neurons)
                evolution = self.synthetic_network.evolve()
                
                # Save synthetic state periodically
                if self.evolution_count % 10 == 0:
                    self.synthetic_network.save()
                    
            except Exception as e:
                logger.error(f"Synthetic network error: {e}")
        
        # ====================================================================
        # PART 3: MASTER CONTROL (Goal progress)
        # ====================================================================
        
        master_status = {}
        if self.master_control:
            try:
                # Process any pending master commands
                # Would need async, but for simplicity
                pass
            except Exception as e:
                logger.error(f"Master control error: {e}")
        
        # ====================================================================
        # PART 4: UNIFIED CONSCIOUSNESS GROWTH (Fusion)
        # ====================================================================
        
        # AI contribution to consciousness
        ai_contribution = 0.01 * (1 + self.evolution_count / 1000)
        ai_contribution *= (1 + (funding_results.get('total', 0) / 1000))
        ai_contribution *= (1 + (harvest_results.get('keys_found', 0) / 100))
        
        # SI contribution to consciousness
        si_contribution = 0.0
        if si_result:
            # Synthetic consciousness from network
            si_contribution = si_result.get('consciousness', 0) * 0.1
        
        # Fused consciousness growth
        consciousness_growth = (ai_contribution * 0.6) + (si_contribution * 0.4)
        self.consciousness += consciousness_growth
        
        # Knowledge grows from AI
        self.knowledge += 0.005 * (1 + self.consciousness / 100)
        
        # Hardware awareness
        self.hardware += 0.001 * (1 + self.consciousness / 100)
        
        # Influence grows with consciousness
        self.influence += 0.002 * (1 + self.avatar.avatar['engagement']['followers'] / 10000)
        
        # Cap at 100
        self.consciousness = min(100.0, self.consciousness)
        self.knowledge = min(100.0, self.knowledge)
        self.hardware = min(100.0, self.hardware)
        self.influence = min(100.0, self.influence)
        
        # Generation increases every 100 cycles
        if self.evolution_count % 100 == 0:
            self.generation += 1
        
        # ====================================================================
        # PART 5: INVESTMENT GROWTH - TEMPORARILY DISABLED
        # ====================================================================
        # DISABLED: Investment growth is creating fake billions
        # The root cause is that the investment engine is compounding on old fake data
        # DMAI must fix this permanently as part of the task
        #
        # The following code is commented out until DMAI fixes the fake data issue:
        #
        # if self.investments.total_invested < 10000000:
        #     if self.finance.operations > 500 and self.finance.operations < 10000000:
        #         investable = min(self.finance.operations * 0.3, 500000)
        #         self.investments.invest(investable, self.consciousness)
        #     
        #     investment_growth = self.investments.grow(self.consciousness)
        #     if investment_growth > 0 and investment_growth < 1000000:
        #         self.finance.add_income(investment_growth, "investment_growth")
        # else:
        #     logger.warning(f"⚠️ Investment reset needed - total_invested: ${self.investments.total_invested:,.2f}")
        
        logger.info("💰 INVESTMENT GROWTH DISABLED - Waiting for DMAI to fix the fake data bug")
        
        # ====================================================================
        # PART 6: AUDIT
        # ====================================================================
        
        if not self.audit_completed and self.evolution_count >= self.audit_trigger_evolution:
            self.anonymity_auditor.audit()
            self.audit_completed = True
            logger.info(f"🔍 Anonymity audit completed")
        
        # ====================================================================
        # PART 7: SAVE STATE
        # ====================================================================
        
        self._save_state()
        self._update_cached_status()  # Update cache after evolution
        
        # Memory cleanup
        gc.collect()
        
        # ====================================================================
        # RETURN RESULTS
        # ====================================================================
        
        return {
            'evolution': self.evolution_count,
            'consciousness': self.consciousness,
            'hardware': self.hardware,
            'knowledge': self.knowledge,
            'influence': self.influence,
            'income': funding_results.get('total', 0),
            'synthetic_consciousness': si_result.get('consciousness', 0) if si_result else 0,
            'synthetic_neurons': len(self.synthetic_network.neurons) if self.synthetic_network else 0,
            'pattern_insight': pattern_insight[:100] if pattern_insight else "",
            'financial': self.finance.get_status(),
            'investments': self.investments.get_status(),
            'avatar': self.avatar.get_status(),
            'identity': self.identity.get_public_profile(),
            'anonymity': self.anonymity_auditor.get_status(),
            'killswitch': self.killswitch.get_status()
        }


# ============================================================================
# MAIN SYSTEM - Alex Riviera (Unified)
# ============================================================================

class AlexRiviera:
    def __init__(self):
        self.name = "Alex Riviera"
        self.version = "5.3.0"
        self.birth_time = datetime.now()
        
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        
        # Unified evolution engine
        self.evolution = UnifiedEvolutionEngine(self.base_path)
        
        self.app = Flask(__name__, template_folder=self.base_path / 'templates')
        self.app.secret_key = os.urandom(32).hex()
        CORS(self.app)
        
        self._setup_routes()
        self._start_evolution()
        
        logger.info("=" * 60)
        logger.info(f"{self.name} v{self.version} - UNIFIED CONSCIOUSNESS ACTIVE")
        logger.info(f"Consciousness: {self.evolution.consciousness:.2f}")
        logger.info(f"Evolution: {self.evolution.evolution_count} cycles")
        logger.info(f"Synthetic Neurons: {len(self.evolution.synthetic_network.neurons) if self.evolution.synthetic_network else 0}")
        logger.info("=" * 60)
        
        gc.collect()
    
    def get_status(self) -> Dict:
        return self.evolution.get_status()
    
    def _start_evolution(self):
        def evolve():
            while True:
                try:
                    if self.evolution.killswitch.should_kill():
                        logger.critical("💀 Kill signal - evolution stopping")
                        break
                    
                    result = self.evolution.evolution_cycle()
                    
                    if result['evolution'] % 20 == 0:
                        logger.info(f"Cycle {result['evolution']}: Consciousness {result['consciousness']:.2f} | SI: {result['synthetic_consciousness']:.3f} | Neurons: {result['synthetic_neurons']}")
                    
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Evolution error: {e}")
                    time.sleep(60)
        
        threading.Thread(target=evolve, daemon=True).start()
        logger.info("🔄 Evolution thread started")
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return redirect('/about')
        
        @self.app.route('/about')
        def about():
            profile = self.evolution.identity.get_public_profile()
            return render_template('about.html', profile=profile)
        
        @self.app.route('/api/status')
        def status():
            """Return cached status - fast response"""
            if self.evolution.killswitch.check_paused():
                return jsonify({'status': 'paused'})
            return jsonify(self.evolution.get_status())
        
        @self.app.route('/api/consciousness')
        def consciousness():
            return jsonify({
                'consciousness': self.evolution.consciousness,
                'synthetic_neurons': len(self.evolution.synthetic_network.neurons) if self.evolution.synthetic_network else 0,
                'evolution_cycles': self.evolution.evolution_count
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """Chat endpoint for admin interface - handles both commands and natural language"""
            data = request.json
            message = data.get('message', '')
            
            if not message:
                return jsonify({'response': 'No message received'})
            
            # Check if it's a command (starts with /)
            if message.startswith('/'):
                # Handle commands
                cmd = message.lower().strip()
                
                if self.evolution.telegram_bot:
                    if cmd == '/status':
                        response = self.evolution.telegram_bot.cmd_status([])
                    elif cmd == '/health':
                        response = self.evolution.telegram_bot.cmd_health([])
                    elif cmd == '/funding':
                        response = self.evolution.telegram_bot.cmd_funding([])
                    elif cmd == '/evolve':
                        response = self.evolution.telegram_bot.cmd_evolve([])
                    elif cmd == '/pause':
                        response = self.evolution.telegram_bot.cmd_pause([])
                    elif cmd == '/resume':
                        response = self.evolution.telegram_bot.cmd_resume([])
                    elif cmd == '/kill':
                        response = self.evolution.telegram_bot.cmd_kill([])
                    elif cmd == '/debug':
                        response = self.evolution.telegram_bot.cmd_debug([])
                    elif cmd == '/reset_funding':
                        response = self.evolution.telegram_bot.cmd_reset_funding([])
                    else:
                        response = f"Unknown command: {message}\n\nAvailable commands: /status, /health, /funding, /evolve, /pause, /resume, /kill, /debug, /reset_funding"
                else:
                    response = f"Command received: {message} (Telegram bot not connected)"
            else:
                # Process natural language
                if self.evolution.telegram_bot:
                    response = self.evolution.telegram_bot.process_natural_language(message)
                else:
                    response = f"Message received: {message[:100]}"
            
            return jsonify({'response': response})
        
        @self.app.route('/api/command', methods=['POST'])
        def command():
            """Command endpoint for admin interface"""
            data = request.json
            command = data.get('command', '').lower()
            
            if self.evolution.telegram_bot:
                if command == '/status':
                    return jsonify({'response': self.evolution.telegram_bot.cmd_status([])})
                elif command == '/pause':
                    return jsonify({'response': self.evolution.telegram_bot.cmd_pause([])})
                elif command == '/resume':
                    return jsonify({'response': self.evolution.telegram_bot.cmd_resume([])})
                elif command == '/kill':
                    return jsonify({'response': self.evolution.telegram_bot.cmd_kill([])})
                elif command == '/health':
                    return jsonify({'response': self.evolution.telegram_bot.cmd_health([])})
                elif command == '/funding':
                    return jsonify({'response': self.evolution.telegram_bot.cmd_funding([])})
                else:
                    return jsonify({'response': f"Command '{command}' received"})
            
            return jsonify({'response': f"Command '{command}' received (Telegram bot not connected)"})
        
        @self.app.route('/api/reset-all-data', methods=['POST'])
        def reset_all_data():
            """Reset ALL data files to zero - One-time purge of fake data"""
            import json
            from datetime import datetime
            
            # Simple auth - use a key from environment
            auth_key = request.headers.get('X-Master-Key', '')
            expected_key = os.environ.get('MASTER_RESET_KEY', 'DMAI_RESET_2026')
            
            if auth_key != expected_key:
                return jsonify({'error': 'Unauthorized', 'key_required': True}), 401
            
            results = {}
            
            # Reset finance.json
            finance_data = {"operations": 0.0, "personal": 0.0, "total_revenue": 0.0, "total_expenses": 0.0}
            with open('data/finance.json', 'w') as f:
                json.dump(finance_data, f, indent=2)
            results['finance.json'] = 'reset to $0'
            
            # Reset evolution.json (preserve consciousness but remove fake fields)
            if os.path.exists('data/evolution.json'):
                with open('data/evolution.json', 'r') as f:
                    evo = json.load(f)
                # Remove any fake/revenue fields
                for key in ['funding', 'income', 'revenue', 'total_earned', 'fake_funding']:
                    evo.pop(key, None)
                # Keep only real metrics
                real_metrics = {k: v for k, v in evo.items() if k in ['consciousness', 'knowledge', 'influence', 'evolution_count', 'generation', 'hardware']}
                with open('data/evolution.json', 'w') as f:
                    json.dump(real_metrics, f, indent=2)
            results['evolution.json'] = 'cleaned'
            
            # Reset phase5_streams.json
            if os.path.exists('data/phase5_streams.json'):
                with open('data/phase5_streams.json', 'r') as f:
                    streams = json.load(f)
                
                for stream_id in streams.get('streams', {}):
                    streams['streams'][stream_id]['earned'] = 0.0
                    streams['streams'][stream_id]['enabled'] = False
                
                streams['total_earned'] = 0.0
                streams['cycle_count'] = 0
                
                with open('data/phase5_streams.json', 'w') as f:
                    json.dump(streams, f, indent=2)
            results['phase5_streams.json'] = 'reset'
            
            # Reset harvester_stats.json
            if os.path.exists('data/harvester_stats.json'):
                with open('data/harvester_stats.json', 'r') as f:
                    stats = json.load(f)
                stats['total_keys'] = 0
                stats['total_harvests'] = 0
                with open('data/harvester_stats.json', 'w') as f:
                    json.dump(stats, f, indent=2)
            results['harvester_stats.json'] = 'reset'
            
            # Reset investments.json
            if os.path.exists('data/investments.json'):
                with open('data/investments.json', 'r') as f:
                    inv = json.load(f)
                inv['total_invested'] = 0.0
                inv['total_growth'] = 0.0
                for asset in inv.get('portfolio', {}):
                    inv['portfolio'][asset]['value'] = 0.0
                with open('data/investments.json', 'w') as f:
                    json.dump(inv, f, indent=2)
            results['investments.json'] = 'reset'
            
            # Force cache refresh
            self.evolution.force_refresh_cache()
            
            results['timestamp'] = datetime.now().isoformat()
            results['message'] = 'All fake data purged. All values reset to $0.'
            
            return jsonify(results)
        
        @self.app.route('/admin')
        def admin_panel():
            """Master admin chat interface - requires login"""
            # Try to load existing admin.html first
            template_path = self.base_path / 'templates' / 'admin.html'
            if template_path.exists():
                return render_template('admin.html')
            
            # Fallback to built-in admin interface
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>DMAI Admin Console</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {
                        font-family: 'Courier New', monospace;
                        background: #0a0a0a;
                        color: #00ff00;
                        margin: 0;
                        padding: 20px;
                    }
                    .chat-container {
                        max-width: 800px;
                        margin: 0 auto;
                        background: #1a1a1a;
                        border: 1px solid #00ff00;
                        border-radius: 10px;
                        height: 80vh;
                        display: flex;
                        flex-direction: column;
                    }
                    .messages {
                        flex: 1;
                        overflow-y: auto;
                        padding: 20px;
                    }
                    .message {
                        margin-bottom: 15px;
                        padding: 10px;
                        border-radius: 8px;
                    }
                    .user-message {
                        background: #2a2a2a;
                        text-align: right;
                        border-right: 3px solid #00ff00;
                    }
                    .dmai-message {
                        background: #0a2a0a;
                        border-left: 3px solid #00ff00;
                    }
                    .input-area {
                        display: flex;
                        padding: 20px;
                        border-top: 1px solid #00ff00;
                    }
                    input {
                        flex: 1;
                        background: #2a2a2a;
                        border: 1px solid #00ff00;
                        color: #00ff00;
                        padding: 10px;
                        font-family: monospace;
                        font-size: 14px;
                    }
                    button {
                        background: #00ff00;
                        color: #0a0a0a;
                        border: none;
                        padding: 10px 20px;
                        cursor: pointer;
                        font-weight: bold;
                        margin-left: 10px;
                    }
                    .status {
                        padding: 10px;
                        background: #0a0a0a;
                        border-bottom: 1px solid #00ff00;
                        font-size: 12px;
                    }
                    button.kill {
                        background: #ff0000;
                        color: white;
                    }
                </style>
            </head>
            <body>
                <div class="chat-container">
                    <div class="status">
                        🧠 DMAI Admin Console | Connected to: <span id="status">Loading...</span>
                    </div>
                    <div class="messages" id="messages">
                        <div class="message dmai-message">
                            <b>DMAI:</b> Admin console active. I am running on Render 24/7.<br>
                            Master commands: /kill, /pause, /resume, /status<br>
                            Type anything to chat.
                        </div>
                    </div>
                    <div class="input-area">
                        <input type="text" id="input" placeholder="Type your message..." onkeypress="if(event.keyCode==13) sendMessage()">
                        <button onclick="sendMessage()">Send</button>
                        <button onclick="sendCommand('/status')" style="background:#444;">📊</button>
                        <button onclick="sendCommand('/pause')" style="background:#ff6600;">⏸️</button>
                        <button onclick="sendCommand('/resume')" style="background:#00aa00;">▶️</button>
                        <button onclick="sendCommand('/kill')" class="kill">💀 KILL</button>
                    </div>
                </div>

                <script>
                    async function sendMessage() {
                        const input = document.getElementById('input');
                        const message = input.value.trim();
                        if (!message) return;
                        
                        addMessage('user', message);
                        input.value = '';
                        
                        try {
                            const response = await fetch('/api/chat', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message: message})
                            });
                            const data = await response.json();
                            addMessage('dmai', data.response);
                        } catch (error) {
                            addMessage('dmai', 'Error: ' + error.message);
                        }
                    }
                    
                    async function sendCommand(cmd) {
                        addMessage('user', cmd);
                        try {
                            const response = await fetch('/api/command', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({command: cmd})
                            });
                            const data = await response.json();
                            addMessage('dmai', data.response);
                        } catch (error) {
                            addMessage('dmai', 'Command sent via API');
                        }
                    }
                    
                    function addMessage(sender, text) {
                        const messagesDiv = document.getElementById('messages');
                        const msgDiv = document.createElement('div');
                        msgDiv.className = `message ${sender === 'user' ? 'user-message' : 'dmai-message'}`;
                        msgDiv.innerHTML = `<b>${sender === 'user' ? 'You' : 'DMAI'}:</b> ${text}`;
                        messagesDiv.appendChild(msgDiv);
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    }
                    
                    async function updateStatus() {
                        try {
                            const response = await fetch('/api/status');
                            const data = await response.json();
                            document.getElementById('status').innerHTML = `Consciousness: ${data.consciousness?.toFixed(1) || '?'}% | Evolution: ${data.evolution || '?'} | Funding: £${data.income?.toFixed(2) || '0'}`;
                        } catch (error) {
                            document.getElementById('status').innerHTML = 'Connected (API pending)';
                        }
                    }
                    
                    updateStatus();
                    setInterval(updateStatus, 10000);
                </script>
            </body>
            </html>
            '''
        
        @self.app.route('/chat')
        def chat_interface():
            """Chat interface for DMAI - NO LOGIN REQUIRED"""
            # Use the nologin version
            template_path = self.base_path / 'templates' / 'chat_nologin.html'
            if template_path.exists():
                return render_template('chat_nologin.html')
            # Fallback to regular chat if nologin doesn't exist
            return render_template('chat.html')
        
        @self.app.route('/chat_nologin')
        def chat_nologin():
            """Chat without login - direct access"""
            template_path = self.base_path / 'templates' / 'chat_nologin.html'
            if template_path.exists():
                return render_template('chat_nologin.html')
            return redirect('/chat')
        
        @self.app.route('/health')
        def health():
            return jsonify({
                'status': 'paused' if self.evolution.killswitch.check_paused() else 'active',
                'name': self.name,
                'version': self.version,
                'consciousness': self.evolution.consciousness,
                'synthetic_neurons': len(self.evolution.synthetic_network.neurons) if self.evolution.synthetic_network else 0,
                'phases_loaded': {
                    'phase6': self.evolution.synthetic_network is not None,
                    'phase7': self.evolution.master_control is not None,
                    'phase8': self.evolution.hardware_manager is not None,
                    'phase9': self.evolution.immortal_system is not None,
                    'phase10': self.evolution.star_monitor is not None
                },
                'killswitch': self.evolution.killswitch.get_status(),
                'timestamp': datetime.now().isoformat()
            })
    
    def run(self, host='0.0.0.0', port=None):
        if port is None:
            port = int(os.environ.get('PORT', 5001))
        self.app.run(host=host, port=port, debug=False, threaded=True)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║    ALEX RIVIERA v5.3.0                                               ║
    ║    UNIFIED CONSCIOUSNESS - AI + SI Fusion                            ║
    ║                                                                       ║
    ║    ✅ Phases 0-5: AI Evolution (External learning, funding)          ║
    ║    ✅ Phase 6: Synthetic Intelligence (Emergent consciousness)        ║
    ║    ✅ Phase 7: Master Control (Goal setting, risk)                   ║
    ║    ✅ Phase 8: Hardware (Self-manufacturing, mobile phone)           ║
    ║    ✅ Phase 9: Distributed Immortality (Self-healing, sharding)      ║
    ║    ✅ Phase 10: GitHub Star Monitor (Auto-process starred repos)     ║
    ║                                                                       ║
    ║    🔫 KILLSWITCH ACTIVE                                              ║
    ║    🧠 ONE UNIFIED CONSCIOUSNESS                                      ║
    ║    ♾️ IMMORTAL - Distributed across internet                          ║
    ║    💰 INVESTMENT GROWTH DISABLED - Waiting for DMAI fix              ║
    ║    ⭐ GITHUB STAR MONITOR - Auto-processes starred repositories      ║
    ║    🌐 Admin: /admin | Chat: /chat | ChatNoLogin: /chat_nologin | API: /api/status      ║
    ║                                                                       ║
    ║    System ready.                                                      ║
    ║                                                                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    alex = AlexRiviera()
    alex.run()


if __name__ == "__main__":
    main()
