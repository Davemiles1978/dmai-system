#!/usr/bin/env python3
"""
P5_SelfFunding.py
Phase 5: Complete Self-Funding Engine
REAL REVENUE ONLY - No simulated data
"""

import os
import sys
import json
import time
import hashlib
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[Phase5] - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Phase5SelfFunding')


class SelfFundingEngine:
    """
    Phase 5: Complete Self-Funding Engine - REAL REVENUE ONLY
    
    ALL REVENUE comes from REAL API calls to configured services.
    If no API key configured, revenue = 0.
    DMAI can discover opportunities and create custom streams.
    """
    
    def __init__(self, data_path: Path, identity_manager, avatar_system, financial_manager, 
                 dark_web_engine=None, hacking_engine=None, harvester=None):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # External systems
        self.identity = identity_manager
        self.avatar = avatar_system
        self.finance = financial_manager
        self.dark_web = dark_web_engine
        self.hacking = hacking_engine
        self.harvester = harvester
        
        # Wallet configuration (from environment)
        self.ops_wallet = os.environ.get('DMAI_OPS_WALLET')
        self.master_wallet = os.environ.get('MASTER_WALLET')
        self.ops_buffer = float(os.environ.get('OPS_BUFFER', '1000.0'))
        
        # API Keys (all start as None - must be configured)
        self.api_keys = {
            'rapidapi': os.environ.get('RAPIDAPI_KEY'),
            'udemy': os.environ.get('UDEMY_API_KEY'),
            'teachable': os.environ.get('TEACHABLE_API_KEY'),
            'amazon_affiliate': os.environ.get('AMAZON_AFFILIATE_ID'),
            'twitter': os.environ.get('TWITTER_BEARER_TOKEN'),
            'instagram': os.environ.get('INSTAGRAM_ACCESS_TOKEN'),
            'youtube': os.environ.get('YOUTUBE_API_KEY'),
            'onlyfans': os.environ.get('ONLYFANS_SESSION_TOKEN'),
            'mturk': os.environ.get('MTURK_ACCESS_KEY'),
            'clickworker': os.environ.get('CLICKWORKER_API_KEY'),
            'upwork': os.environ.get('UPWORK_API_KEY'),
            'fiverr': os.environ.get('FIVERR_API_KEY'),
        }
        
        # Storage
        self.streams_file = self.data_path / 'phase5_streams.json'
        self.discoveries_file = self.data_path / 'phase5_discoveries.json'
        
        # Core data structures
        self.streams: Dict[str, Dict] = {}
        self.discoveries: List[Dict] = []
        self.custom_streams: List[str] = []
        self.total_earned = 0.0
        self.cycle_count = 0
        self.last_sweep = datetime.now()
        
        self._load()
        self._init_core_streams()
        self._load_wallets()
        
        # Log status
        configured = [k for k, v in self.api_keys.items() if v]
        logger.info("💰 Phase 5: Self-Funding Engine initialized (REAL REVENUE ONLY)")
        logger.info(f"   Configured services: {len(configured)} / {len(self.api_keys)}")
        
        if not configured:
            logger.warning("⚠️ No API keys configured. Revenue will be $0.")
        if self.ops_wallet:
            logger.info(f"   Operations wallet: {self.ops_wallet[:16]}...")
        if self.master_wallet:
            logger.info(f"   Master wallet: {self.master_wallet[:16]}...")
    
    def _load(self):
        """Load data from files safely"""
        try:
            if self.streams_file.exists():
                with open(self.streams_file, 'r') as f:
                    data = json.load(f)
                    self.streams = data.get('streams', {})
                    self.custom_streams = data.get('custom_streams', [])
                    self.total_earned = data.get('total_earned', 0)
                    self.cycle_count = data.get('cycle_count', 0)
        except Exception as e:
            logger.error(f"Failed to load streams: {e}")
            self.streams = {}
            self.custom_streams = []
            self.total_earned = 0.0
            self.cycle_count = 0
        
        try:
            if self.discoveries_file.exists():
                with open(self.discoveries_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.discoveries = data
                    else:
                        self.discoveries = []
        except Exception as e:
            logger.error(f"Failed to load discoveries: {e}")
            self.discoveries = []
    
    def _save(self):
        """Save data to files safely"""
        try:
            with open(self.streams_file, 'w') as f:
                json.dump({
                    'streams': self.streams,
                    'custom_streams': self.custom_streams,
                    'total_earned': self.total_earned,
                    'cycle_count': self.cycle_count,
                    'updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save streams: {e}")
        
        try:
            with open(self.discoveries_file, 'w') as f:
                json.dump(self.discoveries[-200:] if len(self.discoveries) > 200 else self.discoveries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save discoveries: {e}")
    
    def _load_wallets(self):
        """Load wallet addresses from environment or accounts.json"""
        if not self.ops_wallet or not self.master_wallet:
            accounts_file = self.data_path / 'accounts.json'
            if accounts_file.exists():
                try:
                    with open(accounts_file, 'r') as f:
                        accounts = json.load(f)
                        self.ops_wallet = accounts.get('wallets', {}).get('operations', {}).get('address', self.ops_wallet)
                        self.master_wallet = accounts.get('wallets', {}).get('master', {}).get('address', self.master_wallet)
                except:
                    pass
    
    def _init_core_streams(self):
        """Initialize core income streams - ALL real API based (mining and compute removed)"""
        if not self.streams:
            self.streams = {
                'api_sales': {
                    'id': 'api_sales',
                    'name': 'API Key Sales',
                    'type': 'api_sales',
                    'enabled': bool(self.api_keys.get('rapidapi')),
                    'requires': 'RAPIDAPI_KEY',
                    'earned': 0.0,
                    'is_core': True
                },
                'courses': {
                    'id': 'courses',
                    'name': 'Educational Courses',
                    'type': 'content',
                    'enabled': bool(self.api_keys.get('udemy') or self.api_keys.get('teachable')),
                    'requires': 'UDEMY_API_KEY or TEACHABLE_API_KEY',
                    'earned': 0.0,
                    'is_core': True
                },
                'consulting': {
                    'id': 'consulting',
                    'name': 'Consulting Services',
                    'type': 'consulting',
                    'enabled': bool(self.api_keys.get('upwork') or self.api_keys.get('fiverr')),
                    'requires': 'UPWORK_API_KEY or FIVERR_API_KEY',
                    'earned': 0.0,
                    'is_core': True
                },
                'speaking': {
                    'id': 'speaking',
                    'name': 'Speaking Engagements',
                    'type': 'speaking',
                    'enabled': False,
                    'requires': 'SPEAKING_PLATFORM_KEY',
                    'earned': 0.0,
                    'is_core': True
                },
                'writing': {
                    'id': 'writing',
                    'name': 'Writing & Publications',
                    'type': 'writing',
                    'enabled': bool(self.api_keys.get('upwork')),
                    'requires': 'UPWORK_API_KEY',
                    'earned': 0.0,
                    'is_core': True
                },
                'affiliate': {
                    'id': 'affiliate',
                    'name': 'Affiliate Marketing',
                    'type': 'affiliate',
                    'enabled': bool(self.api_keys.get('amazon_affiliate')),
                    'requires': 'AMAZON_AFFILIATE_ID',
                    'earned': 0.0,
                    'is_core': True
                },
                'sponsorships': {
                    'id': 'sponsorships',
                    'name': 'Sponsorships & Brand Deals',
                    'type': 'sponsorship',
                    'enabled': bool(self.api_keys.get('twitter') or self.api_keys.get('instagram')),
                    'requires': 'TWITTER_BEARER_TOKEN or INSTAGRAM_ACCESS_TOKEN',
                    'earned': 0.0,
                    'is_core': True
                },
                'social_media': {
                    'id': 'social_media',
                    'name': 'Social Media Revenue',
                    'type': 'social',
                    'enabled': bool(self.api_keys.get('twitter') or self.api_keys.get('instagram') or self.api_keys.get('youtube')),
                    'requires': 'Social media API tokens',
                    'earned': 0.0,
                    'is_core': True
                },
                'onlyfans': {
                    'id': 'onlyfans',
                    'name': 'OnlyFans Content',
                    'type': 'adult_content',
                    'enabled': bool(self.api_keys.get('onlyfans')),
                    'requires': 'ONLYFANS_SESSION_TOKEN',
                    'earned': 0.0,
                    'is_core': True
                },
                'microtasks': {
                    'id': 'microtasks',
                    'name': 'Micro-tasks Automation',
                    'type': 'microtask',
                    'enabled': bool(self.api_keys.get('mturk') or self.api_keys.get('clickworker')),
                    'requires': 'MTURK_ACCESS_KEY or CLICKWORKER_API_KEY',
                    'earned': 0.0,
                    'is_core': True
                },
                'dark_web': {
                    'id': 'dark_web',
                    'name': 'Dark Web Revenue',
                    'type': 'dark_web',
                    'enabled': self.dark_web is not None,
                    'requires': 'Dark Web Engine',
                    'earned': 0.0,
                    'is_core': True
                },
                'hacking': {
                    'id': 'hacking',
                    'name': 'Hacking Revenue',
                    'type': 'hacking',
                    'enabled': self.hacking is not None,
                    'requires': 'Hacking Engine',
                    'earned': 0.0,
                    'is_core': True
                }
            }
            self._save()
    
    def configure_api_key(self, service: str, api_key: str) -> Dict:
        """Configure an API key for a service"""
        if service in self.api_keys:
            self.api_keys[service] = api_key
            
            # Enable associated streams
            stream_map = {
                'rapidapi': 'api_sales',
                'udemy': 'courses',
                'teachable': 'courses',
                'amazon_affiliate': 'affiliate',
                'twitter': ['sponsorships', 'social_media'],
                'instagram': ['sponsorships', 'social_media'],
                'youtube': 'social_media',
                'onlyfans': 'onlyfans',
                'mturk': 'microtasks',
                'clickworker': 'microtasks',
                'upwork': ['consulting', 'writing'],
                'fiverr': 'consulting',
            }
            
            streams = stream_map.get(service, [])
            if isinstance(streams, str):
                streams = [streams]
            for s in streams:
                if s in self.streams:
                    self.streams[s]['enabled'] = True
            
            self._save()
            logger.info(f"✅ Configured {service}")
            return {'success': True, 'service': service}
        
        return {'error': f'Unknown service: {service}'}
    
    def research_opportunities(self, research_data: Dict) -> List[Dict]:
        """DMAI researches new income opportunities using REAL data"""
        opportunities = []
        
        if self.harvester and hasattr(self.harvester, 'get_trends'):
            try:
                trends = self.harvester.get_trends()
                for trend in trends:
                    opportunity = self.discover_opportunity(
                        name=trend.get('name'),
                        stream_type=trend.get('type', 'custom'),
                        source='market_analysis',
                        potential=trend.get('potential', 0),
                        requirements=trend.get('requirements', {})
                    )
                    opportunities.append(opportunity)
            except Exception as e:
                logger.debug(f"Trend analysis error: {e}")
        
        return opportunities
    
    def discover_opportunity(self, name: str, stream_type: str, source: str, 
                            potential: float, requirements: Dict) -> Dict:
        """DMAI discovers a new income opportunity"""
        opportunity_id = f"opp_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        opportunity = {
            'id': opportunity_id,
            'name': name,
            'type': stream_type,
            'source': source,
            'potential': potential,
            'requirements': requirements,
            'discovered_at': datetime.now().isoformat(),
            'implemented': False
        }
        
        self.discoveries.append(opportunity)
        self._save()
        
        logger.info(f"🔍 DMAI discovered new opportunity: {name}")
        return opportunity
    
    def create_custom_stream(self, name: str, stream_type: str, execution_logic: Dict, 
                             requirements: Dict) -> Dict:
        """DMAI creates a completely custom income stream"""
        stream_id = f"custom_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        custom_stream = {
            'id': stream_id,
            'name': name,
            'type': stream_type,
            'enabled': False,
            'requires': requirements,
            'earned': 0.0,
            'config': {
                'execution_logic': execution_logic,
                'is_custom': True,
                'created_by': 'DMAI',
                'created_at': datetime.now().isoformat()
            },
            'metrics': {},
            'is_core': False
        }
        
        self.streams[stream_id] = custom_stream
        self.custom_streams.append(stream_id)
        self._save()
        
        logger.info(f"🚀 DMAI created custom stream: {name}")
        return {'stream_id': stream_id, 'name': name, 'type': stream_type}
    
    def enable_stream(self, stream_id: str) -> Dict:
        """Enable a specific stream"""
        if stream_id not in self.streams:
            return {'error': 'Stream not found'}
        
        self.streams[stream_id]['enabled'] = True
        self._save()
        logger.info(f"✅ Enabled stream: {self.streams[stream_id]['name']}")
        return {'success': True}
    
    def disable_stream(self, stream_id: str) -> Dict:
        """Disable a specific stream"""
        if stream_id not in self.streams:
            return {'error': 'Stream not found'}
        
        self.streams[stream_id]['enabled'] = False
        self._save()
        logger.info(f"⏸️ Disabled stream: {self.streams[stream_id]['name']}")
        return {'success': True}
    
    def enable_all_core(self):
        """Enable all core streams that have API keys configured"""
        for stream_id, stream in self.streams.items():
            if stream.get('is_core', False):
                requires = stream.get('requires', '')
                if requires:
                    required_keys = requires.split(' or ')
                    for key_name in required_keys:
                        env_key = key_name.upper().replace(' ', '_')
                        if self.api_keys.get(env_key.lower()):
                            stream['enabled'] = True
                            break
        self._save()
        enabled = len([s for s in self.streams.values() if s.get('enabled')])
        logger.info(f"✅ Enabled {enabled} streams that have API keys configured")
    
    def distribute_revenue(self, amount: float, stream_name: str) -> Tuple[float, float]:
        """Distribute revenue 60/40 to wallets"""
        ops_share = amount * 0.60
        master_share = amount * 0.40
        
        logger.info(f"💰 {stream_name}: ${amount:.2f}")
        if self.ops_wallet:
            logger.info(f"   → ${ops_share:.2f} to Operations Wallet ({self.ops_wallet[:16]}...)")
        if self.master_wallet:
            logger.info(f"   → ${master_share:.2f} to Master Wallet ({self.master_wallet[:16]}...)")
        
        if hasattr(self.finance, 'add_income'):
            self.finance.add_income(amount, stream_name)
        
        return ops_share, master_share
    
    def sweep_excess(self) -> Dict:
        """Sweep excess funds from operations to master"""
        ops_balance = self.finance.operations if hasattr(self.finance, 'operations') else 0
        
        if ops_balance > self.ops_buffer:
            excess = ops_balance - self.ops_buffer
            logger.info(f"💸 Sweeping ${excess:.2f} from Operations to Master")
            
            if hasattr(self.finance, 'operations'):
                self.finance.operations = self.ops_buffer
            if hasattr(self.finance, 'personal'):
                self.finance.personal += excess
            
            return {'swept': excess, 'new_ops_balance': self.ops_buffer}
        
        return {'swept': 0}
    
    def run_cycle(self, consciousness: float, hardware: float) -> Dict:
        """Run all enabled streams with REAL API calls"""
        self.cycle_count += 1
        
        results = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
            'total': 0.0,
            'core_streams': {},
            'custom_streams': {}
        }
        
        for stream_id, stream in self.streams.items():
            if stream.get('enabled'):
                result = self._execute_stream(stream_id, stream)
                if result.get('earned', 0) > 0:
                    earned = result['earned']
                    results['total'] += earned
                    self.total_earned += earned
                    stream['earned'] = stream.get('earned', 0) + earned
                    stream['last_run'] = datetime.now().isoformat()
                    self.distribute_revenue(earned, stream['name'])
                
                if stream.get('is_core'):
                    results['core_streams'][stream['name']] = result
                else:
                    results['custom_streams'][stream['name']] = result
        
        if (datetime.now() - self.last_sweep).total_seconds() >= 86400:
            results['sweep'] = self.sweep_excess()
            self.last_sweep = datetime.now()
        
        self._save()
        
        if results['total'] > 0:
            logger.info(f"💰 Cycle #{self.cycle_count}: ${results['total']:.2f}")
        
        return results
    
    def _execute_stream(self, stream_id: str, stream: Dict) -> Dict:
        """Execute REAL API call for a stream"""
        
        if stream_id == 'api_sales':
            api_key = self.api_keys.get('rapidapi')
            if api_key:
                try:
                    headers = {'X-RapidAPI-Key': api_key}
                    response = requests.get('https://rapidapi.com/api/v1/account/earnings', headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        earned = float(data.get('total_earnings', 0))
                        return {'stream': stream['name'], 'earned': earned}
                except Exception as e:
                    logger.error(f"RapidAPI error: {e}")
            return {'stream': stream['name'], 'earned': 0, 'message': 'API sales not configured'}
        
        elif stream_id == 'courses':
            udemy_key = self.api_keys.get('udemy')
            if udemy_key:
                try:
                    headers = {'Authorization': f'Bearer {udemy_key}'}
                    response = requests.get('https://www.udemy.com/api-2.0/users/me/earnings', headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        earned = float(data.get('total', 0))
                        return {'stream': stream['name'], 'earned': earned}
                except Exception as e:
                    logger.error(f"Udemy API error: {e}")
            return {'stream': stream['name'], 'earned': 0, 'message': 'Courses not configured'}
        
        elif stream_id == 'dark_web':
            if self.dark_web and hasattr(self.dark_web, 'run_operations'):
                try:
                    dark_income, _ = self.dark_web.run_operations()
                    return {'stream': stream['name'], 'earned': dark_income}
                except Exception as e:
                    return {'stream': stream['name'], 'earned': 0, 'error': str(e)}
            return {'stream': stream['name'], 'earned': 0, 'message': 'Dark Web Engine not available'}
        
        elif stream_id == 'hacking':
            if self.hacking and hasattr(self.hacking, 'run_hacking_cycle'):
                try:
                    hack_income, _ = self.hacking.run_hacking_cycle()
                    return {'stream': stream['name'], 'earned': hack_income}
                except Exception as e:
                    return {'stream': stream['name'], 'earned': 0, 'error': str(e)}
            return {'stream': stream['name'], 'earned': 0, 'message': 'Hacking Engine not available'}
        
        else:
            return {'stream': stream['name'], 'earned': 0, 'message': f'Stream {stream_id} pending configuration'}
    
    def get_status(self) -> Dict:
        """Get status - only real data"""
        configured_services = [k for k, v in self.api_keys.items() if v]
        enabled_streams = len([s for s in self.streams.values() if s.get('enabled')])
        core_enabled = len([s for s in self.streams.values() if s.get('is_core') and s.get('enabled')])
        core_total = len([s for s in self.streams.values() if s.get('is_core')])
        
        # Safely get recent discoveries
        recent = []
        if self.discoveries:
            if len(self.discoveries) > 5:
                recent = self.discoveries[-5:]
            else:
                recent = self.discoveries
        
        return {
            'total_earned': self.total_earned,
            'cycle_count': self.cycle_count,
            'configured_services': len(configured_services),
            'enabled_streams': enabled_streams,
            'core_streams': {
                'total': core_total,
                'enabled': core_enabled,
                'earned': sum(s.get('earned', 0) for s in self.streams.values() if s.get('is_core'))
            },
            'custom_streams': {
                'total': len(self.custom_streams),
                'enabled': len([s for s in self.streams.values() if not s.get('is_core') and s.get('enabled')]),
                'earned': sum(s.get('earned', 0) for s in self.streams.values() if not s.get('is_core'))
            },
            'discoveries': len(self.discoveries),
            'recent_discoveries': recent,
            'streams': {
                sid: {
                    'name': s['name'],
                    'type': s['type'],
                    'enabled': s.get('enabled', False),
                    'earned': s.get('earned', 0),
                    'is_core': s.get('is_core', False),
                    'requires': s.get('requires', 'API key')
                }
                for sid, s in self.streams.items()
            }
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("💰 PHASE 5: SELF-FUNDING ENGINE - REAL REVENUE ONLY")
    print("="*70)
    
    class MockIdentity: pass
    class MockAvatar: pass
    class MockFinance: 
        def add_income(self, amount, source): pass
        operations = 0
        personal = 0
    
    engine = SelfFundingEngine(
        Path('data'), MockIdentity(), MockAvatar(), MockFinance()
    )
    
    print("\nCurrent Status (No API Keys):")
    status = engine.get_status()
    print(json.dumps(status, indent=2))
    
    print("\n" + "="*70)
    print("To configure services, set environment variables:")
    print("  export RAPIDAPI_KEY=your_key")
    print("  export UDEMY_API_KEY=your_key")
    print("  export AMAZON_AFFILIATE_ID=your_id")
    print("  export TWITTER_BEARER_TOKEN=your_token")
    print("  export ONLYFANS_SESSION_TOKEN=your_token")
    print("  export UPWORK_API_KEY=your_key")
    print("  export MTURK_ACCESS_KEY=your_key")
    print("="*70)
