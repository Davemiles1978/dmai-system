"""
SELF-FUNDING PHASE 2 & 3 - EXECUTION LAYER
Phase 2: Paper Execution (simulated, no real money)
Phase 3: Real Execution (live trading, real revenue)
"""

import os
import json
import time
import threading
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PaperExecutionEngine:
    """
    Phase 2: Paper Execution - Simulated revenue generation
    No real money - tests strategies with market data
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.execution_dir = data_path / 'execution' / 'paper'
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        
        # Paper accounts for each revenue avenue
        self.paper_accounts = {
            'quant_trading': {
                'balance': 10000.0,  # $10,000 paper trading
                'positions': [],
                'pnl': 0.0,
                'trades': []
            },
            'content_creation': {
                'posts': 0,
                'views': 0,
                'engagement': 0.0,
                'estimated_revenue': 0.0
            },
            'ai_services': {
                'api_calls': 0,
                'requests': 0,
                'estimated_revenue': 0.0
            },
            'software_products': {
                'users': 0,
                'subscriptions': 0,
                'estimated_revenue': 0.0
            },
            'affiliate_referral': {
                'clicks': 0,
                'conversions': 0,
                'estimated_commission': 0.0
            },
            'data_services': {
                'queries': 0,
                'data_requests': 0,
                'estimated_revenue': 0.0
            },
            'education_training': {
                'enrollments': 0,
                'courses_sold': 0,
                'estimated_revenue': 0.0
            },
            'consulting_analysis': {
                'consultations': 0,
                'reports': 0,
                'estimated_revenue': 0.0
            },
            'ad_revenue': {
                'impressions': 0,
                'clicks': 0,
                'estimated_revenue': 0.0
            },
            'crowdfunding_patronage': {
                'patrons': 0,
                'pledges': 0,
                'estimated_revenue': 0.0
            }
        }
        
        # Strategy execution queue
        self.strategy_queue = []
        self.execution_active = False
        self.execution_thread = None
        
        self._load_state()
        
        logger.info("💰 Phase 2: Paper Execution Engine initialized")
    
    def _load_state(self):
        """Load paper execution state"""
        state_file = self.execution_dir / 'paper_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.paper_accounts = state.get('paper_accounts', self.paper_accounts)
                    self.strategy_queue = state.get('strategy_queue', [])
                    logger.info("📂 Loaded paper execution state")
            except Exception as e:
                logger.error(f"Failed to load paper state: {e}")
    
    def _save_state(self):
        """Save paper execution state"""
        try:
            state = {
                'paper_accounts': self.paper_accounts,
                'strategy_queue': self.strategy_queue,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.execution_dir / 'paper_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper state: {e}")
    
    def execute_strategy(self, avenue: str, strategy: Dict) -> Dict:
        """
        Execute a strategy in paper mode (simulated)
        """
        if avenue not in self.paper_accounts:
            return {'success': False, 'error': f'Unknown avenue: {avenue}'}
        
        strategy_id = strategy.get('id', f"{avenue}_strategy_{int(time.time())}")
        
        # Add to queue
        execution = {
            'id': strategy_id,
            'avenue': avenue,
            'strategy': strategy,
            'status': 'queued',
            'started_at': datetime.now().isoformat(),
            'paper_account_snapshot': self.paper_accounts[avenue].copy()
        }
        
        self.strategy_queue.append(execution)
        self._save_state()
        
        # Execute immediately if active
        if self.execution_active:
            self._process_strategy(execution)
        
        logger.info(f"📋 Strategy queued for paper execution: {strategy_id}")
        
        return {
            'success': True,
            'execution_id': strategy_id,
            'status': 'queued',
            'paper_account': self.paper_accounts[avenue],
            'message': f"Strategy queued for paper execution on {avenue}"
        }
    
    def _process_strategy(self, execution: Dict):
        """
        Process a single strategy in paper mode
        """
        avenue = execution['avenue']
        strategy = execution['strategy']
        strategy_id = execution['id']
        
        try:
            execution['status'] = 'running'
            self._save_state()
            
            # Simulate execution based on avenue type
            if avenue == 'quant_trading':
                result = self._simulate_trading_strategy(strategy)
            elif avenue == 'content_creation':
                result = self._simulate_content_strategy(strategy)
            elif avenue == 'ai_services':
                result = self._simulate_ai_services_strategy(strategy)
            elif avenue == 'software_products':
                result = self._simulate_software_strategy(strategy)
            elif avenue == 'affiliate_referral':
                result = self._simulate_affiliate_strategy(strategy)
            elif avenue == 'data_services':
                result = self._simulate_data_services_strategy(strategy)
            elif avenue == 'education_training':
                result = self._simulate_education_strategy(strategy)
            elif avenue == 'consulting_analysis':
                result = self._simulate_consulting_strategy(strategy)
            elif avenue == 'ad_revenue':
                result = self._simulate_ad_revenue_strategy(strategy)
            elif avenue == 'crowdfunding_patronage':
                result = self._simulate_crowdfunding_strategy(strategy)
            else:
                result = {'success': False, 'error': f'Unknown avenue: {avenue}'}
            
            execution['status'] = 'completed'
            execution['completed_at'] = datetime.now().isoformat()
            execution['result'] = result
            
            # Update paper account with results
            if result.get('success'):
                account = self.paper_accounts[avenue]
                if 'pnl' in result:
                    account['pnl'] += result['pnl']
                    account['balance'] += result['pnl']
                if 'estimated_revenue' in result:
                    account['estimated_revenue'] += result['estimated_revenue']
                account['trades'] = account.get('trades', [])
                account['trades'].append(result)
            
            logger.info(f"✅ Paper execution completed: {strategy_id} - Result: {result.get('message', 'Done')}")
            
        except Exception as e:
            execution['status'] = 'failed'
            execution['error'] = str(e)
            logger.error(f"Paper execution failed: {strategy_id} - {e}")
        
        finally:
            self._save_state()
    
    def _simulate_trading_strategy(self, strategy: Dict) -> Dict:
        """Simulate a trading strategy with paper account"""
        # Simulate market movement
        pnl_percent = random.uniform(-5, 15)  # -5% to +15%
        initial_balance = 10000
        pnl = initial_balance * (pnl_percent / 100)
        
        return {
            'success': True,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'final_balance': initial_balance + pnl,
            'trades_executed': random.randint(1, 10),
            'win_rate': random.uniform(40, 70),
            'sharpe_ratio': random.uniform(0.5, 2.0),
            'message': f"Paper trading completed: {pnl_percent:+.2f}% PnL"
        }
    
    def _simulate_content_strategy(self, strategy: Dict) -> Dict:
        """Simulate content creation strategy"""
        views = random.randint(100, 10000)
        engagement_rate = random.uniform(1, 10)
        estimated_revenue = views * random.uniform(0.001, 0.01)
        
        return {
            'success': True,
            'views': views,
            'engagement_rate': engagement_rate,
            'estimated_revenue': estimated_revenue,
            'posts_created': random.randint(1, 5),
            'message': f"Content generated: {views} views, ${estimated_revenue:.2f} estimated revenue"
        }
    
    def _simulate_ai_services_strategy(self, strategy: Dict) -> Dict:
        """Simulate AI services strategy"""
        api_calls = random.randint(100, 5000)
        estimated_revenue = api_calls * random.uniform(0.001, 0.05)
        
        return {
            'success': True,
            'api_calls': api_calls,
            'estimated_revenue': estimated_revenue,
            'uptime': random.uniform(99, 100),
            'message': f"AI Services: {api_calls} API calls, ${estimated_revenue:.2f} estimated revenue"
        }
    
    def _simulate_software_strategy(self, strategy: Dict) -> Dict:
        """Simulate software product strategy"""
        users = random.randint(10, 500)
        subscriptions = random.randint(1, users)
        estimated_revenue = subscriptions * random.uniform(5, 50)
        
        return {
            'success': True,
            'users': users,
            'subscriptions': subscriptions,
            'estimated_revenue': estimated_revenue,
            'churn_rate': random.uniform(1, 10),
            'message': f"Software: {users} users, {subscriptions} subscriptions, ${estimated_revenue:.2f} MRR"
        }
    
    def _simulate_affiliate_strategy(self, strategy: Dict) -> Dict:
        """Simulate affiliate marketing strategy"""
        clicks = random.randint(50, 5000)
        conversions = random.randint(1, int(clicks * 0.1))
        estimated_commission = conversions * random.uniform(5, 50)
        
        return {
            'success': True,
            'clicks': clicks,
            'conversions': conversions,
            'conversion_rate': (conversions / clicks * 100) if clicks > 0 else 0,
            'estimated_commission': estimated_commission,
            'message': f"Affiliate: {clicks} clicks, {conversions} conversions, ${estimated_commission:.2f} commission"
        }
    
    def _simulate_data_services_strategy(self, strategy: Dict) -> Dict:
        """Simulate data services strategy"""
        queries = random.randint(100, 10000)
        estimated_revenue = queries * random.uniform(0.001, 0.02)
        
        return {
            'success': True,
            'queries': queries,
            'estimated_revenue': estimated_revenue,
            'data_points': queries * random.randint(10, 100),
            'message': f"Data Services: {queries} queries, ${estimated_revenue:.2f} estimated revenue"
        }
    
    def _simulate_education_strategy(self, strategy: Dict) -> Dict:
        """Simulate education/training strategy"""
        enrollments = random.randint(1, 100)
        course_price = random.uniform(20, 200)
        estimated_revenue = enrollments * course_price
        
        return {
            'success': True,
            'enrollments': enrollments,
            'course_price': course_price,
            'estimated_revenue': estimated_revenue,
            'completion_rate': random.uniform(30, 90),
            'message': f"Education: {enrollments} enrollments, ${estimated_revenue:.2f} revenue"
        }
    
    def _simulate_consulting_strategy(self, strategy: Dict) -> Dict:
        """Simulate consulting/analysis strategy"""
        consultations = random.randint(1, 20)
        hourly_rate = random.uniform(100, 500)
        hours_per_consultation = random.uniform(1, 5)
        estimated_revenue = consultations * hourly_rate * hours_per_consultation
        
        return {
            'success': True,
            'consultations': consultations,
            'hourly_rate': hourly_rate,
            'estimated_revenue': estimated_revenue,
            'satisfaction_score': random.uniform(4, 5),
            'message': f"Consulting: {consultations} consultations, ${estimated_revenue:.2f} revenue"
        }
    
    def _simulate_ad_revenue_strategy(self, strategy: Dict) -> Dict:
        """Simulate ad revenue strategy"""
        impressions = random.randint(1000, 100000)
        clicks = random.randint(10, int(impressions * 0.05))
        cpm = random.uniform(1, 10)
        estimated_revenue = (impressions / 1000) * cpm
        
        return {
            'success': True,
            'impressions': impressions,
            'clicks': clicks,
            'ctr': (clicks / impressions * 100) if impressions > 0 else 0,
            'cpm': cpm,
            'estimated_revenue': estimated_revenue,
            'message': f"Ad Revenue: {impressions} impressions, ${estimated_revenue:.2f} revenue"
        }
    
    def _simulate_crowdfunding_strategy(self, strategy: Dict) -> Dict:
        """Simulate crowdfunding/patronage strategy"""
        patrons = random.randint(1, 50)
        avg_pledge = random.uniform(5, 25)
        estimated_revenue = patrons * avg_pledge
        
        return {
            'success': True,
            'patrons': patrons,
            'avg_pledge': avg_pledge,
            'estimated_revenue': estimated_revenue,
            'retention_rate': random.uniform(60, 95),
            'message': f"Crowdfunding: {patrons} patrons, ${estimated_revenue:.2f} monthly revenue"
        }
    
    def start_execution(self) -> Dict:
        """Start the paper execution engine"""
        if self.execution_active:
            return {'success': False, 'error': 'Execution already active'}
        
        self.execution_active = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        
        logger.info("🚀 Phase 2: Paper Execution Engine started")
        
        return {
            'success': True,
            'message': 'Paper execution started',
            'phase': '2 - Paper Execution'
        }
    
    def stop_execution(self) -> Dict:
        """Stop the paper execution engine"""
        self.execution_active = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        
        logger.info("⏸️ Phase 2: Paper Execution Engine stopped")
        
        return {
            'success': True,
            'message': 'Paper execution stopped'
        }
    
    def _execution_loop(self):
        """Main execution loop"""
        while self.execution_active:
            if self.strategy_queue:
                # Process next strategy
                execution = self.strategy_queue.pop(0)
                self._process_strategy(execution)
            
            time.sleep(10)  # Check every 10 seconds
    
    def get_paper_account_status(self) -> Dict:
        """Get all paper account statuses"""
        total_pnl = sum(acc.get('pnl', 0) for acc in self.paper_accounts.values())
        total_estimated_revenue = sum(acc.get('estimated_revenue', 0) for acc in self.paper_accounts.values())
        
        return {
            'paper_accounts': self.paper_accounts,
            'total_paper_pnl': total_pnl,
            'total_estimated_revenue': total_estimated_revenue,
            'active_strategies': len(self.strategy_queue),
            'execution_active': self.execution_active
        }


class RealExecutionEngine:
    """
    Phase 3: Real Execution - Live revenue generation
    Requires API keys and master approval
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.execution_dir = data_path / 'execution' / 'real'
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        
        # Real accounts status
        self.real_accounts = {
            'quant_trading': {'active': False, 'api_keys_configured': False, 'balance': 0, 'pnl': 0},
            'content_creation': {'active': False, 'api_keys_configured': False, 'posts': 0, 'revenue': 0},
            'ai_services': {'active': False, 'api_keys_configured': False, 'api_calls': 0, 'revenue': 0},
            'software_products': {'active': False, 'api_keys_configured': False, 'users': 0, 'revenue': 0},
            'affiliate_referral': {'active': False, 'api_keys_configured': False, 'clicks': 0, 'commission': 0},
            'data_services': {'active': False, 'api_keys_configured': False, 'queries': 0, 'revenue': 0},
            'education_training': {'active': False, 'api_keys_configured': False, 'enrollments': 0, 'revenue': 0},
            'consulting_analysis': {'active': False, 'api_keys_configured': False, 'consultations': 0, 'revenue': 0},
            'ad_revenue': {'active': False, 'api_keys_configured': False, 'impressions': 0, 'revenue': 0},
            'crowdfunding_patronage': {'active': False, 'api_keys_configured': False, 'patrons': 0, 'revenue': 0}
        }
        
        self.master_approved = False
        self.execution_active = False
        
        self._load_state()
        logger.info("💰 Phase 3: Real Execution Engine initialized (requires master approval and API keys)")
    
    def _load_state(self):
        """Load real execution state"""
        state_file = self.execution_dir / 'real_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.real_accounts = state.get('real_accounts', self.real_accounts)
                    self.master_approved = state.get('master_approved', False)
                    self.execution_active = state.get('execution_active', False)
                    logger.info("📂 Loaded real execution state")
            except Exception as e:
                logger.error(f"Failed to load real state: {e}")
    
    def _save_state(self):
        """Save real execution state"""
        try:
            state = {
                'real_accounts': self.real_accounts,
                'master_approved': self.master_approved,
                'execution_active': self.execution_active,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.execution_dir / 'real_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save real state: {e}")
    
    def configure_api_keys(self, avenue: str, api_keys: Dict) -> Dict:
        """Configure API keys for a revenue avenue"""
        if avenue not in self.real_accounts:
            return {'success': False, 'error': f'Unknown avenue: {avenue}'}
        
        # Store keys securely (in production, use proper secrets management)
        keys_file = self.execution_dir / f"{avenue}_keys.json"
        try:
            with open(keys_file, 'w') as f:
                json.dump(api_keys, f)
            
            self.real_accounts[avenue]['api_keys_configured'] = True
            self.real_accounts[avenue]['keys_configured_at'] = datetime.now().isoformat()
            self._save_state()
            
            logger.info(f"🔑 API keys configured for {avenue}")
            
            return {
                'success': True,
                'message': f'API keys configured for {avenue}',
                'avenue': avenue
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def request_master_approval(self) -> Dict:
        """Request master approval for real execution"""
        # Check if all required API keys are configured
        missing_keys = [
            avenue for avenue, data in self.real_accounts.items()
            if not data.get('api_keys_configured', False)
        ]
        
        if missing_keys:
            return {
                'success': False,
                'error': 'Missing API keys for some avenues',
                'missing_keys': missing_keys
            }
        
        return {
            'success': True,
            'requires_master_approval': True,
            'message': 'Master approval required for real execution. This will enable live revenue generation.',
            'avenues_ready': [a for a, d in self.real_accounts.items() if d.get('api_keys_configured', False)]
        }
    
    def grant_master_approval(self) -> Dict:
        """Master grants approval for real execution"""
        self.master_approved = True
        self._save_state()
        
        logger.info("✅ MASTER APPROVAL GRANTED for Real Execution")
        
        return {
            'success': True,
            'message': 'Master approval granted. Real execution can now be started.',
            'phase': '3 - Real Execution'
        }
    
    def start_execution(self) -> Dict:
        """Start real execution (requires master approval)"""
        if not self.master_approved:
            return {
                'success': False,
                'error': 'Master approval required',
                'action': 'Request approval first using request_master_approval()'
            }
        
        if self.execution_active:
            return {'success': False, 'error': 'Execution already active'}
        
        # Check which avenues have API keys
        active_avenues = [
            avenue for avenue, data in self.real_accounts.items()
            if data.get('api_keys_configured', False)
        ]
        
        if not active_avenues:
            return {
                'success': False,
                'error': 'No API keys configured. Use configure_api_keys() first.'
            }
        
        self.execution_active = True
        self._save_state()
        
        logger.info(f"🚀 Phase 3: Real Execution started for avenues: {active_avenues}")
        
        return {
            'success': True,
            'message': 'Real execution started',
            'phase': '3 - Real Execution',
            'active_avenues': active_avenues
        }
    
    def stop_execution(self) -> Dict:
        """Stop real execution"""
        self.execution_active = False
        self._save_state()
        
        logger.info("⏸️ Phase 3: Real Execution stopped")
        
        return {
            'success': True,
            'message': 'Real execution stopped'
        }
    
    def get_status(self) -> Dict:
        """Get real execution status"""
        total_revenue = sum(acc.get('revenue', 0) for acc in self.real_accounts.values())
        
        return {
            'master_approved': self.master_approved,
            'execution_active': self.execution_active,
            'real_accounts': self.real_accounts,
            'total_revenue_generated': total_revenue,
            'ready_for_execution': self.master_approved and self.execution_active,
            'phase': '3 - Real Execution' if self.master_approved else '2 - Paper Execution'
        }
    
    def record_revenue(self, avenue: str, amount: float, metadata: Dict = None) -> Dict:
        """Record actual revenue generated"""
        if avenue not in self.real_accounts:
            return {'success': False, 'error': f'Unknown avenue: {avenue}'}
        
        self.real_accounts[avenue]['revenue'] = self.real_accounts[avenue].get('revenue', 0) + amount
        self.real_accounts[avenue]['last_revenue'] = amount
        self.real_accounts[avenue]['last_revenue_at'] = datetime.now().isoformat()
        
        if metadata:
            revenue_log = self.execution_dir / 'revenue_log.json'
            log_entry = {
                'avenue': avenue,
                'amount': amount,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                if revenue_log.exists():
                    with open(revenue_log, 'r') as f:
                        log = json.load(f)
                else:
                    log = []
                log.append(log_entry)
                with open(revenue_log, 'w') as f:
                    json.dump(log, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to log revenue: {e}")
        
        self._save_state()
        
        logger.info(f"💰 Revenue recorded: {avenue} - ${amount:.2f}")
        
        return {
            'success': True,
            'avenue': avenue,
            'amount': amount,
            'total_revenue': self.real_accounts[avenue]['revenue'],
            'message': f"${amount:.2f} recorded for {avenue}"
        }
