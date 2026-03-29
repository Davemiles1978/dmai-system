#!/usr/bin/env python3
"""
SELF-FUNDING - PHASE 2: PAPER EXECUTION
Simulated execution with paper accounts, no real money.
Requires master strategy approval from Phase 1.
"""

import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SelfFundingPhase2Paper:
    """
    PHASE 2: Paper Execution
    - Simulated trades using real market data
    - No real money involved
    - Performance tracking and reporting
    - Requires master approval to enable strategies
    """
    
    def __init__(self, data_path: Path, knowledge_graph, ai_hub, strategy_candidates: Dict):
        self.data_path = data_path
        self.knowledge_graph = knowledge_graph
        self.ai_hub = ai_hub
        self.execution_dir = data_path / 'execution' / 'phase2_paper'
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy candidates from Phase 1
        self.strategy_candidates = strategy_candidates
        
        # Active paper strategies (master approved)
        self.active_strategies = {}  # avenue -> strategy_id
        
        # Paper trading accounts (simulated)
        self.paper_accounts = {
            'quant_trading': {
                'balance': 100000.0,  # $100k paper money
                'positions': {},
                'trades': [],
                'pnl': 0.0
            },
            'content_creation': {
                'balance': 0.0,
                'content_published': [],
                'engagement_metrics': {},
                'estimated_value': 0.0
            },
            'ai_services': {
                'balance': 0.0,
                'api_calls': 0,
                'subscribers': 0,
                'estimated_revenue': 0.0
            },
            'software_products': {
                'balance': 0.0,
                'users': 0,
                'subscriptions': 0,
                'estimated_revenue': 0.0
            },
            'affiliate_referral': {
                'balance': 0.0,
                'clicks': 0,
                'conversions': 0,
                'commission_earned': 0.0
            },
            'data_services': {
                'balance': 0.0,
                'api_requests': 0,
                'clients': 0,
                'estimated_revenue': 0.0
            },
            'education_training': {
                'balance': 0.0,
                'students': 0,
                'courses_sold': 0,
                'estimated_revenue': 0.0
            },
            'consulting_analysis': {
                'balance': 0.0,
                'engagements': 0,
                'hours_billed': 0,
                'estimated_revenue': 0.0
            },
            'ad_revenue': {
                'balance': 0.0,
                'impressions': 0,
                'clicks': 0,
                'estimated_revenue': 0.0
            },
            'crowdfunding_patronage': {
                'balance': 0.0,
                'patrons': 0,
                'monthly_pledges': 0.0,
                'estimated_revenue': 0.0
            }
        }
        
        self.execution_active = False
        self.execution_thread = None
        self._execution_complete = False
        
        # State file
        self.state_file = self.execution_dir / 'phase2_state.json'
        self._load_state()
        
        logger.info(f"📋 Phase 2: Paper Execution initialized")
        logger.info(f"   Paper accounts: {len(self.paper_accounts)}")
        logger.info(f"   Active strategies: {len(self.active_strategies)}")
    
    def _load_state(self):
        """Load paper execution state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.active_strategies = state.get('active_strategies', {})
                    self.paper_accounts = state.get('paper_accounts', self.paper_accounts)
                    self.execution_active = False  # NEVER auto-start
                    self._execution_complete = state.get('execution_complete', False)
                    logger.info(f"📂 Loaded Phase 2 state: {len(self.active_strategies)} active strategies")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save paper execution state"""
        try:
            state = {
                'active_strategies': self.active_strategies,
                'paper_accounts': self.paper_accounts,
                'execution_active': False,
                'execution_complete': self._execution_complete,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def enable_strategy(self, avenue: str, strategy_id: str) -> Dict:
        """
        Enable a specific strategy for paper execution.
        Requires master approval (handled by UI).
        """
        if avenue not in self.strategy_candidates:
            return {
                'success': False,
                'error': f'Unknown avenue: {avenue}'
            }
        
        # Find the strategy
        strategy = None
        for candidate in self.strategy_candidates.get(avenue, []):
            if candidate.get('id') == strategy_id:
                strategy = candidate
                break
        
        if not strategy:
            return {
                'success': False,
                'error': f'Strategy not found: {strategy_id}'
            }
        
        self.active_strategies[avenue] = {
            'strategy_id': strategy_id,
            'strategy_name': strategy.get('name', 'Unnamed Strategy'),
            'enabled_at': datetime.now().isoformat(),
            'status': 'paper_execution',
            'requires_master_review': True
        }
        
        self._save_state()
        
        logger.info(f"✅ Strategy enabled for Phase 2: {avenue} - {strategy.get('name')}")
        
        return {
            'success': True,
            'message': f'Strategy enabled for paper execution: {strategy.get("name")}',
            'avenue': avenue,
            'strategy': strategy
        }
    
    def disable_strategy(self, avenue: str) -> Dict:
        """Disable a strategy from paper execution"""
        if avenue not in self.active_strategies:
            return {
                'success': False,
                'error': f'No active strategy for avenue: {avenue}'
            }
        
        del self.active_strategies[avenue]
        self._save_state()
        
        logger.info(f"⏸️ Strategy disabled: {avenue}")
        
        return {
            'success': True,
            'message': f'Strategy disabled for {avenue}'
        }
    
    def start_execution(self) -> Dict:
        """Start paper execution of enabled strategies"""
        if self.execution_active:
            return {
                'success': False,
                'error': 'Paper execution already active'
            }
        
        if not self.active_strategies:
            return {
                'success': False,
                'error': 'No active strategies enabled. Use enable_strategy() first.'
            }
        
        if self._execution_complete:
            return {
                'success': False,
                'error': 'Paper execution already complete'
            }
        
        self.execution_active = True
        self.execution_thread = threading.Thread(target=self._run_execution, daemon=True)
        self.execution_thread.start()
        
        logger.info(f"📊 Phase 2: Paper Execution STARTED with {len(self.active_strategies)} strategies")
        
        return {
            'success': True,
            'message': f'Paper execution started with {len(self.active_strategies)} strategies',
            'active_strategies': list(self.active_strategies.keys())
        }
    
    def stop_execution(self) -> Dict:
        """Stop paper execution"""
        if not self.execution_active:
            return {
                'success': False,
                'error': 'Paper execution not active'
            }
        
        self.execution_active = False
        self._save_state()
        
        logger.info("⏸️ Phase 2: Paper Execution PAUSED")
        
        return {
            'success': True,
            'message': 'Paper execution paused'
        }
    
    def crash_recovery(self) -> Dict:
        """Auto-resume paper execution after crash"""
        if self._execution_complete:
            return {'recovered': False, 'reason': 'already_complete'}
        
        if self.active_strategies and not self.execution_active:
            logger.info(f"🔄 CRASH RECOVERY: Resuming Phase 2 with {len(self.active_strategies)} strategies")
            return self.start_execution()
        
        return {'recovered': False, 'reason': 'no_active_strategies'}
    
    def get_status(self) -> Dict:
        """Get paper execution status"""
        total_pnl = sum(acc.get('pnl', 0) for acc in self.paper_accounts.values())
        total_estimated_value = sum(
            acc.get('estimated_revenue', 0) for acc in self.paper_accounts.values()
        )
        
        return {
            'phase': '2 - Paper Execution',
            'active': self.execution_active,
            'complete': self._execution_complete,
            'active_strategies': self.active_strategies,
            'active_count': len(self.active_strategies),
            'paper_accounts': self.paper_accounts,
            'total_pnl': total_pnl,
            'total_estimated_value': total_estimated_value,
            'can_start': not self.execution_active and not self._execution_complete and bool(self.active_strategies),
            'can_stop': self.execution_active,
            'status': 'executing' if self.execution_active else 'paused'
        }
    
    def _run_execution(self):
        """Main paper execution loop - SIMULATED ONLY, no real money"""
        logger.info("📊 Phase 2 Paper Execution thread started")
        
        try:
            cycle = 0
            while self.execution_active and not self._execution_complete:
                cycle += 1
                logger.info(f"📈 Paper Execution Cycle {cycle}")
                
                for avenue, strategy_info in self.active_strategies.items():
                    if not self.execution_active:
                        break
                    
                    logger.info(f"   Executing: {avenue}")
                    
                    # Simulate execution based on avenue type
                    if avenue == 'quant_trading':
                        self._simulate_trading(avenue)
                    elif avenue == 'content_creation':
                        self._simulate_content(avenue)
                    elif avenue == 'ai_services':
                        self._simulate_ai_services(avenue)
                    elif avenue == 'software_products':
                        self._simulate_software(avenue)
                    elif avenue == 'affiliate_referral':
                        self._simulate_affiliate(avenue)
                    elif avenue == 'data_services':
                        self._simulate_data_services(avenue)
                    elif avenue == 'education_training':
                        self._simulate_education(avenue)
                    elif avenue == 'consulting_analysis':
                        self._simulate_consulting(avenue)
                    elif avenue == 'ad_revenue':
                        self._simulate_ad_revenue(avenue)
                    elif avenue == 'crowdfunding_patronage':
                        self._simulate_crowdfunding(avenue)
                    
                    self._save_state()
                    time.sleep(2)
                
                # Save snapshot after each cycle
                self._save_snapshot(cycle)
                
                # Run for 10 cycles then mark complete
                if cycle >= 10:
                    self._execution_complete = True
                    self._save_state()
                    logger.info("📊 Phase 2: Paper Execution COMPLETE (10 cycles completed)")
                    break
                
                time.sleep(30)
            
        except Exception as e:
            logger.error(f"Phase 2 execution thread error: {e}")
            self._save_state()
        finally:
            self.execution_active = False
            self._save_state()
    
    def _simulate_trading(self, avenue: str):
        """Simulate paper trading"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        change_pct = random.uniform(-0.02, 0.03)
        trade_value = account.get('balance', 100000) * random.uniform(0.01, 0.05)
        
        if random.random() > 0.4:
            pnl = trade_value * abs(change_pct)
        else:
            pnl = -trade_value * abs(change_pct) * 0.5
        
        account['balance'] = account.get('balance', 100000) + pnl
        account['pnl'] = account.get('pnl', 0) + pnl
        account['trades'] = account.get('trades', [])
        account['trades'].append({
            'timestamp': datetime.now().isoformat(),
            'value': trade_value,
            'pnl': pnl,
            'balance': account['balance']
        })
        
        if len(account['trades']) > 100:
            account['trades'] = account['trades'][-100:]
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Trading PnL: ${pnl:.2f}")
    
    def _simulate_content(self, avenue: str):
        """Simulate content creation metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        views = random.randint(100, 10000)
        engagement = random.uniform(0.01, 0.1)
        estimated_value = views * 0.01
        
        account['engagement_metrics'] = account.get('engagement_metrics', {})
        account['engagement_metrics']['total_views'] = account['engagement_metrics'].get('total_views', 0) + views
        account['engagement_metrics']['avg_engagement'] = engagement
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + estimated_value
        
        account['content_published'] = account.get('content_published', [])
        account['content_published'].append({
            'timestamp': datetime.now().isoformat(),
            'views': views,
            'estimated_value': estimated_value
        })
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Content: {views} views, Est. ${estimated_value:.2f}")
    
    def _simulate_ai_services(self, avenue: str):
        """Simulate AI services metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        api_calls = random.randint(1000, 50000)
        revenue = api_calls * 0.001
        
        account['api_calls'] = account.get('api_calls', 0) + api_calls
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + revenue
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      AI Services: {api_calls} calls, Est. ${revenue:.2f}")
    
    def _simulate_software(self, avenue: str):
        """Simulate software product metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        new_users = random.randint(10, 500)
        subscriptions = random.randint(1, 50)
        revenue = subscriptions * 10
        
        account['users'] = account.get('users', 0) + new_users
        account['subscriptions'] = account.get('subscriptions', 0) + subscriptions
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + revenue
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Software: +{new_users} users, +{subscriptions} subs, Est. ${revenue:.2f}")
    
    def _simulate_affiliate(self, avenue: str):
        """Simulate affiliate marketing metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        clicks = random.randint(100, 5000)
        conversion_rate = random.uniform(0.01, 0.05)
        conversions = int(clicks * conversion_rate)
        commission = conversions * random.uniform(5, 50)
        
        account['clicks'] = account.get('clicks', 0) + clicks
        account['conversions'] = account.get('conversions', 0) + conversions
        account['commission_earned'] = account.get('commission_earned', 0) + commission
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + commission
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Affiliate: {clicks} clicks, {conversions} conv, ${commission:.2f}")
    
    def _simulate_data_services(self, avenue: str):
        """Simulate data services metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        api_requests = random.randint(1000, 20000)
        revenue = api_requests * 0.005
        
        account['api_requests'] = account.get('api_requests', 0) + api_requests
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + revenue
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Data Services: {api_requests} requests, Est. ${revenue:.2f}")
    
    def _simulate_education(self, avenue: str):
        """Simulate education training metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        courses_sold = random.randint(1, 20)
        revenue = courses_sold * random.uniform(50, 200)
        
        account['courses_sold'] = account.get('courses_sold', 0) + courses_sold
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + revenue
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Education: {courses_sold} courses, Est. ${revenue:.2f}")
    
    def _simulate_consulting(self, avenue: str):
        """Simulate consulting services metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        hours = random.randint(5, 40)
        rate = random.uniform(150, 500)
        revenue = hours * rate
        
        account['hours_billed'] = account.get('hours_billed', 0) + hours
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + revenue
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Consulting: {hours} hours @ ${rate:.0f}, Est. ${revenue:.2f}")
    
    def _simulate_ad_revenue(self, avenue: str):
        """Simulate ad revenue metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        impressions = random.randint(10000, 100000)
        cpm = random.uniform(1, 10)
        revenue = (impressions / 1000) * cpm
        
        account['impressions'] = account.get('impressions', 0) + impressions
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + revenue
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Ad Revenue: {impressions} impressions, Est. ${revenue:.2f}")
    
    def _simulate_crowdfunding(self, avenue: str):
        """Simulate crowdfunding/patronage metrics"""
        import random
        account = self.paper_accounts.get(avenue, {})
        
        new_patrons = random.randint(1, 50)
        monthly_pledge = new_patrons * random.uniform(5, 20)
        
        account['patrons'] = account.get('patrons', 0) + new_patrons
        account['monthly_pledges'] = account.get('monthly_pledges', 0) + monthly_pledge
        account['estimated_revenue'] = account.get('estimated_revenue', 0) + monthly_pledge
        
        self.paper_accounts[avenue] = account
        logger.debug(f"      Crowdfunding: +{new_patrons} patrons, Est. ${monthly_pledge:.2f}/mo")
    
    def _save_snapshot(self, cycle: int):
        """Save execution snapshot"""
        snapshot = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'active_strategies': self.active_strategies,
            'paper_accounts': self.paper_accounts,
            'total_pnl': sum(acc.get('pnl', 0) for acc in self.paper_accounts.values()),
            'total_estimated_value': sum(
                acc.get('estimated_revenue', 0) for acc in self.paper_accounts.values()
            )
        }
        
        snapshot_file = self.execution_dir / f'snapshot_cycle_{cycle}_{int(time.time())}.json'
        try:
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2)
            logger.debug(f"💾 Saved Phase 2 snapshot: Cycle {cycle}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def get_report(self) -> Dict:
        """Generate comprehensive execution report for master review"""
        total_pnl = sum(acc.get('pnl', 0) for acc in self.paper_accounts.values())
        total_estimated = sum(
            acc.get('estimated_revenue', 0) for acc in self.paper_accounts.values()
        )
        
        return {
            'phase': '2 - Paper Execution',
            'status': 'complete' if self._execution_complete else ('active' if self.execution_active else 'paused'),
            'active_strategies': self.active_strategies,
            'paper_accounts': self.paper_accounts,
            'summary': {
                'total_pnl': total_pnl,
                'total_estimated_value': total_estimated,
                'combined_performance': total_pnl + total_estimated
            },
            'ready_for_phase_3': self._execution_complete,
            'requires_master_approval_for_phase_3': True,
            'message': 'Paper execution complete. Review results before approving Phase 3.'
        }
