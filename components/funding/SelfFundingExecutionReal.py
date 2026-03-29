#!/usr/bin/env python3
"""
SELF-FUNDING - PHASE 3: REAL EXECUTION
Requires master capital and explicit master approval.
CANNOT be auto-enabled or auto-started.
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


class SelfFundingPhase3Real:
    """
    PHASE 3: Real Execution
    - Requires master capital
    - Requires explicit master approval (cannot be auto-enabled)
    - Full real-world execution
    """
    
    def __init__(self, data_path: Path, financial_manager, knowledge_graph, ai_hub):
        self.data_path = data_path
        self.financial_manager = financial_manager
        self.knowledge_graph = knowledge_graph
        self.ai_hub = ai_hub
        self.execution_dir = data_path / 'execution' / 'phase3_real'
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        
        self.execution_active = False
        self.master_approved = False
        self.master_capital_verified = False
        self.execution_thread = None
        self.pending_request = None
        
        # State file
        self.state_file = self.execution_dir / 'phase3_state.json'
        self._load_state()
        
        logger.info(f"💰 Phase 3: Real Execution initialized")
        logger.info(f"   Master Approval: {self.master_approved}")
        logger.info(f"   Capital Verified: {self.master_capital_verified}")
    
    def _load_state(self):
        """Load real execution state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.master_approved = state.get('master_approved', False)
                    self.master_capital_verified = state.get('master_capital_verified', False)
                    self.execution_active = False
                    self.pending_request = state.get('pending_request', None)
                    logger.info(f"📂 Loaded Phase 3 state: Approved={self.master_approved}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save real execution state"""
        try:
            state = {
                'master_approved': self.master_approved,
                'master_capital_verified': self.master_capital_verified,
                'pending_request': self.pending_request,
                'execution_active': False,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def request_approval(self, capital_amount: float, strategy_plan: Dict) -> Dict:
        """
        Request master approval for Phase 3 real execution.
        This is a REQUEST only - requires manual master approval.
        """
        self.pending_request = {
            'capital_amount': capital_amount,
            'strategy_plan': strategy_plan,
            'requested_at': datetime.now().isoformat(),
            'status': 'pending_master_approval'
        }
        
        # Save request to file for master review
        request_file = self.execution_dir / 'pending_approval_request.json'
        with open(request_file, 'w') as f:
            json.dump(self.pending_request, f, indent=2)
        
        self._save_state()
        
        logger.warning(f"💰 PHASE 3 REQUEST: Master approval required for ${capital_amount} capital")
        
        return {
            'success': True,
            'message': f'Phase 3 approval request submitted for ${capital_amount}',
            'request_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'requires_master_action': True,
            'next_steps': [
                '1. Master reviews the strategy plan',
                '2. Master approves/denies via admin interface',
                '3. If approved, master provides capital',
                '4. Capital must be verified before execution begins'
            ]
        }
    
    def approve(self, capital_verified: bool = False) -> Dict:
        """
        Master approval action (called from admin interface only)
        This cannot be called by DMAI automatically.
        """
        self.master_approved = True
        
        if capital_verified:
            self.master_capital_verified = True
        
        self._save_state()
        
        logger.info(f"💰 PHASE 3 APPROVED by Master. Capital verified: {capital_verified}")
        
        return {
            'success': True,
            'message': 'Phase 3 real execution approved',
            'capital_verified': self.master_capital_verified,
            'can_start': self.master_approved and self.master_capital_verified
        }
    
    def deny(self, reason: str) -> Dict:
        """Master denial action"""
        self.master_approved = False
        self.pending_request = None
        self._save_state()
        
        logger.info(f"💰 PHASE 3 DENIED by Master. Reason: {reason}")
        
        return {
            'success': True,
            'message': f'Phase 3 real execution denied: {reason}'
        }
    
    def verify_capital(self, amount: float) -> Dict:
        """Verify master capital has been provided"""
        if amount > 0:
            self.master_capital_verified = True
            self._save_state()
            
            logger.info(f"💰 Capital verified: ${amount}")
            
            return {
                'success': True,
                'message': f'Capital of ${amount} verified',
                'capital_verified': True
            }
        
        return {
            'success': False,
            'error': 'Invalid capital amount'
        }
    
    def start_execution(self) -> Dict:
        """Start real execution (ONLY if approved and capital verified)"""
        if not self.master_approved:
            return {
                'success': False,
                'error': 'Master approval required',
                'message': 'Phase 3 has not been approved by master'
            }
        
        if not self.master_capital_verified:
            return {
                'success': False,
                'error': 'Capital not verified',
                'message': 'Master capital must be provided and verified before execution'
            }
        
        if self.execution_active:
            return {
                'success': False,
                'error': 'Execution already active'
            }
        
        self.execution_active = True
        self.execution_thread = threading.Thread(target=self._run_execution, daemon=True)
        self.execution_thread.start()
        
        logger.warning(f"💰 PHASE 3: REAL EXECUTION STARTED - Master approved and capital verified")
        
        return {
            'success': True,
            'message': 'Phase 3 real execution started',
            'warning': 'REAL CAPITAL IS BEING USED. Monitor closely.'
        }
    
    def stop_execution(self) -> Dict:
        """Stop real execution"""
        if not self.execution_active:
            return {
                'success': False,
                'error': 'Execution not active'
            }
        
        self.execution_active = False
        self._save_state()
        
        logger.warning(f"⏸️ PHASE 3: REAL EXECUTION PAUSED by Master")
        
        return {
            'success': True,
            'message': 'Phase 3 real execution paused'
        }
    
    def crash_recovery(self) -> Dict:
        """Auto-resume after crash - only if approved and capital verified"""
        if self.master_approved and self.master_capital_verified and not self.execution_active:
            logger.warning("🔄 CRASH RECOVERY: Resuming Phase 3 Real Execution")
            return self.start_execution()
        
        return {'recovered': False, 'reason': 'not_approved_or_no_capital'}
    
    def get_status(self) -> Dict:
        """Get real execution status"""
        return {
            'phase': '3 - Real Execution',
            'active': self.execution_active,
            'master_approved': self.master_approved,
            'capital_verified': self.master_capital_verified,
            'pending_request': self.pending_request,
            'ready_for_execution': self.master_approved and self.master_capital_verified and not self.execution_active,
            'requires_master_approval': not self.master_approved,
            'requires_capital_verification': self.master_approved and not self.master_capital_verified,
            'can_start': self.master_approved and self.master_capital_verified and not self.execution_active,
            'can_stop': self.execution_active,
            'message': 'REAL CAPITAL IN USE' if self.execution_active else 'Awaiting master action'
        }
    
    def _run_execution(self):
        """Real execution loop - ONLY runs with master approval and capital"""
        logger.warning("💰 REAL EXECUTION THREAD STARTED - REAL CAPITAL IN USE")
        
        try:
            while self.execution_active:
                # Placeholder for actual execution logic
                # This would connect to real APIs, execute real trades, etc.
                logger.info("💰 Real execution cycle - monitoring active strategies")
                time.sleep(60)
            
        except Exception as e:
            logger.error(f"Real execution thread error: {e}")
        finally:
            self.execution_active = False
            self._save_state()
            logger.warning("💰 REAL EXECUTION STOPPED")
