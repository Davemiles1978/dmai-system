#!/usr/bin/env python3
"""
SELF-FUNDING ORCHESTRATOR
Unified interface for all three phases:
- Phase 1: Knowledge Acquisition (SelfFundingTraining)
- Phase 2: Paper Execution (SelfFundingPhase2Paper)
- Phase 3: Real Execution (SelfFundingPhase3Real)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

from .SelfFundingTraining import SelfFundingTraining
from .SelfFundingExecutionPaper import SelfFundingPhase2Paper
from .SelfFundingExecutionReal import SelfFundingPhase3Real

logger = logging.getLogger(__name__)


class SelfFundingOrchestrator:
    """
    Unified orchestrator for all self-funding phases.
    Phase 1: Knowledge Acquisition (always available)
    Phase 2: Paper Execution (requires master strategy approval)
    Phase 3: Real Execution (requires master approval + capital)
    """
    
    def __init__(self, data_path: Path, financial_manager, knowledge_graph, ai_hub):
        self.data_path = data_path
        self.financial_manager = financial_manager
        self.knowledge_graph = knowledge_graph
        self.ai_hub = ai_hub
        
        # Phase 1: Knowledge Acquisition
        self.training = SelfFundingTraining(data_path, financial_manager, knowledge_graph, ai_hub)
        
        # Phase 2: Paper Execution (initially None, created when Phase 1 complete)
        self.paper: Optional[SelfFundingPhase2Paper] = None
        
        # Phase 3: Real Execution
        self.real = SelfFundingPhase3Real(data_path, financial_manager, knowledge_graph, ai_hub)
        
        self.current_phase = 1  # Start with Phase 1
        
        logger.info("💰 Self-Funding Orchestrator initialized")
    
    # ========================================================================
    # Phase 1: Knowledge Acquisition
    # ========================================================================
    
    def start_learning(self, avenue: str = None) -> Dict:
        """Start Phase 1 knowledge acquisition"""
        return self.training.start_learning(avenue)
    
    def stop_learning(self) -> Dict:
        """Stop Phase 1 knowledge acquisition"""
        return self.training.stop_learning()
    
    def get_learning_status(self) -> Dict:
        """Get Phase 1 status"""
        return self.training.get_status()
    
    def get_strategy_candidates(self, avenue: str = None) -> Dict:
        """Get strategy candidates from Phase 1"""
        return self.training.get_strategy_candidates(avenue)
    
    def fix_concept_counting(self) -> Dict:
        """Fix concept counting - passes through to training"""
        return self.training.fix_concept_counting()
    
    def get_phase_2_requirements(self) -> Dict:
        return self.training.get_phase_2_requirements()
    
    # ========================================================================
    # Phase 2: Paper Execution
    # ========================================================================
    
    def transition_to_phase_2(self, approved_strategies: Dict = None) -> Dict:
        """
        Transition to Phase 2 after Phase 1 complete.
        Master must approve which strategies to enable.
        """
        if not self.training._ready_for_phase_2():
            return {
                'success': False,
                'error': 'Phase 1 not complete',
                'progress': self.training.get_progress(),
                'required': 'Complete all revenue avenue learning first'
            }
        
        # Initialize Phase 2 with strategy candidates from Phase 1
        strategy_candidates = self.training.strategy_candidates
        
        self.paper = SelfFundingPhase2Paper(
            self.data_path,
            self.knowledge_graph,
            self.ai_hub,
            strategy_candidates
        )
        
        # If strategies are pre-approved, enable them
        if approved_strategies:
            for avenue, strategy_id in approved_strategies.items():
                self.paper.enable_strategy(avenue, strategy_id)
        
        self.current_phase = 2
        
        logger.info(f"🔄 Transitioned to Phase 2: Paper Execution")
        
        return {
            'success': True,
            'message': 'Transitioned to Phase 2',
            'phase_2_status': self.paper.get_status()
        }
    
    def enable_strategy(self, avenue: str, strategy_id: str) -> Dict:
        """Enable a strategy for Phase 2 paper execution"""
        if not self.paper:
            return {
                'success': False,
                'error': 'Phase 2 not initialized. Call transition_to_phase_2() first.'
            }
        return self.paper.enable_strategy(avenue, strategy_id)
    
    def disable_strategy(self, avenue: str) -> Dict:
        """Disable a strategy from Phase 2"""
        if not self.paper:
            return {
                'success': False,
                'error': 'Phase 2 not initialized'
            }
        return self.paper.disable_strategy(avenue)
    
    def start_phase_2(self) -> Dict:
        """Start Phase 2 paper execution"""
        if not self.paper:
            return {
                'success': False,
                'error': 'Phase 2 not initialized. Call transition_to_phase_2() first.'
            }
        return self.paper.start_execution()
    
    def stop_phase_2(self) -> Dict:
        """Stop Phase 2 paper execution"""
        if not self.paper:
            return {
                'success': False,
                'error': 'Phase 2 not initialized'
            }
        return self.paper.stop_execution()
    
    def get_phase_2_status(self) -> Dict:
        """Get Phase 2 status"""
        if not self.paper:
            return {'phase': '2 - Paper Execution', 'initialized': False}
        return self.paper.get_status()
    
    # ========================================================================
    # Phase 3: Real Execution
    # ========================================================================
    
    def transition_to_phase_3(self, capital_amount: float, strategy_plan: Dict) -> Dict:
        """
        Request transition to Phase 3.
        This triggers approval request - does NOT automatically start.
        """
        if not self.paper or not self.paper._execution_complete:
            return {
                'success': False,
                'error': 'Phase 2 not complete',
                'phase_2_status': self.paper.get_status() if self.paper else 'not_initialized'
            }
        
        self.current_phase = 3
        return self.real.request_approval(capital_amount, strategy_plan)
    
    def approve_phase_3(self, capital_verified: bool = False) -> Dict:
        """Master approval for Phase 3 (admin only)"""
        return self.real.approve(capital_verified)
    
    def deny_phase_3(self, reason: str) -> Dict:
        """Master denial for Phase 3 (admin only)"""
        return self.real.deny(reason)
    
    def verify_phase_3_capital(self, amount: float) -> Dict:
        """Verify capital for Phase 3"""
        return self.real.verify_capital(amount)
    
    def start_phase_3(self) -> Dict:
        """Start Phase 3 real execution"""
        return self.real.start_execution()
    
    def stop_phase_3(self) -> Dict:
        """Stop Phase 3 real execution"""
        return self.real.stop_execution()
    
    def get_phase_3_status(self) -> Dict:
        """Get Phase 3 status"""
        return self.real.get_status()
    
    # ========================================================================
    # Unified Methods
    # ========================================================================
    
    def crash_recovery(self) -> Dict:
        """Auto-resume after system crash"""
        recovered = []
        
        # Check Phase 1
        result = self.training.crash_recovery()
        if result.get('recovered'):
            recovered.append('phase_1')
        
        # Check Phase 2
        if self.paper:
            result = self.paper.crash_recovery()
            if result.get('recovered'):
                recovered.append('phase_2')
        
        # Check Phase 3
        result = self.real.crash_recovery()
        if result.get('recovered'):
            recovered.append('phase_3')
        
        return {
            'recovered': len(recovered) > 0,
            'recovered_phases': recovered
        }
    
    def get_status(self) -> Dict:
        """Get unified status across all phases"""
        return {
            'current_phase': self.current_phase,
            'phase_1': self.training.get_status(),
            'phase_2': self.get_phase_2_status(),
            'phase_3': self.get_phase_3_status(),
            'phase_2_available': self.training._ready_for_phase_2(),
            'phase_3_available': self.paper and self.paper._execution_complete if self.paper else False
        }
    def status(self) -> Dict:
        """Backward compatibility alias - returns flat Phase 1 status"""
        return self.training.get_status()


    
    def get_comprehensive_report(self) -> Dict:
        """Get comprehensive report across all phases"""
        return {
            'phase_1': self.training.get_learning_summary(),
            'phase_2': self.paper.get_report() if self.paper else {'status': 'not_started'},
            'phase_3': self.real.get_status(),
            'current_phase': self.current_phase
        }
