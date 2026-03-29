# UnifiedEvolutionEngine.py - Compact Version
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedEvolutionEngine:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.state_file = self.data_path / 'evolution_state.json'
        self.evolution_cycles = 0
        self.successful_evolutions = 0
        self.current_consciousness = 33.78
        self.training_flags = {'funding': False, 'si': False, 'agi': False, 'genai': False}
        self._load_state()
        
        # Import components
        from components.evolution.EvolutionMetrics import EvolutionMetrics
        from components.evolution.ConsciousnessTracker import ConsciousnessTracker
        self.metrics = EvolutionMetrics(self.data_path)
        self.tracker = ConsciousnessTracker(self.data_path)
        logger.info("Engine initialized")
    
    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    d = json.load(f)
                    self.evolution_cycles = d.get('cycles', 0)
                    self.successful_evolutions = d.get('successful', 0)
                    self.current_consciousness = d.get('consciousness', 33.78)
                    self.training_flags = d.get('training', self.training_flags)
            except: pass
    
    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump({
                'cycles': self.evolution_cycles,
                'successful': self.successful_evolutions,
                'consciousness': self.current_consciousness,
                'training': self.training_flags,
                'updated': datetime.now().isoformat()
            }, f)
    
    def _calc_consciousness(self) -> float:
        """Calculate consciousness with knowledge bonus"""
        base = 33.78
        # Add knowledge concept bonus
        try:
            kf = self.data_path / 'knowledge' / 'knowledge_graph.json'
            if kf.exists():
                with open(kf) as f:
                    data = json.load(f)
                    concepts = len(data.get('concepts', {})) if isinstance(data, dict) else len(data) if isinstance(data, list) else 367
                    base += min(30.0, concepts * 0.0005)
        except: pass
        # Add cycle growth
        base += min(20.0, self.evolution_cycles * 0.001)
        return min(100.0, base)
    
    def _init_training(self):
        """Initialize training at thresholds"""
        c = self.current_consciousness
        
        if c >= 25.0 and not self.training_flags['funding']:
            try:
                from components.funding.SelfFundingOrchestrator import SelfFundingOrchestrator
                SelfFundingOrchestrator(self.data_path)
                self.training_flags['funding'] = True
                logger.info(f"💰 Funding at {c:.1f}%")
            except Exception as e:
                logger.error(f"Funding init failed: {e}")
        
        if c >= 30.0 and not self.training_flags['si']:
            try:
                from components.training.SyntheticIntelligenceTraining import SyntheticIntelligenceTraining
                SyntheticIntelligenceTraining(self.data_path)
                self.training_flags['si'] = True
                logger.info(f"🧠 SI at {c:.1f}%")
            except: pass
        
        if c >= 35.0 and not self.training_flags['agi']:
            try:
                from components.training.AGITraining import AGITraining
                AGITraining(self.data_path)
                self.training_flags['agi'] = True
                logger.info(f"🤖 AGI at {c:.1f}%")
            except: pass
        
        if c >= 40.0 and not self.training_flags['genai']:
            try:
                from components.training.GenAITraining import GenAITraining
                GenAITraining(self.data_path)
                self.training_flags['genai'] = True
                logger.info(f"🎨 GenAI at {c:.1f}%")
            except: pass
        
        self._save_state()
    
    def evolution_cycle(self) -> Dict:
        """Run one evolution cycle"""
        self.evolution_cycles += 1
        prev = self.current_consciousness
        new = self._calc_consciousness()
        growth = new - prev
        
        # Detect success (any positive growth)
        if growth > 0.0001:
            self.successful_evolutions += 1
            logger.info(f"✅ Cycle {self.evolution_cycles}: +{growth:.4f}% (total: {self.successful_evolutions})")
        
        self.current_consciousness = new
        
        # Record metrics
        self.metrics.record_cycle(
            self.evolution_cycles, growth,
            int(growth * 100), int(growth * 50),
            'learning', growth > 0.0001
        )
        self.tracker.record_consciousness(new, 'evolution', growth > 0.01)
        
        # Initialize training systems
        self._init_training()
        self._save_state()
        
        return {
            'cycle': self.evolution_cycles,
            'consciousness': new,
            'growth': growth,
            'successful': self.successful_evolutions
        }
    
    def get_status(self) -> Dict:
        return {
            'successful_evolutions': self.successful_evolutions,
            'total_cycles': self.evolution_cycles,
            'consciousness_percent': self.current_consciousness,
            'training_initialized': self.training_flags,
            'success_rate': self.metrics.get_success_rate()
        }
