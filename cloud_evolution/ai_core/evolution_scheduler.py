#!/usr/bin/env python3
"""Automatic evolution scheduler for DMAI"""
import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).parent.parent))))))

from ai_core.core_brain import DMAIBrain
from ai_core.evolution_engine import EvolutionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolutionScheduler:
    """Manages when DMAI evolves"""
    
    def __init__(self):
        self.brain = DMAIBrain()
        self.engine = EvolutionEngine()
        self.evolution_log = 'ai_core/evolution/schedule_log.json'
        self.load_schedule()
        
    def load_schedule(self):
        """Load evolution schedule"""
        if os.path.exists(self.evolution_log):
            with open(self.evolution_log, 'r') as f:
                self.schedule = json.load(f)
        else:
            self.schedule = {
                "last_evolution": None,
                "next_evolution": None,
                "evolution_count": 0,
                "schedule_type": "adaptive",  # adaptive, daily, weekly
                "performance_threshold": 0.7
            }
    
    def save_schedule(self):
        with open(self.evolution_log, 'w') as f:
            json.dump(self.schedule, f, indent=2)
    
    def should_evolve_now(self):
        """Determine if it's time to evolve"""
        now = datetime.now()
        
        # Check if we've ever evolved
        if not self.schedule["last_evolution"]:
            return True
        
        last = datetime.fromisoformat(self.schedule["last_evolution"])
        
        # Adaptive scheduling based on performance
        if self.schedule["schedule_type"] == "adaptive":
            performance = self.get_current_performance()
            
            if performance < 0.5:  # Poor performance - evolve sooner
                return (now - last) > timedelta(hours=12)
            elif performance < 0.7:  # Medium performance - evolve daily
                return (now - last) > timedelta(days=1)
            else:  # Good performance - evolve weekly
                return (now - last) > timedelta(days=7)
        
        # Simple daily evolution
        elif self.schedule["schedule_type"] == "daily":
            return (now - last) > timedelta(days=1)
        
        # Simple weekly evolution
        elif self.schedule["schedule_type"] == "weekly":
            return (now - last) > timedelta(days=7)
        
        return False
    
    def get_current_performance(self):
        """Get current performance metrics"""
        # In real implementation, this would track actual performance
        # For now, estimate based on capability levels
        if self.brain.capabilities:
            avg_level = sum(c.get("level", 0) for c in self.brain.capabilities.values()) / len(self.brain.capabilities)
            return avg_level
        return 0.5
    
    def evolve(self):
        """Run evolution cycle"""
        logger.info("🧬 Starting scheduled evolution")
        
        # Get current metrics
        metrics = {
            "accuracy": self.brain.calculate_accuracy(),
            "speed": self.brain.calculate_speed(),
            "knowledge_gaps": len(self.brain.knowledge_base.get("learned", [])),
            "avg_capability": self.get_current_performance()
        }
        
        brain_state = {
            "capabilities": self.brain.capabilities,
            "knowledge": self.brain.knowledge_base
        }
        
        # Run evolution
        result = self.engine.evolve(brain_state, metrics)
        
        # Update schedule
        now = datetime.now()
        self.schedule["last_evolution"] = now.isoformat()
        self.schedule["evolution_count"] += 1
        
        # Calculate next evolution
        if self.schedule["schedule_type"] == "adaptive":
            if result.get("score", 0) > 0.7:
                next_evo = now + timedelta(days=7)
            elif result.get("score", 0) > 0.5:
                next_evo = now + timedelta(days=1)
            else:
                next_evo = now + timedelta(hours=12)
        else:
            next_evo = now + timedelta(days=1)
        
        self.schedule["next_evolution"] = next_evo.isoformat()
        self.save_schedule()
        
        logger.info(f"✅ Evolution complete. Generation: {self.engine.generation}")
        return result
    
    def run_forever(self):
        """Run scheduler continuously"""
        logger.info("🚀 Evolution scheduler started")
        
        while True:
            try:
                if self.should_evolve_now():
                    self.evolve()
                
                # Check every hour
                time.sleep(3600)
                
            except KeyboardInterrupt:
                logger.info("Stopping evolution scheduler")
                break
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                time.sleep(3600)

if __name__ == "__main__":
    scheduler = EvolutionScheduler()
    
    # Run once now for testing
    if "--now" in sys.argv:
        print("Running evolution now...")
        result = scheduler.evolve()
        print(f"Evolution complete: {result}")
    
    # Run forever
    elif "--daemon" in sys.argv:
        scheduler.run_forever()
    
    else:
        print("Usage: python evolution_scheduler.py [--now] [--daemon]")
        print("  --now    : Run evolution once immediately")
        print("  --daemon : Run continuously in background")
