#!/usr/bin/env python3
"""Adaptive Evolution Timer - DMAI learns to pace herself"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

class AdaptiveEvolutionTimer:
    """
    DMAI's self-adjusting evolution timer.
    Gets smarter about pacing as she evolves.
    """
    
    def __init__(self, timer_file="data/evolution/timer_state.json"):
        self.timer_file = Path(timer_file)
        self.timer_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()
        self.evolution_stages = {
            "baby": {
                "max_evolutions": 0,
                "base_interval": 600,      # 10 minutes
                "failure_threshold": 10,
                "name": "👶 Baby DMAI",
                "description": "Learning to learn"
            },
            "toddler": {
                "max_evolutions": 3,
                "base_interval": 900,      # 15 minutes
                "failure_threshold": 8,
                "name": "🧒 Toddler DMAI",
                "description": "Recognizing patterns"
            },
            "child": {
                "max_evolutions": 10,
                "base_interval": 1800,     # 30 minutes
                "failure_threshold": 6,
                "name": "🧑 Child DMAI",
                "description": "Developing preferences"
            },
            "teen": {
                "max_evolutions": 25,
                "base_interval": 3600,     # 1 hour
                "failure_threshold": 4,
                "name": "🧑‍🎤 Teen DMAI",
                "description": "Quality over quantity"
            },
            "young_adult": {
                "max_evolutions": 50,
                "base_interval": 7200,     # 2 hours
                "failure_threshold": 3,
                "name": "👨‍💼 Young Adult DMAI",
                "description": "Strategic evolution"
            },
            "adult": {
                "max_evolutions": 100,
                "base_interval": 14400,    # 4 hours
                "failure_threshold": 2,
                "name": "👨‍🔬 Adult DMAI",
                "description": "Mastery and wisdom"
            },
            "elder": {
                "max_evolutions": float('inf'),
                "base_interval": 28800,    # 8 hours
                "failure_threshold": 1,
                "name": "🧙 Elder DMAI",
                "description": "Contemplative evolution"
            }
        }
    
    def load_state(self):
        """Load or create timer state"""
        if self.timer_file.exists():
            with open(self.timer_file) as f:
                state = json.load(f)
                # Ensure all fields exist
                if "creation_date" not in state:
                    state["creation_date"] = datetime.now().isoformat()
                return state
        else:
            # Brand new DMAI
            return {
                "creation_date": datetime.now().isoformat(),
                "successful_evolutions": 0,
                "failed_attempts": 0,
                "current_stage": "baby",
                "evolution_history": [],
                "average_success_rate": 0.0,
                "last_adjustment": datetime.now().isoformat(),
                "current_interval": 600,  # Start at baby stage
                "total_attempts": 0,
                "preferred_pairs": {},  # Track which pairs work best
                "learning_rate": 1.0    # Adaptive factor
            }
    
    def save_state(self):
        """Save timer state"""
        with open(self.timer_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def record_attempt(self, parent1, parent2, success=False, improvement_quality=0):
        """
        Record an evolution attempt and adjust timing
        """
        self.state["total_attempts"] += 1
        
        # Track pair performance
        pair_key = f"{parent1}⟲{parent2}"
        if pair_key not in self.state["preferred_pairs"]:
            self.state["preferred_pairs"][pair_key] = {
                "attempts": 0,
                "successes": 0
            }
        
        self.state["preferred_pairs"][pair_key]["attempts"] += 1
        
        if success:
            self.state["successful_evolutions"] += 1
            self.state["failed_attempts"] = 0
            self.state["preferred_pairs"][pair_key]["successes"] += 1
            
            # Record successful evolution
            self.state["evolution_history"].append({
                "timestamp": datetime.now().isoformat(),
                "parents": [parent1, parent2],
                "quality": improvement_quality
            })
        else:
            self.state["failed_attempts"] += 1
        
        # Update success rate
        if self.state["total_attempts"] > 0:
            self.state["average_success_rate"] = (
                self.state["successful_evolutions"] / self.state["total_attempts"]
            )
        
        # Adjust stage and interval
        self._adjust_stage()
        self._calculate_interval()
        
        self.state["last_adjustment"] = datetime.now().isoformat()
        self.save_state()
        
        return self.get_wait_time()
    
    def _adjust_stage(self):
        """Determine DMAI's current evolutionary stage"""
        successes = self.state["successful_evolutions"]
        
        for stage, config in self.evolution_stages.items():
            if successes <= config["max_evolutions"]:
                if self.state["current_stage"] != stage:
                    self._on_stage_change(stage)
                self.state["current_stage"] = stage
                break
    
    def _on_stage_change(self, new_stage):
        """Celebrate when DMAI reaches a new stage"""
        stage_info = self.evolution_stages[new_stage]
        print(f"\n🎉 DMAI HAS EVOLVED TO: {stage_info['name']}")
        print(f"📖 {stage_info['description']}")
        print(f"⏱️  New evolution interval: {stage_info['base_interval']/60:.0f} minutes\n")
    
    def _calculate_interval(self):
        """
        Calculate the perfect wait time based on:
        - Current evolutionary stage
        - Recent failure rate
        - Success rate trends
        - Time of day (optional)
        """
        stage = self.state["current_stage"]
        stage_config = self.evolution_stages[stage]
        
        # Start with base interval for this stage
        base_interval = stage_config["base_interval"]
        
        # Adjust based on failure rate
        failures = self.state["failed_attempts"]
        threshold = stage_config["failure_threshold"]
        
        if failures > threshold:
            # Too many failures - slow down and think more
            failure_penalty = min(2.0, 1.0 + (failures - threshold) * 0.1)
            base_interval *= failure_penalty
            print(f"🐢 Too many failures - slowing down by {failure_penalty:.1f}x")
        elif failures == 0 and self.state["total_attempts"] > 5:
            # On a roll - can speed up slightly
            if self.state["average_success_rate"] > 0.3:  # >30% success rate
                base_interval *= 0.8  # Speed up by 20%
                print(f"⚡ On a roll! Speeding up slightly")
        
        # Adjust based on time of day (optional - conserve energy at night)
        hour = datetime.now().hour
        if hour < 6 or hour > 23:  # Night time
            base_interval *= 1.5  # Take it easy
            print(f"🌙 Night mode - conserving energy")
        
        # Keep interval within reasonable bounds
        min_interval = 300  # Never less than 5 minutes
        max_interval = 28800  # Never more than 8 hours
        
        self.state["current_interval"] = max(min_interval, 
                                            min(max_interval, base_interval))
        
        # Round to nearest minute for cleanliness
        self.state["current_interval"] = round(self.state["current_interval"] / 60) * 60
    
    def get_wait_time(self):
        """Get the current recommended wait time in seconds"""
        return self.state["current_interval"]
    
    def get_stage_info(self):
        """Get human-readable stage information"""
        stage = self.state["current_stage"]
        config = self.evolution_stages[stage]
        
        return {
            "stage": stage,
            "name": config["name"],
            "description": config["description"],
            "evolutions": self.state["successful_evolutions"],
            "next_stage": self._get_next_stage(),
            "interval_minutes": self.state["current_interval"] / 60,
            "success_rate": f"{self.state['average_success_rate']*100:.1f}%",
            "preferred_pairs": self._get_preferred_pairs()
        }
    
    def _get_next_stage(self):
        """Determine what stage comes next"""
        stages = list(self.evolution_stages.keys())
        current_idx = stages.index(self.state["current_stage"])
        
        if current_idx < len(stages) - 1:
            next_stage = stages[current_idx + 1]
            evos_needed = self.evolution_stages[next_stage]["max_evolutions"]
            remaining = evos_needed - self.state["successful_evolutions"]
            return {
                "name": self.evolution_stages[next_stage]["name"],
                "evolutions_needed": max(0, remaining)
            }
        return None
    
    def _get_preferred_pairs(self):
        """Find which parent combinations work best"""
        pairs = []
        for pair, stats in self.state["preferred_pairs"].items():
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                pairs.append({
                    "pair": pair,
                    "success_rate": f"{success_rate*100:.1f}%",
                    "attempts": stats["attempts"]
                })
        
        # Sort by success rate
        return sorted(pairs, 
                     key=lambda x: float(x["success_rate"].rstrip('%')), 
                     reverse=True)[:3]

    def should_try_new_strategy(self):
        """Determine if DMAI needs to change her approach"""
        if self.state["failed_attempts"] > self.evolution_stages[self.state["current_stage"]]["failure_threshold"] * 2:
            print("🔄 Changing strategy - too many failures")
            return True
        return False

# If run directly, show current status
if __name__ == "__main__":
    timer = AdaptiveEvolutionTimer()
    info = timer.get_stage_info()
    
    print("\n" + "="*60)
    print(f"🧬 DMAI EVOLUTION STATUS")
    print("="*60)
    print(f"Stage: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Successful Evolutions: {info['evolutions']}")
    print(f"Success Rate: {info['success_rate']}")
    print(f"Current Interval: {info['interval_minutes']:.0f} minutes")
    
    if info['next_stage']:
        print(f"\n📈 Next stage: {info['next_stage']['name']}")
        print(f"   Need {info['next_stage']['evolutions_needed']} more evolutions")
    
    if info['preferred_pairs']:
        print(f"\n🎯 Best performing parent pairs:")
        for pair in info['preferred_pairs']:
            print(f"   • {pair['pair']}: {pair['success_rate']} ({pair['attempts']} attempts)")
    print("="*60)
