#!/usr/bin/env python3
"""
Dynamic Evolution Scheduler
Adjusts evolution frequency based on system metrics
"""
import time
import json
import logging
import psutil
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger('dynamic_scheduler')

class DynamicScheduler:
    def __init__(self, base_interval=3600):  # 1 hour base
        self.base_interval = base_interval
        self.min_interval = 300   # 5 minutes
        self.max_interval = 7200   # 2 hours
        self.last_adjustment = datetime.now()
        self.metrics_history = []
        
    def calculate_optimal_interval(self):
        """Calculate optimal evolution interval based on current conditions"""
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Check API rate limits from harvester logs
        api_availability = self.check_api_availability()
        
        # Check recent evolution success rate
        success_rate = self.get_evolution_success_rate()
        
        # Calculate adjustment factor
        adjustment = 1.0
        
        # High CPU/memory usage -> increase interval
        if cpu_percent > 80 or memory_percent > 80:
            adjustment *= 1.5
            logger.info(f"High resource usage (CPU: {cpu_percent}%, RAM: {memory_percent}%) -> slowing down")
        
        # API rate limits low -> decrease interval
        if api_availability < 0.3:  # Less than 30% API capacity
            adjustment *= 2.0
            logger.info(f"API rate limits low ({api_availability:.1%}) -> slowing down")
        elif api_availability > 0.7:  # More than 70% API capacity
            adjustment *= 0.8
            logger.info(f"Good API availability ({api_availability:.1%}) -> speeding up")
        
        # High success rate -> decrease interval (more frequent evolutions)
        if success_rate > 0.7:
            adjustment *= 0.7
            logger.info(f"High evolution success rate ({success_rate:.1%}) -> speeding up")
        elif success_rate < 0.3:
            adjustment *= 1.3
            logger.info(f"Low evolution success rate ({success_rate:.1%}) -> slowing down")
        
        # Calculate new interval
        new_interval = self.base_interval * adjustment
        
        # Clamp to min/max
        new_interval = max(self.min_interval, min(self.max_interval, new_interval))
        
        # Store metrics
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'cpu': cpu_percent,
            'memory': memory_percent,
            'api_availability': api_availability,
            'success_rate': success_rate,
            'interval': new_interval
        })
        
        # Keep last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return int(new_interval)
    
    def check_api_availability(self):
        """Check GitHub API rate limit availability"""
        try:
            import requests
            response = requests.get('https://api.github.com/rate_limit')
            if response.status_code == 200:
                data = response.json()
                remaining = data['rate']['remaining']
                limit = data['rate']['limit']
                return remaining / limit
        except:
            pass
        return 1.0  # Default to full availability if can't check
    
    def get_evolution_success_rate(self):
        """Calculate evolution success rate from recent cycles"""
        log_file = Path('logs/evolution.log')
        if not log_file.exists():
            return 0.5
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
            
            successes = 0
            total = 0
            for line in lines:
                if 'intelligence increased' in line:
                    successes += 1
                    total += 1
                elif 'Evolution cycle complete' in line:
                    total += 1
            
            return successes / max(total, 1)
        except:
            return 0.5
    
    def save_state(self):
        """Save scheduler state"""
        state = {
            'last_adjustment': self.last_adjustment.isoformat(),
            'metrics_history': self.metrics_history[-10:],  # Last 10 adjustments
            'current_interval': self.base_interval
        }
        
        with open('logs/scheduler_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def run(self):
        """Main scheduler loop"""
        logger.info("Starting dynamic evolution scheduler")
        
        while True:
            optimal_interval = self.calculate_optimal_interval()
            
            if optimal_interval != self.base_interval:
                logger.info(f"Adjusting evolution interval: {self.base_interval}s -> {optimal_interval}s")
                self.base_interval = optimal_interval
                self.last_adjustment = datetime.now()
                
                # Update evolution_engine.py config
                self.update_evolution_config(optimal_interval)
            
            self.save_state()
            time.sleep(300)  # Check every 5 minutes
    
    def update_evolution_config(self, interval):
        """Update evolution engine with new interval"""
        # This would modify the evolution_engine.py or its config
        # For now, just log it
        logger.info(f"Evolution interval set to {interval} seconds")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    scheduler = DynamicScheduler()
    
    # Run once to test
    interval = scheduler.calculate_optimal_interval()
    print(f"Optimal interval: {interval} seconds ({interval/60:.1f} minutes)")
