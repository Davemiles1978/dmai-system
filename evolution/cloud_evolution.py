#!/usr/bin/env python3
"""Cloud-optimized evolution service with heartbeat monitoring"""

import os
import time
import json
import requests
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from adaptive_timer import AdaptiveEvolutionTimer

class CloudEvolutionService:
    """Runs DMAI evolution in the cloud with 24/7 monitoring"""
    
    def __init__(self):
        self.timer = AdaptiveEvolutionTimer()
        self.heartbeat_url = os.getenv('HEARTBEAT_URL')  # For monitoring
        self.service_name = "dmai-evolution"
        self.version = "1.0"
        
    def send_heartbeat(self, status):
        """Let monitoring know we're alive"""
        if self.heartbeat_url:
            try:
                requests.post(self.heartbeat_url, json={
                    'service': self.service_name,
                    'status': status,
                    'timestamp': datetime.now().isoformat(),
                    'stage': self.timer.get_stage_info()['name'],
                    'evolutions': self.timer.state['successful_evolutions']
                })
            except:
                pass  # Silent fail - heartbeat is optional
    
    def run_cycle(self):
        """Execute one evolution cycle with proper error handling"""
        try:
            print(f"\n🧬 Cycle at {datetime.now().isoformat()}")
            
            # Run the evolution engine
            result = subprocess.run(
                [sys.executable, "evolution/evolution_engine.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Check for success
            success = "✅ Improvement passed all tests!" in result.stdout
            
            # Record with timer
            wait_time = self.timer.record_attempt(
                "DMAI", "DMAI",
                success=success
            )
            
            # Log output
            self.log_cycle(result, success)
            
            return wait_time
            
        except subprocess.TimeoutExpired:
            print("⚠️ Evolution cycle timed out")
            self.send_heartbeat("timeout")
            return 300  # Wait 5 minutes on timeout
            
        except Exception as e:
            print(f"❌ Error in evolution cycle: {e}")
            self.send_heartbeat("error")
            return 600  # Wait 10 minutes on error
    
    def log_cycle(self, result, success):
        """Log cycle results for debugging"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'stage': self.timer.state['current_stage'],
            'evolutions': self.timer.state['successful_evolutions'],
            'output': result.stdout[-500:] if result.stdout else '',
            'errors': result.stderr[-500:] if result.stderr else ''
        }
        
        # Write to log file
        with open('logs/cloud_evolution.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def run_forever(self):
        """Main loop - runs 24/7"""
        print(f"\n{'='*60}")
        print(f"☁️ DMAI CLOUD EVOLUTION SERVICE STARTED")
        print(f"PID: {os.getpid()}")
        print(f"Stage: {self.timer.get_stage_info()['name']}")
        print(f"{'='*60}\n")
        
        cycle_count = 0
        self.send_heartbeat("started")
        
        while True:
            try:
                cycle_count += 1
                print(f"\n{'─'*40}")
                print(f"Cycle #{cycle_count}")
                print(f"{'─'*40}")
                
                # Run evolution
                wait_time = self.run_cycle()
                
                # Send heartbeat
                self.send_heartbeat("running")
                
                # Show next cycle info
                info = self.timer.get_stage_info()
                print(f"\n⏱️  Next cycle in {wait_time/60:.1f} minutes")
                print(f"📊 Status: {info['name']} | {info['success_rate']} success")
                print(f"📈 Evolutions: {info['evolutions']}")
                
                # Smart sleep - wake up every minute to check for signals
                for minute in range(int(wait_time / 60)):
                    time.sleep(60)
                    
                    # Optional: Could check for commands here
                    if Path("/tmp/stop_evolution").exists():
                        print("\n🛑 Stop signal received")
                        self.send_heartbeat("stopped")
                        Path("/tmp/stop_evolution").unlink()
                        return
                        
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down cloud evolution service")
                self.send_heartbeat("stopped")
                break
                
            except Exception as e:
                print(f"❌ Critical error: {e}")
                self.send_heartbeat("crashed")
                time.sleep(300)  # Wait 5 minutes before restart

if __name__ == "__main__":
    service = CloudEvolutionService()
    service.run_forever()
