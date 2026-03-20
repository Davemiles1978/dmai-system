#!/usr/bin/env python3
"""
Adaptive Cleanup System - Cleans based on evolution rate
More frequent cleaning when evolution is fast
"""

import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
from evolution_cleanup import EvolutionCleanup

class AdaptiveCleanup:
    """
    Cleans up evolved systems based on production rate
    More frequent cleanup = faster evolution
    """
    
    def __init__(self):
        self.cleanup = EvolutionCleanup()
        self.config_file = Path("data/evolution/cleanup_config.json")
        self.load_config()
        
    def load_config(self):
        """Load or create cleanup configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {
                "last_cleanup": datetime.now().isoformat(),
                "systems_at_last_cleanup": 0,
                "cleanup_interval_hours": 24,  # Start with daily
                "growth_rate": 0,
                "cleanup_history": []
            }
    
    def save_config(self):
        """Save cleanup configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def calculate_growth_rate(self):
        """Calculate how fast systems are being created"""
        current_systems = len(list(Path("agents/evolved").iterdir()))
        
        last_count = self.config["systems_at_last_cleanup"]
        if last_count == 0:
            return 0
        
        # Calculate hours since last cleanup
        last_time = datetime.fromisoformat(self.config["last_cleanup"])
        hours_passed = (datetime.now() - last_time).total_seconds() / 3600
        
        if hours_passed == 0:
            return 0
        
        # Systems per hour growth rate
        growth_rate = (current_systems - last_count) / hours_passed
        return max(0, growth_rate)
    
    def determine_cleanup_interval(self, growth_rate):
        """
        Set cleanup frequency based on growth rate
        Faster growth = more frequent cleanup
        """
        if growth_rate > 5:  # More than 5 systems per hour
            return 1  # Clean every hour!
        elif growth_rate > 2:  # 2-5 systems per hour
            return 3  # Every 3 hours
        elif growth_rate > 1:  # 1-2 systems per hour
            return 6  # Every 6 hours
        elif growth_rate > 0.5:  # 0.5-1 systems per hour
            return 12  # Every 12 hours
        else:
            return 24  # Daily default
    
    def should_cleanup(self):
        """Determine if cleanup is needed now"""
        last_time = datetime.fromisoformat(self.config["last_cleanup"])
        hours_since = (datetime.now() - last_time).total_seconds() / 3600
        
        return hours_since >= self.config["cleanup_interval_hours"]
    
    def adaptive_cleanup(self, dry_run=True):
        """
        Run cleanup with adaptive frequency
        Returns True if cleanup was performed
        """
        print("\n" + "="*70)
        print("🔄 ADAPTIVE CLEANUP CHECK")
        print("="*70)
        
        # Calculate current growth rate
        current_systems = len(list(Path("agents/evolved").iterdir()))
        growth_rate = self.calculate_growth_rate()
        
        print(f"\n📊 Current Systems: {current_systems}")
        print(f"📈 Growth Rate: {growth_rate:.2f} systems/hour")
        
        # Update cleanup interval based on growth
        new_interval = self.determine_cleanup_interval(growth_rate)
        if new_interval != self.config["cleanup_interval_hours"]:
            print(f"⏱️  Adjusting cleanup interval: {self.config['cleanup_interval_hours']}h → {new_interval}h")
            self.config["cleanup_interval_hours"] = new_interval
        
        # Check if cleanup is needed
        if self.should_cleanup():
            print(f"\n🧹 CLEANUP NEEDED (last was {self.config['cleanup_interval_hours']}h ago)")
            
            # Run the actual cleanup
            kept, deleted = self.cleanup.cleanup(dry_run=dry_run)
            
            # Update config
            self.config["last_cleanup"] = datetime.now().isoformat()
            self.config["systems_at_last_cleanup"] = current_systems
            self.config["cleanup_history"].append({
                "timestamp": datetime.now().isoformat(),
                "kept": len(kept) if kept else 0,
                "deleted": len(deleted) if deleted else 0,
                "growth_rate": growth_rate
            })
            
            self.save_config()
            return True
        else:
            next_cleanup = datetime.fromisoformat(self.config["last_cleanup"]) + timedelta(hours=self.config["cleanup_interval_hours"])
            print(f"\n⏳ No cleanup needed yet")
            print(f"   Next cleanup: {next_cleanup.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Interval: {self.config['cleanup_interval_hours']} hours")
            return False
    
    def get_cleanup_schedule(self):
        """Get human-readable cleanup schedule"""
        interval = self.config["cleanup_interval_hours"]
        if interval == 1:
            return "⏱️  Every hour"
        elif interval == 3:
            return "⏱️  Every 3 hours"
        elif interval == 6:
            return "⏱️  Every 6 hours"
        elif interval == 12:
            return "⏱️  Twice daily"
        else:
            return "⏱️  Daily"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Cleanup Manager")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--execute", action="store_true", help="Actually delete files")
    parser.add_argument("--schedule", action="store_true", help="Show cleanup schedule")
    
    args = parser.parse_args()
    
    cleanup = AdaptiveCleanup()
    
    if args.schedule:
        print(f"\n📅 Cleanup Schedule: {cleanup.get_cleanup_schedule()}")
        print(f"   Last cleanup: {cleanup.config['last_cleanup']}")
        print(f"   Systems at last: {cleanup.config['systems_at_last_cleanup']}")
    elif args.execute:
        cleanup.adaptive_cleanup(dry_run=False)
    else:
        cleanup.adaptive_cleanup(dry_run=True)
