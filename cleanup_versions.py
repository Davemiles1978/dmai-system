#!/usr/bin/env python3
"""
Intelligent Version Cleanup System
Removes obsolete evolution versions while preserving:
- Best versions (always keep)
- Checkpoints (configurable retention)
- Recent generations (last N)
"""

import os
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta

class VersionCleaner:
    def __init__(self):
        self.checkpoints_path = Path("checkpoints")
        self.best_path = self.checkpoints_path / "best_versions"
        self.log_file = Path("cleanup.log")
        
        # Configuration - ADJUST THESE VALUES
        self.config = {
            # Keep these many most recent generations
            "keep_last_generations": 5,
            
            # Keep checkpoints newer than this many days
            "max_age_days": 30,
            
            # Always keep best versions
            "keep_best_versions": True,
            
            # Keep best version per category/repo
            "keep_best_per_repo": True,
            
            # Delete versions with score below this (if older than threshold)
            "min_score_threshold": 0.8,
            
            # Dry run mode (set to False to actually delete)
            "dry_run": True
        }
        
    def log(self, message):
        """Log messages to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def get_all_generations(self):
        """Get all generation folders with their metadata"""
        generations = []
        
        if not self.checkpoints_path.exists():
            return generations
            
        for gen_dir in self.checkpoints_path.glob("generation_*"):
            try:
                gen_num = int(gen_dir.name.replace("generation_", ""))
                meta_file = gen_dir / "metadata.json"
                
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                # Get creation time
                created = datetime.fromtimestamp(gen_dir.stat().st_ctime)
                
                generations.append({
                    "number": gen_num,
                    "path": gen_dir,
                    "created": created,
                    "metadata": metadata,
                    "size_mb": self.get_dir_size_mb(gen_dir)
                })
            except Exception as e:
                self.log(f"Error reading {gen_dir}: {e}")
        
        # Sort by generation number
        generations.sort(key=lambda x: x["number"])
        return generations
    
    def get_dir_size_mb(self, path):
        """Calculate directory size in MB"""
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024 * 1024)
    
    def get_best_scores(self):
        """Load best scores from file"""
        scores_file = self.checkpoints_path / "best_scores.json"
        if scores_file.exists():
            with open(scores_file, 'r') as f:
                return json.load(f)
        return {}
    
    def identify_obsolete_versions(self, generations):
        """Identify which generations can be deleted"""
        obsolete = []
        keep = []
        
        best_scores = self.get_best_scores()
        latest_generations = sorted([g["number"] for g in generations], reverse=True)
        
        # Always keep the most recent N generations
        keep_gens = set(latest_generations[:self.config["keep_last_generations"]])
        
        for gen in generations:
            gen_num = gen["number"]
            
            # Always keep most recent generations
            if gen_num in keep_gens:
                keep.append({"reason": f"Recent generation (top {self.config['keep_last_generations']})", **gen})
                continue
            
            # Keep if it's a best version for any repo
            if self.config["keep_best_versions"]:
                is_best = False
                for repo, data in best_scores.items():
                    if data.get("generation") == gen_num:
                        is_best = True
                        break
                if is_best:
                    keep.append({"reason": "Contains best version", **gen})
                    continue
            
            # Check age
            age_days = (datetime.now() - gen["created"]).days
            if age_days > self.config["max_age_days"]:
                obsolete.append({"reason": f"Older than {self.config['max_age_days']} days", **gen})
                continue
            
            # Check if it's low scoring and old enough
            # You could add more sophisticated scoring here
            
            # If not kept by any rule, mark for deletion (but not too aggressively)
            if gen_num < latest_generations[0] - 10:  # More than 10 generations old
                obsolete.append({"reason": "Old generation without special status", **gen})
            else:
                keep.append({"reason": "Within recent range", **gen})
        
        return keep, obsolete
    
    def cleanup(self):
        """Main cleanup function"""
        self.log("=" * 60)
        self.log("Starting version cleanup")
        self.log(f"Configuration: {json.dumps(self.config, indent=2)}")
        self.log("=" * 60)
        
        generations = self.get_all_generations()
        self.log(f"Found {len(generations)} generation folders")
        
        if not generations:
            self.log("No generations found")
            return
        
        # Calculate total size
        total_size = sum(g["size_mb"] for g in generations)
        self.log(f"Total size: {total_size:.2f} MB")
        
        # Identify obsolete versions
        keep, obsolete = self.identify_obsolete_versions(generations)
        
        self.log(f"\n📊 ANALYSIS RESULTS:")
        self.log(f"  Keep: {len(keep)} generations ({sum(k['size_mb'] for k in keep):.2f} MB)")
        self.log(f"  Delete: {len(obsolete)} generations ({sum(o['size_mb'] for o in obsolete):.2f} MB)")
        
        if obsolete:
            self.log("\n🗑️ GENERATIONS TO DELETE:")
            for o in obsolete:
                self.log(f"  • Gen {o['number']} - {o['reason']} - {o['size_mb']:.2f} MB")
            
            if not self.config["dry_run"]:
                # Actually delete
                for o in obsolete:
                    self.log(f"Deleting generation {o['number']}...")
                    shutil.rmtree(o["path"])
                self.log(f"\n✅ Deleted {len(obsolete)} generations")
                self.log(f"Freed {(sum(o['size_mb'] for o in obsolete)):.2f} MB")
            else:
                self.log("\n⚠️ DRY RUN MODE - No files deleted")
                self.log("Set 'dry_run': False in config to actually delete")
        
        # Save cleanup report
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "total_generations": len(generations),
            "kept": len(keep),
            "deleted": len(obsolete),
            "freed_mb": sum(o["size_mb"] for o in obsolete) if not self.config["dry_run"] else 0,
            "dry_run": self.config["dry_run"]
        }
        
        report_file = self.checkpoints_path / "cleanup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"\n📝 Report saved to {report_file}")
        self.log("=" * 60)
        
        return report

if __name__ == "__main__":
    cleaner = VersionCleaner()
    
    # You can run with different modes
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        cleaner.config["dry_run"] = False
        print("⚠️ EXECUTE MODE - Files WILL be deleted!")
        confirm = input("Type 'DELETE' to confirm: ")
        if confirm == "DELETE":
            cleaner.cleanup()
        else:
            print("Cleanup cancelled")
    else:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print("Run with --execute to actually delete files")
        cleaner.cleanup()
