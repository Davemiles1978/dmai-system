#!/usr/bin/env python3
"""
Intelligent Cleanup System for DMAI Evolution
Prevents overload while preserving the best systems
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import heapq

class EvolutionCleanup:
    """
    Smart cleanup that keeps the best and deletes the rest.
    Uses multiple strategies to prevent overload.
    """
    
    def __init__(self):
        self.evolved_dir = Path("agents/evolved")
        self.cleanup_log = Path("data/evolution/cleanup_history.json")
        self.cleanup_log.parent.mkdir(exist_ok=True)
        
        # Load cleanup history
        self.history = self._load_history()
        
        # Cleanup thresholds
        self.max_systems = 30  # Maximum total evolved systems
        self.max_generations = 5  # Keep only last 5 generations
        self.min_quality_score = 0.6  # Minimum quality to keep
        
    def _load_history(self):
        """Load cleanup history"""
        if self.cleanup_log.exists():
            with open(self.cleanup_log) as f:
                return json.load(f)
        return {"cleanups": [], "deleted": [], "kept": []}
    
    def _save_history(self):
        """Save cleanup history"""
        with open(self.cleanup_log, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def analyze_systems(self):
        """
        Analyze all evolved systems and score them
        Returns list of systems with scores
        """
        if not self.evolved_dir.exists():
            return []
        
        systems = []
        for system_dir in self.evolved_dir.iterdir():
            if not system_dir.is_dir():
                continue
            
            # Get evolution record
            record_file = system_dir / "evolution_record.json"
            if not record_file.exists():
                continue
                
            with open(record_file) as f:
                record = json.load(f)
            
            # Calculate system score
            score = self._calculate_score(system_dir, record)
            
            # Get generation from name or record
            generation = self._extract_generation(system_dir.name, record)
            
            systems.append({
                "name": system_dir.name,
                "path": str(system_dir),
                "generation": generation,
                "score": score,
                "age": datetime.now() - datetime.fromisoformat(record.get("timestamp", datetime.now().isoformat())),
                "record": record
            })
        
        # Sort by score (highest first)
        systems.sort(key=lambda x: x["score"], reverse=True)
        return systems
    
    def _calculate_score(self, system_dir, record):
        """
        Calculate a quality score for a system
        Higher is better (keep), lower is worse (delete)
        """
        score = 0.0
        
        # 1. Base score from improvement quality
        improvements = record.get("improvements", [])
        for imp in improvements:
            score += imp.get("quality", 0.5)
        
        # 2. Bonus for unique capabilities
        caps = set()
        for imp in improvements:
            if "capability" in imp:
                caps.add(imp["capability"])
        score += len(caps) * 0.2
        
        # 3. Penalty for very old systems
        created = datetime.fromisoformat(record.get("timestamp", datetime.now().isoformat()))
        age_days = (datetime.now() - created).days
        score -= age_days * 0.05
        
        # 4. Bonus for being a parent of other systems
        parent_count = self._count_as_parent(system_dir.name)
        score += parent_count * 0.3
        
        # 5. Check for manifest existence (better if it has one)
        if (system_dir / "manifest.json").exists():
            score += 0.5
        
        return max(0.0, score)
    
    def _extract_generation(self, name, record):
        """Extract generation number from name or record"""
        # Try from name (e.g., "system_gen27")
        import re
        match = re.search(r'gen(\d+)', name)
        if match:
            return int(match.group(1))
        
        # Fallback to record
        return record.get("new_generation", 1)
    
    def _count_as_parent(self, system_name):
        """Count how many newer systems have this as parent"""
        count = 0
        base_name = system_name.split('_gen')[0]  # Remove generation suffix
        
        for other_dir in self.evolved_dir.iterdir():
            if not other_dir.is_dir() or other_dir.name == system_name:
                continue
            
            record_file = other_dir / "evolution_record.json"
            if not record_file.exists():
                continue
                
            with open(record_file) as f:
                record = json.load(f)
            
            parents = record.get("parents", [])
            if any(base_name in str(p) for p in parents):
                count += 1
        
        return count
    
    def cleanup(self, dry_run=True):
        """
        Perform intelligent cleanup
        If dry_run=True, only report what would be deleted
        """
        print("\n" + "="*70)
        print("🧹 EVOLUTION CLEANUP ANALYSIS")
        print("="*70)
        
        systems = self.analyze_systems()
        print(f"\n📊 Analyzing {len(systems)} evolved systems...")
        
        if not systems:
            print("✅ No systems to clean up")
            return
        
        # Strategy 1: Keep only top N systems by score
        keep_count = min(self.max_systems, len(systems))
        keep_systems = systems[:keep_count]
        delete_candidates = systems[keep_count:]
        
        print(f"\n📈 Score distribution:")
        for i, sys in enumerate(systems[:10], 1):
            print(f"   {i:2d}. {sys['name']:40} Score: {sys['score']:.2f} Gen: {sys['generation']}")
        
        # Strategy 2: Remove very old generations (if not in top keep)
        for system in systems[keep_count:]:
            if system["generation"] < self._get_max_generation() - self.max_generations:
                if system not in delete_candidates:
                    delete_candidates.append(system)
        
        # Strategy 3: Remove low-quality systems
        for system in systems[keep_count:]:
            if system["score"] < self.min_quality_score:
                if system not in delete_candidates:
                    delete_candidates.append(system)
        
        # Remove duplicates
        delete_candidates = list({sys["name"]: sys for sys in delete_candidates}.values())
        
        print(f"\n🗑️  Candidates for deletion: {len(delete_candidates)}")
        
        if dry_run:
            print("\n📋 DRY RUN - No files will be deleted")
            print("   Systems that WOULD be deleted:")
            for sys in delete_candidates[:10]:  # Show first 10
                print(f"   • {sys['name']} (score: {sys['score']:.2f}, gen: {sys['generation']})")
            if len(delete_candidates) > 10:
                print(f"   ... and {len(delete_candidates) - 10} more")
        else:
            # Actually delete
            for system in delete_candidates:
                path = Path(system["path"])
                if path.exists():
                    shutil.rmtree(path)
                    self.history["deleted"].append({
                        "name": system["name"],
                        "score": system["score"],
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"✅ Deleted: {system['name']}")
            
            # Record kept systems
            for system in keep_systems:
                self.history["kept"].append({
                    "name": system["name"],
                    "score": system["score"],
                    "timestamp": datetime.now().isoformat()
                })
            
            self.history["cleanups"].append({
                "timestamp": datetime.now().isoformat(),
                "deleted": len(delete_candidates),
                "kept": len(keep_systems)
            })
            
            self._save_history()
        
        print("\n" + "="*70)
        print(f"📊 SUMMARY: Would keep {len(keep_systems)}, would delete {len(delete_candidates)}")
        print("="*70)
        
        return keep_systems, delete_candidates
    
    def _get_max_generation(self):
        """Get the highest generation number"""
        max_gen = 0
        for system in self.analyze_systems():
            if system["generation"] > max_gen:
                max_gen = system["generation"]
        return max_gen
    
    def auto_cleanup(self, threshold_days=7):
        """
        Automatic cleanup based on age and quality
        Run this periodically (e.g., weekly)
        """
        print(f"\n🔄 Running automatic cleanup (older than {threshold_days} days)...")
        
        systems = self.analyze_systems()
        threshold = datetime.now() - timedelta(days=threshold_days)
        
        to_delete = []
        for system in systems:
            # Delete if too old AND low quality
            if system["age"] > threshold and system["score"] < 0.5:
                to_delete.append(system)
        
        if to_delete:
            print(f"🗑️  Deleting {len(to_delete)} old, low-quality systems...")
            for system in to_delete:
                path = Path(system["path"])
                if path.exists():
                    shutil.rmtree(path)
                    print(f"   ✅ Deleted: {system['name']}")
        else:
            print("✅ No systems meet auto-cleanup criteria")
    
    def consolidate_evolved_systems(self):
        """
        Move best evolved systems to a 'hall of fame'
        and create symbolic links for easy access
        """
        hall_of_fame = Path("agents/hall_of_fame")
        hall_of_fame.mkdir(exist_ok=True)
        
        systems = self.analyze_systems()
        best_systems = systems[:5]  # Top 5 systems
        
        print(f"\n🏆 Promoting top 5 systems to Hall of Fame:")
        for system in best_systems:
            link_name = hall_of_fame / system['name']
            if not link_name.exists():
                # Create symlink to the actual system
                try:
                    link_name.symlink_to(Path(system['path']).absolute())
                    print(f"   ✅ {system['name']} (score: {system['score']:.2f})")
                except:
                    print(f"   ⚠️ Could not create link for {system['name']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evolution Cleanup Manager")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--execute", action="store_true", help="Actually delete files")
    parser.add_argument("--auto", action="store_true", help="Run automatic cleanup")
    parser.add_argument("--consolidate", action="store_true", help="Create Hall of Fame")
    
    args = parser.parse_args()
    
    cleanup = EvolutionCleanup()
    
    if args.auto:
        cleanup.auto_cleanup()
    elif args.consolidate:
        cleanup.consolidate_evolved_systems()
    elif args.execute:
        cleanup.cleanup(dry_run=False)
    else:
        cleanup.cleanup(dry_run=True)  # Default to dry run
