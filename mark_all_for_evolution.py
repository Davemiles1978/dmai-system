#!/usr/bin/env python3
"""
MARK ALL AI REPOS FOR EVOLUTION
This script adds EVOLVE-BLOCK markers around every Python file
Run this once to prepare all your repos for evolution
"""

import os
import sys
from pathlib import Path

class EvolutionMarker:
    def __init__(self):
        self.repos_path = Path.cwd() / "repos"
        self.stats = {
            "files_found": 0,
            "files_marked": 0,
            "files_skipped": 0,
            "errors": 0
        }
    
    def mark_python_file(self, filepath):
        """Add EVOLVE markers to a Python file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Skip if already marked
            if '# EVOLVE-BLOCK-START' in content:
                print(f"⏭️  Already marked: {filepath.name}")
                self.stats["files_skipped"] += 1
                return False
            
            # Add markers around entire file
            marked_content = f"""# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

{content}

# EVOLVE-BLOCK-END
"""
            
            # Write back
            with open(filepath, 'w') as f:
                f.write(marked_content)
            
            print(f"✅ Marked: {filepath}")
            self.stats["files_marked"] += 1
            return True
            
        except Exception as e:
            print(f"❌ Error marking {filepath}: {e}")
            self.stats["errors"] += 1
            return False
    
    def mark_all_repos(self):
        """Find and mark all Python files in all repos"""
        print("\n🔍 Scanning for Python files in all repos...")
        print("=" * 60)
        
        # Walk through all repos
        python_files = []
        for root, dirs, files in os.walk(self.repos_path):
            # Skip hidden directories and venv
            if '/.' in root or 'venv' in root:
                continue
            
            for file in files:
                if file.endswith('.py'):
                    full_path = Path(root) / file
                    python_files.append(full_path)
        
        self.stats["files_found"] = len(python_files)
        
        if not python_files:
            print("❌ No Python files found in repos folder!")
            return
        
        print(f"📁 Found {len(python_files)} Python files to process...\n")
        
        # Mark each file
        for py_file in python_files:
            relative_path = py_file.relative_to(self.repos_path)
            print(f"📄 Processing: {relative_path}")
            self.mark_python_file(py_file)
        
        # Print summary
        self.print_summary()
    
    def mark_reverse_engineered(self):
        """Specifically mark the reverse engineered models"""
        rev_path = self.repos_path / "reverse_engineered"
        if rev_path.exists():
            print("\n🎯 Marking reverse engineered models...")
            for model_dir in rev_path.iterdir():
                if model_dir.is_dir():
                    for py_file in model_dir.glob("*.py"):
                        self.mark_python_file(py_file)
    
    def create_evolution_config(self):
        """Create a configuration file for the evolution system"""
        config_path = self.repos_path / "evolution_config.json"
        
        config = {
            "repos": [],
            "evolution_settings": {
                "cycle_interval_hours": 1,
                "checkpoint_frequency": 10,
                "min_improvement_threshold": 0.01,
                "max_generations_per_repo": 1000
            },
            "reverse_engineered": {
                "enabled": True,
                "update_frequency_days": 7,
                "api_monitoring": True
            }
        }
        
        # Add all repos to config
        for item in self.repos_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                config["repos"].append({
                    "name": item.name,
                    "path": str(item.relative_to(self.repos_path)),
                    "files_marked": True,
                    "active": True
                })
        
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        print(f"\n📝 Evolution config created: {config_path}")
    
    def print_summary(self):
        """Print marking statistics"""
        print("\n" + "=" * 60)
        print("📊 MARKING SUMMARY")
        print("=" * 60)
        print(f"📁 Total Python files found: {self.stats['files_found']}")
        print(f"✅ Newly marked: {self.stats['files_marked']}")
        print(f"⏭️  Already marked: {self.stats['files_skipped']}")
        print(f"❌ Errors: {self.stats['errors']}")
        
        if self.stats['files_marked'] > 0:
            print("\n🎉 Success! Your repos are now ready for evolution!")
    
    def run(self):
        """Run the complete marking process"""
        print("\n🚀 PREPARING AI REPOS FOR EVOLUTION")
        print("=" * 60)
        
        # Mark all Python files
        self.mark_all_repos()
        
        # Specifically mark reverse engineered models
        self.mark_reverse_engineered()
        
        # Create evolution config
        self.create_evolution_config()
        
        print("\n✅ All repos are now ready for evolution!")

if __name__ == "__main__":
    marker = EvolutionMarker()
    marker.run()
