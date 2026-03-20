#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Version Merger for DMAI Evolution
Merges external provider updates with internally evolved versions
Creates "super evolved" versions that combine the best of both
"""

import json
import difflib
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

class VersionMerger:
    def __init__(self):
        self.evolution_dir = "/Users/davidmiles/Desktop/dmai-system/evolution"
        self.versions_dir = f"{self.evolution_dir}/versions"
        self.merged_dir = f"{self.evolution_dir}/merged_versions"
        self.history_file = f"{self.evolution_dir}/version_history.json"
        
        # Create directories
        Path(self.versions_dir).mkdir(parents=True, exist_ok=True)
        Path(self.merged_dir).mkdir(parents=True, exist_ok=True)
        
        self.load_history()
    
    def load_history(self):
        """Load version history"""
        if Path(self.history_file).exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'internal_versions': {},
                'provider_versions': {},
                'merged_versions': {},
                'evolution_lineage': {}
            }
    
    def save_history(self):
        """Save version history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_internal_version(self, evaluator_name):
        """Get the latest internally evolved version"""
        if evaluator_name in self.history['internal_versions']:
            versions = self.history['internal_versions'][evaluator_name]
            if versions:
                return versions[-1]  # Latest version
        return None
    
    def get_provider_version(self, evaluator_name, provider_version):
        """Get or create provider version entry"""
        key = f"{evaluator_name}_provider_{provider_version}"
        if key in self.history['provider_versions']:
            return self.history['provider_versions'][key]
        return None
    
    def compare_versions(self, internal_code, provider_code):
        """Compare internal and provider code to find differences"""
        if not internal_code or not provider_code:
            return None
        
        # Split into lines
        internal_lines = internal_code.splitlines()
        provider_lines = provider_code.splitlines()
        
        # Generate diff
        diff = list(difflib.unified_diff(
            internal_lines, 
            provider_lines,
            fromfile='internal_evolved',
            tofile='provider_new',
            lineterm=''
        ))
        
        # Analyze differences
        stats = {
            'total_lines_diff': len(diff),
            'internal_unique': [],
            'provider_unique': [],
            'common_improvements': []
        }
        
        # Parse diff to understand what's different
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                stats['provider_unique'].append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                stats['internal_unique'].append(line[1:])
        
        return stats
    
    def merge_versions(self, evaluator_name, internal_path, provider_info):
        """
        Intelligently merge internal evolved version with provider update
        Creates a "super evolved" version that combines the best of both
        """
        print(f"\n🔄 Merging {evaluator_name} versions...")
        print(f"   Internal: {internal_path}")
        print(f"   Provider: v{provider_info['version']}")
        
        # Read both versions
        with open(internal_path, 'r') as f:
            internal_code = f.read()
        
        # Provider code would come from API/download
        # For now, simulate with a template
        provider_code = self.simulate_provider_code(evaluator_name, provider_info['version'])
        
        # Compare them
        diff_stats = self.compare_versions(internal_code, provider_code)
        
        if not diff_stats:
            print("   ⚠️  No differences found")
            return None
        
        print(f"\n   📊 Analysis:")
        print(f"      • Lines different: {diff_stats['total_lines_diff']}")
        print(f"      • Internal unique improvements: {len(diff_stats['internal_unique'])}")
        print(f"      • Provider unique features: {len(diff_stats['provider_unique'])}")
        
        # Intelligent merging strategy
        merged_code = self.intelligent_merge(
            evaluator_name,
            internal_code,
            provider_code,
            diff_stats
        )
        
        # Create merged version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_filename = f"{evaluator_name}_super_evolved_v{provider_info['version']}_{timestamp}.py"
        merged_path = f"{self.merged_dir}/{merged_filename}"
        
        with open(merged_path, 'w') as f:
            f.write(merged_code)
        
        # Record the merge
        merge_record = {
            'evaluator': evaluator_name,
            'timestamp': timestamp,
            'internal_version': internal_path,
            'provider_version': provider_info['version'],
            'merged_path': merged_path,
            'diff_stats': diff_stats,
            'merge_strategy': 'intelligent_merge'
        }
        
        # Update history
        if evaluator_name not in self.history['merged_versions']:
            self.history['merged_versions'][evaluator_name] = []
        
        self.history['merged_versions'][evaluator_name].append(merge_record)
        
        # Track lineage
        lineage_key = f"{evaluator_name}_{timestamp}"
        self.history['evolution_lineage'][lineage_key] = {
            'parent_internal': internal_path,
            'parent_provider': provider_info,
            'child_merged': merged_path,
            'merge_time': timestamp
        }
        
        self.save_history()
        
        print(f"\n   ✅ Created super evolved version:")
        print(f"      • {merged_filename}")
        print(f"      • Combines internal improvements with provider v{provider_info['version']}")
        
        return merged_path
    
    def intelligent_merge(self, evaluator_name, internal_code, provider_code, diff_stats):
        """
        Intelligently merge code, preferring internal improvements
        but incorporating beneficial provider updates
        """
        # This is where the real AI magic happens
        # For now, we'll use a heuristic approach
        
        # Strategy: Keep internal improvements, add provider features
        internal_lines = internal_code.splitlines()
        provider_lines = provider_code.splitlines()
        
        merged_lines = []
        
        # Track which lines we've processed
        internal_idx = 0
        provider_idx = 0
        
        while internal_idx < len(internal_lines) or provider_idx < len(provider_lines):
            internal_line = internal_lines[internal_idx] if internal_idx < len(internal_lines) else None
            provider_line = provider_lines[provider_idx] if provider_idx < len(provider_lines) else None
            
            # If lines match, take either
            if internal_line == provider_line:
                merged_lines.append(internal_line)
                internal_idx += 1
                provider_idx += 1
            elif internal_line and provider_line and internal_line.strip() and provider_line.strip():
                # Different lines - need to decide which to keep
                
                # Check if internal line is an improvement marker
                if '# IMPROVED BY' in internal_line or '# EVOLVED' in internal_line:
                    merged_lines.append(internal_line)
                    internal_idx += 1
                # Check if provider line is a new feature
                elif '# PROVIDER UPDATE' in provider_line or '# NEW FEATURE' in provider_line:
                    merged_lines.append(provider_line)
                    provider_idx += 1
                # Default to keeping internal version (our evolved code)
                else:
                    merged_lines.append(internal_line)
                    internal_idx += 1
                    # Don't advance provider_idx - we'll check again
            elif internal_line:
                merged_lines.append(internal_line)
                internal_idx += 1
            elif provider_line:
                # Add provider lines only if they seem beneficial
                if 'import' in provider_line or 'def ' in provider_line or 'class ' in provider_line:
                    merged_lines.append(provider_line)
                provider_idx += 1
        
        # Add merge metadata
        header = f"""
# =============================================
# SUPER EVOLVED VERSION
# Generated: {datetime.now().isoformat()}
# Evaluator: {evaluator_name}
# 
# This version combines:
#   • Internal evolution improvements ({len(diff_stats['internal_unique'])} unique lines)
#   • Provider v{provider_info['version']} features ({len(diff_stats['provider_unique'])} unique lines)
# =============================================

"""
        
        return header + "\n".join(merged_lines)
    
    def simulate_provider_code(self, evaluator_name, version):
        """Simulate provider code for testing"""
        # In production, this would fetch actual provider code
        return f'''# PROVIDER UPDATE v{version}
# New features from {evaluator_name} provider

import json
import time

class {evaluator_name.capitalize()}Provider:
    def __init__(self):
        self.version = "{version}"
        self.new_features = True
    
    def enhanced_api(self):
        """New API method from provider"""
        return {{"status": "updated", "version": self.version}}
    
    # PROVIDER UPDATE: Added new functionality
    def provider_specific_optimization(self):
        return "Optimized by provider"
'''
    
    def get_evolution_lineage(self, evaluator_name):
        """Get the complete evolution lineage for an evaluator"""
        lineage = []
        
        for key, record in self.history['evolution_lineage'].items():
            if evaluator_name in key:
                lineage.append(record)
        
        # Sort by timestamp
        lineage.sort(key=lambda x: x['merge_time'])
        
        return lineage
    
    def should_replace_internal(self, evaluator_name, merged_path):
        """
        Decide whether to replace internal version with merged version
        Only after validation and testing
        """
        # In production, this would run tests comparing performance
        # For now, we'll simulate with a heuristic
        
        # Check if we have at least 3 successful merges
        merged_count = len(self.history['merged_versions'].get(evaluator_name, []))
        
        if merged_count >= 3:
            # After 3 merges, the merged version should be significantly better
            print(f"\n   📈 {evaluator_name} has undergone {merged_count} successful merges")
            print(f"   ✅ Ready to promote merged version as primary")
            return True
        
        return False

# Singleton instance
version_merger = VersionMerger()

if __name__ == "__main__":
    print("🧬 DMAI VERSION MERGER")
    print("=" * 50)
    
    # Test with a sample evaluator
    test_evaluator = "gemini"
    test_internal = "/Users/davidmiles/Desktop/dmai-system/evolution/evaluators/gemini_evaluator.py"
    test_provider = {
        'version': '2.6.0',
        'features': ['multimodal_enhanced', 'faster_inference']
    }
    
    merged = version_merger.merge_versions(test_evaluator, test_internal, test_provider)
    
    if merged:
        print(f"\n✅ Super evolved version created: {merged}")
        
        # Check lineage
        lineage = version_merger.get_evolution_lineage(test_evaluator)
        print(f"\n📊 Evolution lineage: {len(lineage)} generations")
        
        # Check if ready to replace
        if version_merger.should_replace_internal(test_evaluator, merged):
            print("\n   ⚠️  This evaluator is ready for promotion to primary version!")
