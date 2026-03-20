#!/usr/bin/env python3
"""
DMAI VALIDATING AUTOPILOT - Runs test sanitizer before healing
"""

from healing_autopilot_fixed_v7 import HealingAutopilot
import subprocess
import sys
import json
import os

class ValidatingAutopilot(HealingAutopilot):
    def __init__(self):
        super().__init__()
        self.validation_results = {}
    
    def pre_build_sanitize(self):
        """Run test sanitizer before any build"""
        print("\n🧹 Running test sanitizer...")
        result = subprocess.run(
            [sys.executable, "autonomy/sanitize_tests.py"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return result.returncode == 0
    
    def run_validation(self):
        """Run the full component validator"""
        print("\n🔍 Running full system validation...")
        result = subprocess.run(
            [sys.executable, "autonomy/validate_components.py"],
            capture_output=True,
            text=True
        )
        
        try:
            with open('autonomy/validation_results.json', 'r') as f:
                self.validation_results = json.load(f)
            
            passed = len(self.validation_results.get('passed', []))
            failed = len(self.validation_results.get('failed', []))
            
            print(f"   ✅ Passed: {passed}")
            print(f"   ❌ Failed: {failed}")
            
            return failed == 0
        except:
            print("   ⚠️ Could not parse validation results")
            return False
    
    def build_with_healing(self, comp_id):
        """Sanitize, build, then validate"""
        # Always sanitize before building
        self.pre_build_sanitize()
        
        # Now build with healing
        success = super().build_with_healing(comp_id)
        
        if success:
            print("\n🧪 Running post-build validation...")
            validation_passed = self.run_validation()
            
            if not validation_passed:
                print("⚠️  Validation failed after build!")
                if comp_id in self.completed:
                    self.completed.remove(comp_id)
                if comp_id not in self.failed:
                    self.failed.append(comp_id)
                self.save_state()
                return False
        
        return success
    
    def run_healing_autopilot(self):
        """Override to add initial sanitize"""
        self.pre_build_sanitize()
        super().run_healing_autopilot()
    
    def print_status(self):
        """Print status with validation info"""
        super().print_status()
        
        if self.validation_results:
            passed = len(self.validation_results.get('passed', []))
            failed = len(self.validation_results.get('failed', []))
            print(f"\n📊 Validation: ✅ {passed} passed, ❌ {failed} failed")

if __name__ == "__main__":
    builder = ValidatingAutopilot()
    
    print("\nSelect mode:")
    print("  [m] Manual")
    print("  [a] Regular Autopilot")
    print("  [h] Healing Autopilot")
    print("  [v] Validating Autopilot (with test sanitizer)")
    
    choice = input("\nChoice: ").lower()
    
    if choice == 'v':
        builder.run_healing_autopilot()
    elif choice == 'h':
        builder.run_healing_autopilot()
    elif choice == 'a':
        builder.run_autopilot()
    else:
        builder.run()
