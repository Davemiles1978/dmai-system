#!/usr/bin/env python3
"""
DMAI HEALING AUTOPILOT WITH VALIDATION
Runs full validation after each component
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
    
    def run_validation(self):
        """Run the full component validator"""
        print("\n🔍 Running full system validation...")
        result = subprocess.run(
            [sys.executable, "autonomy/validate_components.py"],
            capture_output=True,
            text=True
        )
        
        # Parse results
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
        """Build and then validate"""
        success = super().build_with_healing(comp_id)
        
        if success:
            print("\n🧪 Running post-build validation...")
            validation_passed = self.run_validation()
            
            if not validation_passed:
                print("⚠️  Validation failed after build!")
                # Move component back to failed for review
                if comp_id in self.completed:
                    self.completed.remove(comp_id)
                if comp_id not in self.failed:
                    self.failed.append(comp_id)
                self.save_state()
                return False
        
        return success
    
    def print_status(self):
        """Print status with validation info"""
        super().print_status()
        
        # Show validation summary if available
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
    print("  [v] Validating Autopilot (with post-build validation)")
    
    choice = input("\nChoice: ").lower()
    
    if choice == 'v':
        builder.run_healing_autopilot()
    elif choice == 'h':
        builder.run_healing_autopilot()
    elif choice == 'a':
        builder.run_autopilot()
    else:
        builder.run()
