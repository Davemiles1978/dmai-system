#!/usr/bin/env python3
"""
DMAI Component Validator
Tests that built components actually exist and work
"""

import os
import sys
import importlib.util
import json
from pathlib import Path

class ComponentValidator:
    def __init__(self):
        self.components_dir = Path("components")
        self.tests_dir = Path("tests")
        self.results = {
            "passed": [],
            "failed": [],
            "missing": [],
            "details": {}
        }
        
    def validate_component(self, comp_path):
        """Validate a single component file"""
        try:
            # Check if file exists
            if not comp_path.exists():
                return False, "File missing"
            
            # Try to import it as a module
            module_name = comp_path.stem
            spec = importlib.util.spec_from_file_location(module_name, comp_path)
            if spec is None or spec.loader is None:
                return False, "Invalid Python module"
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if it has expected attributes
            has_class = any(isinstance(getattr(module, attr), type) 
                          for attr in dir(module) 
                          if not attr.startswith('_'))
            
            return True, "Valid" if has_class else "No class found"
            
        except Exception as e:
            return False, str(e)
    
    def validate_test(self, test_path):
        """Validate a test file"""
        try:
            if not test_path.exists():
                return False, "Test missing"
            
            # Try to import the test
            spec = importlib.util.spec_from_file_location(test_path.stem, test_path)
            if spec is None or spec.loader is None:
                return False, "Invalid test module"
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for unittest.TestCase subclasses
            import unittest
            has_tests = any(issubclass(getattr(module, attr), unittest.TestCase) 
                          for attr in dir(module) 
                          if isinstance(getattr(module, attr), type))
            
            return True, "Has tests" if has_tests else "No test cases"
            
        except Exception as e:
            return False, str(e)
    
    def run_validation(self):
        """Run validation on all components"""
        print("\n" + "="*80)
        print("🔍 DMAI COMPONENT VALIDATION")
        print("="*80)
        
        # Check all phase directories
        for phase_dir in sorted(self.components_dir.glob("phase*")):
            phase = phase_dir.name
            print(f"\n📁 {phase}:")
            
            for comp_file in sorted(phase_dir.glob("*.py")):
                comp_id = comp_file.stem.split('_')[0]  # Extract PXTX
                valid, message = self.validate_component(comp_file)
                
                # Check corresponding test
                test_file = self.tests_dir / f"{comp_id}_test.py"
                test_valid, test_message = self.validate_test(test_file)
                
                status = "✅" if valid and test_valid else "❌"
                print(f"  {status} {comp_file.name}")
                print(f"     Component: {message}")
                print(f"     Test: {test_message if test_valid else '❌ ' + test_message}")
                
                self.results['details'][comp_id] = {
                    'component': {'valid': valid, 'message': message},
                    'test': {'valid': test_valid, 'message': test_message}
                }
                
                if valid and test_valid:
                    self.results['passed'].append(comp_id)
                else:
                    self.results['failed'].append(comp_id)
        
        # Summary
        print("\n" + "="*80)
        print("📊 VALIDATION SUMMARY")
        print("="*80)
        print(f"✅ Passed: {len(self.results['passed'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        print(f"Total: {len(self.results['passed']) + len(self.results['failed'])}")
        
        # Save results
        with open('autonomy/validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\n💾 Results saved to autonomy/validation_results.json")
        
        return self.results

if __name__ == "__main__":
    validator = ComponentValidator()
    validator.run_validation()
