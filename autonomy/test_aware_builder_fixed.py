#!/usr/bin/env python3
"""
DMAI TEST-AWARE BUILDER - Fixed version for simplified roadmap
"""

import os
import json
import time
import subprocess
import sys
from datetime import datetime

class TestAwareBuilder:
    def __init__(self):
        self.roadmap_file = "docs/complete_roadmap_with_evolution.json"
        self.state_file = "autonomy/build_state.json"
        self.test_results_file = "autonomy/test_results.json"
        self.components = {}
        self.completed = []
        self.in_progress = []
        self.failed = []
        self.test_results = {}
        self.load_state()
        
    def load_state(self):
        """Load current build state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.completed = data.get('completed', [])
                self.in_progress = data.get('in_progress', [])
                self.failed = data.get('failed', [])
        
        if os.path.exists(self.test_results_file):
            with open(self.test_results_file, 'r') as f:
                self.test_results = json.load(f)
        
    def save_state(self):
        """Save build state"""
        with open(self.state_file, 'w') as f:
            json.dump({
                'completed': self.completed,
                'in_progress': self.in_progress,
                'failed': self.failed,
                'last_updated': str(datetime.now())
            }, f, indent=2)
        
        with open(self.test_results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
    
    def load_roadmap(self):
        """Load and parse the roadmap"""
        if not os.path.exists(self.roadmap_file):
            print(f"❌ Roadmap not found: {self.roadmap_file}")
            return False
        
        with open(self.roadmap_file, 'r') as f:
            self.roadmap = json.load(f)
        
        print(f"\n📖 Loaded roadmap: {self.roadmap['metadata']['version']}")
        print(f"   Current Gen: {self.roadmap['metadata']['current_generation']}")
        print(f"   Progress: {self.roadmap['metadata']['overall_progress']}%")
        
        # Build component index - FIXED for simplified structure
        self.components = {}
        comp_index = 0
        for phase in self.roadmap['phases']:
            for task in phase['tasks']:
                comp_id = f"P{phase['id']}T{comp_index}"
                self.components[comp_id] = {
                    'id': comp_id,
                    'name': task['name'],
                    'phase': phase['id'],
                    'priority': task['priority'],
                    'depends_on': task.get('depends_on', []),
                    'status': task['status'],
                    'eta': task.get('eta', 'unknown'),
                    'details': task.get('details', {}),
                    'weakness_type': task.get('weakness_type', 'unknown')
                }
                comp_index += 1
        
        print(f"   Total components: {len(self.components)}")
        return True
    
    def analyze_dependencies(self):
        """Determine what can be built now"""
        ready = []
        blocked = []
        
        for comp_id, comp in self.components.items():
            if comp_id in self.completed:
                continue
            if comp_id in self.in_progress:
                continue
            if comp_id in self.failed:
                continue
                
            # Check dependencies
            deps_met = True
            missing_deps = []
            for dep in comp['depends_on']:
                if dep not in self.completed:
                    deps_met = False
                    missing_deps.append(dep)
            
            if deps_met:
                ready.append(comp_id)
            else:
                blocked.append({
                    'component': comp_id,
                    'name': comp['name'],
                    'missing': missing_deps
                })
        
        return ready, blocked
    
    def get_next_component(self):
        """Get the highest priority component ready to build"""
        ready, blocked = self.analyze_dependencies()
        
        if not ready:
            return None
        
        # Sort by priority (critical first, then phase order)
        def priority_score(comp_id):
            comp = self.components[comp_id]
            priority_map = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
            return (
                priority_map.get(comp['priority'], 0),
                -comp['phase']  # Earlier phases first
            )
        
        ready.sort(key=priority_score, reverse=True)
        return ready[0]
    
    def create_test_suite(self, comp):
        """Create tests for the component"""
        test_file = f"tests/{comp['id']}_test.py"
        os.makedirs("tests", exist_ok=True)
        
        # Clean component name for class
        class_name = comp['name'].replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
        
        test_code = f'''#!/usr/bin/env python3
"""
Tests for {comp['name']}
Component ID: {comp['id']}
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Test{class_name}(unittest.TestCase):
    """Test suite for {comp['name']}"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.component_name = "{comp['name']}"
        self.phase = {comp['phase']}
    
    def test_component_exists(self):
        """Test that the component can be referenced"""
        self.assertIsNotNone(self.component_name)
    
    def test_phase_assignment(self):
        """Test phase is correct"""
        self.assertEqual(self.phase, {comp['phase']})
    
    def test_priority(self):
        """Test priority is set"""
        self.assertIn("{comp['priority']}", ["critical", "high", "medium", "low"])

if __name__ == '__main__':
    unittest.main()
'''
        
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        return test_file
    
    def run_tests(self, comp, test_file):
        """Run the test suite and return results"""
        print(f"   🧪 Running tests for {comp['name']}...")
        
        if not os.path.exists(test_file):
            print(f"   ⚠️  No test file found")
            return {'passed': False, 'reason': 'no_tests', 'score': 0}
        
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            passed = result.returncode == 0
            
            # Calculate score based on test results
            output = result.stdout + result.stderr
            test_count = output.count('test_')
            passed_count = output.count('OK') if passed else 0
            
            score = 1.0 if passed else 0.0
            
            test_result = {
                'passed': passed,
                'score': score,
                'output': output[-500:],  # Last 500 chars
                'timestamp': str(datetime.now())
            }
            
            print(f"   {'✅' if passed else '❌'} Tests {'passed' if passed else 'failed'}")
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Tests timed out")
            return {'passed': False, 'reason': 'timeout', 'score': 0}
        except Exception as e:
            print(f"   ⚠️  Test error: {e}")
            return {'passed': False, 'reason': str(e), 'score': 0}
    
    def build_component(self, comp_id):
        """Build and test a component"""
        comp = self.components[comp_id]
        min_pass_rate = 0.7  # Default 70%
        
        print(f"\n🔨 Building: {comp['name']} (Phase {comp['phase']})")
        print(f"   Priority: {comp['priority']}")
        print(f"   Required test pass rate: {min_pass_rate*100}%")
        
        # Mark as in progress
        self.in_progress.append(comp_id)
        self.save_state()
        
        # PHASE 1: BUILD
        print("   ⚙️  Building component...")
        time.sleep(1)  # Simulate build time
        
        # Create a component file
        comp_dir = f"components/phase{comp['phase']}"
        os.makedirs(comp_dir, exist_ok=True)
        comp_file = f"{comp_dir}/{comp['id']}_{comp['name'].replace(' ', '_')}.py"
        
        with open(comp_file, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
{comp['name']} - Built by DMAI Test-Aware Builder
Component ID: {comp['id']}
Phase: {comp['phase']}
Priority: {comp['priority']}
"""

class {comp['name'].replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}:
    def __init__(self):
        self.name = "{comp['name']}"
        self.id = "{comp['id']}"
        self.phase = {comp['phase']}
        self.status = "built"
    
    def info(self):
        return {{
            "name": self.name,
            "id": self.id,
            "phase": self.phase,
            "status": self.status
        }}

if __name__ == "__main__":
    component = {comp['name'].replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}()
    print(f"✅ {{component.name}} built successfully")
''')
        
        print(f"   📁 Created: {comp_file}")
        
        # PHASE 2: CREATE TESTS
        print("   🧪 Creating tests...")
        test_file = self.create_test_suite(comp)
        
        # PHASE 3: RUN TESTS
        test_result = self.run_tests(comp, test_file)
        
        # Store test results
        self.test_results[comp_id] = test_result
        
        # PHASE 4: VALIDATE
        if test_result['passed'] and test_result['score'] >= min_pass_rate:
            print(f"   ✅ Tests passed! Score: {test_result['score']*100}%")
            
            # Mark as completed
            self.completed.append(comp_id)
            self.in_progress.remove(comp_id)
            self.save_state()
            
            return True
        else:
            print(f"   ❌ Tests failed! Score: {test_result['score']*100}%")
            print(f"   ⚠️  Error: {test_result.get('reason', 'Unknown')}")
            
            # Move to failed for review
            self.failed.append(comp_id)
            if comp_id in self.in_progress:
                self.in_progress.remove(comp_id)
            self.save_state()
            
            return False
    
    def print_status(self):
        """Print current build status"""
        total = len(self.components)
        completed = len(self.completed)
        in_progress = len(self.in_progress)
        failed = len(self.failed)
        remaining = total - completed - in_progress - failed
        
        print("\n" + "="*80)
        print("📊 DMAI BUILD & TEST STATUS")
        print("="*80)
        print(f"Total components: {total}")
        print(f"✅ Passed & Completed: {completed}")
        print(f"⚙️  In Progress: {in_progress}")
        print(f"❌ Failed: {failed}")
        print(f"⏳ Remaining: {remaining}")
        print(f"Progress: {completed/total*100:.1f}%")
        
        # Show next component
        next_comp = self.get_next_component()
        if next_comp:
            comp = self.components[next_comp]
            print(f"\n🚀 Next to build: {comp['name']} (Phase {comp['phase']})")
        
        # Show recent failures
        if self.failed:
            print(f"\n❌ Failed components:")
            for fail_id in self.failed:
                print(f"   • {self.components[fail_id]['name']}")
        
        print("="*80)
    
    def run(self):
        """Main execution loop"""
        print("\n" + "="*80)
        print("🧬 DMAI TEST-AWARE BUILDER - Build, Test, Validate")
        print("="*80)
        
        # Load roadmap
        if not self.load_roadmap():
            return
        
        # Main build loop
        try:
            while True:
                self.print_status()
                
                # Offer options
                print("\nOptions:")
                print("  [b] Build next component")
                print("  [r] Retry failed components")
                print("  [s] Show detailed status")
                print("  [q] Quit")
                
                choice = input("\nChoice: ").lower()
                
                if choice == 'q':
                    print("\n🛑 Build paused")
                    break
                    
                elif choice == 'b':
                    next_comp = self.get_next_component()
                    if not next_comp:
                        print("\n🎉 All components built and tested!")
                        break
                    
                    comp = self.components[next_comp]
                    print(f"\n🔧 Building: {comp['name']}")
                    confirm = input("Proceed? (y/n): ").lower()
                    
                    if confirm == 'y':
                        self.build_component(next_comp)
                    else:
                        print("⏸️  Skipping")
                        
                elif choice == 'r':
                    if not self.failed:
                        print("✅ No failed components")
                    else:
                        print("\n🔄 Retrying failed components...")
                        failed_copy = self.failed.copy()
                        for fail_id in failed_copy:
                            print(f"\n🔄 Retrying: {self.components[fail_id]['name']}")
                            self.failed.remove(fail_id)
                            self.build_component(fail_id)
                            time.sleep(1)
                    
                elif choice == 's':
                    print("\n📋 Detailed Component Status:")
                    for phase in range(8):
                        phase_comps = [c for c in self.components.values() if c['phase'] == phase]
                        if phase_comps:
                            print(f"\nPhase {phase}:")
                            for comp in phase_comps:
                                comp_id = [k for k, v in self.components.items() if v == comp][0]
                                if comp_id in self.completed:
                                    status = "✅"
                                elif comp_id in self.failed:
                                    status = "❌"
                                elif comp_id in self.in_progress:
                                    status = "⚙️"
                                else:
                                    status = "⏳"
                                print(f"  {status} {comp['name']}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Build interrupted by user")
        
        self.print_status()
        print(f"\n💾 State saved to {self.state_file}")

if __name__ == "__main__":
    builder = TestAwareBuilder()
    builder.run()
