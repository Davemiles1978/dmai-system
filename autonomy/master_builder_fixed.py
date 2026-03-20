#!/usr/bin/env python3
"""
DMAI MASTER BUILDER - Fixed version for clean roadmap
"""

import os
import json
import time
from datetime import datetime

class MasterBuilder:
    def __init__(self):
        self.roadmap_file = "docs/complete_roadmap_clean.json"
        self.state_file = "autonomy/build_state.json"
        self.components = {}
        self.completed = []
        self.in_progress = []
        self.failed = []
        self.load_state()
        
    def load_state(self):
        """Load current build state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.completed = data.get('completed', [])
                self.in_progress = data.get('in_progress', [])
                self.failed = data.get('failed', [])
        
    def save_state(self):
        """Save build state"""
        with open(self.state_file, 'w') as f:
            json.dump({
                'completed': self.completed,
                'in_progress': self.in_progress,
                'failed': self.failed,
                'last_updated': str(datetime.now())
            }, f, indent=2)
    
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
        
        # Build component index
        for phase in self.roadmap['phases']:
            for task in phase['tasks']:
                # Use task['id'] which exists in the roadmap
                comp_id = task['id']
                self.components[comp_id] = {
                    'id': comp_id,
                    'name': task['name'],
                    'phase': phase['id'],
                    'priority': task['priority'],
                    'depends_on': task.get('depends_on', []),
                    'status': task['status'],
                    'eta': task.get('eta', 'unknown'),
                    'details': task.get('details', {}),
                    'code_template': task.get('code_template'),
                    'requirements': task.get('requirements', [])
                }
        
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
        
        # Sort by priority
        def priority_score(comp_id):
            comp = self.components[comp_id]
            priority_map = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
            return (
                priority_map.get(comp['priority'], 0),
                comp['phase']  # Earlier phases first
            )
        
        ready.sort(key=priority_score, reverse=True)
        return ready[0]
    
    def build_component(self, comp_id):
        """Build a specific component"""
        comp = self.components[comp_id]
        
        print(f"\n🔧 Building: {comp['name']} (Phase {comp['phase']})")
        print(f"   Priority: {comp['priority']}")
        print(f"   ETA: {comp['eta']}")
        
        # Mark as in progress
        self.in_progress.append(comp_id)
        self.save_state()
        
        # Determine component type and build accordingly
        success = False
        
        if 'Design' in comp['name'] or 'design' in comp['name'].lower():
            success = self.create_design_doc(comp)
            
        elif 'Implement' in comp['name'] and comp.get('code_template'):
            success = self.generate_code_file(comp)
            
        elif 'Deploy' in comp['name']:
            success = self.create_deployment_plan(comp)
            
        elif 'Create' in comp['name'] or 'Document' in comp['name'] or 'Research' in comp['name']:
            success = self.create_documentation(comp)
            
        elif 'Test' in comp['name']:
            success = self.create_test_plan(comp)
            
        else:
            success = self.create_placeholder(comp)
        
        if success:
            self.completed.append(comp_id)
            if comp_id in self.in_progress:
                self.in_progress.remove(comp_id)
            self.save_state()
            print(f"✅ Completed: {comp['name']}")
        else:
            print(f"❌ Failed: {comp['name']}")
            if comp_id not in self.failed:
                self.failed.append(comp_id)
            if comp_id in self.in_progress:
                self.in_progress.remove(comp_id)
            self.save_state()
        
        return success
    
    def create_design_doc(self, comp):
        """Create a design document"""
        doc_dir = "docs/design"
        os.makedirs(doc_dir, exist_ok=True)
        
        doc_path = f"{doc_dir}/{comp['id']}_{comp['name'].replace(' ', '_').replace('#', '').replace('(', '').replace(')', '')}.md"
        
        details = comp.get('details', {})
        content = f"""# {comp['name']}
**Component ID:** {comp['id']}
**Phase:** {comp['phase']}
**Priority:** {comp['priority']}
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Description
Design document for {comp['name']}

## Requirements
"""
        if isinstance(details, dict) and 'components' in details:
            for req in details.get('components', []):
                content += f"- {req}\n"
        elif isinstance(details, dict) and 'requirements' in details:
            for req in details.get('requirements', []):
                content += f"- {req}\n"
        
        if 'provider' in details:
            content += f"\n## Provider\n{details['provider']}\n"
        
        content += f"""
## Dependencies
"""
        for dep in comp['depends_on']:
            content += f"- {dep}\n"
        
        with open(doc_path, 'w') as f:
            f.write(content)
        
        print(f"   📁 Created design doc: {doc_path}")
        return True
    
    def generate_code_file(self, comp):
        """Generate a code file from template"""
        template_path = comp.get('code_template')
        if not template_path:
            # Create a default path
            template_path = f"components/phase{comp['phase']}/{comp['id']}_{comp['name'].replace(' ', '_').replace('#', '').replace('(', '').replace(')', '')}.py"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        # Generate basic code structure
        class_name = comp['name'].replace(' ', '').replace('#', '').replace('(', '').replace(')', '').replace('-', '_')
        
        code = f'''#!/usr/bin/env python3
"""
{comp['name']} - Auto-generated by DMAI Master Builder
Component ID: {comp['id']}
Phase: {comp['phase']}
"""

class {class_name}:
    """
    {comp['name']}
    """
    
    def __init__(self):
        self.name = "{comp['name']}"
        self.component_id = "{comp['id']}"
        self.phase = {comp['phase']}
        self.created = "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.status = "initialized"
        
    def info(self):
        """Get component information"""
        return {{
            "name": self.name,
            "id": self.component_id,
            "phase": self.phase,
            "created": self.created,
            "status": self.status
        }}
    
    def __repr__(self):
        return f"<{self.name} ({self.component_id})>"

if __name__ == "__main__":
    instance = {class_name}()
    print(f"✅ {{instance.name}} created")
    print(instance.info())
'''
        
        with open(template_path, 'w') as f:
            f.write(code)
        
        print(f"   📁 Created code file: {template_path}")
        return True
    
    def create_deployment_plan(self, comp):
        """Create a deployment plan"""
        doc_dir = "docs/deployment"
        os.makedirs(doc_dir, exist_ok=True)
        
        doc_path = f"{doc_dir}/{comp['id']}_{comp['name'].replace(' ', '_').replace('#', '').replace('(', '').replace(')', '')}.md"
        
        content = f"""# {comp['name']} - Deployment Plan
**Component ID:** {comp['id']}
**Phase:** {comp['phase']}
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Deployment Steps
1. Prepare environment
2. Validate dependencies
3. Execute deployment
4. Verify deployment
5. Update status

## Dependencies
"""
        for dep in comp['depends_on']:
            content += f"- {dep}\n"
        
        with open(doc_path, 'w') as f:
            f.write(content)
        
        print(f"   📁 Created deployment plan: {doc_path}")
        return True
    
    def create_documentation(self, comp):
        """Create documentation"""
        doc_dir = "docs/documentation"
        os.makedirs(doc_dir, exist_ok=True)
        
        doc_path = f"{doc_dir}/{comp['id']}_{comp['name'].replace(' ', '_').replace('#', '').replace('(', '').replace(')', '')}.md"
        
        content = f"""# {comp['name']}
**Component ID:** {comp['id']}
**Phase:** {comp['phase']}
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
{comp['name']} documentation

## Requirements
"""
        if 'requirements' in comp and comp['requirements']:
            for req in comp['requirements']:
                content += f"- {req}\n"
        
        content += f"""
## Dependencies
"""
        for dep in comp['depends_on']:
            content += f"- {dep}\n"
        
        with open(doc_path, 'w') as f:
            f.write(content)
        
        print(f"   📁 Created documentation: {doc_path}")
        return True
    
    def create_test_plan(self, comp):
        """Create a test plan"""
        doc_dir = "docs/testing"
        os.makedirs(doc_dir, exist_ok=True)
        
        doc_path = f"{doc_dir}/{comp['id']}_{comp['name'].replace(' ', '_').replace('#', '').replace('(', '').replace(')', '')}.md"
        
        content = f"""# {comp['name']} - Test Plan
**Component ID:** {comp['id']}
**Phase:** {comp['phase']}
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Cases
1. Unit tests
2. Integration tests
3. Validation tests

## Success Criteria
- All tests pass
- No regressions
- Performance meets requirements

## Dependencies
"""
        for dep in comp['depends_on']:
            content += f"- {dep}\n"
        
        with open(doc_path, 'w') as f:
            f.write(content)
        
        print(f"   📁 Created test plan: {doc_path}")
        return True
    
    def create_placeholder(self, comp):
        """Create a placeholder for generic tasks"""
        placeholder_dir = "autonomy/placeholders"
        os.makedirs(placeholder_dir, exist_ok=True)
        
        placeholder_path = f"{placeholder_dir}/{comp['id']}.txt"
        
        with open(placeholder_path, 'w') as f:
            f.write(f"""Component: {comp['name']}
ID: {comp['id']}
Phase: {comp['phase']}
Priority: {comp['priority']}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: Placeholder - manual implementation needed
Dependencies: {', '.join(comp['depends_on'])}
""")
        
        print(f"   📁 Created placeholder: {placeholder_path}")
        return True
    
    def print_status(self):
        """Print current build status"""
        total = len(self.components)
        completed = len(self.completed)
        in_progress = len(self.in_progress)
        failed = len(self.failed)
        remaining = total - completed - in_progress - failed
        
        print("\n" + "="*70)
        print("📊 DMAI BUILD STATUS")
        print("="*70)
        print(f"Total components: {total}")
        print(f"✅ Completed: {completed}")
        print(f"⚙️  In Progress: {in_progress}")
        print(f"❌ Failed: {failed}")
        print(f"⏳ Remaining: {remaining}")
        print(f"Progress: {completed/total*100:.1f}%")
        
        # Show next component
        next_comp = self.get_next_component()
        if next_comp:
            comp = self.components[next_comp]
            print(f"\n🚀 Next to build: {comp['name']} (Phase {comp['phase']})")
        
        # Show blocked components
        ready, blocked = self.analyze_dependencies()
        if blocked:
            print(f"\n⛔ Blocked components: {len(blocked)}")
            for b in blocked[:3]:
                print(f"   • {b['name']} waiting for: {', '.join(b['missing'])}")
        
        print("="*70)
    
    def run(self):
        """Main execution loop"""
        print("\n" + "="*70)
        print("🧬 DMAI MASTER BUILDER - Self-Constructing System")
        print("="*70)
        
        # Load roadmap
        if not self.load_roadmap():
            return
        
        # Main build loop
        try:
            while True:
                self.print_status()
                
                # Get next component
                next_comp = self.get_next_component()
                if not next_comp:
                    print("\n🎉 All possible components built!")
                    break
                
                # Ask for confirmation
                comp = self.components[next_comp]
                print(f"\n🔧 Ready to build: {comp['name']}")
                response = input("Build now? (y/n/q): ").lower()
                
                if response == 'q':
                    print("\n🛑 Build paused")
                    break
                elif response == 'y':
                    self.build_component(next_comp)
                else:
                    print("⏸️  Skipping")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Build interrupted by user")
        
        self.print_status()
        print(f"\n💾 State saved to {self.state_file}")

if __name__ == "__main__":
    builder = MasterBuilder()
    builder.run()
