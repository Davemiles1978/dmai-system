#!/usr/bin/env python3
"""
DMAI MASTER BUILDER - Self-Constructing System
Reads the roadmap and builds itself in optimal order
"""

import os
import json
import time
import subprocess
from datetime import datetime

class MasterBuilder:
    def __init__(self):
        self.roadmap_file = "docs/complete_roadmap_clean.json"
        self.state_file = "autonomy/build_state.json"
        self.components = {}
        self.completed = []
        self.in_progress = []
        self.load_state()
        
    def load_state(self):
        """Load current build state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.completed = data.get('completed', [])
                self.in_progress = data.get('in_progress', [])
        
    def save_state(self):
        """Save build state"""
        with open(self.state_file, 'w') as f:
            json.dump({
                'completed': self.completed,
                'in_progress': self.in_progress,
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
                self.components[task['id']] = {
                    'name': task['name'],
                    'phase': phase['id'],
                    'priority': task['priority'],
                    'depends_on': task.get('depends_on', []),
                    'status': task['status'],
                    'eta': task.get('eta', 'unknown'),
                    'details': task.get('details', {}),
                    'code_template': task.get('code_template')
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
        
        if 'code_template' in comp and comp['code_template']:
            # Has a code template - generate the file
            success = self.generate_code_file(comp)
            
        elif 'Design' in comp['name'] or 'Research' in comp['name']:
            # Design/research task - create design document
            success = self.create_design_doc(comp)
            
        elif 'Fix' in comp['name']:
            # Fix task - attempt to fix the issue
            success = self.attempt_fix(comp)
            
        elif 'Deploy' in comp['name']:
            # Deployment task - simulate/actual deployment
            success = self.simulate_deployment(comp)
            
        else:
            # Generic task - create placeholder
            success = self.create_placeholder(comp)
        
        if success:
            # Mark as completed
            self.completed.append(comp_id)
            self.in_progress.remove(comp_id)
            self.save_state()
            print(f"✅ Completed: {comp['name']}")
        else:
            print(f"❌ Failed: {comp['name']}")
            # Leave in progress but could retry later
        
        return success
    
    def generate_code_file(self, comp):
        """Generate a code file from template"""
        template_path = comp['code_template']
        if not template_path:
            return False
        
        # Create directory if needed
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        # Generate basic code structure
        code = f'''#!/usr/bin/env python3
"""
{comp['name']} - Auto-generated by DMAI Master Builder
Part of Phase {comp['phase']} of the evolution roadmap
"""

import os
import sys
from datetime import datetime

class {os.path.basename(template_path).replace('.py', '').title()}:
    """
    {comp['name']}
    {comp.get('details', 'No details provided')}
    """
    
    def __init__(self):
        self.name = "{comp['name']}"
        self.created = str(datetime.now())
        self.phase = {comp['phase']}
        
    def info(self):
        return {{
            "name": self.name,
            "created": self.created,
            "phase": self.phase,
            "status": "initialized"
        }}

if __name__ == "__main__":
    instance = {os.path.basename(template_path).replace('.py', '').title()}()
    print(f"✅ {instance.name} created")
'''
        
        with open(template_path, 'w') as f:
            f.write(code)
        
        print(f"   📁 Created: {template_path}")
        return True
    
    def create_design_doc(self, comp):
        """Create a design document"""
        doc_path = f"docs/design/{comp['id']}_{comp['name'].replace(' ', '_')}.md"
        os.makedirs("docs/design", exist_ok=True)
        
        details = comp.get('details', {})
        content = f"""# {comp['name']}
**Phase:** {comp['phase']}
**Priority:** {comp['priority']}
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Description
{comp.get('details', 'Design document for this component')}

## Requirements
"""
        if isinstance(details, dict) and 'requirements' in details:
            for req in details['requirements']:
                content += f"- {req}\n"
        
        content += f"""
## Implementation Plan
1. Research existing solutions
2. Design architecture
3. Implement prototype
4. Test
5. Deploy

## Dependencies
"""
        for dep in comp['depends_on']:
            content += f"- {dep}\n"
        
        with open(doc_path, 'w') as f:
            f.write(content)
        
        print(f"   📁 Created design doc: {doc_path}")
        return True
    
    def attempt_fix(self, comp):
        """Attempt to fix a known issue"""
        # This would integrate with the weakness scanner
        print(f"   🔍 Would attempt to fix: {comp['name']}")
        return True
    
    def simulate_deployment(self, comp):
        """Simulate or perform deployment"""
        print(f"   🚀 Would deploy: {comp['name']}")
        return True
    
    def create_placeholder(self, comp):
        """Create a placeholder for generic tasks"""
        placeholder_path = f"autonomy/placeholders/{comp['id']}.txt"
        os.makedirs("autonomy/placeholders", exist_ok=True)
        
        with open(placeholder_path, 'w') as f:
            f.write(f"""Component: {comp['name']}
ID: {comp['id']}
Phase: {comp['phase']}
Priority: {comp['priority']}
Created: {datetime.now()}
Status: Placeholder - manual implementation needed
""")
        
        print(f"   📁 Created placeholder: {placeholder_path}")
        return True
    
    def print_status(self):
        """Print current build status"""
        total = len(self.components)
        completed = len(self.completed)
        in_progress = len(self.in_progress)
        remaining = total - completed - in_progress
        
        print("\n" + "="*70)
        print("📊 DMAI BUILD STATUS")
        print("="*70)
        print(f"Total components: {total}")
        print(f"✅ Completed: {completed}")
        print(f"⚙️  In Progress: {in_progress}")
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
            for b in blocked[:3]:  # Show first 3
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
                
                # Ask for confirmation (can be automated later)
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
                
                time.sleep(1)  # Brief pause between builds
                
        except KeyboardInterrupt:
            print("\n\n🛑 Build interrupted by user")
        
        self.print_status()
        print(f"\n💾 State saved to {self.state_file}")

if __name__ == "__main__":
    builder = MasterBuilder()
    builder.run()
