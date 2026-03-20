#!/usr/bin/env python3
"""
Rename all component files to match roadmap numbering
"""

import os
import re
import shutil
import glob
import json

print('='*60)
print('🔢 RENUMBERING COMPONENTS TO MATCH ROADMAP')
print('='*60)

# Load roadmap
with open('docs/complete_roadmap_clean.json', 'r') as f:
    roadmap = json.load(f)

# Build correct ID to name mapping
correct_ids = {}
for phase in roadmap['phases']:
    for task in phase['tasks']:
        correct_ids[task['id']] = task['name']

print(f'Roadmap has {len(correct_ids)} components')

# Create backup
backup_dir = f'components/backup_before_renumber_{os.popen("date +%Y%m%d_%H%M%S").read().strip()}'
print(f'\n📦 Creating backup in {backup_dir}')
shutil.copytree('components', backup_dir)

# Process each phase
for phase in range(8):
    phase_dir = f'components/phase{phase}'
    if not os.path.exists(phase_dir):
        continue
    
    print(f'\n📁 Phase {phase}:')
    files = glob.glob(f'{phase_dir}/*.py')
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Extract current ID from file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for component_id
        match = re.search(r'component_id\s*=\s*[\'"](P[0-9]T[0-9]+)[\'"]', content)
        if not match:
            print(f'  ⚠️  {filename}: No component_id found')
            continue
        
        current_id = match.group(1)
        
        # Check if this ID is in roadmap
        if current_id in correct_ids:
            # ID is correct, just ensure filename matches
            correct_name = correct_ids[current_id]
            safe_name = correct_name.replace(' ', '_').replace('#', '').replace('(', '').replace(')', '').replace('-', '_')
            new_filename = f"{current_id}_{safe_name}.py"
            new_path = os.path.join(phase_dir, new_filename)
            
            if file_path != new_path:
                os.rename(file_path, new_path)
                print(f'  🔄 {filename} -> {new_filename}')
            else:
                print(f'  ✅ {filename} (already correct)')
        else:
            # This ID is not in roadmap - needs to be mapped
            print(f'  ❌ {filename}: ID {current_id} not in roadmap')
            
            # Try to determine correct ID from context
            # For Phase 1: P1T11 should be P1T8? Let's map based on position
            phase_num = current_id[1]
            
            # Create mapping for common mismatches
            id_map = {
                # Phase 1
                'P1T11': 'P1T8',  # Test sync protocol
                'P1T12': 'P1T9',  # Implement master_control.py
                'P1T13': 'P1T10', # Test fragment recreation
                'P1T4': 'P1T4',   # Already correct
                'P1T6': 'P1T3',   # Create identity persona
                'P1T7': 'P1T7',   # Deploy Engine #2 (Oracle)
                'P1T8': 'P1T5',   # Implement validator.py
                'P1T9': 'P1T6',   # Deploy Engine #1 (AWS)
                
                # Phase 2
                'P2T14': 'P2T1',  # Create Privacy.com account
                'P2T15': 'P2T2',  # Create Coinbase account
                'P2T16': 'P2T3',  # Get virtual card(s)
                'P2T17': 'P2T4',  # Document KYC requirements
                'P2T18': 'P2T5',  # Create Revolut account
                'P2T19': 'P2T6',  # Test cloud payment
                
                # Phase 3
                'P3T20': 'P3T1',  # Implement provider_manager.py
                'P3T21': 'P3T2',  # Automate AWS account creation
                'P3T22': 'P3T3',  # Automate GCP account creation
                'P3T23': 'P3T4',  # Automate Azure account creation
                'P3T24': 'P3T5',  # Automate Oracle account creation
                'P3T25': 'P3T6',  # Deploy fragment spawning
                'P3T26': 'P3T7',  # Implement no-co-location audit
                
                # Phase 4
                'P4T27': 'P4T1',  # Implement traffic_masquerade.py
                'P4T28': 'P4T2',  # Implement identity_rotation.py
                'P4T29': 'P4T3',  # Implement honeypot_detector.py
                # P4T4 and P4T5 missing
                
                # Phase 5
                'P5T32': 'P5T1',  # Research Monero mining viability
                'P5T34': 'P5T3',  # Research micro-task automation
                'P5T36': 'P5T5',  # Research compute rental
                # P5T2, P5T4, P5T6, P5T7 missing
                
                # Phase 6
                'P6T39': 'P6T1',  # Implement distributed crawling
                # P6T2-P6T6 missing
                
                # Phase 7
                'P7T49': 'P7T5',  # Dual recovery maintenance
                'P7T50': 'P7T6',  # Master Control authentication
                # P7T1-P7T4 missing
            }
            
            if current_id in id_map:
                new_id = id_map[current_id]
                if new_id in correct_ids:
                    new_name = correct_ids[new_id]
                    safe_name = new_name.replace(' ', '_').replace('#', '').replace('(', '').replace(')', '').replace('-', '_')
                    new_filename = f"{new_id}_{safe_name}.py"
                    new_path = os.path.join(phase_dir, new_filename)
                    
                    # Update content with new ID
                    new_content = content.replace(f'component_id = "{current_id}"', f'component_id = "{new_id}"')
                    new_content = new_content.replace(f"component_id = '{current_id}'", f"component_id = '{new_id}'")
                    
                    with open(new_path, 'w') as f:
                        f.write(new_content)
                    
                    # Remove old file
                    os.remove(file_path)
                    print(f'  🔄 {filename} -> {new_filename} (ID: {current_id} -> {new_id})')
                else:
                    print(f'  ❌ Mapped ID {new_id} not in roadmap')
            else:
                print(f'  ❌ No mapping for {current_id}')

# Now create missing components
print('\n🔧 Creating missing components...')

# Track which IDs we have
existing_ids = set()
for phase in range(8):
    phase_dir = f'components/phase{phase}'
    if os.path.exists(phase_dir):
        for file in glob.glob(f'{phase_dir}/*.py'):
            with open(file, 'r') as f:
                content = f.read()
                match = re.search(r'component_id\s*=\s*[\'"](P[0-9]T[0-9]+)[\'"]', content)
                if match:
                    existing_ids.add(match.group(1))

# Find missing IDs
missing_ids = set(correct_ids.keys()) - existing_ids
if missing_ids:
    print(f'\nMissing {len(missing_ids)} components:')
    for comp_id in sorted(missing_ids):
        name = correct_ids[comp_id]
        phase = comp_id[1]
        print(f'  • {comp_id}: {name}')
        
        # Create file
        phase_dir = f'components/phase{phase}'
        os.makedirs(phase_dir, exist_ok=True)
        
        safe_name = name.replace(' ', '_').replace('#', '').replace('(', '').replace(')', '').replace('-', '_')
        filename = f"{phase_dir}/{comp_id}_{safe_name}.py"
        
        with open(filename, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
{name} - Component {comp_id}
"""

class {safe_name}:
    def __init__(self):
        self.name = "{name}"
        self.component_id = "{comp_id}"
        self.status = "initialized"
        self.depends_on = []
        
    def info(self):
        return {{
            "name": self.name,
            "id": self.component_id,
            "status": self.status
        }}

if __name__ == "__main__":
    component = {safe_name}()
    print(f"✅ {{component.name}} created")
''')
        print(f'    ✅ Created {filename}')
else:
    print('✅ All components already exist!')

print('\n' + '='*60)
print('✅ Renumbering complete!')
print('='*60)
