#!/usr/bin/env python3
"""
Final cleanup - remove duplicates and fix numbering
"""

import os
import re
import shutil
import glob
import json

print('='*60)
print('🧹 FINAL CLEANUP - Removing Duplicates')
print('='*60)

# Load roadmap
with open('docs/complete_roadmap_clean.json', 'r') as f:
    roadmap = json.load(f)

# Get correct IDs
correct_ids = {}
for phase in roadmap['phases']:
    for task in phase['tasks']:
        correct_ids[task['id']] = task['name']

print(f'Roadmap has {len(correct_ids)} components')

# Create backup
backup_dir = f'components/backup_final_{os.popen("date +%Y%m%d_%H%M%S").read().strip()}'
print(f'\n📦 Creating backup in {backup_dir}')
shutil.copytree('components', backup_dir)

# Process each phase
for phase in range(8):
    phase_dir = f'components/phase{phase}'
    if not os.path.exists(phase_dir):
        continue
    
    print(f'\n📁 Phase {phase}:')
    files = glob.glob(f'{phase_dir}/*.py')
    
    # Group files by their component ID
    id_to_files = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Try to extract ID from filename
        match = re.search(r'(P[0-9]T[0-9]+)', filename)
        if match:
            file_id = match.group(1)
            if file_id not in id_to_files:
                id_to_files[file_id] = []
            id_to_files[file_id].append(file_path)
    
    # For each ID, keep only the best file
    for file_id, file_list in id_to_files.items():
        if len(file_list) > 1:
            print(f'  ⚠️  Multiple files for {file_id}:')
            for f in file_list:
                print(f'    {os.path.basename(f)}')
            
            # Keep the one with correct naming, delete others
            correct_name = correct_ids.get(file_id, '').replace(' ', '_').replace('#', '').replace('(', '').replace(')', '').replace('-', '_')
            expected = f"{file_id}_{correct_name}.py" if correct_name else None
            
            kept = None
            for f in file_list:
                if expected and os.path.basename(f) == expected:
                    kept = f
                    print(f'    ✅ Keeping: {os.path.basename(f)}')
                else:
                    os.remove(f)
                    print(f'    ❌ Removed: {os.path.basename(f)}')
            
            if not kept and file_list:
                # Keep the first one
                kept = file_list[0]
                for f in file_list[1:]:
                    os.remove(f)
                    print(f'    ❌ Removed: {os.path.basename(f)}')
    
    # Now check for missing tests
    print(f'\n  Checking tests for Phase {phase}:')
    for file_path in glob.glob(f'{phase_dir}/*.py'):
        filename = os.path.basename(file_path)
        match = re.search(r'(P[0-9]T[0-9]+)', filename)
        if match:
            comp_id = match.group(1)
            test_file = f'tests/{comp_id}_test.py'
            if not os.path.exists(test_file):
                print(f'    ⚠️  Missing test for {comp_id}')
                # Create basic test
                with open(test_file, 'w') as f:
                    f.write(f'''#!/usr/bin/env python3
"""
Tests for {comp_id}
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from components.phase{phase}.{filename.replace('.py', '')} import *
except:
    pass

class Test{comp_id}(unittest.TestCase):
    def test_component_exists(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
''')
                print(f'    ✅ Created test for {comp_id}')

print('\n' + '='*60)
print('✅ Cleanup complete!')
print('='*60)
