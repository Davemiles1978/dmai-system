#!/usr/bin/env python3
"""
Fix all component files to have correct component_id declarations
"""

import os
import re
import glob

# Load roadmap to get correct component names and IDs
import json
with open('docs/complete_roadmap_clean.json', 'r') as f:
    roadmap = json.load(f)

# Create mapping of component IDs to names
component_names = {}
for phase in roadmap['phases']:
    for task in phase['tasks']:
        component_names[task['id']] = task['name']

print('='*60)
print('🔧 FIXING COMPONENT IDS')
print('='*60)

fixed_count = 0
total_count = 0

# Process all component files
for phase_dir in sorted(glob.glob('components/phase*')):
    phase = phase_dir.replace('components/phase', '')
    print(f'\n📁 Phase {phase}:')
    
    for file_path in sorted(glob.glob(f'{phase_dir}/*.py')):
        filename = os.path.basename(file_path)
        total_count += 1
        
        # Try to extract component ID from filename
        match = re.search(r'(P[0-9]T[0-9]+)', filename)
        if not match:
            print(f'  ⚠️  {filename}: No ID in filename')
            continue
            
        comp_id = match.group(1)
        
        # Get correct name from roadmap
        comp_name = component_names.get(comp_id, 'Unknown Component')
        
        # Read current content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if component_id is already set correctly
        has_correct_id = False
        if f'component_id = "{comp_id}"' in content or f"component_id = '{comp_id}'" in content:
            has_correct_id = True
        
        if has_correct_id:
            print(f'  ✅ {filename}: ID already correct')
            continue
        
        # Add or fix component_id
        lines = content.split('\n')
        new_lines = []
        in_class = False
        class_found = False
        indent = ''
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            # Find class definition
            if not class_found and line.strip().startswith('class '):
                class_found = True
                # Calculate indentation
                indent = line[:len(line) - len(line.lstrip())]
                
                # Add __init__ if not present
                next_line = lines[i+1] if i+1 < len(lines) else ''
                if 'def __init__' not in next_line:
                    new_lines.append(f'{indent}    def __init__(self):')
                    new_lines.append(f'{indent}        self.name = "{comp_name}"')
                    new_lines.append(f'{indent}        self.component_id = "{comp_id}"')
                    new_lines.append(f'{indent}        self.status = "initialized"')
                    new_lines.append('')
        
        # Write back
        new_content = '\n'.join(new_lines)
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f'  🔧 {filename}: Added component_id="{comp_id}"')
        fixed_count += 1

print('\n' + '='*60)
print(f'📊 SUMMARY: Fixed {fixed_count} of {total_count} files')
print('='*60)
