#!/usr/bin/env python3
"""Fix Python import paths in all DMAI scripts"""
import os
import re
from pathlib import Path

def fix_script(filepath):
    """Add path setup to script if needed"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if script already has path setup
    has_path_setup = 'sys.path.insert' in content or 'sys.path.append' in content
    
    if has_path_setup:
        # Update existing path setup to use correct path
        content = re.sub(
            r'sys\.path\.(insert|append)\([^)]*',
            f'sys.path.insert(0, str(Path(__file__).parent.parent))).parent.parent))',
            content
        )
        modified = True
    else:
        # Add path setup after imports
        lines = content.split('\n')
        new_lines = []
        import_section = True
        added = False
        
        for line in lines:
            new_lines.append(line)
            if import_section and line.startswith(('import ', 'from ')):
                continue
            elif import_section and not added:
                # Add path setup after imports
                new_lines.extend([
                    '',
                    '# Add project root to path',
                    'import sys',
                    'from pathlib import Path',
                    'sys.path.insert(0, str(Path(__file__).parent.parent))).parent.parent))',
                    ''
                ])
                added = True
                import_section = False
            else:
                import_section = False
        
        content = '\n'.join(new_lines)
        modified = added
    
    if modified:
        # Create backup
        backup = filepath + '.bak'
        if not os.path.exists(backup):
            os.rename(filepath, backup)
        
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    print("🔧 Fixing Python import paths in DMAI scripts")
    print("="*60)
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip venv and hidden directories
        if 'venv' in root or '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(python_files)} Python files")
    
    # Fix each file
    fixed = 0
    for filepath in python_files:
        if fix_script(filepath):
            print(f"✅ Fixed: {filepath}")
            fixed += 1
    
    print("\n" + "="*60)
    print(f"📊 Summary: Fixed {fixed} files")
    print("✅ Backups created with .bak extension")

if __name__ == "__main__":
    main()
