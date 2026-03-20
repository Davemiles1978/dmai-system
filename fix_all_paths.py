#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Comprehensive path fixer for DMAI migration"""
import os
import re
from pathlib import Path

def fix_paths_in_file(filepath):
    """Replace all instances of old path with new path in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Original path patterns to replace
        old_patterns = [
            (r'/Users/davidmiles/Desktop/AI-Evolution-System', '/Users/davidmiles/Desktop/dmai-system'),
            (r'~/Desktop/AI-Evolution-System', '~/Desktop/dmai-system'),
            (r'AI-Evolution-System', 'dmai-system'),
            (r'"AI-Evolution-System"', '"dmai-system"'),
            (r"'AI-Evolution-System'", "'dmai-system'"),
        ]
        
        new_content = content
        for pattern, replacement in old_patterns:
            new_content = new_content.replace(pattern, replacement)
        
        # Also fix any import statements that might reference the old structure
        new_content = re.sub(r'from\s+AI-Evolution-System\.', 'from dmai-system.', new_content)
        new_content = re.sub(r'import\s+AI-Evolution-System\.', 'import dmai-system.', new_content)
        
        if new_content != content:
            # Create backup
            backup = filepath + '.pathbak'
            if not os.path.exists(backup):
                os.rename(filepath, backup)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
    except Exception as e:
        print(f"  ⚠️  Error processing {filepath}: {e}")
    return False

def main():
    print("🔧 DMAI PATH FIXER")
    print("="*60)
    print("Fixing all references from 'AI-Evolution-System' to 'dmai-system'")
    
    fixed_files = 0
    total_files = 0
    
    # File extensions to process
    extensions = ['.py', '.sh', '.json', '.md', '.txt', '.conf', '.service', '.command', '.yml', '.yaml']
    
    for root, dirs, files in os.walk('.'):
        # Skip virtual environment and git
        if 'venv' in root or '.git' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                total_files += 1
                if fix_paths_in_file(filepath):
                    fixed_files += 1
                    print(f"✅ Fixed: {filepath}")
    
    print("\n" + "="*60)
    print(f"📊 Summary:")
    print(f"   Total files scanned: {total_files}")
    print(f"   Files fixed: {fixed_files}")
    print(f"   Backups created with .pathbak extension")
    print("="*60)
    
    # Specifically check critical files
    print("\n🔍 Checking critical files:")
    critical_files = [
        'dmai_core.py',
        'dmai_service.sh',
        'dmai.sh',
        'scripts/start_all_daemons.sh',
        'voice/dmai_voice.py',
        'voice/dmai_voice_with_learning.py',
        'evolution/evolution_engine.py',
        'requirements.txt'
    ]
    
    for cf in critical_files:
        if os.path.exists(cf):
            print(f"   📄 {cf}: OK")
        else:
            print(f"   ❌ {cf}: MISSING")
    
    print("\n✅ Path fixing complete!")
    print("\nNext steps:")
    print("1. Run: source venv/bin/activate")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: ./scripts/start_all_daemons.sh")

if __name__ == "__main__":
    main()
