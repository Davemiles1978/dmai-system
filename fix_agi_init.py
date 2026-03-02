#!/usr/bin/env python3
"""
Fix duplicate __init__ methods in agi_orchestrator.py
"""
import re

with open('agi_orchestrator.py', 'r') as f:
    content = f.read()

# Find all __init__ methods
import_pattern = r'def __init__\(self, base_path: str = "shared_data/agi_evolution"\):.*?(?=def |$)'
matches = list(re.finditer(import_pattern, content, re.DOTALL))

if len(matches) >= 2:
    print(f"Found {len(matches)} __init__ methods")
    
    # Keep the first one (with all components)
    first_init = matches[0].group()
    
    # Remove subsequent __init__ methods
    for match in matches[1:]:
        content = content.replace(match.group(), '')
    
    # Also remove any duplicate code at the end
    content = re.sub(r'    def __init__\(.*?\):.*?(?=def |$)', '', content, flags=re.DOTALL, count=1)
    
    with open('agi_orchestrator.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed duplicate __init__ methods")
    
    # Verify the fix
    with open('agi_orchestrator.py', 'r') as f:
        new_content = f.read()
        final_count = len(re.findall(r'def __init__\(', new_content))
        print(f"Final __init__ count: {final_count}")
else:
    print("No duplicate __init__ methods found")
