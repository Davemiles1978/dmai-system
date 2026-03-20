#!/usr/bin/env python3
"""
Fix dmai_daemon_fixed.py - Remove evolution_engine from SERVICES since it's run by dual_launcher
"""

import re
from pathlib import Path

daemon_path = Path("/Users/davidmiles/Desktop/dmai-system/scripts/dmai_daemon_fixed.py")

if not daemon_path.exists():
    print(f"❌ Daemon not found at {daemon_path}")
    exit(1)

# Read current content
with open(daemon_path, 'r') as f:
    content = f.read()

# Find the SERVICES list and remove evolution_engine entry
# The SERVICES list starts with "SERVICES = [" and ends with "]"
pattern = r'SERVICES = \[(.*?)\]'
match = re.search(pattern, content, re.DOTALL)

if match:
    services_text = match.group(1)
    
    # Split into individual service entries
    entries = []
    current_entry = []
    bracket_count = 0
    
    for line in services_text.split('\n'):
        bracket_count += line.count('{') - line.count('}')
        current_entry.append(line)
        
        if bracket_count == 0 and line.strip().endswith('},'):
            entries.append('\n'.join(current_entry))
            current_entry = []
    
    # Filter out evolution_engine
    new_entries = []
    for entry in entries:
        if '"evolution_engine"' not in entry and '"name": "evolution_engine"' not in entry:
            new_entries.append(entry)
    
    # Rebuild SERVICES list
    new_services = 'SERVICES = [\n' + '\n'.join(new_entries) + '\n]'
    content = content.replace(match.group(0), new_services)
    
    print(f"✅ Removed evolution_engine from SERVICES list")
    print(f"   Original entries: {len(entries)}")
    print(f"   New entries: {len(new_entries)}")
    
    # Also update the initial cleanup to not try to kill evolution_engine
    content = content.replace(
        'os.system("pkill -f \'evolution_engine.py\' 2>/dev/null")',
        '# evolution_engine is run by dual_launcher, not directly'
    )
    
    # Write back
    with open(daemon_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated daemon configuration")
else:
    print("❌ Could not find SERVICES list in daemon file")

