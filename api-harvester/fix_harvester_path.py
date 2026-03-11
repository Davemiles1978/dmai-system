#!/usr/bin/env python3
"""
Fix harvester path in the daemon
"""
import os
import re

daemon_path = "scripts/dmai_daemon_fixed.py"
correct_path = "/Users/davidmiles/Desktop/dmai-system/api-harvester"

# Read the file
with open(daemon_path, 'r') as f:
    content = f.read()

# Replace the path
new_content = re.sub(
    r'HARVESTER_ROOT\s*=\s*["\'].*?["\']',
    f'HARVESTER_ROOT = "{correct_path}"',
    content
)

# Also fix any other references to the old path
new_content = new_content.replace(
    '/Users/davidmiles/Desktop/api-harvester',
    correct_path
)

# Write back
with open(daemon_path, 'w') as f:
    f.write(new_content)

print(f"✅ Updated HARVESTER_ROOT to: {correct_path}")

# Also update any launchd plists that might have the wrong path
launchd_dir = os.path.expanduser("~/Library/LaunchAgents")
for file in os.listdir(launchd_dir):
    if 'dmai' in file.lower() and file.endswith('.plist'):
        plist_path = os.path.join(launchd_dir, file)
        with open(plist_path, 'r') as f:
            plist_content = f.read()
        
        if '/api-harvester' in plist_content or '/api_harvester' in plist_content:
            new_plist = plist_content.replace(
                '/Users/davidmiles/Desktop/api-harvester',
                correct_path
            )
            new_plist = new_plist.replace(
                '/Users/davidmiles/Desktop/dmai-system/api_harvester',
                correct_path
            )
            with open(plist_path, 'w') as f:
                f.write(new_plist)
            print(f"✅ Updated {file}")
