#!/usr/bin/env python3
"""
Fix harvester path in the daemon
Run this from the dmai-system directory
"""
import os
import re

daemon_path = "scripts/dmai_daemon_fixed.py"
correct_path = "/Users/davidmiles/Desktop/dmai-system/api-harvester"

print(f"📝 Fixing paths in {daemon_path}...")

# Check if file exists
if not os.path.exists(daemon_path):
    print(f"❌ Error: {daemon_path} not found!")
    print(f"Current directory: {os.getcwd()}")
    exit(1)

# Read the file
with open(daemon_path, 'r') as f:
    content = f.read()

# Replace the HARVESTER_ROOT line
original = content
new_content = re.sub(
    r'HARVESTER_ROOT\s*=\s*["\'].*?["\']',
    f'HARVESTER_ROOT = "{correct_path}"',
    content,
    flags=re.MULTILINE
)

# Also fix any other references to the old path
new_content = new_content.replace(
    '/Users/davidmiles/Desktop/api-harvester',
    correct_path
)

if new_content != original:
    # Write back
    with open(daemon_path, 'w') as f:
        f.write(new_content)
    print(f"✅ Updated HARVESTER_ROOT to: {correct_path}")
else:
    print("⚠️  No changes needed - path may already be correct")

# Now update launchd plists
print("\n📝 Checking launchd plists...")
launchd_dir = os.path.expanduser("~/Library/LaunchAgents")
updated = 0

for file in os.listdir(launchd_dir):
    if 'dmai' in file.lower() and file.endswith('.plist'):
        plist_path = os.path.join(launchd_dir, file)
        
        try:
            with open(plist_path, 'r') as f:
                plist_content = f.read()
            
            original_plist = plist_content
            new_plist = plist_content.replace(
                '/Users/davidmiles/Desktop/api-harvester',
                correct_path
            )
            new_plist = new_plist.replace(
                '/Users/davidmiles/Desktop/dmai-system/api_harvester',
                correct_path
            )
            
            if new_plist != original_plist:
                with open(plist_path, 'w') as f:
                    f.write(new_plist)
                print(f"✅ Updated {file}")
                updated += 1
        except Exception as e:
            print(f"⚠️  Could not read {file}: {e}")

print(f"\n📊 Summary:")
print(f"  • Updated {updated} launchd plist(s)")

# Kill any running daemons
print("\n🔄 Restarting daemon...")
os.system("pkill -f dmai_daemon_fixed.py 2>/dev/null")
print("✅ Old processes killed")

print("\n🚀 You can now start the daemon with:")
print("   python3 scripts/dmai_daemon_fixed.py start")
