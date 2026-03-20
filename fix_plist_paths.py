#!/usr/bin/env python3
"""
Update all launchd plists to point to new api-harvester location
"""

import os
import plistlib
import subprocess

def fix_plists():
    launchd_dir = os.path.expanduser("~/Library/LaunchAgents")
    correct_path = "/Users/davidmiles/Desktop/dmai-system/api-harvester"
    
    print("\n🔧 Updating launchd plists...")
    
    for file in os.listdir(launchd_dir):
        if 'dmai' in file.lower() and file.endswith('.plist'):
            plist_path = os.path.join(launchd_dir, file)
            
            try:
                with open(plist_path, 'rb') as f:
                    plist = plistlib.load(f)
                
                modified = False
                
                # Update WorkingDirectory
                if 'WorkingDirectory' in plist:
                    old = plist['WorkingDirectory']
                    if 'api-harvester' in old or 'api_harvester' in old:
                        plist['WorkingDirectory'] = correct_path
                        modified = True
                        print(f"  📝 {file}: {old} -> {correct_path}")
                
                if modified:
                    with open(plist_path, 'wb') as f:
                        plistlib.dump(plist, f)
                    
                    # Reload service
                    service = file.replace('.plist', '')
                    subprocess.run(['launchctl', 'unload', plist_path])
                    subprocess.run(['launchctl', 'load', plist_path])
                    print(f"  ✅ Reloaded {service}")
                    
            except Exception as e:
                print(f"  ❌ Error with {file}: {e}")

if __name__ == "__main__":
    fix_plists()
