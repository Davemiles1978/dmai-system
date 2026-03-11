#!/usr/bin/env python3
"""
Quick check of api_harvester location
"""

import os

def check_paths():
    desktop = os.path.expanduser("~/Desktop")
    
    paths_to_check = [
        "/Users/davidmiles/Desktop/dmai-system/api_harvester",
        "/Users/davidmiles/Desktop/dmai-system/api-harvester",
        "/Users/davidmiles/Desktop/AI-Evolution-System/api-harvester",
        "/Users/davidmiles/Desktop/api-harvester"
    ]
    
    print("\n🔍 Checking api-harvester locations:")
    print("=" * 50)
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ FOUND: {path}")
            if os.path.isdir(path):
                try:
                    items = os.listdir(path)[:5]
                    print(f"   └─ Contains: {', '.join(items)}")
                except:
                    print("   └─ (cannot list contents)")
        else:
            print(f"❌ NOT FOUND: {path}")
    
    # Specifically check the one you mentioned
    suggested = "/Users/davidmiles/Desktop/dmai-system/api_harvester"
    if os.path.exists(suggested):
        print("\n✅ The api_harvester folder EXISTS at the suggested location!")
        print("\n📋 This means:")
        print("   • The code is in the right parent directory (dmai-system)")
        print("   • But the folder name uses underscore (_) instead of hyphen (-)")
        print("   • Launchd is looking for 'api-harvester' (with hyphen)")
        
        print("\n🔧 FIX OPTIONS:")
        print("   1. Create a symlink (quick fix):")
        print(f"      ln -s {suggested} /Users/davidmiles/Desktop/dmai-system/api-harvester")
        print("\n   2. Rename the folder (cleaner):")
        print(f"      mv {suggested} /Users/davidmiles/Desktop/dmai-system/api-harvester")
        print("\n   3. Update launchd plists to look for api_harvester instead")
    else:
        print(f"\n❌ {suggested} NOT FOUND - please verify the exact path")

if __name__ == "__main__":
    check_paths()
