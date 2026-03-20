#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
DMAI Status Verifier - Check what's actually working
"""

import os
import sys
import json
import importlib
from pathlib import Path

def check_module(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return "✅ IMPORT OK"
    except ImportError as e:
        return f"❌ IMPORT FAIL: {e}"
    except Exception as e:
        return f"⚠️ ERROR: {e}"

def check_file_exists(filepath):
    """Check if a file exists"""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        return f"✅ EXISTS ({size} bytes)"
    return "❌ NOT FOUND"

def main():
    print("\n" + "="*60)
    print("DMAI SYSTEM VERIFICATION - MARCH 5, 2026")
    print("="*60)
    
    # Core modules
    print("\n📦 CORE MODULES:")
    print(f"  safety.py: {check_module('safety')}")
    print(f"  music_learner.py: {check_module('music_learner')}")
    
    # Voice system
    print("\n🎤 VOICE SYSTEM:")
    print(f"  voice/dmai_voice_with_learning.py: {check_file_exists('voice/dmai_voice_with_learning.py')}")
    print(f"  voice/enroll_master_comprehensive.py: {check_file_exists('voice/enroll_master_comprehensive.py')}")
    print(f"  voice/enroll_master_improved.py: {check_file_exists('voice/enroll_master_improved.py')}")
    print(f"  voice/enroll_master.py: {check_file_exists('voice/enroll_master.py')}")
    
    # Check for voice profiles
    print("\n🔐 VOICE PROFILES:")
    profile_files = list(Path("data").glob("voice_*.json")) if Path("data").exists() else []
    if profile_files:
        for f in profile_files:
            print(f"  {f.name}: ✅ EXISTS")
    else:
        print("  No voice profiles found ❌")
    
    # Music system
    print("\n🎵 MUSIC SYSTEM:")
    print(f"  music_identifier.py: {check_file_exists('music_identifier.py')}")
    
    # Check music data
    music_data = Path("data/music")
    if music_data.exists():
        artists_file = music_data / "artists.json"
        if artists_file.exists():
            with open(artists_file) as f:
                artists = json.load(f).get("artists", {})
            print(f"  Artists known: {len(artists)}")
        else:
            print("  No music data yet ❌")
    
    # Cloud UI
    print("\n☁️ CLOUD UI:")
    print(f"  cloud_web_ui.py: {check_file_exists('cloud_web_ui.py')}")
    print(f"  dmai_web_ui.py: {check_file_exists('dmai_web_ui.py')}")
    
    # Mobile
    print("\n📱 MOBILE INTEGRATION:")
    print("  No mobile files found ❌")
    
    # Reports
    print("\n📊 REPORTS:")
    print(f"  daily_report.py: {check_file_exists('daily_report.py')}")
    
    # Check for running processes
    print("\n🔄 RUNNING PROCESSES:")
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    dmai_processes = [line for line in result.stdout.split('\n') if 'dmai' in line.lower()]
    if dmai_processes:
        for proc in dmai_processes[:3]:  # Show first 3
            print(f"  {proc[:80]}...")
    else:
        print("  No DMAI processes running ❌")
    
    print("\n" + "="*60)
    print("RECOMMENDED NEXT ACTIONS:")
    print("1. Run voice enrollment: python voice/enroll_master_comprehensive.py")
    print("2. Start background listener: python background_listener.py")
    print("3. Test cloud UI: python cloud_web_ui.py")
    print("4. Verify music learner: python -c 'from music_learner import develop_dmai_taste; develop_dmai_taste()'")
    print("="*60)

if __name__ == "__main__":
    main()
