#!/usr/bin/env python3
"""
Test script to diagnose service issues
"""

import subprocess
import time
import sys
from pathlib import Path

def test_service(name, cmd):
    print(f"\n🔍 Testing {name}...")
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment to see if it crashes
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ {name} started successfully (PID: {process.pid})")
            # Kill it
            process.terminate()
            time.sleep(1)
            return True
        else:
            # Get error output
            stdout, stderr = process.communicate()
            print(f"❌ {name} failed to start")
            if stderr:
                print(f"Error: {stderr[:500]}")
            return False
    except Exception as e:
        print(f"❌ Error testing {name}: {e}")
        return False

# Test evolution engine
venv_python = Path("/Users/davidmiles/Desktop/dmai-system/venv/bin/python3")
evolution_path = Path("/Users/davidmiles/Desktop/dmai-system/evolution/continuous_advanced_evolution.py")

if evolution_path.exists():
    test_service("evolution_engine", [str(venv_python), str(evolution_path), "--server"])
else:
    print(f"❌ Evolution engine not found at {evolution_path}")

# Test dual launcher
dual_path = Path("/Users/davidmiles/Desktop/dmai-system/evolution/dual_launcher.py")
if dual_path.exists():
    test_service("dual_launcher", [str(venv_python), str(dual_path)])
else:
    print(f"❌ Dual launcher not found at {dual_path}")

print("\n📊 Test complete")
