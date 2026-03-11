#!/usr/bin/env python3
"""
Fix for dmai_daemon_fixed.py - Improve process detection
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

# Replace is_process_alive with a more robust version
new_is_alive = '''def is_process_alive(pid, service_name=None):
    """Check if process is alive using multiple methods"""
    if not pid:
        return False
    
    # Method 1: Try kill(0) to check if process exists
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    
    # Method 2: Check if it's a zombie
    try:
        result = subprocess.run(['ps', '-o', 'state=', '-p', str(pid)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            state = result.stdout.strip()
            if state == 'Z':  # Zombie
                return False
    except:
        pass
    
    # Method 3: For evolution_engine, also check if port 9003 is listening
    if service_name == 'evolution_engine':
        try:
            # Check if port 9003 is in use
            result = subprocess.run(['lsof', '-i', ':9003'], 
                                  capture_output=True, text=True)
            if '9003' in result.stdout and str(pid) in result.stdout:
                return True
            # Also check if any process named evolution is running
            result = subprocess.run(['pgrep', '-f', 'continuous_advanced_evolution'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except:
            pass
    
    # Method 4: Check process name
    if service_name:
        try:
            result = subprocess.run(['ps', '-p', str(pid), '-o', 'comm='],
                                  capture_output=True, text=True)
            cmd = result.stdout.strip().lower()
            
            # Map service names to expected process names
            name_map = {
                'evolution_engine': ['continuous_advanced_evolution', 'evolution_engine', 'python'],
                'dual_launcher': ['dual_launcher', 'python'],
                'harvester_daemon': ['harvester', 'python'],
                'harvester_api': ['api_server', 'python'],
                'book_reader': ['book_reader', 'python'],
                'web_researcher': ['web_researcher', 'python'],
                'dark_researcher': ['dark_researcher', 'python'],
                'music_learner': ['music_learner', 'python'],
                'voice_service': ['dmai_voice', 'python'],
            }
            
            expected = name_map.get(service_name, [service_name])
            if any(name in cmd for name in expected):
                return True
        except:
            pass
    
    # If we got here, the process exists but might be misidentified
    return True

def check_service(service):
    """Check if service is running - enhanced version"""
    # First check by PID
    if service.get("pid") and is_process_alive(service["pid"], service["name"]):
        return True
    
    # If PID check fails, try to find the process by name
    try:
        name_map = {
            'evolution_engine': 'continuous_advanced_evolution',
            'dual_launcher': 'dual_launcher',
            'harvester_daemon': 'harvester',
            'harvester_api': 'api_server',
            'book_reader': 'book_reader',
            'web_researcher': 'web_researcher',
            'dark_researcher': 'dark_researcher',
            'music_learner': 'music_learner',
            'voice_service': 'dmai_voice',
        }
        
        search_term = name_map.get(service["name"], service["name"])
        result = subprocess.run(['pgrep', '-f', search_term], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split()
            if pids:
                # Update the service with the new PID
                service["pid"] = int(pids[0])
                service["last_restart"] = time.time()
                return True
    except:
        pass
    
    return False'''

# Replace the functions
content = re.sub(
    r'def is_process_alive\(pid\).*?(?=\n\S|\Z)',
    new_is_alive,
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'def check_service\(service\).*?(?=\n\S|\Z)',
    'def check_service(service):\n    """Check if service is running"""\n    return is_process_alive(service.get("pid"), service.get("name"))',
    content,
    flags=re.DOTALL
)

# Write back
with open(daemon_path, 'w') as f:
    f.write(content)

print("✅ Updated daemon with improved process detection")
