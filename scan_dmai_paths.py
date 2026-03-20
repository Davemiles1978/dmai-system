#!/usr/bin/env python3
"""
DMAI System Path Scanner
Scans all DMAI-related directories and identifies paths being used
"""

import os
import json
import subprocess
import plistlib
from datetime import datetime
from pathlib import Path

def scan_dmai_directories():
    """Find all DMAI-related directories on the system"""
    base_paths = [
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~")
    ]
    
    dmai_dirs = []
    patterns = ['dmai', 'DMAI', 'api-harvester', 'evolution', 'AGI']
    
    for base in base_paths:
        if os.path.exists(base):
            try:
                for item in os.listdir(base):
                    item_path = os.path.join(base, item)
                    if os.path.isdir(item_path):
                        # Check if directory name contains any pattern
                        if any(pattern.lower() in item.lower() for pattern in patterns):
                            dmai_dirs.append({
                                'path': item_path,
                                'name': item,
                                'exists': True,
                                'size': get_dir_size(item_path) if os.path.exists(item_path) else 0
                            })
            except PermissionError:
                continue
    
    return dmai_dirs

def get_dir_size(path):
    """Get directory size in MB"""
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except (PermissionError, OSError):
        pass
    return round(total / (1024 * 1024), 2)  # Convert to MB

def scan_launchd_services():
    """Find all launchd services related to DMAI"""
    services = []
    
    # Check user launch agents
    launch_agent_dir = os.path.expanduser("~/Library/LaunchAgents")
    if os.path.exists(launch_agent_dir):
        for file in os.listdir(launch_agent_dir):
            if 'dmai' in file.lower() or 'api' in file.lower() or 'evolution' in file.lower():
                plist_path = os.path.join(launch_agent_dir, file)
                try:
                    with open(plist_path, 'rb') as f:
                        plist = plistlib.load(f)
                    
                    services.append({
                        'name': file,
                        'path': plist_path,
                        'program': plist.get('Program', plist.get('ProgramArguments', ['Unknown'])[0]),
                        'working_dir': plist.get('WorkingDirectory', 'Not specified')
                    })
                except Exception as e:
                    services.append({
                        'name': file,
                        'path': plist_path,
                        'error': str(e)
                    })
    
    return services

def scan_running_processes():
    """Find running DMAI-related processes"""
    processes = []
    
    try:
        # Get all Python processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if any(term in line.lower() for term in ['dmai', 'api-harvester', 'evolution']):
                parts = line.split()
                if len(parts) > 10:
                    processes.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'command': ' '.join(parts[10:]),
                        'user': parts[0]
                    })
    except Exception as e:
        processes.append({'error': str(e)})
    
    return processes

def scan_config_files():
    """Find all configuration files"""
    config_files = []
    search_paths = [
        os.path.expanduser("~/Desktop/dmai-system"),
        os.path.expanduser("~")
    ]
    
    for base in search_paths:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                for file in files:
                    if file.endswith(('.json', '.yaml', '.yml', '.conf', '.env', '.ini')):
                        if any(term in root.lower() + file.lower() for term in ['dmai', 'api', 'evolution']):
                            filepath = os.path.join(root, file)
                            config_files.append({
                                'path': filepath,
                                'name': file,
                                'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
                            })
    
    return config_files

def check_path_exists(path):
    """Check if a specific path exists"""
    return {
        'path': path,
        'exists': os.path.exists(path),
        'is_dir': os.path.isdir(path) if os.path.exists(path) else False,
        'is_file': os.path.isfile(path) if os.path.exists(path) else False
    }

def main():
    print("🔍 DMAI System Path Scanner")
    print("=" * 50)
    print(f"Scan started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'directories': scan_dmai_directories(),
        'launchd_services': scan_launchd_services(),
        'running_processes': scan_running_processes(),
        'config_files': scan_config_files(),
        'critical_paths': {
            '/Users/davidmiles/Desktop/api-harvester': check_path_exists('/Users/davidmiles/Desktop/api-harvester'),
            '/Users/davidmiles/Desktop/dmai-system': check_path_exists('/Users/davidmiles/Desktop/dmai-system'),
            '/Users/davidmiles/Desktop/AI-Evolution-System': check_path_exists('/Users/davidmiles/Desktop/AI-Evolution-System'),
            '/Users/davidmiles/Desktop/AI-Evolution-System-old-backup': check_path_exists('/Users/davidmiles/Desktop/AI-Evolution-System-old-backup')
        }
    }
    
    # Print summary
    print("📁 DMAI Directories Found:")
    for d in results['directories']:
        status = "✅" if d['exists'] else "❌"
        print(f"  {status} {d['path']} ({d['size']} MB)")
    
    print(f"\n⚙️  Launchd Services Found: {len(results['launchd_services'])}")
    for s in results['launchd_services']:
        print(f"  • {s['name']}")
        print(f"    Program: {s.get('program', 'Unknown')}")
        print(f"    Working Dir: {s.get('working_dir', 'Unknown')}")
    
    print(f"\n🔄 Running Processes: {len(results['running_processes'])}")
    for p in results['running_processes'][:5]:  # Show first 5
        print(f"  • PID {p['pid']}: {p['command'][:80]}...")
    
    print("\n🔍 Critical Path Check:")
    for path, info in results['critical_paths'].items():
        status = "✅ EXISTS" if info['exists'] else "❌ MISSING"
        print(f"  {status} - {path}")
    
    # Save to file
    output_file = f"dmai_path_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Full scan saved to: {output_file}")
    
    # Also save a readable text version
    txt_file = f"dmai_path_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(txt_file, 'w') as f:
        f.write(f"DMAI System Path Scan - {results['timestamp']}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CRITICAL PATHS:\n")
        for path, info in results['critical_paths'].items():
            f.write(f"  {path}: {'EXISTS' if info['exists'] else 'MISSING'}\n")
        
        f.write("\nLAUNCHD SERVICES:\n")
        for s in results['launchd_services']:
            f.write(f"\n  {s['name']}\n")
            f.write(f"    Path: {s['path']}\n")
            f.write(f"    Program: {s.get('program', 'N/A')}\n")
            f.write(f"    Working Dir: {s.get('working_dir', 'N/A')}\n")
    
    print(f"📄 Readable report saved to: {txt_file}")
    
    # Immediate action needed based on findings
    print("\n🚨 IMMEDIATE ACTIONS NEEDED:")
    if not results['critical_paths']['/Users/davidmiles/Desktop/api-harvester']['exists']:
        print("  ⚠️  CRITICAL: api-harvester path is missing!")
        print("     Run ACTION 2 next to locate/clone api-harvester")
    
    for s in results['launchd_services']:
        working_dir = s.get('working_dir', '')
        if '/Users/davidmiles/Desktop/api-harvester' in working_dir:
            print(f"  ⚠️  Service {s['name']} points to missing path")

if __name__ == "__main__":
    main()
