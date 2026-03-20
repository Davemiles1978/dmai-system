#!/usr/bin/env python3
"""
FAST DMAI Path Scanner - High-level only
Scans parent folders and critical paths quickly
"""

import os
import json
import subprocess
from datetime import datetime

def quick_dir_check():
    """Quick check of top-level directories only"""
    desktop = os.path.expanduser("~/Desktop")
    critical_paths = [
        os.path.join(desktop, "api-harvester"),
        os.path.join(desktop, "dmai-system"),
        os.path.join(desktop, "AI-Evolution-System"),
        os.path.join(desktop, "AI-Evolution-System-old-backup"),
        os.path.join(desktop, "my workspace"),
        os.path.join(desktop, "dmai-production")
    ]
    
    results = []
    for path in critical_paths:
        if os.path.exists(path):
            # Just get basic info without recursion
            try:
                items = os.listdir(path)[:5]  # First 5 items only
                item_count = len(os.listdir(path))
            except:
                items = []
                item_count = 0
            
            results.append({
                'path': path,
                'exists': True,
                'is_directory': os.path.isdir(path),
                'item_count': item_count,
                'sample_items': items
            })
        else:
            results.append({
                'path': path,
                'exists': False,
                'is_directory': False
            })
    
    return results

def check_launchd_quick():
    """Quick check of launchd plists"""
    launchd_dir = os.path.expanduser("~/Library/LaunchAgents")
    services = []
    
    if os.path.exists(launchd_dir):
        for file in os.listdir(launchd_dir):
            if any(term in file.lower() for term in ['dmai', 'api', 'evolution']):
                plist_path = os.path.join(launchd_dir, file)
                # Just read the first few lines instead of parsing full plist
                try:
                    with open(plist_path, 'r') as f:
                        content = f.read(2000)  # First 2000 chars
                        
                    # Extract working directory
                    if 'WorkingDirectory' in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'WorkingDirectory' in line:
                                if i+1 < len(lines):
                                    wd_line = lines[i+1].strip()
                                    if '<string>' in wd_line:
                                        working_dir = wd_line.replace('<string>', '').replace('</string>', '').strip()
                                        services.append({
                                            'name': file,
                                            'working_dir': working_dir,
                                            'exists': os.path.exists(working_dir) if working_dir else False
                                        })
                except:
                    services.append({'name': file, 'error': 'Could not read'})
    
    return services

def check_running_processes_quick():
    """Quick check of running processes"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        dmai_processes = []
        for line in lines:
            if any(term in line.lower() for term in ['dmai', 'api-harvester', 'evolution']):
                parts = line.split()
                if len(parts) > 10:
                    dmai_processes.append({
                        'pid': parts[1],
                        'command': parts[10][:50] + '...' if len(parts[10]) > 50 else parts[10]
                    })
        
        return dmai_processes[:10]  # First 10 only
    except:
        return []

def check_symlinks():
    """Check for any symlinks that might cause issues"""
    desktop = os.path.expanduser("~/Desktop")
    symlinks = []
    
    for item in os.listdir(desktop):
        item_path = os.path.join(desktop, item)
        if os.path.islink(item_path):
            target = os.readlink(item_path)
            if any(term in item.lower() + target.lower() for term in ['dmai', 'api', 'evolution']):
                symlinks.append({
                    'name': item,
                    'target': target,
                    'target_exists': os.path.exists(target)
                })
    
    return symlinks

def main():
    print("\n🔍 FAST DMAI PATH SCANNER")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # 1. Critical directories
    print("📁 CRITICAL DIRECTORIES:")
    for item in quick_dir_check():
        if item['exists']:
            print(f"  ✅ {item['path']}")
            print(f"     └─ {item['item_count']} items (sample: {', '.join(item['sample_items'])})\n")
        else:
            print(f"  ❌ {item['path']} - MISSING\n")
    
    # 2. Launchd services
    print("\n⚙️  LAUNCHD SERVICES:")
    services = check_launchd_quick()
    for s in services:
        if 'working_dir' in s:
            status = "✅" if s['exists'] else "❌"
            print(f"  {status} {s['name']} -> {s['working_dir']}")
    
    # 3. Running processes
    print("\n🔄 RUNNING PROCESSES:")
    procs = check_running_processes_quick()
    for p in procs[:5]:  # Show first 5 only
        print(f"  • PID {p['pid']}: {p['command']}")
    
    # 4. Symlinks
    print("\n🔗 SYMLINKS:")
    links = check_symlinks()
    for link in links:
        status = "✅" if link['target_exists'] else "❌"
        print(f"  {status} {link['name']} -> {link['target']}")
    
    # 5. Immediate fix needed
    print("\n🚨 CRITICAL ISSUES:")
    desktop = os.path.expanduser("~/Desktop")
    
    # Check if api-harvester exists anywhere
    api_paths = []
    for item in os.listdir(desktop):
        if 'api-harvester' in item.lower():
            api_paths.append(os.path.join(desktop, item))
    
    if not api_paths:
        print("  ❌ api-harvester NOT FOUND anywhere on Desktop")
        print("     ACTION REQUIRED: Need to clone api-harvester")
    else:
        print(f"  ✅ api-harvester found at: {api_paths[0]}")
    
    # Check for services pointing to wrong path
    for s in services:
        if 'working_dir' in s and not s['exists']:
            if '/api-harvester' in s['working_dir']:
                print(f"  ⚠️  Service {s['name']} points to missing path: {s['working_dir']}")
    
    # Save minimal report
    report = {
        'timestamp': datetime.now().isoformat(),
        'critical_dirs': quick_dir_check(),
        'services': services,
        'processes': procs,
        'symlinks': links
    }
    
    report_file = f"fast_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Minimal report saved to: {report_file}")
    print("\n⚠️  NEXT STEP: Run ACTION 2 to fix api-harvester")

if __name__ == "__main__":
    main()
