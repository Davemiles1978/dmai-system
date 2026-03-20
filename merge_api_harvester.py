#!/usr/bin/env python3
"""
Complete api-harvester consolidation script
Merges all api-harvester code from multiple sources into one location
"""

import os
import shutil
import filecmp
from datetime import datetime
from pathlib import Path

# Define paths
DMAI_SYSTEM_PATH = "/Users/davidmiles/Desktop/dmai-system"
OLD_EVOLUTION_PATH = "/Users/davidmiles/Desktop/AI-Evolution-System"
OLD_BACKUP_PATH = "/Users/davidmiles/Desktop/AI-Evolution-System-old-backup"

# All source locations of api-harvester
SOURCE_PATHS = [
    {
        'name': 'Current (dmai-system)',
        'path': os.path.join(DMAI_SYSTEM_PATH, "api_harvester"),
        'priority': 3  # Highest priority - this is our target
    },
    {
        'name': 'Evolution System',
        'path': os.path.join(OLD_EVOLUTION_PATH, "api-harvester"),
        'priority': 2
    },
    {
        'name': 'Evolution Backup',
        'path': os.path.join(OLD_BACKUP_PATH, "api-harvester"),
        'priority': 1  # Lowest priority
    }
]

# Target location (what we want)
TARGET_PATH = os.path.join(DMAI_SYSTEM_PATH, "api-harvester")

def scan_all_sources():
    """Scan all sources and return which exist"""
    print("\n🔍 SCANNING ALL SOURCE LOCATIONS")
    print("=" * 50)
    
    valid_sources = []
    for source in SOURCE_PATHS:
        exists = os.path.exists(source['path'])
        status = "✅" if exists else "❌"
        print(f"{status} {source['name']}: {source['path']}")
        
        if exists:
            # Quick stats
            try:
                file_count = sum(len(files) for _, _, files in os.walk(source['path']))
                print(f"   └─ {file_count} files")
                valid_sources.append({**source, 'file_count': file_count})
            except:
                print(f"   └─ (cannot count files)")
                valid_sources.append({**source, 'file_count': 0})
    
    return valid_sources

def create_backup():
    """Create timestamped backup of current api_harvester"""
    current = SOURCE_PATHS[0]['path']  # Current location
    if os.path.exists(current):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{current}_backup_{timestamp}"
        print(f"\n📦 Creating backup of current api_harvester...")
        shutil.copytree(current, backup_path)
        print(f"   ✅ Backup created at: {backup_path}")
        
        # Also create a backup report
        with open(f"{backup_path}/BACKUP_INFO.txt", 'w') as f:
            f.write(f"Backup created: {datetime.now()}\n")
            f.write(f"Original location: {current}\n")
            f.write("This is a safe backup before merging with other sources.\n")
        
        return backup_path
    return None

def compare_two_dirs(dir1, dir2, name1, name2):
    """Compare two directories and return differences"""
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        return None
    
    print(f"\n📊 Comparing: {name1} vs {name2}")
    print(f"   {dir1}")
    print(f"   {dir2}")
    
    comparison = filecmp.dircmp(dir1, dir2)
    
    # Get detailed stats
    only_in_1 = []
    only_in_2 = []
    
    # Recursively check for nested differences
    for item in comparison.left_only:
        item_path = os.path.join(dir1, item)
        if os.path.isfile(item_path):
            only_in_1.append(item)
        else:
            only_in_1.append(f"{item}/ (directory)")
    
    for item in comparison.right_only:
        item_path = os.path.join(dir2, item)
        if os.path.isfile(item_path):
            only_in_2.append(item)
        else:
            only_in_2.append(f"{item}/ (directory)")
    
    results = {
        'only_in_first': only_in_1,
        'only_in_second': only_in_2,
        'differ': comparison.diff_files,
        'common': comparison.common_files
    }
    
    print(f"\n   Results:")
    print(f"   • Only in {name1}: {len(only_in_1)} items")
    if only_in_1:
        print(f"     Samples: {', '.join(only_in_1[:5])}")
    
    print(f"   • Only in {name2}: {len(only_in_2)} items")
    if only_in_2:
        print(f"     Samples: {', '.join(only_in_2[:5])}")
    
    print(f"   • Files that differ: {len(comparison.diff_files)}")
    if comparison.diff_files:
        print(f"     Samples: {', '.join(comparison.diff_files[:5])}")
    
    return results

def merge_from_source(source_dir, source_name, target_dir):
    """Merge files from a source directory into target"""
    if not os.path.exists(source_dir) or not os.path.exists(target_dir):
        return [], []
    
    print(f"\n🔄 Merging from {source_name}...")
    
    comparison = compare_two_dirs(target_dir, source_dir, "Target", source_name)
    if not comparison:
        return [], []
    
    merged_files = []
    conflicts = []
    
    # Copy files that are only in source
    for item in comparison['only_in_second']:
        # Clean up the item name (remove directory marker if present)
        clean_item = item.replace("/ (directory)", "")
        src = os.path.join(source_dir, clean_item)
        dst = os.path.join(target_dir, clean_item)
        
        if os.path.isdir(src):
            if not os.path.exists(dst):
                print(f"   📁 Copying directory: {clean_item}/")
                shutil.copytree(src, dst)
                merged_files.append(f"{clean_item}/ (directory)")
            else:
                # Directory exists, we need to merge recursively
                print(f"   📁 Merging directory: {clean_item}/")
                # Recursively merge this directory
                sub_merged, sub_conflicts = merge_from_source(
                    src, f"{source_name}/{clean_item}", dst
                )
                merged_files.extend([f"{clean_item}/{f}" for f in sub_merged])
                conflicts.extend([f"{clean_item}/{c}" for c in sub_conflicts])
        else:
            print(f"   📄 Copying file: {clean_item}")
            shutil.copy2(src, dst)
            merged_files.append(clean_item)
    
    # Handle files that differ
    for file in comparison['differ']:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, file)
        
        # Create a merged version with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{dst}.from_{source_name.replace(' ', '_')}_{timestamp}"
        
        print(f"\n   ⚠️  Conflict: {file}")
        print(f"      • Target size: {os.path.getsize(dst)} bytes")
        print(f"      • {source_name} size: {os.path.getsize(src)} bytes")
        
        # Backup the current file
        shutil.copy2(dst, backup_file)
        print(f"      • Backed up target to: {backup_file}")
        
        # For now, keep the newer file (based on modification time)
        src_mtime = os.path.getmtime(src)
        dst_mtime = os.path.getmtime(dst)
        
        if src_mtime > dst_mtime:
            print(f"      • {source_name} version is newer, copying...")
            shutil.copy2(src, dst)
            conflicts.append({'file': file, 'source': source_name, 'action': 'replaced_with_newer', 'backup': backup_file})
        else:
            print(f"      • Target version is newer, keeping current")
            conflicts.append({'file': file, 'source': source_name, 'action': 'kept_current', 'backup': backup_file})
    
    return merged_files, conflicts

def rename_to_hyphen():
    """Rename api_harvester to api-harvester"""
    current = SOURCE_PATHS[0]['path']  # Current underscore version
    
    if os.path.exists(current) and not os.path.exists(TARGET_PATH):
        print(f"\n📝 Renaming api_harvester to api-harvester...")
        os.rename(current, TARGET_PATH)
        print(f"   ✅ Renamed: {current} -> {TARGET_PATH}")
        return True
    elif os.path.exists(TARGET_PATH):
        # Check if it's the same as current (symlink or duplicate)
        if os.path.islink(TARGET_PATH):
            print(f"\n⚠️  Target path is a symlink, checking...")
            if os.readlink(TARGET_PATH) == current:
                print(f"   Symlink already points to current location")
                return True
        else:
            print(f"\n⚠️  Target path already exists: {TARGET_PATH}")
            # Ask user what to do
            response = input("   Overwrite? (yes/no): ")
            if response.lower() == 'yes':
                shutil.rmtree(TARGET_PATH)
                os.rename(current, TARGET_PATH)
                print(f"   ✅ Renamed after removing existing")
                return True
    return False

def update_symlinks():
    """Update or create symlinks for compatibility"""
    # Create symlink from underscore to hyphen for compatibility
    underscore_link = os.path.join(DMAI_SYSTEM_PATH, "api_harvester")
    if os.path.islink(underscore_link):
        os.unlink(underscore_link)
    
    if not os.path.exists(underscore_link):
        os.symlink(TARGET_PATH, underscore_link)
        print(f"✅ Created symlink: {underscore_link} -> {TARGET_PATH}")
    
    # Check for any other expected locations
    other_expected = [
        "/Users/davidmiles/Desktop/api-harvester",
        "/Users/davidmiles/Desktop/api_harvester"
    ]
    
    for expected in other_expected:
        if not os.path.exists(expected):
            print(f"⚠️  Expected location missing: {expected}")
            response = input(f"   Create symlink here? (yes/no): ")
            if response.lower() == 'yes':
                os.symlink(TARGET_PATH, expected)
                print(f"   ✅ Created: {expected}")

def create_migration_report(backup_path, all_merges, all_conflicts):
    """Create a detailed report of what was done"""
    report_path = os.path.join(DMAI_SYSTEM_PATH, f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("API-HARVESTER COMPLETE MIGRATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SOURCES SCANNED:\n")
        for source in SOURCE_PATHS:
            exists = os.path.exists(source['path'])
            status = "FOUND" if exists else "NOT FOUND"
            f.write(f"  • {source['name']}: {source['path']} - {status}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("MIGRATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"BACKUP CREATED:\n")
        f.write(f"  {backup_path}\n\n")
        
        f.write("FILES MERGED BY SOURCE:\n")
        for source_name, files in all_merges.items():
            if files:
                f.write(f"\n  📦 From {source_name}:\n")
                for file in files[:20]:  # Limit to first 20
                    f.write(f"    ✅ {file}\n")
                if len(files) > 20:
                    f.write(f"    ... and {len(files) - 20} more\n")
        
        f.write("\nCONFLICT RESOLUTIONS:\n")
        for conflict in all_conflicts:
            f.write(f"\n  ⚠️  {conflict['file']}\n")
            f.write(f"     Source: {conflict['source']}\n")
            f.write(f"     Action: {conflict['action']}\n")
            f.write(f"     Backup: {conflict['backup']}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("FINAL LOCATION:\n")
        f.write(f"  {TARGET_PATH}\n")
        
        if os.path.exists(TARGET_PATH):
            total_files = sum(len(files) for _, _, files in os.walk(TARGET_PATH))
            f.write(f"  Total files: {total_files}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("NEXT STEPS:\n")
        f.write("1. Review this report carefully\n")
        f.write("2. Test that all services work with new location\n")
        f.write("3. After verification, archive old folders:\n")
        f.write(f"   • {OLD_EVOLUTION_PATH}\n")
        f.write(f"   • {OLD_BACKUP_PATH}\n")
        f.write("4. Run the path fix script to update launchd\n")
    
    print(f"\n📄 Migration report saved to: {report_path}")
    return report_path

def main():
    print("\n" + "="*60)
    print("🚀 API-HARVESTER COMPLETE MIGRATION UTILITY")
    print("="*60)
    print("Merging from: AI-Evolution-System and AI-Evolution-System-old-backup")
    print("Target: dmai-system/api-harvester")
    print("="*60)
    
    # Step 1: Scan all sources
    print("\n🔍 Step 1: Scanning all sources...")
    valid_sources = scan_all_sources()
    
    if len(valid_sources) < 2:
        print("\n⚠️  Only one source found. Nothing to merge.")
        return
    
    # Step 2: Create backup
    print("\n💾 Step 2: Creating backup...")
    backup_path = create_backup()
    
    # Step 3: Merge from each source (in priority order, excluding current)
    all_merges = {}
    all_conflicts = []
    
    # Sort sources by priority (highest first) but skip the first one (current)
    sources_to_merge = sorted(valid_sources[1:], key=lambda x: x['priority'], reverse=True)
    
    print("\n🔄 Step 3: Merging from other sources...")
    for source in sources_to_merge:
        if os.path.exists(source['path']):
            print(f"\n--- Merging from {source['name']} ---")
            merged, conflicts = merge_from_source(
                source['path'], 
                source['name'], 
                SOURCE_PATHS[0]['path']  # Current location
            )
            all_merges[source['name']] = merged
            all_conflicts.extend(conflicts)
    
    # Step 4: Rename to hyphen version
    print("\n📝 Step 4: Standardizing folder name...")
    if rename_to_hyphen():
        print("   ✅ Successfully renamed to api-harvester")
    else:
        print("   ⚠️  Using existing api-harvester folder")
    
    # Step 5: Update symlinks
    print("\n🔗 Step 5: Creating compatibility symlinks...")
    update_symlinks()
    
    # Step 6: Create report
    print("\n📋 Step 6: Generating report...")
    if backup_path:
        report_path = create_migration_report(backup_path, all_merges, all_conflicts)
    
    # Step 7: Final verification
    print("\n✅ Step 7: Final verification")
    print("-" * 40)
    
    if os.path.exists(TARGET_PATH):
        total_files = sum(len(files) for _, _, files in os.walk(TARGET_PATH))
        print(f"   ✅ Target location: {TARGET_PATH}")
        print(f"   📦 Total files: {total_files}")
        
        # Show what old locations still exist
        old_locations = []
        for source in valid_sources[1:]:  # Skip current
            if os.path.exists(source['path']):
                old_locations.append(f"   • {source['name']}: {source['path']}")
        
        if old_locations:
            print(f"\n⚠️  Old locations still exist:")
            for loc in old_locations:
                print(loc)
            print("\n   After verifying everything works, you can archive these:")
            print("   mv ~/Desktop/AI-Evolution-System ~/Desktop/AI-Evolution-System-archived")
            print("   mv ~/Desktop/AI-Evolution-System-old-backup ~/Desktop/AI-Evolution-System-old-backup-archived")
    
    print("\n" + "="*60)
    print("✅ MIGRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    # Ask for confirmation
    print("\n⚠️  WARNING: This will merge files from ALL old locations into dmai-system")
    print("   Sources to check:")
    print(f"   • {OLD_EVOLUTION_PATH}/api-harvester")
    print(f"   • {OLD_BACKUP_PATH}/api-harvester")
    
    response = input("\nDo you want to continue? (yes/no): ")
    
    if response.lower() == 'yes':
        main()
    else:
        print("Migration cancelled.")
