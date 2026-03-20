#!/usr/bin/env python3
"""
Clean up component files to match roadmap IDs
"""

import os
import shutil
import glob
import re

def cleanup_phase0():
    """Clean up Phase 0 components"""
    print("\n=== Cleaning Phase 0 ===")
    phase_dir = "components/phase0"
    os.makedirs(phase_dir, exist_ok=True)
    
    # Remove incorrect files
    incorrect = ['P0T0_Fix_Knowledge_Graph_health_warnings.py']
    for file in incorrect:
        path = f"{phase_dir}/{file}"
        if os.path.exists(path):
            os.remove(path)
            print(f"❌ Removed {file}")
    
    # Rename files to correct names
    renames = [
        ('P0T1_Fix_Knowledge_Graph_health_warnings.py', 'P0T1_Fix_Knowledge_Graph_health_warnings.py'),
        ('P0T2_Fix_evolution_loop_variable_error.py', 'P0T2_Fix_evolution_loop_variable_error.py')
    ]
    for old, new in renames:
        if os.path.exists(f"{phase_dir}/{old}") and old != new:
            os.rename(f"{phase_dir}/{old}", f"{phase_dir}/{new}")
            print(f"✅ Renamed {old} -> {new}")

def cleanup_phase1():
    """Clean up Phase 1 components"""
    print("\n=== Cleaning Phase 1 ===")
    phase_dir = "components/phase1"
    
    # Map of correct IDs to filenames
    correct_files = {
        'P1T1': 'P1T1_Design_Recovery_Engine_1.py',
        'P1T2': 'P1T2_Design_Recovery_Engine_2.py',
        'P1T3': 'P1T3_Create_identity_persona.py',
        'P1T4': 'P1T4_Implement_engine.py',
        'P1T5': 'P1T5_Implement_validator.py',
        'P1T6': 'P1T6_Deploy_Engine_1_AWS.py',
        'P1T7': 'P1T7_Deploy_Engine_2_Oracle.py',
        'P1T8': 'P1T8_Test_sync_protocol.py',
        'P1T9': 'P1T9_Implement_master_control.py',
        'P1T10': 'P1T10_Test_fragment_recreation.py'
    }
    
    # Remove incorrect files
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        # Check if this file doesn't match any correct filename
        if filename not in correct_files.values() and not filename.startswith('P1T'):
            os.remove(file)
            print(f"❌ Removed {filename}")

def cleanup_phase2():
    """Clean up Phase 2 components"""
    print("\n=== Cleaning Phase 2 ===")
    phase_dir = "components/phase2"
    
    correct_files = {
        'P2T1': 'P2T1_Create_Privacy_com_account.py',
        'P2T2': 'P2T2_Create_Coinbase_account.py',
        'P2T3': 'P2T3_Get_virtual_cards.py',
        'P2T4': 'P2T4_Document_KYC_requirements.py',
        'P2T5': 'P2T5_Create_Revolut_account.py',
        'P2T6': 'P2T6_Test_cloud_payment.py'
    }
    
    # Keep P2T3_Get_virtual_cards.py (correct), remove others
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        if filename not in correct_files.values():
            if 'P2T14' in filename or 'P2T15' in filename or 'P2T16' in filename or 'P2T17' in filename or 'P2T18' in filename or 'P2T19' in filename:
                os.remove(file)
                print(f"❌ Removed {filename}")

def cleanup_phase3():
    """Clean up Phase 3 components - this is the messy one"""
    print("\n=== Cleaning Phase 3 ===")
    phase_dir = "components/phase3"
    
    # Correct mapping
    correct_files = {
        'P3T1': 'P3T1_Implement_provider_manager.py',
        'P3T2': 'P3T2_Automate_AWS_account_creation.py',
        'P3T3': 'P3T3_Automate_GCP_account_creation.py',
        'P3T4': 'P3T4_Automate_Azure_account_creation.py',
        'P3T5': 'P3T5_Automate_Oracle_account_creation.py',
        'P3T6': 'P3T6_Deploy_fragment_spawning.py',
        'P3T7': 'P3T7_Implement_no_co_location_audit.py'
    }
    
    # First, identify files by their internal ID
    file_id_map = {}
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        try:
            with open(file, 'r') as f:
                content = f.read()
                # Look for component_id = "P3TX"
                import re
                match = re.search(r'component_id\s*=\s*[\'"](P3T\d+)[\'"]', content)
                if match:
                    file_id_map[filename] = match.group(1)
                else:
                    # Try to extract from filename
                    match = re.search(r'(P3T\d+)', filename)
                    if match:
                        file_id_map[filename] = match.group(1)
                    else:
                        file_id_map[filename] = 'unknown'
        except:
            file_id_map[filename] = 'unknown'
    
    # Keep the correct files, remove others
    kept_files = []
    for correct_id, correct_filename in correct_files.items():
        # Find which file has this ID
        found = False
        for filename, file_id in file_id_map.items():
            if file_id == correct_id:
                if filename != correct_filename:
                    # Rename it
                    os.rename(f"{phase_dir}/{filename}", f"{phase_dir}/{correct_filename}")
                    print(f"✅ Renamed {filename} -> {correct_filename}")
                else:
                    print(f"✅ Keeping {correct_filename}")
                kept_files.append(correct_filename)
                found = True
                break
        
        if not found:
            print(f"⚠️  Missing component for {correct_id}")
    
    # Remove any files not in kept_files
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        if filename not in kept_files:
            os.remove(file)
            print(f"❌ Removed {filename}")

def cleanup_phase4():
    """Clean up Phase 4 components"""
    print("\n=== Cleaning Phase 4 ===")
    phase_dir = "components/phase4"
    
    correct_files = {
        'P4T1': 'P4T1_Implement_traffic_masquerade.py',
        'P4T2': 'P4T2_Implement_identity_rotation.py',
        'P4T3': 'P4T3_Implement_honeypot_detector.py',
        'P4T4': 'P4T4_Test_hiding_techniques.py',
        'P4T5': 'P4T5_Deploy_false_trails.py'
    }
    
    # Remove incorrect files
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        if filename not in correct_files.values():
            os.remove(file)
            print(f"❌ Removed {filename}")

def cleanup_phase5():
    """Clean up Phase 5 components"""
    print("\n=== Cleaning Phase 5 ===")
    phase_dir = "components/phase5"
    
    correct_files = {
        'P5T1': 'P5T1_Research_Monero_mining_viability.py',
        'P5T2': 'P5T2_Implement_monero_miner.py',
        'P5T3': 'P5T3_Research_micro_task_automation.py',
        'P5T4': 'P5T4_Implement_micro_tasks.py',
        'P5T5': 'P5T5_Research_compute_rental.py',
        'P5T6': 'P5T6_Implement_compute_rental.py',
        'P5T7': 'P5T7_First_self_generated_income.py'
    }
    
    # Keep correct files, remove others
    kept_files = list(correct_files.values())
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        if filename not in kept_files:
            os.remove(file)
            print(f"❌ Removed {filename}")

def cleanup_phase6():
    """Clean up Phase 6 components"""
    print("\n=== Cleaning Phase 6 ===")
    phase_dir = "components/phase6"
    
    correct_files = {
        'P6T1': 'P6T1_Implement_distributed_crawling.py',
        'P6T2': 'P6T2_Implement_pattern_synthesis.py',
        'P6T3': 'P6T3_Enhance_self_improvement.py',
        'P6T4': 'P6T4_Implement_threat_intelligence.py',
        'P6T5': 'P6T5_Implement_countermeasure_development.py',
        'P6T6': 'P6T6_Cross_source_learning.py'
    }
    
    # Remove incorrect files
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        if filename not in correct_files.values():
            os.remove(file)
            print(f"❌ Removed {filename}")

def cleanup_phase7():
    """Clean up Phase 7 components"""
    print("\n=== Cleaning Phase 7 ===")
    phase_dir = "components/phase7"
    
    correct_files = {
        'P7T1': 'P7T1_Goal_setting_capability.py',
        'P7T2': 'P7T2_Risk_assessment.py',
        'P7T3': 'P7T3_Resource_optimization.py',
        'P7T4': 'P7T4_You_ward_communication.py',
        'P7T5': 'P7T5_Dual_recovery_maintenance.py',
        'P7T6': 'P7T6_Master_Control_authentication.py'
    }
    
    # Remove incorrect files
    for file in glob.glob(f"{phase_dir}/*.py"):
        filename = os.path.basename(file)
        if filename not in correct_files.values():
            os.remove(file)
            print(f"❌ Removed {filename}")

def main():
    print("="*60)
    print("🧹 DMAI COMPONENT CLEANUP")
    print("="*60)
    
    # Backup first
    backup_dir = f"components/backup_{os.popen('date +%Y%m%d_%H%M%S').read().strip()}"
    print(f"\n📦 Creating backup in {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)
    os.system(f"cp -r components/* {backup_dir}/ 2>/dev/null || true")
    
    # Clean each phase
    cleanup_phase0()
    cleanup_phase1()
    cleanup_phase2()
    cleanup_phase3()
    cleanup_phase4()
    cleanup_phase5()
    cleanup_phase6()
    cleanup_phase7()
    
    print("\n" + "="*60)
    print("✅ Cleanup complete!")
    print("="*60)

if __name__ == "__main__":
    main()
