# test_training_simple.py
#!/usr/bin/env python3
"""
Simple Training Test - Tests if training systems actually progress
Run this from the dmai-system directory
"""

import os
import sys
import time
from pathlib import Path

# Change to the dmai-system directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add current directory to path
sys.path.insert(0, os.getcwd())

print(f"Working directory: {os.getcwd()}")

# Create test data directory
test_data_dir = Path("./test_training_data")
test_data_dir.mkdir(exist_ok=True)

try:
    # Import training module
    from components.training.AGITrainingProgram import TrainingProgramOrchestrator
    print("✅ Successfully imported AGI Training module")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class SimpleTrainingTest:
    """Simple test to see if training actually progresses"""
    
    def __init__(self):
        self.orchestrator = TrainingProgramOrchestrator(test_data_dir)
        self.training_sessions = {}
    
    def test_agi_training(self):
        """Test AGI training progress"""
        print("\n" + "=" * 60)
        print("TESTING AGI TRAINING")
        print("=" * 60)
        
        # Create a training program
        print("\n📝 Creating AGI training program...")
        result = self.orchestrator.create_from_template('customer_service', {})
        
        if not result.get('success'):
            print(f"   ❌ Failed to create program: {result.get('error', 'Unknown')}")
            return False
        
        program_id = result['program_id']
        print(f"   ✅ Program created: {program_id}")
        
        # Start training
        print("   🎓 Starting training...")
        session = self.orchestrator.training_program.train_new_system(program_id, {})
        
        if not session.get('success'):
            print(f"   ❌ Failed to start: {session.get('error', 'Unknown')}")
            return False
        
        session_id = session['session_id']
        print(f"   ✅ Session started: {session_id}")
        self.training_sessions['agi'] = session_id
        
        # Monitor progress
        print("\n📊 Monitoring progress (20 seconds)...")
        previous_progress = 0
        for i in range(10):
            time.sleep(2)
            status = self.orchestrator.training_program.get_training_status(session_id)
            
            if status.get('success'):
                progress = status.get('progress', 0)
                print(f"   Cycle {i+1}: Progress = {progress:.1f}% | Status = {status.get('status', 'unknown')}")
                
                if progress > previous_progress:
                    previous_progress = progress
            else:
                print(f"   Cycle {i+1}: Error - {status.get('error', 'Unknown')}")
        
        # Check if progress was made
        if previous_progress > 0:
            print(f"\n✅ AGI TRAINING IS WORKING! Progress: {previous_progress:.1f}%")
            return True
        else:
            print(f"\n⚠️ AGI TRAINING IS NOT PROGRESSING! Still at 0%")
            return False
    
    def test_llm_training(self):
        """Test LLM training progress"""
        print("\n" + "=" * 60)
        print("TESTING LLM TRAINING")
        print("=" * 60)
        
        try:
            from components.llm_training.LLMTrainingProgram import LLMTrainingOrchestrator
            
            llm_orchestrator = LLMTrainingOrchestrator(test_data_dir)
            
            print("\n📝 Creating LLM training program...")
            result = llm_orchestrator.create_from_template('customer_support', {})
            
            if not result.get('success'):
                print(f"   ❌ Failed to create program: {result.get('error', 'Unknown')}")
                return False
            
            program_id = result['program_id']
            print(f"   ✅ Program created: {program_id}")
            
            print("   🎓 Starting training...")
            session = llm_orchestrator.llm_training.train_llm(program_id, {})
            
            if not session.get('success'):
                print(f"   ❌ Failed to start: {session.get('error', 'Unknown')}")
                return False
            
            session_id = session['session_id']
            print(f"   ✅ Session started: {session_id}")
            
            print("\n📊 Monitoring progress (10 seconds)...")
            previous_progress = 0
            for i in range(5):
                time.sleep(2)
                status = llm_orchestrator.llm_training.get_training_status(session_id)
                
                if status.get('success'):
                    progress = status.get('progress', 0)
                    print(f"   Cycle {i+1}: Progress = {progress:.1f}% | Status = {status.get('status', 'unknown')}")
                    
                    if progress > previous_progress:
                        previous_progress = progress
                else:
                    print(f"   Cycle {i+1}: Error - {status.get('error', 'Unknown')}")
            
            if previous_progress > 0:
                print(f"\n✅ LLM TRAINING IS WORKING! Progress: {previous_progress:.1f}%")
                return True
            else:
                print(f"\n⚠️ LLM TRAINING IS NOT PROGRESSING! Still at 0%")
                return False
                
        except ImportError as e:
            print(f"❌ LLM Training module not found: {e}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


if __name__ == '__main__':
    print("=" * 60)
    print("SIMPLE TRAINING TEST")
    print("=" * 60)
    print("\nThis test will create and start training systems to see if they progress.")
    print("It will monitor for 20 seconds to detect progress.\n")
    
    test = SimpleTrainingTest()
    
    # Test AGI training first
    agi_result = test.test_agi_training()
    
    # Test LLM training second
    llm_result = test.test_llm_training()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"AGI Training: {'✅ WORKING' if agi_result else '❌ NOT PROGRESSING'}")
    print(f"LLM Training: {'✅ WORKING' if llm_result else '❌ NOT PROGRESSING'}")
    
    if not agi_result:
        print("\n🔧 AGI TRAINING FIX NEEDED")
        print("   The AGI training system is being created but not making progress.")
        print("   This explains why training appears to 'complete' quickly - it's not actually training.")
