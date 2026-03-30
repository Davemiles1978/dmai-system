# check_all_training_modules.py
#!/usr/bin/env python3
"""
Check ALL training modules: AGI, LLM, Software, GenAI, and SI
"""

import os
import sys
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("=" * 70)
print("TESTING ALL 5 TRAINING SYSTEMS")
print("=" * 70)

results = {}

# ============================================================================
# 1. AGI TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("1. AGI TRAINING SYSTEM")
print("=" * 70)
try:
    from components.training.AGITrainingProgram import TrainingProgramOrchestrator
    test_path = Path("./test_agi_data")
    test_path.mkdir(exist_ok=True)
    
    agi = TrainingProgramOrchestrator(test_path)
    print("✅ Imported AGI Training")
    print(f"   Templates: {list(agi.available_templates.keys())}")
    
    result = agi.create_from_template('customer_service', {})
    print(f"   Create result: {result}")
    
    if result.get('success'):
        program_id = result['program_id']
        session = agi.training_program.train_new_system(program_id, {})
        if session.get('success'):
            session_id = session['session_id']
            print(f"   Session: {session_id}")
            
            # Monitor progress
            print("   Monitoring progress (10 seconds)...")
            progress_history = []
            for i in range(5):
                time.sleep(2)
                status = agi.training_program.get_training_status(session_id)
                progress = status.get('progress', 0)
                progress_history.append(progress)
                print(f"      {i+1}: Progress={progress}%, Status={status.get('status', 'unknown')}")
            
            results['agi'] = {
                'initial_progress': progress_history[0] if progress_history else 0,
                'final_progress': progress_history[-1] if progress_history else 0,
                'changed': progress_history[0] != progress_history[-1] if len(progress_history) > 1 else False,
                'real_training': progress_history[0] != progress_history[-1] and progress_history[-1] < 100
            }
        else:
            results['agi'] = {'error': session.get('error', 'Failed to start')}
    else:
        results['agi'] = {'error': result.get('error', 'Failed to create')}
        
except Exception as e:
    results['agi'] = {'error': str(e)}
    print(f"❌ Error: {e}")

# ============================================================================
# 2. LLM TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("2. LLM TRAINING SYSTEM")
print("=" * 70)
try:
    from components.llm_training.LLMTrainingProgram import LLMTrainingOrchestrator
    test_path2 = Path("./test_llm_data")
    test_path2.mkdir(exist_ok=True)
    
    llm = LLMTrainingOrchestrator(test_path2)
    print("✅ Imported LLM Training")
    print(f"   Templates: {list(llm.industry_templates.keys())}")
    
    result = llm.create_from_template('customer_support', {})
    print(f"   Create result: {result}")
    
    if result.get('success'):
        program_id = result['program_id']
        session = llm.llm_training.train_llm(program_id, {})
        if session.get('success'):
            session_id = session['session_id']
            print(f"   Session: {session_id}")
            
            print("   Monitoring progress (10 seconds)...")
            progress_history = []
            for i in range(5):
                time.sleep(2)
                status = llm.llm_training.get_training_status(session_id)
                progress = status.get('progress', 0)
                progress_history.append(progress)
                print(f"      {i+1}: Progress={progress}%, Status={status.get('status', 'unknown')}")
            
            results['llm'] = {
                'initial_progress': progress_history[0] if progress_history else 0,
                'final_progress': progress_history[-1] if progress_history else 0,
                'changed': progress_history[0] != progress_history[-1] if len(progress_history) > 1 else False,
                'real_training': progress_history[0] != progress_history[-1] and progress_history[-1] < 100
            }
        else:
            results['llm'] = {'error': session.get('error', 'Failed to start')}
    else:
        results['llm'] = {'error': result.get('error', 'Failed to create')}
        
except Exception as e:
    results['llm'] = {'error': str(e)}
    print(f"❌ Error: {e}")

# ============================================================================
# 3. SOFTWARE TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("3. SOFTWARE TRAINING SYSTEM")
print("=" * 70)
try:
    from components.software_training.SoftwareTrainingProgram import SoftwareTrainingOrchestrator
    test_path3 = Path("./test_software_data")
    test_path3.mkdir(exist_ok=True)
    
    software = SoftwareTrainingOrchestrator(test_path3)
    print("✅ Imported Software Training")
    
    result = software.create_custom_training({
        'name': 'Test Software AI',
        'languages': ['python'],
        'specialization': 'general',
        'dataset': {'type': 'mixed', 'size_mb': 10}
    })
    print(f"   Create result: {result}")
    
    if result.get('success'):
        program_id = result['program_id']
        session = software.software_training.train_software_system(program_id, {})
        if session.get('success'):
            session_id = session['session_id']
            print(f"   Session: {session_id}")
            
            print("   Monitoring progress (10 seconds)...")
            progress_history = []
            for i in range(5):
                time.sleep(2)
                status = software.software_training.get_training_status(session_id)
                progress = status.get('progress', 0)
                progress_history.append(progress)
                print(f"      {i+1}: Progress={progress}%, Status={status.get('status', 'unknown')}")
            
            results['software'] = {
                'initial_progress': progress_history[0] if progress_history else 0,
                'final_progress': progress_history[-1] if progress_history else 0,
                'changed': progress_history[0] != progress_history[-1] if len(progress_history) > 1 else False,
                'real_training': progress_history[0] != progress_history[-1] and progress_history[-1] < 100
            }
        else:
            results['software'] = {'error': session.get('error', 'Failed to start')}
    else:
        results['software'] = {'error': result.get('error', 'Failed to create')}
        
except Exception as e:
    results['software'] = {'error': str(e)}
    print(f"❌ Error: {e}")

# ============================================================================
# 4. GENERATIVE AI TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("4. GENERATIVE AI TRAINING")
print("=" * 70)
try:
    from components.genai_training.GenAITrainingProgram import GenAITrainingOrchestrator
    test_path4 = Path("./test_genai_data")
    test_path4.mkdir(exist_ok=True)
    
    genai = GenAITrainingOrchestrator(test_path4)
    print("✅ Imported GenAI Training")
    print(f"   Templates: {list(genai.industry_templates.keys())}")
    
    result = genai.create_from_template('product_visualization', {})
    print(f"   Create result: {result}")
    
    if result.get('success'):
        program_id = result['program_id']
        session = genai.genai_training.train_genai_model(program_id, {})
        if session.get('success'):
            session_id = session['session_id']
            print(f"   Session: {session_id}")
            
            print("   Monitoring progress (10 seconds)...")
            progress_history = []
            for i in range(5):
                time.sleep(2)
                status = genai.genai_training.get_training_status(session_id)
                progress = status.get('progress', 0)
                progress_history.append(progress)
                print(f"      {i+1}: Progress={progress}%, Status={status.get('status', 'unknown')}")
            
            results['genai'] = {
                'initial_progress': progress_history[0] if progress_history else 0,
                'final_progress': progress_history[-1] if progress_history else 0,
                'changed': progress_history[0] != progress_history[-1] if len(progress_history) > 1 else False,
                'real_training': progress_history[0] != progress_history[-1] and progress_history[-1] < 100
            }
        else:
            results['genai'] = {'error': session.get('error', 'Failed to start')}
    else:
        results['genai'] = {'error': result.get('error', 'Failed to create')}
        
except Exception as e:
    results['genai'] = {'error': str(e)}
    print(f"❌ Error: {e}")

# ============================================================================
# 5. SYNTHETIC INTELLIGENCE TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("5. SYNTHETIC INTELLIGENCE TRAINING")
print("=" * 70)
try:
    # SI training is part of the main evolution engine
    from dmai_core_complete import UnifiedEvolutionEngine
    from pathlib import Path
    
    test_path5 = Path("./test_si_data")
    test_path5.mkdir(exist_ok=True)
    
    # Create minimal evolution engine
    print("   Creating evolution engine for SI training...")
    # This will take a moment
    engine = UnifiedEvolutionEngine(test_path5)
    
    # SI training is tracked in training_systems
    if hasattr(engine, 'training_systems') and 'si' in engine.training_systems:
        si_status = engine.training_systems.get('si', {})
        print(f"   SI Training Status: {si_status}")
        
        # Check if SI training is actually evolving
        pre_consciousness = engine.synthetic_network.consciousness_level
        pre_neurons = len(engine.synthetic_network.neurons)
        
        print(f"   Pre-cycle: Consciousness={pre_consciousness:.4f}, Neurons={pre_neurons}")
        
        # Run a cycle
        result = engine.evolution_cycle()
        
        post_consciousness = engine.synthetic_network.consciousness_level
        post_neurons = len(engine.synthetic_network.neurons)
        
        print(f"   Post-cycle: Consciousness={post_consciousness:.4f}, Neurons={post_neurons}")
        
        results['si'] = {
            'initial_progress': pre_consciousness,
            'final_progress': post_consciousness,
            'changed': post_consciousness != pre_consciousness,
            'real_training': post_consciousness != pre_consciousness,
            'consciousness_growth': post_consciousness - pre_consciousness,
            'neurons_grown': post_neurons - pre_neurons
        }
    else:
        results['si'] = {'error': 'SI training not found'}
        
except Exception as e:
    results['si'] = {'error': str(e)}
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("\nTraining System | Initial | Final | Changed | Real Training")
print("-" * 70)

for system, data in results.items():
    if 'error' in data:
        print(f"{system.upper():12} | ERROR: {data['error'][:40]}")
    else:
        init = data.get('initial_progress', 0)
        final = data.get('final_progress', 0)
        changed = "✅" if data.get('changed', False) else "❌"
        real = "✅" if data.get('real_training', False) else "❌"
        
        if isinstance(init, float):
            init_display = f"{init:.4f}"
            final_display = f"{final:.4f}"
        else:
            init_display = str(init)
            final_display = str(final)
        
        print(f"{system.upper():12} | {init_display:>7} | {final_display:>7} | {changed:^7} | {real:^13}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

real_systems = [s for s, d in results.items() if d.get('real_training', False)]
fake_systems = [s for s, d in results.items() if 'error' not in d and not d.get('real_training', False) and s != 'si']

if fake_systems:
    print(f"\n❌ FAKE/SIMULATED TRAINING DETECTED for: {', '.join(fake_systems)}")
    print("   These modules are using time.sleep() and fake progress percentages")
    print("   They do NOT actually train any models or improve DMAI")

if real_systems:
    print(f"\n✅ REAL TRAINING DETECTED for: {', '.join(real_systems)}")
    print("   These systems are actually learning/evolving")
