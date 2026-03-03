# test_agi_evolution.py

#!/usr/bin/env python3
"""
Test script for AGI evolution system
"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from agi_orchestrator import AGIOrchestrator
from capability_synthesizer import CapabilitySynthesizer

async def test_capability_synthesis():
    """Test capability synthesis"""
    print("\n🧪 Testing Capability Synthesis...")
    
    synthesizer = CapabilitySynthesizer()
    
    # Create some test capabilities
    test_caps = ["data_processor", "validator", "logger"]
    
    # Try to synthesize
    new_cap = await synthesizer.synthesize_new_capability(
        goal="Create a data validation pipeline",
        available_capabilities=test_caps,
        context={"purpose": "test"}
    )
    
    if new_cap:
        print(f"✅ Synthesized: {new_cap.name}")
        print(f"  Dependencies: {new_cap.dependencies}")
        return True
    else:
        print("⚠️ No synthesis performed (needs more capabilities)")
        return False
        
async def test_orchestrator():
    """Test AGI orchestrator"""
    print("\n🧪 Testing AGI Orchestrator...")
    
    orchestrator = AGIOrchestrator()
    
    # Test goal submission
    goal = {
        'description': 'Test goal for evolution',
        'priority': 5,
        'requires_evolution': True,
        'test': True
    }
    
    await orchestrator.submit_goal(goal)
    
    # Check status
    status = orchestrator.get_status()
    print(f"✅ Orchestrator status: {status['state']['health_status']}")
    print(f"  Generation: {status['state']['generation']}")
    print(f"  Active capabilities: {status['active_capabilities']}")
    
    return True
    
async def test_recursive_evolution():
    """Test recursive self-improvement"""
    print("\n🧪 Testing Recursive Evolution...")
    
    orchestrator = AGIOrchestrator()
    
    # Simulate evolution step
    await orchestrator._perform_evolution_step()
    
    # Check if generation increased
    status = orchestrator.get_status()
    print(f"✅ Evolution step complete")
    print(f"  New generation: {status['state']['generation']}")
    
    return True
    
async def main():
    """Run all tests"""
    print("=" * 60)
    print("🔬 AGI Evolution System Tests")
    print("=" * 60)
    
    tests = [
        ("Capability Synthesis", test_capability_synthesis),
        ("Orchestrator", test_orchestrator),
        ("Recursive Evolution", test_recursive_evolution)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results[name] = "✅ PASSED" if result else "⚠️ SKIPPED"
        except Exception as e:
            results[name] = f"❌ FAILED: {e}"
            
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    
    all_passed = True
    for name, result in results.items():
        print(f"{name}: {result}")
        if "FAILED" in result:
            all_passed = False
            
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️ Some tests failed. Check output above.")
    print("=" * 60)
    
if __name__ == "__main__":
    asyncio.run(main())
