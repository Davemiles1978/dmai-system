#!/usr/bin/env python3
"""
Test all evaluators with proper class name handling
"""

import os
import sys
import importlib.util
import json

# Add project root to path
project_root = "/Users/davidmiles/Desktop/AI-Evolution-System"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TARGET = os.path.join(project_root, "evolution/evaluators/base_evaluator.py")

evaluators = [
    ('gemini', 'GeminiEvaluator'),
    ('grok', 'GrokEvaluator'),
    ('gpt', 'GptEvaluator'),
    ('claude', 'ClaudeEvaluator'),
    ('deepseek_v4', 'DeepseekV4Evaluator'),
    ('qwen3', 'Qwen3Evaluator'),
    ('kimi_k2', 'KimiK2Evaluator'),
    ('kimi_dev', 'KimiDevEvaluator'),
    ('llama4_scout', 'Llama4ScoutEvaluator'),
    ('alpamayo', 'AlpamayoEvaluator'),
    ('nova', 'NovaEvaluator'),
    ('mistral', 'MistralEvaluator')
]

print("🧪 TESTING ALL EVALUATORS")
print("=" * 60)
print(f"Target: {TARGET}\n")

success_count = 0
fail_count = 0

for module_name, class_name in evaluators:
    try:
        # Construct module path
        module_path = os.path.join(project_root, "evolution", "evaluators", f"{module_name}_evaluator.py")
        
        if not os.path.exists(module_path):
            print(f"❌ {module_name.upper():12} | File not found: {module_path}")
            fail_count += 1
            continue
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get class
        evaluator_class = getattr(module, class_name)
        
        # Create instance and test
        evaluator = evaluator_class()
        result = evaluator.evaluate(TARGET)
        
        # Check if result has expected structure
        if 'score' in result and 'suggestions' in result:
            print(f"✅ {module_name.upper():12} | Score: {result['score']:3} | Suggestions: {len(result['suggestions'])}")
            success_count += 1
        else:
            print(f"⚠️ {module_name.upper():12} | Invalid result format")
            fail_count += 1
        
    except Exception as e:
        print(f"❌ {module_name.upper():12} | Error: {str(e)[:50]}")
        fail_count += 1

print("\n" + "=" * 60)
print(f"✅ Testing complete: {success_count} passed, {fail_count} failed")
print("=" * 60)

# Optional: Show evolution stats
try:
    from evolution.randomizer import randomizer
    stats = randomizer.get_evolution_stats()
    print("\n📊 Evolution System Stats:")
    print(f"   Total evaluators: {stats['total_evaluators']}")
    print(f"   External research weight: {stats['external_research_weight']}")
except:
    pass
