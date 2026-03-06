#!/usr/bin/env python3
"""
Test all evaluators on a sample file
"""

import os
import sys
import importlib.util

# Add the project root to Python path
project_root = "/Users/davidmiles/Desktop/AI-Evolution-System"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Sample target file to evaluate
TARGET = os.path.join(project_root, "evolution/evaluators/base_evaluator.py")

# Define evaluators with their correct class names
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
print("=" * 50)
print(f"Target: {TARGET}\n")

for evaluator_name, class_name in evaluators:
    try:
        # Construct the module path
        module_path = os.path.join(project_root, "evolution", "evaluators", f"{evaluator_name}_evaluator.py")
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(f"{evaluator_name}_evaluator", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the evaluator class using the correct class name
        evaluator_class = getattr(module, class_name)
        
        # Create instance and evaluate
        evaluator = evaluator_class()
        result = evaluator.evaluate(TARGET)
        
        print(f"✅ {evaluator_name.upper():12} | Score: {result['score']:3} | Suggestions: {len(result['suggestions'])} | Improved: {result['improved']}")
        
    except Exception as e:
        print(f"❌ {evaluator_name.upper():12} | Error: {e}")

print("\n" + "=" * 50)
print("✅ Testing complete")
        

