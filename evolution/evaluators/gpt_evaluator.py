#!/usr/bin/env python3
"""
GPT Evaluator for DMAI Evolution System
"""

import sys
import os

# Add project root to path
project_root = "/Users/davidmiles/Desktop/dmai-system"
if project_root not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent)))

try:
    from evolution.evaluators.base_evaluator import BaseEvaluator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent)))
    from evolution.evaluators.base_evaluator import BaseEvaluator

class GptEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("gpt", version="1.0.0")
    
    def generate_suggestions(self, analysis):
        suggestions = super().generate_suggestions(analysis)
        suggestions.append("Add gpt-specific optimizations")
        return suggestions

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        evaluator = GptEvaluator()
        result = evaluator.evaluate(target)
        print(result)
    else:
        print(f"Usage: gpt_evaluator.py <target_file>")
