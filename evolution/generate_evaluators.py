#!/usr/bin/env python3
"""
Generate all evaluator files with proper content
"""

import os

evaluators = [
    'gemini', 'grok', 'gpt', 'claude', 'deepseek_v4', 
    'qwen3', 'kimi_k2', 'kimi_dev', 'llama4_scout', 
    'alpamayo', 'nova', 'mistral'
]

base_dir = "/Users/davidmiles/Desktop/dmai-system/evolution/evaluators"

# Template for each evaluator
template = '''#!/usr/bin/env python3
\"\"\"
{name_upper} Evaluator for DMAI Evolution System
\"\"\"

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

class {class_name}(BaseEvaluator):
    def __init__(self):
        super().__init__("{name}", version="1.0.0")
    
    def generate_suggestions(self, analysis):
        suggestions = super().generate_suggestions(analysis)
        suggestions.append("Add {name}-specific optimizations")
        return suggestions

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        evaluator = {class_name}()
        result = evaluator.evaluate(target)
        print(result)
    else:
        print(f"Usage: {name}_evaluator.py <target_file>")
'''

for name in evaluators:
    # Create class name (e.g., deepseek_v4 -> DeepseekV4Evaluator)
    parts = name.split('_')
    class_name = ''.join(p.capitalize() for p in parts) + 'Evaluator'
    
    filename = f"{base_dir}/{name}_evaluator.py"
    
    content = template.format(
        name_upper=name.upper(),
        name=name,
        class_name=class_name
    )
    
    with open(filename, 'w') as f:
        f.write(content)
    
    os.chmod(filename, 0o755)
    print(f"✅ Created {filename}")

print("\n🎯 All evaluator files created successfully!")
print(f"Total: {len(evaluators)} evaluators")
