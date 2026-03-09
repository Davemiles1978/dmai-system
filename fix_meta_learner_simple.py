# Simple fix for meta_learner.py

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

with open('meta_learner.py', 'r') as f:
    content = f.read()

# Make analyze_evolution_cycle async if it's not already
if 'def analyze_evolution_cycle' in content and 'async def analyze_evolution_cycle' not in content:
    content = content.replace('def analyze_evolution_cycle', 'async def analyze_evolution_cycle')
    print("✅ Made analyze_evolution_cycle async")

# Add a safe learn_from_evolution method
if 'async def learn_from_evolution' not in content:
    # Find the end of the class and add the method
    import re
    class_end = re.search(r'class MetaLearner:.*?\n\n', content, re.DOTALL)
    if class_end:
        insert_pos = class_end.end()
        method = """

    async def learn_from_evolution(self, evolution_record):
        \"\"\"Learn from evolution records\"\"\"
        print(f"🔄 MetaLearner: Learning from evolution at generation {evolution_record.get('generation', 'unknown')}")
        try:
            # Try to call analyze_evolution_cycle (which should now be async)
            return await self.analyze_evolution_cycle(evolution_record)
        except Exception as e:
            print(f"❌ Error in learn_from_evolution: {e}")
            return {"status": "error", "message": str(e)}
"""
        content = content[:insert_pos] + method + content[insert_pos:]
        print("✅ Added learn_from_evolution method")

with open('meta_learner.py', 'w') as f:
    f.write(content)
print("✅ meta_learner.py fixed")
