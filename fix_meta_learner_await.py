# Read and fix meta_learner.py
with open('meta_learner.py', 'r') as f:
    content = f.read()

# Check if analyze_evolution_cycle is defined as async
if 'def analyze_evolution_cycle' in content and 'async def analyze_evolution_cycle' not in content:
    # Replace with async version
    content = content.replace('def analyze_evolution_cycle', 'async def analyze_evolution_cycle')
    print("✅ Made analyze_evolution_cycle async")

# Also fix the learn_from_evolution method to handle both sync and async
old_method = """    async def learn_from_evolution(self, evolution_record):
        """Wrapper for analyze_evolution_cycle to match orchestrator expectations"""
        print(f"🔄 MetaLearner: Learning from evolution at generation {evolution_record.get('generation', 'unknown')}")
        return await self.analyze_evolution_cycle(evolution_record)"""

new_method = """    async def learn_from_evolution(self, evolution_record):
        """Learn from evolution records"""
        print(f"🔄 MetaLearner: Learning from evolution at generation {evolution_record.get('generation', 'unknown')}")
        # Check if analyze_evolution_cycle is async
        if hasattr(self.analyze_evolution_cycle, '__code__') and self.analyze_evolution_cycle.__code__.co_flags & 0x80:
            # It's async, await it
            return await self.analyze_evolution_cycle(evolution_record)
        else:
            # It's sync, call it directly
            return self.analyze_evolution_cycle(evolution_record)"""

content = content.replace(old_method, new_method)

with open('meta_learner.py', 'w') as f:
    f.write(content)
print("✅ Fixed meta_learner.py await issue")
