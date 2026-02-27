# Read the file
with open('meta_learner.py', 'r') as f:
    content = f.read()

# Find the end of the class and insert the method
if 'class MetaLearner:' in content:
    # Look for the last method or the class end
    lines = content.split('\n')
    new_lines = []
    in_class = False
    added = False
    
    for line in lines:
        new_lines.append(line)
        if 'class MetaLearner:' in line:
            in_class = True
        elif in_class and not added and line.strip() and not line.startswith(' ') and not line.startswith('\t') and line.strip() != '':
            # We've exited the class, add the method before this line
            new_lines.insert(-1, '')
            new_lines.insert(-1, '    async def learn_from_evolution(self, evolution_record):')
            new_lines.insert(-1, '        """Learn from evolution records"""')
            new_lines.insert(-1, '        print(f"🔄 MetaLearner: Learning from evolution at generation {evolution_record.get(\'generation\', \'unknown\')}")')
            new_lines.insert(-1, '        return await self.analyze_evolution_cycle(evolution_record)')
            new_lines.insert(-1, '')
            added = True
            in_class = False
    
    with open('meta_learner.py', 'w') as f:
        f.write('\n'.join(new_lines))
    print("✅ Fixed meta_learner.py")
