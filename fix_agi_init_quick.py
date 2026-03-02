#!/usr/bin/env python3
"""
Quick fix for AGI Orchestrator initialization
"""
import re

with open('agi_orchestrator.py', 'r') as f:
    content = f.read()

# Check if self_healer is being initialized
if 'self.self_healer = SelfHealer()' not in content:
    print("❌ self_healer initialization missing")
    
    # Find the __init__ method and add it
    init_pattern = r'def __init__\(self.*?\).*?:(.*?)(?=def |$)'
    match = re.search(init_pattern, content, re.DOTALL)
    
    if match:
        init_body = match.group(1)
        if 'self.self_healer' not in init_body:
            # Add self_healer initialization
            new_init = init_body.replace(
                'self.knowledge_graph = KnowledgeGraph()',
                'self.knowledge_graph = KnowledgeGraph()\n        self.self_healer = SelfHealer()'
            )
            content = content.replace(init_body, new_init)
            print("✅ Added self_healer to __init__")
else:
    print("✅ self_healer already in __init__")

# Check for duplicate __init__ methods
init_count = len(re.findall(r'def __init__\(', content))
if init_count > 1:
    print(f"⚠️ Found {init_count} __init__ methods - keeping first one")
    # Keep only the first __init__
    inits = list(re.finditer(r'def __init__\(.*?\):.*?(?=def |$)', content, re.DOTALL))
    if len(inits) > 1:
        # Remove subsequent __init__ methods
        for init in inits[1:]:
            content = content.replace(init.group(), '')
        print("✅ Removed duplicate __init__ methods")

with open('agi_orchestrator.py', 'w') as f:
    f.write(content)
print("✅ Fix applied")
