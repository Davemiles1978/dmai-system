import re

with open('app.py', 'r') as f:
    content = f.read()

# Split by route definitions
pattern = r'(@app\.route\([^)]+\)\s*\n\s*def\s+(\w+)\()'
matches = list(re.finditer(pattern, content, re.MULTILINE))

# Keep track of seen endpoints
seen = set()
last_end = 0
new_content = []

print("Found routes:")
for match in matches:
    endpoint = match.group(2)
    print(f"  {endpoint} at position {match.start()}")
    
    # Add text before this route
    new_content.append(content[last_end:match.start()])
    
    if endpoint not in seen:
        seen.add(endpoint)
        new_content.append(match.group(0))
        print(f"  ✅ Keeping {endpoint}")
    else:
        print(f"  ❌ Removing duplicate {endpoint}")
    
    last_end = match.end()

# Add remaining content
new_content.append(content[last_end:])

# Write cleaned file
with open('app.py.clean', 'w') as f:
    f.write(''.join(new_content))

print("\n✅ Cleaned app.py.clean created!")
