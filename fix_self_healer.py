# Read the file
with open('self_healer.py', 'r') as f:
    content = f.read()

# Find the end of the class and insert the method
if 'class SelfHealer:' in content:
    lines = content.split('\n')
    new_lines = []
    in_class = False
    added = False
    
    for line in lines:
        new_lines.append(line)
        if 'class SelfHealer:' in line:
            in_class = True
        elif in_class and not added and line.strip() and not line.startswith(' ') and not line.startswith('\t') and line.strip() != '':
            # We've exited the class, add the method before this line
            new_lines.insert(-1, '')
            new_lines.insert(-1, '    async def heal_component(self, component_name, error_info):')
            new_lines.insert(-1, '        """Attempt to heal a component"""')
            new_lines.insert(-1, '        print(f"🩺 SelfHealer: Attempting to heal {component_name}")')
            new_lines.insert(-1, '        # Try to find the component')
            new_lines.insert(-1, '        if hasattr(self, component_name):')
            new_lines.insert(-1, '            component = getattr(self, component_name)')
            new_lines.insert(-1, '            return await self.safe_update(')
            new_lines.insert(-1, '                lambda: self.check_system_health(),')
            new_lines.insert(-1, '                component,')
            new_lines.insert(-1, '                test_suite="basic"')
            new_lines.insert(-1, '            )')
            new_lines.insert(-1, '        backup_id = await self.create_backup(component_name)')
            new_lines.insert(-1, '        return {')
            new_lines.insert(-1, '            "status": "healing_initiated",')
            new_lines.insert(-1, '            "component": component_name,')
            new_lines.insert(-1, '            "backup_id": backup_id,')
            new_lines.insert(-1, '            "message": "Component not directly accessible, backup created"')
            new_lines.insert(-1, '        }')
            new_lines.insert(-1, '')
            added = True
            in_class = False
    
    with open('self_healer.py', 'w') as f:
        f.write('\n'.join(new_lines))
    print("✅ Fixed self_healer.py")
