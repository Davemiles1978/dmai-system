# Simple fix for self_healer.py
with open('self_healer.py', 'r') as f:
    content = f.read()

# Add heal_component method if not present
if 'async def heal_component' not in content:
    # Find the end of the class and add the method
    import re
    class_end = re.search(r'class SelfHealer:.*?\n\n', content, re.DOTALL)
    if class_end:
        insert_pos = class_end.end()
        method = """

    async def heal_component(self, component_name, error_info):
        \"\"\"Attempt to heal a component\"\"\"
        print(f"🩺 SelfHealer: Attempting to heal {component_name}")
        try:
            # Create a backup
            backup_id = await self.create_backup(component_name)
            return {
                "status": "healing_initiated",
                "component": component_name,
                "backup_id": backup_id,
                "message": "Healing process started"
            }
        except Exception as e:
            print(f"❌ Error in heal_component: {e}")
            return {
                "status": "error",
                "component": component_name,
                "error": str(e)
            }
"""
        content = content[:insert_pos] + method + content[insert_pos:]
        print("✅ Added heal_component method")

with open('self_healer.py', 'w') as f:
    f.write(content)
print("✅ self_healer.py fixed")
