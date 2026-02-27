# Read and fix self_healer.py
with open('self_healer.py', 'r') as f:
    content = f.read()

# Update heal_component to handle the backup_id properly
old_method = """    async def heal_component(self, component_name, error_info):
        """Attempt to heal a component"""
        print(f"🩺 SelfHealer: Attempting to heal {component_name}")
        # Try to find the component
        if hasattr(self, component_name):
            component = getattr(self, component_name)
            return await self.safe_update(
                lambda: self.check_system_health(),
                component,
                test_suite="basic"
            )
        backup_id = await self.create_backup(component_name)
        return {
            "status": "healing_initiated",
            "component": component_name,
            "backup_id": backup_id,
            "message": "Component not directly accessible, backup created"
        }"""

new_method = """    async def heal_component(self, component_name, error_info):
        """Attempt to heal a component"""
        print(f"🩺 SelfHealer: Attempting to heal {component_name}")
        try:
            # Try to find the component
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                result = await self.safe_update(
                    lambda: self.check_system_health(),
                    component,
                    test_suite="basic"
                )
                return result
            
            # Create backup and return status
            backup_id = await self.create_backup(component_name)
            return {
                "status": "healing_initiated",
                "component": component_name,
                "backup_id": backup_id,
                "message": "Component not directly accessible, backup created"
            }
        except Exception as e:
            print(f"❌ Error in heal_component: {e}")
            return {
                "status": "error",
                "component": component_name,
                "error": str(e)
            }"""

content = content.replace(old_method, new_method)

with open('self_healer.py', 'w') as f:
    f.write(content)
print("✅ Fixed self_healer.py error handling")
