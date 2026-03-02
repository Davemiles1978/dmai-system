#!/usr/bin/env python3
import re

with open('launch_agi.py', 'r') as f:
    content = f.read()

# Check if method exists but is outside class
if 'def _safe_start_orchestrator' in content:
    # Make sure it's inside the AGILauncher class
    if 'class AGILauncher:' in content:
        # Find class definition and ensure method is inside
        lines = content.split('\n')
        in_class = False
        class_found = False
        method_added = False
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            if 'class AGILauncher:' in line:
                class_found = True
                in_class = True
                continue
            if in_class and line.strip() and not line.startswith(' ') and not line.startswith('\t') and line.strip() != '':
                # We've exited the class, add method before this line
                if not method_added:
                    new_lines.insert(-1, '')
                    new_lines.insert(-1, '    async def _safe_start_orchestrator(self):')
                    new_lines.insert(-1, '        """Safely start orchestrator regardless of method name"""')
                    new_lines.insert(-1, '        if hasattr(self.orchestrator, "start"):')
                    new_lines.insert(-1, '            return await self.orchestrator.start()')
                    new_lines.insert(-1, '        elif hasattr(self.orchestrator, "run"):')
                    new_lines.insert(-1, '            print("⚠️ Using run() instead of start()")')
                    new_lines.insert(-1, '            return await self.orchestrator.run()')
                    new_lines.insert(-1, '        else:')
                    new_lines.insert(-1, '            for method_name in ["start", "run", "begin", "execute", "launch"]:')
                    new_lines.insert(-1, '                if hasattr(self.orchestrator, method_name):')
                    new_lines.insert(-1, '                    print(f"⚠️ Using {method_name}() as fallback")')
                    new_lines.insert(-1, '                    method = getattr(self.orchestrator, method_name)')
                    new_lines.insert(-1, '                    if asyncio.iscoroutinefunction(method):')
                    new_lines.insert(-1, '                        return await method()')
                    new_lines.insert(-1, '                    else:')
                    new_lines.insert(-1, '                        return method()')
                    new_lines.insert(-1, '            raise AttributeError("Orchestrator has no start/run method")')
                    new_lines.insert(-1, '')
                    method_added = True
                in_class = False
        
        with open('launch_agi.py', 'w') as f:
            f.write('\n'.join(new_lines))
        print("✅ Fixed launch_agi.py - added _safe_start_orchestrator inside class")
