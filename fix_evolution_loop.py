#!/usr/bin/env python3
"""
Fix the evolution loop variable error
"""
import re

with open('agi_orchestrator.py', 'r') as f:
    content = f.read()

# Find the evolution_loop method and fix the variable issue
loop_pattern = r'async def _evolution_loop\(self\).*?while not self\.evolution_queue\.empty\(\):.*?evolution_task = await self\.evolution_queue\.get\(\)(.*?)(?=await asyncio\.sleep|$)'
match = re.search(loop_pattern, content, re.DOTALL)

if match:
    old_code = match.group(0)
    
    # Replace with fixed version that initializes result
    new_code = '''async def _evolution_loop(self):
        """Main evolution loop for recursive self-improvement"""
        while True:
            try:
                if await self._should_evolve():
                    await self._perform_evolution_step()
                    
                while not self.evolution_queue.empty():
                    evolution_task = await self.evolution_queue.get()
                    result = None  # Initialize result
                    try:
                        result = await self._process_evolution_task(evolution_task)
                    except Exception as e:
                        print(f"❌ Error processing task: {e}")
                        result = {"error": str(e)}
                    finally:
                        self.evolution_queue.task_done()
                    
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._handle_error("evolution_loop", e)
                await asyncio.sleep(300)'''
    
    content = content.replace(old_code, new_code)
    print("✅ Fixed evolution loop variable error")
else:
    print("❌ Could not find evolution_loop method")

with open('agi_orchestrator.py', 'w') as f:
    f.write(content)
