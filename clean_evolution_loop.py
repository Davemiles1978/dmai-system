#!/usr/bin/env python3
"""
Clean up the evolution loop - remove duplicate and fix syntax
"""
import re

with open('agi_orchestrator.py', 'r') as f:
    content = f.read()

# Find the entire evolution_loop method
pattern = r'async def _evolution_loop\(self\).*?while True:.*?(?=\n\s*def|\n\s*@|\n\s*$)'
match = re.search(pattern, content, re.DOTALL)

if match:
    old_method = match.group(0)
    
    # Create clean version
    new_method = '''async def _evolution_loop(self):
        """Main evolution loop for recursive self-improvement"""
        while True:
            try:
                if await self._should_evolve():
                    await self._perform_evolution_step()
                    
                while not self.evolution_queue.empty():
                    evolution_task = await self.evolution_queue.get()
                    result = None
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
    
    content = content.replace(old_method, new_method)
    print("✅ Replaced evolution loop with clean version")
else:
    print("❌ Could not find evolution loop method")

with open('agi_orchestrator.py', 'w') as f:
    f.write(content)
