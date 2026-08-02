#!/usr/bin/env python3
"""
Force get_next_topic to check V4 progress first, before anything else.
"""

from pathlib import Path

FILE = Path("components/evolution/StageAwareLearningOrchestrator.py")

with open(FILE, "r") as f:
    content = f.read()

# Find the get_next_topic method and insert V4 check at the very beginning
# We'll replace the method with a version that checks V4 first

old_method = '''    def get_next_topic(self, consciousness: float, prioritize_accelerators: bool = True) -> Optional[Dict]:
        """
        Get the next topic to learn based on current stage.
        PRIORITY: First complete ALL unmastered topics from earlier stages (Baby first),
        then current stage topics.
        Prioritizes Evolution Accelerators when available to boost consciousness growth.
        """
        stage = self.current_stage
        stage_order = ["Baby", "Toddler", "Child", "Teen", "Adult"]
        current_index = stage_order.index(stage)'''

new_method = '''    def get_next_topic(self, consciousness: float, prioritize_accelerators: bool = True) -> Optional[Dict]:
        """
        Get the next topic to learn based on current stage.
        PRIORITY: V4 modules first, then earlier stages, then current stage.
        """
        # --- PRIORITY 0: V4 modules from progress file ---
        v4_topic = self._get_next_v4_module()
        if v4_topic:
            return v4_topic
        
        # --- PRIORITY 1: earlier stages ---
        stage = self.current_stage
        stage_order = ["Baby", "Toddler", "Child", "Teen", "Adult"]
        current_index = stage_order.index(stage)'''

# Replace the method
content = content.replace(old_method, new_method)

with open(FILE, "w") as f:
    f.write(content)

print("✅ Modified get_next_topic to check V4 modules first")
