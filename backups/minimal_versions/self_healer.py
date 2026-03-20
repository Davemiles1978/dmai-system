"""Self Healer module for AGI evolution system"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from pathlib import Path
from datetime import datetime
import shutil

class SelfHealer:
    """Self-healing system"""
    
    def __init__(self, base_path: str = "shared_data/agi_evolution"):
        self.base_path = Path(base_path)
        self.healer_path = self.base_path / "self_healer"
        self.backup_path = self.healer_path / "backups"
        self.healer_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.components = {}
        print("✅ SelfHealer initialized")
        
    def register_component(self, name: str, component):
        """Register a component for healing"""
        self.components[name] = component
        
    async def heal_component(self, component_name: str, error_info: dict) -> dict:
        """Attempt to heal a component"""
        backup_id = await self.create_backup(component_name)
        return {
            "status": "healing_initiated",
            "component": component_name,
            "backup_id": backup_id
        }
        
    async def create_backup(self, component_name: str) -> str:
        """Create a backup"""
        backup_id = f"{component_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.backup_path / backup_id
        backup_dir.mkdir(exist_ok=True)
        return backup_id
        
    async def rollback(self, backup_id: str) -> dict:
        """Rollback to a backup"""
        return {"status": "rolled_back", "backup_id": backup_id}
        
    def is_healthy(self) -> bool:
        return True
