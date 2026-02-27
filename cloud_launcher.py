#!/usr/bin/env python3
"""
Render-optimized launcher for AGI Evolution System
Runs 24/7 with Render.com persistence
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import json

# Setup Render-compatible logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/data/evolution.log') if os.path.exists('/var/data') else logging.FileHandler('evolution.log')
    ]
)

logger = logging.getLogger('render_agi')

class RenderAGILauncher:
    def __init__(self):
        self.is_render = os.environ.get('RENDER', False)
        self.data_path = Path('/var/data') if self.is_render else Path('./data')
        self.setup_persistence()
        
    def setup_persistence(self):
        """Setup persistent storage for Render"""
        # Create persistent directories
        persistent_dirs = [
            self.data_path / 'shared_data',
            self.data_path / 'shared_data/agi_evolution',
            self.data_path / 'shared_data/agi_evolution/capabilities',
            self.data_path / 'shared_data/agi_evolution/patterns',
            self.data_path / 'shared_data/agi_evolution/synthesis',
            self.data_path / 'shared_data/agi_evolution/orchestrator_state',
            self.data_path / 'shared_data/agi_evolution/evolution_history',
            self.data_path / 'shared_checkpoints',
            self.data_path / 'agi',
            self.data_path / 'agi/backups',
            self.data_path / 'agi/health',
            self.data_path / 'agi/models',
            self.data_path / 'agi/test_results',
        ]
        
        for path in persistent_dirs:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created: {path}")
        
        # Create symlinks for Render
        if self.is_render:
            for dir_name in ['shared_data', 'shared_checkpoints', 'agi']:
                target = self.data_path / dir_name
                link = Path(f'/opt/render/project/src/{dir_name}')
                if not link.exists():
                    link.symlink_to(target, target_is_directory=True)
                    logger.info(f"🔗 Created symlink: {link} -> {target}")
        
        # Save deployment info
        deploy_info = {
            'deployed_at': datetime.now().isoformat(),
            'platform': 'render' if self.is_render else 'local',
            'generation': int(os.environ.get('GENERATION_START', 5)),
            'data_path': str(self.data_path),
            'repo': 'https://github.com/Davemiles1978/dmai-system.git'
        }
        
        with open(self.data_path / 'deploy_info.json', 'w') as f:
            json.dump(deploy_info, f, indent=2)
            
    async def run(self):
        """Run the AGI system"""
        from launch_agi import AGILauncher
        
        logger.info("="*60)
        logger.info("☁️ Starting AGI Evolution System on Render.com")
        logger.info("="*60)
        
        # Import and configure launcher
        launcher = AGILauncher()
        
        try:
            # Override paths for Render
            if self.is_render:
                launcher.orchestrator.base_path = self.data_path / 'shared_data' / 'agi_evolution'
                launcher.orchestrator.checkpoint_path = self.data_path / 'shared_checkpoints'
            
            await launcher.initialize()
            await launcher.start()
            
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            # Don't exit - Render will restart
            await asyncio.sleep(60)
            await self.run()

async def main():
    launcher = RenderAGILauncher()
    await launcher.run()

if __name__ == "__main__":
    asyncio.run(main())
