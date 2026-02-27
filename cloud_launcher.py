#!/usr/bin/env python3
"""
Render-optimized launcher for AGI Evolution System
Runs 24/7 with persistent disk storage
"""

import asyncio
import logging
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
import json
import signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/data/evolution.log')
    ]
)

logger = logging.getLogger('render_agi')

class RenderAGILauncher:
    def __init__(self):
        self.data_path = Path('/var/data')
        self.running = True
        self.setup_persistence()
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Handle shutdown gracefully"""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info("🛑 Shutdown signal received, saving state...")
        self.running = False
        
    def setup_persistence(self):
        """Setup persistent storage on Render disk"""
        logger.info("📁 Setting up persistent storage...")
        
        # Create all necessary directories on persistent disk
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
            logger.info(f"  ✅ {path}")
        
        # Create symlinks from working directory to persistent disk
        symlinks = [
            ('shared_data', self.data_path / 'shared_data'),
            ('shared_checkpoints', self.data_path / 'shared_checkpoints'),
            ('agi', self.data_path / 'agi'),
        ]
        
        for link_name, target in symlinks:
            link_path = Path(link_name)
            if link_path.exists():
                if link_path.is_symlink():
                    logger.info(f"  🔗 Symlink already exists: {link_name}")
                else:
                    logger.warning(f"  ⚠️ {link_name} exists but is not a symlink")
            else:
                link_path.symlink_to(target, target_is_directory=True)
                logger.info(f"  🔗 Created symlink: {link_name} -> {target}")
        
        # Save deployment info
        deploy_info = {
            'deployed_at': datetime.now().isoformat(),
            'platform': 'render',
            'generation': int(os.environ.get('GENERATION_START', 5)),
            'data_path': str(self.data_path),
            'disk_size': '10GB',
            'repo': 'https://github.com/Davemiles1978/dmai-system.git'
        }
        
        info_file = self.data_path / 'deploy_info.json'
        with open(info_file, 'w') as f:
            json.dump(deploy_info, f, indent=2)
        logger.info(f"  ✅ Deployment info saved")
        
        # Load previous state if exists
        self.load_previous_state()
        
    def load_previous_state(self):
        """Load previous evolution state from disk"""
        state_file = self.data_path / 'shared_data/agi_evolution/orchestrator_state/current_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"📂 Loaded previous state - Generation {state.get('generation', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to load previous state: {e}")
        else:
            logger.info("🆕 No previous state found, starting fresh")
    
    async def run(self):
        """Run the AGI system"""
        # Import here to avoid circular imports
        from launch_agi import AGILauncher
        
        logger.info("="*60)
        logger.info("🤖 Starting AGI Evolution System on Render")
        logger.info("="*60)
        
        # Initialize and run the system
        launcher = AGILauncher()
        
        try:
            # Override paths to use persistent disk
            launcher.orchestrator.base_path = self.data_path / 'shared_data' / 'agi_evolution'
            launcher.orchestrator.checkpoint_path = self.data_path / 'shared_checkpoints'
            
            await launcher.initialize()
            
            # Start health monitoring in background
            asyncio.create_task(self.health_monitor())
            
            await launcher.start()
            
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            # Keep trying to restart
            await asyncio.sleep(10)
            if self.running:
                await self.run()
    
    async def health_monitor(self):
        """Monitor system health and log stats"""
        while self.running:
            try:
                # Check disk usage
                disk_usage = shutil.disk_usage(self.data_path)
                free_gb = disk_usage.free / (1024**3)
                total_gb = disk_usage.total / (1024**3)
                
                logger.info(f"💾 Disk: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
                
                # Check if we have recent checkpoints
                checkpoints = list((self.data_path / 'shared_checkpoints').glob('gen_*'))
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    logger.info(f"📌 Latest checkpoint: {latest.name}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(3600)

async def main():
    launcher = RenderAGILauncher()
    await launcher.run()

if __name__ == "__main__":
    asyncio.run(main())
