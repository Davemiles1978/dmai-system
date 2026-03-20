#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Render-optimized launcher for AGI Evolution System
"""
import asyncio
import logging
import os
import sys
import traceback
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

# ============= RENDER-COMPATIBLE SETUP =============
# On Render free tier, we can only write to:
# - /tmp (ephemeral - cleared on restart)
# - The app's own directory (may not be writable)
# - /opt/render/project/data (only on paid instances)

# Determine the best data path
if os.path.exists('/opt/render/project/data') and os.access('/opt/render/project/data', os.W_OK):
    # Paid instance with persistent disk
    DATA_ROOT = Path('/opt/render/project/data')
    print(f"✅ Using persistent disk: {DATA_ROOT}")
else:
    # Free instance - use /tmp (ephemeral)
    DATA_ROOT = Path('/tmp/agi_evolution_data')
    print(f"⚠️ Using ephemeral storage (free tier): {DATA_ROOT}")

# Create data directory
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Setup logging path
LOG_FILE = DATA_ROOT / 'evolution.log'

# Setup logging with safe file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger('render_agi')
logger.info(f"🚀 Logging initialized to: {LOG_FILE}")
logger.info(f"📁 Data root: {DATA_ROOT}")
# ====================================================

class RenderAGILauncher:
    def __init__(self):
        self.data_root = DATA_ROOT
        self.launcher = None
        self.setup_directories()
        self.setup_symlinks()
    
    def setup_directories(self):
        """Create necessary directories"""
        try:
            dirs = [
                self.data_root / 'shared_data',
                self.data_root / 'shared_data/agi_evolution',
                self.data_root / 'shared_data/agi_evolution/capabilities',
                self.data_root / 'shared_data/agi_evolution/patterns',
                self.data_root / 'shared_data/agi_evolution/synthesis',
                self.data_root / 'shared_data/agi_evolution/orchestrator_state',
                self.data_root / 'shared_data/agi_evolution/evolution_history',
                self.data_root / 'shared_checkpoints',
                self.data_root / 'agi',
                self.data_root / 'agi/backups',
                self.data_root / 'agi/health',
                self.data_root / 'agi/models',
                self.data_root / 'agi/test_results',
            ]
            
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created: {d}")
                
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            raise
    
    def setup_symlinks(self):
        """Create symlinks from working directory to data root"""
        try:
            # Remove existing directories if they're not symlinks
            for dir_name in ['shared_data', 'shared_checkpoints', 'agi']:
                path = Path(dir_name)
                if path.exists():
                    if not path.is_symlink():
                        logger.info(f"📦 Backing up existing {dir_name} before symlink")
                        backup_name = f"{dir_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        shutil.move(str(path), backup_name)
                        logger.info(f"  Moved to {backup_name}")
                
                # Create symlink to data root
                target = self.data_root / dir_name
                if not path.exists():
                    path.symlink_to(target, target_is_directory=True)
                    logger.info(f"🔗 Created symlink: {path} -> {target}")
                
        except Exception as e:
            logger.error(f"Symlink setup failed: {e}")
            # Non-fatal, continue
    
    async def seed_if_needed(self):
        """Seed knowledge graph if empty"""
        try:
            from knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            if len(kg.graph.nodes) <= 1:  # Only root node
                logger.info("🌱 Seeding knowledge graph with initial concepts...")
                kg.add_concept("evolution", "core_process", {"description": "System evolution"})
                kg.add_concept("mutation", "core_operation", {"description": "Code mutation"})
                kg.add_concept("selection", "core_operation", {"description": "Fitness selection"})
                kg.add_concept("capability", "core_concept", {"description": "System capability"})
                kg.add_relationship("evolution", "mutation", "uses")
                kg.add_relationship("evolution", "selection", "uses")
                kg.add_relationship("mutation", "capability", "creates")
                kg.save()
                logger.info("✅ Knowledge graph seeded")
        except Exception as e:
            logger.error(f"Failed to seed knowledge graph: {e}")
    
    async def initialize(self):
        """Initialize the AGI launcher"""
        logger.info("🚀 Initializing AGI Launcher...")
        
        # Import here to catch import errors
        from launch_agi import AGILauncher
        
        self.launcher = AGILauncher()
        
        # Override paths for persistent storage
        if hasattr(self.launcher, 'orchestrator') and self.launcher.orchestrator:
            self.launcher.orchestrator.base_path = self.data_root / 'shared_data' / 'agi_evolution'
            self.launcher.orchestrator.checkpoint_path = self.data_root / 'shared_checkpoints'
            logger.info(f"📁 Data path: {self.launcher.orchestrator.base_path}")
            logger.info(f"📁 Checkpoint path: {self.launcher.orchestrator.checkpoint_path}")
        else:
            logger.warning("⚠️ Orchestrator not yet initialized, paths will be set later")
        
        # Seed knowledge graph if needed
        await self.seed_if_needed()
        
        await self.launcher.initialize()
        logger.info("✅ AGI Launcher initialized")
    
    async def run(self):
        """Run the AGI system"""
        try:
            logger.info("="*60)
            logger.info("🤖 Starting AGI Evolution System on Render")
            logger.info("="*60)
            
            # Initialize first
            await self.initialize()
            
            # Then start
            await self.launcher.start()
            
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            logger.error(f"Python path: {sys.path}")
            raise
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
            logger.error(traceback.format_exc())
            raise

async def main():
    launcher = RenderAGILauncher()
    await launcher.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
