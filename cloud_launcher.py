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

# ============= SAFE LOGGING SETUP =============
# Try multiple locations for log file, with fallbacks
LOG_PATHS = [
    '/var/data/evolution.log',           # Preferred persistent location
    '/tmp/evolution.log',                 # Temporary location (cleared on restart)
    'evolution.log'                        # Current directory (may not be writable)
]

log_file = None
for potential_path in LOG_PATHS:
    try:
        # Test if we can write to this location
        test_path = Path(potential_path).parent
        test_path.mkdir(parents=True, exist_ok=True)
        
        # Try to create a test file
        with open(potential_path, 'a') as f:
            f.write('')
        
        log_file = potential_path
        print(f"✅ Using log file: {log_file}")
        break
    except (OSError, PermissionError, IOError):
        continue

if log_file is None:
    # Ultimate fallback - use temp directory
    log_dir = tempfile.gettempdir()
    log_file = os.path.join(log_dir, 'evolution.log')
    print(f"⚠️ Using fallback log file: {log_file}")

# Setup logging with safe file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('render_agi')
logger.info(f"🚀 Logging initialized to: {log_file}")
# ==============================================

class RenderAGILauncher:
    def __init__(self):
        self.data_path = Path('/var/data')
        self.launcher = None
        self.setup_directories()
        self.setup_symlinks()
    
    def setup_directories(self):
        """Create necessary directories"""
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            dirs = [
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
            
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created: {d}")
                
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            raise
    
    def setup_symlinks(self):
        """Create symlinks from working directory to persistent disk"""
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
                
                # Create symlink
                target = self.data_path / dir_name
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
        
        # Override paths for persistent storage AFTER initialization
        if hasattr(self.launcher, 'orchestrator') and self.launcher.orchestrator:
            self.launcher.orchestrator.base_path = self.data_path / 'shared_data' / 'agi_evolution'
            self.launcher.orchestrator.checkpoint_path = self.data_path / 'shared_checkpoints'
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
