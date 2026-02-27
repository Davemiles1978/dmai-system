# launch_agi.py

#!/usr/bin/env python3
"""
AGI System Launcher - Starts all components for recursive self-improvement
"""

import asyncio
import signal
import sys
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from agi_orchestrator import AGIOrchestrator
from knowledge_graph import KnowledgeGraph
from meta_learner import MetaLearner
from self_healer import SelfHealer
from data_validator import DataValidator
from capability_synthesizer import CapabilitySynthesizer

class AGILauncher:
    """Launches and manages all AGI components"""
    
    def __init__(self):
        self.orchestrator = None
        self.running = False
        self.start_time = datetime.now()
        
    async def initialize(self):
        """Initialize all components"""
        print("=" * 60)
        print("🚀 AGI System Launcher v1.0")
        print("=" * 60)
        
        # Check environment
        await self._check_environment()
        
        # Initialize orchestrator
        print("\n📡 Initializing AGI Orchestrator...")
        self.orchestrator = AGIOrchestrator()
        
        # Register shutdown handler
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        print("✅ Initialization complete")
        
    async def _check_environment(self):
        """Check the environment is properly set up"""
        print("\n🔍 Checking environment...")
        
        # Check required directories
        required_dirs = [
            "shared_data/agi_evolution",
            "shared_checkpoints",
            "shared_data/agi_evolution/capabilities",
            "shared_data/agi_evolution/patterns",
            "shared_data/agi_evolution/synthesis"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path}")
            
        # Check for latest checkpoint
        checkpoints = list(Path("shared_checkpoints").glob("gen_*"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"  ✅ Latest checkpoint: {latest.name}")
        else:
            print("  ⚠️ No checkpoints found - starting fresh")
            
        print("✅ Environment check complete")
        
    async def start(self):
        """Start all AGI components"""
        print("\n" + "=" * 60)
        print("🎯 Starting AGI System...")
        print("=" * 60)
        
        self.running = True
        
        # Start orchestrator
        await self.orchestrator.start()
        
        # Submit initial goals
        await self._submit_initial_goals()
        
        # Display status
        await self._display_status()
        
        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
            
    async def _submit_initial_goals(self):
        """Submit initial evolution goals"""
        print("\n📋 Submitting initial goals...")
        
        goals = [
            {
                'description': 'Improve capability synthesis成功率',
                'priority': 10,
                'requires_evolution': True,
                'context': {'target': 'synthesis_success_rate', 'threshold': 0.8}
            },
            {
                'description': 'Create meta-learning optimization capability',
                'priority': 9,
                'requires_evolution': True,
                'context': {'base_capabilities': ['learning', 'optimization']}
            },
            {
                'description': 'Implement self-healing prediction',
                'priority': 8,
                'requires_evolution': True,
                'context': {'type': 'predictive_healing'}
            },
            {
                'description': 'Enhance data validation with AI',
                'priority': 7,
                'requires_evolution': True,
                'context': {'ai_enhanced': True}
            }
        ]
        
        for goal in goals:
            await self.orchestrator.submit_goal(goal)
            print(f"  ✅ {goal['description']}")
            
    async def _display_status(self):
        """Display current system status"""
        status = self.orchestrator.get_status()
        
        print("\n" + "=" * 60)
        print("📊 System Status")
        print("=" * 60)
        print(f"Generation: {status['state']['generation']}")
        print(f"Health: {status['state']['health_status']}")
        print(f"Active Capabilities: {status['active_capabilities']}")
        print(f"Pending Goals: {status['pending_goals']}")
        print(f"Learning Rate: {status['state']['learning_rate']:.3f}")
        print(f"Exploration Rate: {status['state']['exploration_rate']:.3f}")
        
        print("\nComponents:")
        for comp, health in status['components'].items():
            print(f"  {comp}: {health}")
            
        print("\n" + "=" * 60)
        print("System is running. Press Ctrl+C to stop.")
        print("=" * 60)
        
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\n🛑 Shutdown signal received...")
        self.running = False
        
    async def shutdown(self):
        """Gracefully shutdown all components"""
        print("\n" + "=" * 60)
        print("🛑 Shutting down AGI System...")
        print("=" * 60)
        
        if self.orchestrator:
            # Save final state
            self.orchestrator._save_state()
            
        # Calculate uptime
        uptime = datetime.now() - self.start_time
        print(f"Uptime: {uptime}")
        print("✅ Shutdown complete")
        
async def main():
    """Main entry point"""
    launcher = AGILauncher()
    
    try:
        await launcher.initialize()
        await launcher.start()
    except KeyboardInterrupt:
        pass
    finally:
        await launcher.shutdown()
        
if __name__ == "__main__":
    asyncio.run(main())
