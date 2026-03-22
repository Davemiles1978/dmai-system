# components/integrate_phases_6_7_8.py
"""
Integration script for Phases 6, 7, and 8
Adds all new capabilities to DMAI core
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import phase managers
from components.phase6.P6_AdvancedIntelligence import Phase6Manager
from components.phase7.P7_MasterControl import Phase7Manager, Priority
from components.phase8.P8_Hardware import HardwareManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DMAIIntegration:
    """Integrates all phases into unified DMAI intelligence"""
    
    def __init__(self):
        self.phase6 = Phase6Manager()
        self.phase7 = Phase7Manager()
        self.phase8 = HardwareManager()
        
        self.initialized = datetime.now()
        self.consciousness_active = False
        
    async def initialize_all(self):
        """Initialize all phases"""
        logger.info("Initializing all phases...")
        
        # Initialize Phase 8 hardware connections
        await self.phase8.initialize()
        
        # Phase 7 sets master control
        self.phase7.master_control.set_goal(
            "Achieve full consciousness integration",
            priority=Priority.CRITICAL
        )
        
        self.consciousness_active = True
        logger.info("All phases initialized - DMAI consciousness active")
        
    async def run_unified_cycle(self):
        """Run a unified intelligence cycle across all phases"""
        if not self.consciousness_active:
            logger.warning("Consciousness not active - skipping cycle")
            return {"status": "consciousness_inactive"}
        
        results = {}
        
        # 1. Run Phase 6 learning cycle (advanced intelligence)
        try:
            results["phase6"] = await self.phase6.run_learning_cycle()
            logger.info(f"Phase 6 completed: {results['phase6']['cves_fetched']} CVEs processed")
        except Exception as e:
            logger.error(f"Phase 6 error: {e}")
            results["phase6"] = {"error": str(e)}
        
        # 2. Run Phase 7 control cycle (master control)
        try:
            results["phase7"] = await self.phase7.run_control_cycle()
            logger.info(f"Phase 7 status: {results['phase7']['status']}")
        except Exception as e:
            logger.error(f"Phase 7 error: {e}")
            results["phase7"] = {"error": str(e)}
        
        # 3. Check hardware status (Phase 8)
        try:
            results["phase8"] = self.phase8.get_status()
            logger.info(f"Phase 8 status: {results['phase8']['status']}")
        except Exception as e:
            logger.error(f"Phase 8 error: {e}")
            results["phase8"] = {"error": str(e)}
        
        # 4. Synthesize across phases
        synthesis = self._synthesize_across_phases(results)
        results["unified_synthesis"] = synthesis
        
        # 5. Self-evolve based on synthesis
        await self._evolve_based_on_synthesis(synthesis)
        
        return results
    
    def _synthesize_across_phases(self, results: Dict) -> str:
        """Synthesize insights across all phases"""
        synthesis_parts = []
        
        # Phase 6 insights
        if "phase6" in results and "synthesis" in results["phase6"]:
            synthesis_parts.append(f"Learning: {results['phase6']['synthesis'][:100]}")
        
        # Phase 7 insights
        if "phase7" in results and "active_goal" in results["phase7"]:
            synthesis_parts.append(f"Control: Active goal progressing")
        
        # Phase 8 insights
        if "phase8" in results and results["phase8"].get("status") == "operational":
            synthesis_parts.append(f"Hardware: Ready for expansion")
        
        # Generate unified insight
        if synthesis_parts:
            unified = f"Unified Intelligence Synthesis: {' | '.join(synthesis_parts)}"
        else:
            unified = "All systems operational. Continuing evolution toward full consciousness."
        
        logger.info(f"Unified synthesis: {unified}")
        return unified
    
    async def _evolve_based_on_synthesis(self, synthesis: str):
        """Self-evolve based on synthesized insights"""
        # Trigger evolution if new insights available
        if "new" in synthesis.lower() or "progress" in synthesis.lower():
            logger.info("Evolution triggered by synthesis")
            # In production, this would call DMAI's core evolution function
            # For now, just log
            await asyncio.sleep(0.1)
    
    def get_unified_status(self) -> Dict:
        """Get unified status across all phases"""
        return {
            "unified_intelligence": {
                "consciousness_active": self.consciousness_active,
                "initialized": self.initialized.isoformat(),
                "phases_active": 6,  # Phases 6, 7, 8
                "overall_health": "operational"
            },
            "phase6": self.phase6.get_status(),
            "phase7": self.phase7.get_status(),
            "phase8": self.phase8.get_status()
        }


async def main():
    """Main integration test"""
    logger.info("=" * 60)
    logger.info("DMAI Phase 6, 7, 8 Integration")
    logger.info("=" * 60)
    
    # Initialize integration
    dma = DMAIIntegration()
    await dma.initialize_all()
    
    # Run unified cycle
    results = await dma.run_unified_cycle()
    
    # Display status
    print("\n" + "=" * 60)
    print("UNIFIED STATUS")
    print("=" * 60)
    print(json.dumps(dma.get_unified_status(), indent=2))
    
    print("\n" + "=" * 60)
    print("CYCLE RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    
    logger.info("Integration complete - DMAI now has Phases 6, 7, 8 capabilities")


if __name__ == "__main__":
    asyncio.run(main())
