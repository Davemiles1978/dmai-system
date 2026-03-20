#!/usr/bin/env python3
"""
DMAI WORKER - Core intelligence (no web server)
Runs as background worker on Render
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import time
import json
import logging
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - WORKER[DMAI] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('WORKER')

# Import your DMAI core
from dmai_core_clean import DMAIIntelligence

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 DMAI WORKER starting on Render")
    logger.info("="*60)
    
    # Initialize DMAI (this starts the intelligence)
    dmai = DMAIIntelligence()
    
    try:
        # Run forever (this never returns)
        dmai.run()
    except KeyboardInterrupt:
        dmai.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
