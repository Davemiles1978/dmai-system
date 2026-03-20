#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from music_learner import MusicLearner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MUSIC - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/music_learner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MUSIC")

class MusicLearningService:
    def __init__(self):
        self.learner = MusicLearner()
        self.learning_cycle = 0
        
    def run(self):
        logger.info("Music Learner started")
        
        while True:
            try:
                self.learning_cycle += 1
                logger.info(f"Music learning cycle {self.learning_cycle}")
                
                stats = self.learner.get_stats()
                logger.info(f"Known artists: {stats['total_artists_known']}")
                
                # Develop taste every 10 cycles
                if self.learning_cycle % 10 == 0:
                    logger.info("Developing music taste...")
                    develop_dmai_taste()
                
                time.sleep(3600)  # Run every hour
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Music learning error: {e}")
                time.sleep(3600)

if __name__ == "__main__":
    service = MusicLearningService()
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Music learner stopped")
