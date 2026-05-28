"""Meta-learner - uses controlled threads, no unlimited spawning"""
import logging
from components.thread_manager import thread_manager

logger = logging.getLogger(__name__)

class MetaLearnerFixed:
    def __init__(self):
        self.insights = []
        logger.info("🧠 Meta-learner initialized (controlled thread mode)")
    
    def analyze_pattern(self, pattern_data):
        """Analyze learning patterns - runs in controlled thread"""
        # Analysis logic here
        logger.info(f"Analyzing pattern: {pattern_data[:50]}...")
        return {"insight": "Pattern analyzed", "confidence": 0.8}
    
    def schedule_analysis(self, data):
        """Schedule analysis without spawning new threads"""
        thread_manager.submit(lambda: self.analyze_pattern(data))

meta_learner = MetaLearnerFixed()
