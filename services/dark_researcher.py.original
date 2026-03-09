#!/usr/bin/env python3
"""
Dark Web Researcher - DMAI Service
Researches security topics and dark web terminology
Survives independently with auto-restart capability
"""

import os
import sys
import time
import json
import logging
import traceback
import random
from datetime import datetime
from pathlib import Path

# Ensure logs directory exists
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

# Configure logging FIRST so we can see all errors
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detail
    format='%(asctime)s - DARK - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'dark_researcher.log', mode='a'),
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ]
)
logger = logging.getLogger("DARK_RESEARCHER")
logger.info("=" * 60)
logger.info("DARK RESEARCHER STARTING UP")
logger.info(f"PID: {os.getpid()}")
logger.info(f"Time: {datetime.now().isoformat()}")
logger.info("=" * 60)

# Try to import language learner with detailed error handling
try:
    logger.info("Attempting to import LanguageLearner...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logger.info(f"Python path: {sys.path}")
    
    from language_learning.processor.language_learner import LanguageLearner
    logger.info("✅ Successfully imported LanguageLearner")
    LANGUAGE_LEARNER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import LanguageLearner: {e}")
    logger.error(traceback.format_exc())
    LANGUAGE_LEARNER_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Unexpected error importing LanguageLearner: {e}")
    logger.error(traceback.format_exc())
    LANGUAGE_LEARNER_AVAILABLE = False

class DarkWebResearcher:
    """Researches dark web topics and security terminology"""
    
    def __init__(self):
        logger.info("Initializing DarkWebResearcher...")
        self.learner = None
        self.tor_running = False
        self.words_learned = 0
        self.cycle_count = 0
        self.errors = 0
        self.start_time = time.time()
        
        # Topics to research
        self.dark_topics = [
            "cybersecurity", "encryption", "privacy", "anonymity",
            "zero day exploits", "vulnerabilities", "threat intelligence",
            "darknet markets", "cryptocurrency", "blockchain",
            "underground forums", "hacking techniques", "security research",
            "penetration testing", "network security", "malware analysis",
            "digital forensics", "cryptography", "Tor network", "VPN protocols"
        ]
        
        # Initialize language learner if available
        if LANGUAGE_LEARNER_AVAILABLE:
            try:
                logger.info("Initializing LanguageLearner...")
                self.learner = LanguageLearner()
                logger.info("✅ LanguageLearner initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LanguageLearner: {e}")
                logger.error(traceback.format_exc())
                self.learner = None
        else:
            logger.warning("⚠️ LanguageLearner not available - running in standalone mode")
        
        # Check Tor availability
        self.check_tor()
        logger.info("DarkWebResearcher initialization complete")
        
    def check_tor(self):
        """Check if Tor proxy is available"""
        try:
            logger.debug("Checking Tor availability...")
            import requests
            
            # Try with Tor proxy
            session = requests.session()
            session.proxies = {
                'http': 'socks5h://127.0.0.1:9050',
                'https': 'socks5h://127.0.0.1:9050'
            }
            session.timeout = 10
            
            try:
                response = session.get('http://check.torproject.org', timeout=10)
                self.tor_running = True
                logger.info("✅ Tor is running and available")
                return True
            except:
                # Try without Tor
                response = requests.get('https://api.ipify.org?format=json', timeout=10)
                self.tor_running = False
                logger.warning("⚠️ Tor not available - using clearnet only")
                return False
                
        except ImportError:
            logger.warning("⚠️ Requests library not available - cannot check Tor")
            self.tor_running = False
            return False
        except Exception as e:
            logger.error(f"❌ Error checking Tor: {e}")
            self.tor_running = False
            return False
    
    def research_topic(self, topic):
        """Research a topic from available sources"""
        try:
            logger.info(f"Researching topic: {topic}")
            
            # Try Wikipedia API
            try:
                import requests
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
                
                for retry in range(3):
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            text = data.get('extract', '')
                            
                            if text and len(text) > 50:
                                logger.info(f"Got {len(text)} chars from Wikipedia for '{topic}'")
                                
                                # Process with language learner if available
                                if self.learner:
                                    try:
                                        result = self.learner.process_text(
                                            text, 
                                            source=f"dark_research_{topic}"
                                        )
                                        if result and result.get("new_words", 0) > 0:
                                            new = result["new_words"]
                                            self.words_learned += new
                                            logger.info(f"✅ Learned {new} new words from '{topic}'")
                                    except Exception as e:
                                        logger.error(f"Error processing text with learner: {e}")
                                else:
                                    logger.info(f"📚 Would learn from: {topic} (learner unavailable)")
                                
                                return True
                            else:
                                logger.warning(f"Text too short for '{topic}'")
                        else:
                            logger.warning(f"Wikipedia returned {response.status_code} for '{topic}'")
                            
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Request error (attempt {retry+1}/3): {e}")
                        if retry < 2:
                            time.sleep(2)
                            
            except ImportError:
                logger.warning("Requests library not available - cannot research")
                return False
                
        except Exception as e:
            logger.error(f"Research error for '{topic}': {e}")
            logger.debug(traceback.format_exc())
            
        return False
    
    def run(self):
        """Main execution loop - NEVER EXITS"""
        logger.info("🚀 Dark Researcher main loop started")
        logger.info(f"Topics available: {len(self.dark_topics)}")
        
        # Main infinite loop
        while True:
            try:
                # Inner research loop
                self.cycle_count += 1
                logger.info(f"🔍 Starting research cycle #{self.cycle_count}")
                
                # Select random topic
                topic = random.choice(self.dark_topics)
                
                # Research the topic
                self.research_topic(topic)
                
                # Calculate wait time (10-30 minutes)
                wait_time = random.randint(600, 1800)
                logger.info(f"⏳ Next research cycle in {wait_time//60} minutes")
                logger.info(f"📊 Stats: {self.words_learned} words learned, {self.errors} errors")
                
                # Sleep in small chunks to remain responsive
                chunks = wait_time
                for i in range(chunks):
                    time.sleep(1)
                    # Log every 5 minutes during sleep
                    if i > 0 and i % 300 == 0:
                        logger.debug(f"Sleeping... {wait_time - i} seconds remaining")
                        
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt received")
                break
            except Exception as e:
                self.errors += 1
                logger.error(f"💥 Critical error in main loop: {e}")
                logger.error(traceback.format_exc())
                
                # Wait before restarting cycle
                logger.info("Restarting research cycle in 30 seconds...")
                time.sleep(30)
        
        logger.info(f"Dark researcher stopped after {self.cycle_count} cycles")
        logger.info(f"Total words learned: {self.words_learned}")

def main():
    """Entry point with exception catching"""
    researcher = None
    try:
        researcher = DarkWebResearcher()
        researcher.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.critical(f"FATAL UNHANDLED ERROR: {e}")
        logger.critical(traceback.format_exc())
    finally:
        if researcher:
            logger.info(f"Final stats: {researcher.words_learned} words learned")
        logger.info("Dark researcher process exiting")
        # Small delay to allow logs to flush
        time.sleep(0.5)

if __name__ == "__main__":
    main()