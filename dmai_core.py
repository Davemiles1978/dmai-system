#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import subprocess
import threading
import requests
import random
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CORE - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/core.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DMAI-CORE")

class DMAICore:
    def __init__(self):
        self.services = {
            "voice": {
                "script": "voice/dmai_voice_with_learning.py",
                "process": None,
                "restart_count": 0,
                "last_restart": None,
                "enabled": True
            },
            "learner": {
                "script": "services/vocabulary_learner.py",
                "process": None,
                "restart_count": 0,
                "last_restart": None,
                "enabled": True
            },
            "web_researcher": {
                "script": "services/web_researcher.py",
                "process": None,
                "restart_count": 0,
                "last_restart": None,
                "enabled": True
            },
            "dark_researcher": {
                "script": "services/dark_researcher.py",
                "process": None,
                "restart_count": 0,
                "last_restart": None,
                "enabled": True
            },
            "evolution": {
                "script": "services/evolution_engine.py",
                "process": None,
                "restart_count": 0,
                "last_restart": None,
                "enabled": True
            }
        }
        self.running = True
        self.evolution_generation = 1
        self.vocabulary_size = 0
        self.knowledge_base = {}
        
    def start_service(self, name, config):
        try:
            script_path = Path(config["script"])
            if not script_path.exists():
                logger.error(f"Service script not found: {script_path}")
                return False
                
            logger.info(f"Starting {name} service...")
            
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=open(f'logs/{name}.log', 'a'),
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True
            )
            
            config["process"] = process
            config["last_restart"] = datetime.now()
            logger.info(f"{name} started (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False
    
    def check_services(self):
        for name, config in self.services.items():
            if not config["enabled"]:
                continue
                
            process = config.get("process")
            
            if process:
                poll = process.poll()
                if poll is not None:
                    logger.warning(f"{name} died with code {poll}")
                    
                    config["restart_count"] += 1
                    wait_time = min(30 * (2 ** config["restart_count"]), 3600)
                    
                    logger.info(f"Restarting {name} in {wait_time}s...")
                    time.sleep(wait_time)
                    
                    self.start_service(name, config)
            else:
                self.start_service(name, config)
    
    def update_vocabulary_size(self):
        try:
            vocab_file = Path("language_learning/data/vocabulary.json")
            if vocab_file.exists():
                with open(vocab_file, 'r') as f:
                    vocab = json.load(f)
                self.vocabulary_size = len(vocab)
                logger.info(f"Vocabulary: {self.vocabulary_size} words")
        except:
            pass
    
    def self_improve(self):
        logger.info("Running self-improvement cycle...")
        
        improvements = []
        
        if self.vocabulary_size > 10000 and self.evolution_generation < 2:
            improvements.append("Enable deep learning mode")
            self.evolution_generation = 2
            
        if self.vocabulary_size > 50000 and self.evolution_generation < 3:
            improvements.append("Enable cross-domain knowledge synthesis")
            self.evolution_generation = 3
            
        if self.vocabulary_size > 100000 and self.evolution_generation < 4:
            improvements.append("Enable autonomous research direction")
            self.evolution_generation = 4
            
        if improvements:
            logger.info(f"DMAI evolved to generation {self.evolution_generation}")
            for imp in improvements:
                logger.info(f"  • {imp}")
                
            evo_file = Path("data/evolution.json")
            evo_data = {
                "generation": self.evolution_generation,
                "timestamp": datetime.now().isoformat(),
                "improvements": improvements,
                "vocabulary": self.vocabulary_size
            }
            with open(evo_file, 'w') as f:
                json.dump(evo_data, f, indent=2)
    
    def research_web(self):
        try:
            topics = [
                "artificial intelligence", "machine learning", "neural networks",
                "consciousness", "quantum computing", "emerging technology",
                "scientific breakthroughs", "future of computing"
            ]
            
            topic = random.choice(topics)
            logger.info(f"Researching: {topic}")
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Research error: {e}")
    
    def run(self):
        logger.info("="*60)
        logger.info("DMAI CORE INITIALIZED")
        logger.info("="*60)
        
        cycle = 0
        while self.running:
            try:
                cycle += 1
                logger.debug(f"Core cycle {cycle}")
                
                self.check_services()
                
                if cycle % 10 == 0:
                    self.update_vocabulary_size()
                
                if cycle % 120 == 0:
                    self.self_improve()
                
                if cycle % 30 == 0:
                    self.research_web()
                
                if cycle % 10 == 0:
                    self.print_status()
                
                time.sleep(30)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Core error: {e}")
                time.sleep(60)
        
        logger.info("DMAI Core shutting down")
    
    def print_status(self):
        print("\n" + "="*70)
        print(f"DMAI STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"Vocabulary: {self.vocabulary_size} words")
        print(f"Generation: {self.evolution_generation}")
        running = sum(1 for s in self.services.values() if s.get('process'))
        print(f"Services running: {running}")
        print("="*70)

if __name__ == "__main__":
    core = DMAICore()
    try:
        core.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        for name, config in core.services.items():
            if config.get("process"):
                config["process"].terminate()
        logger.info("DMAI Core stopped")
