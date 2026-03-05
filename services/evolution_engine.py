#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - EVOLUTION - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EVOLUTION")

class EvolutionEngine:
    def __init__(self):
        self.generation = self.load_generation()
        self.best_score = 0
        self.improvements = []
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.evolvable_files = self.find_evolvable_files()
        
    def load_generation(self):
        try:
            with open('data/evolution_state.json', 'r') as f:
                data = json.load(f)
                return data.get('generation', 1) + 1
        except:
            return 1
    
    def find_evolvable_files(self):
        """Find Python files that can be evolved"""
        evolvable = []
        
        search_paths = [
            "voice/dmai_voice_with_learning.py",
            "voice/speech_to_text.py",
            "voice/speaker.py",
            "language_learning/processor/language_learner.py",
            "language_learning/listener/ambient_listener.py",
            "services/web_researcher.py",
            "services/dark_researcher.py",
            "services/book_reader.py",
            "services/evolution_engine.py"  # Self-evolution
        ]
        
        for path in search_paths:
            if Path(path).exists():
                evolvable.append(path)
                
        return evolvable
    
    def get_vocabulary_size(self):
        try:
            with open('language_learning/data/vocabulary.json', 'r') as f:
                return len(json.load(f))
        except:
            return 0
    
    def get_performance_metrics(self):
        vocab = self.get_vocabulary_size()
        
        if vocab < 1000:
            score = 1
        elif vocab < 10000:
            score = 2
        elif vocab < 50000:
            score = 3
        elif vocab < 100000:
            score = 4
        else:
            score = 5
            
        return {
            "vocabulary": vocab,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }
    
    def improve_file(self, filepath):
        """Make actual improvements to files"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            improvements_made = []
            original_content = content
            
            # IMPROVEMENT 1: Add retry logic to network operations
            if "requests.get" in content and "retry" not in content:
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if "requests.get" in line:
                        indent = ' ' * (len(line) - len(line.lstrip()))
                        new_lines.append(f"{indent}for retry in range(3):")
                        new_lines.append(f"{indent}    try:")
                        new_lines.append(f"{indent}        {line}")
                        new_lines.append(f"{indent}        break")
                        new_lines.append(f"{indent}    except:")
                        new_lines.append(f"{indent}        if retry == 2:")
                        new_lines.append(f"{indent}            raise")
                        new_lines.append(f"{indent}        time.sleep(1)")
                    else:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
                improvements_made.append("Added retry logic to network calls")
            
            # IMPROVEMENT 2: Add caching for repeated operations
            if "def get_" in content and "cache" not in content:
                lines = content.split('\n')
                new_lines = []
                cache_added = False
                for line in lines:
                    if "def get_" in line and "(" in line and not cache_added:
                        func_name = line.split('def ')[1].split('(')[0]
                        indent = ' ' * (len(line) - len(line.lstrip()))
                        new_lines.append(line)
                        new_lines.append(f"{indent}    # Add caching")
                        new_lines.append(f"{indent}    cache_key = f'{func_name}_{str(args) if 'args' in locals() else ''}'")
                        new_lines.append(f"{indent}    if hasattr(self, '_cache') and cache_key in self._cache:")
                        new_lines.append(f"{indent}        return self._cache[cache_key]")
                        cache_added = True
                    else:
                        new_lines.append(line)
                
                if cache_added:
                    content = '\n'.join(new_lines)
                    improvements_made.append("Added result caching")
            
            # IMPROVEMENT 3: Add better error handling
            if "try:" not in content and "except" in content:
                # Already has some error handling
                pass
            elif "try:" not in content and "def run" in content:
                lines = content.split('\n')
                new_lines = []
                in_run = False
                for line in lines:
                    if "def run" in line:
                        new_lines.append(line)
                        new_lines.append("    try:")
                        in_run = True
                    elif in_run and line.strip() and not line.startswith(' ' * 8):
                        new_lines.append("    except Exception as e:")
                        new_lines.append("        logger.error(f'Error in run loop: {e}')")
                        new_lines.append("        time.sleep(60)")
                        new_lines.append(line)
                        in_run = False
                    elif in_run:
                        new_lines.append("    " + line)
                    else:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
                improvements_made.append("Added error handling to run loop")
            
            # IMPROVEMENT 4: Add performance optimization (batch processing)
            if "for " in content and "item in" in content and "batch" not in content:
                if content.count("for ") > 5:  # Many loops
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        if "for " in line and "item in" in line:
                            new_lines.append(line)
                            indent = ' ' * (len(line) - len(line.lstrip()))
                            new_lines.append(f"{indent}    # Consider batch processing for performance")
                        else:
                            new_lines.append(line)
                    
                    content = '\n'.join(new_lines)
                    improvements_made.append("Added batch processing hint")
            
            if improvements_made and content != original_content:
                # Save backup
                backup_path = self.checkpoint_dir / f"{Path(filepath).name}.gen_{self.generation-1}"
                shutil.copy2(filepath, backup_path)
                
                # Write improved version
                with open(filepath, 'w') as f:
                    f.write(content)
                
                self.improvements.append({
                    "file": filepath,
                    "improvements": improvements_made,
                    "generation": self.generation
                })
                
                logger.info(f"✅ Improved {filepath}: {', '.join(improvements_made)}")
                return True
            else:
                logger.debug(f"No improvements needed for {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to improve {filepath}: {e}")
            return False
    
    def evolve(self):
        """Run evolution cycle - actually improves files"""
        logger.info("="*60)
        logger.info(f"🔄 EVOLUTION CYCLE {self.generation}")
        logger.info("="*60)
        
        metrics = self.get_performance_metrics()
        logger.info(f"Current score: {metrics['score']} (vocab: {metrics['vocabulary']})")
        
        improvements_made = 0
        files_improved = []
        
        for filepath in self.evolvable_files:
            logger.info(f"Analyzing {filepath}...")
            if self.improve_file(filepath):
                improvements_made += 1
                files_improved.append(filepath)
        
        # Save checkpoint
        checkpoint = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "files_improved": files_improved,
            "improvements": self.improvements[-10:] if self.improvements else []
        }
        
        with open(self.checkpoint_dir / f"gen_{self.generation}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        if improvements_made > 0:
            logger.info(f"🎉 Evolution complete! Improved {improvements_made} files")
            self.generation += 1
        else:
            logger.info("No improvements made this cycle")
        
        self.save_state()
    
    def save_state(self):
        state = {
            "generation": self.generation,
            "best_score": self.best_score,
            "total_improvements": len(self.improvements),
            "recent_improvements": self.improvements[-10:],
            "evolvable_files": self.evolvable_files,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("data/evolution_state.json", 'w') as f:
            json.dump(state, f, indent=2)
    
    def run(self):
        logger.info("🚀 Evolution Engine Started")
        logger.info(f"Found {len(self.evolvable_files)} evolvable files")
        logger.info(f"Starting at generation {self.generation}")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                self.evolve()
                
                # Evolve more frequently - every hour
                wait_time = 3600
                logger.info(f"⏰ Next evolution in {wait_time//60} minutes")
                
                # Countdown
                for i in range(wait_time, 0, -60):
                    if i % 3600 == 0:
                        logger.info(f"⏳ {i//3600}h remaining until next evolution")
                    elif i % 600 == 0:
                        logger.info(f"⏳ {i//60}m remaining")
                    time.sleep(60)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Evolution error: {e}")
                time.sleep(3600)

if __name__ == "__main__":
    engine = EvolutionEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info(f"🎯 Evolution stopped at generation {engine.generation}")
        logger.info(f"📊 Total improvements: {len(engine.improvements)}")
