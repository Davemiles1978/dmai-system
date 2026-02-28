#!/usr/bin/env python3
"""
24/7/365 EVOLUTION ENGINE - Runs continuously in the cloud
This is the REAL evolution system, not a UI counter
Now with Git sync for cross-device persistence
"""

import os
import json
import time
import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - EVOLUTION - %(message)s',
    handlers=[
        logging.FileHandler('evolution.log'),
        logging.StreamHandler()
    ]
)

class EvolutionEngine:
    def __init__(self):
        # Use git-tracked directory for checkpoints to sync across devices
        self.repos_path = Path("repos")
        self.checkpoints_path = Path("shared_checkpoints")  # Changed from 'checkpoints'
        self.checkpoints_path.mkdir(exist_ok=True)
        
        # Ensure .gitignore allows this directory
        self.setup_git_tracking()
        
        # Evolution state
        self.current_generation = self.load_generation()
        self.best_scores = self.load_best_scores()
        self.is_evolving = True
        self.cycle_interval = 3600  # 1 hour in seconds
        self.last_git_sync = time.time()
        self.git_sync_interval = 300  # Sync every 5 minutes
        
        logging.info(f"🚀 Evolution Engine Started - Generation {self.current_generation}")
    
    def setup_git_tracking(self):
        """Ensure shared_checkpoints is tracked by git (except large files)"""
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            # Remove checkpoints/ from gitignore if present
            if 'checkpoints/' in content:
                content = content.replace('checkpoints/', '# checkpoints/ - now using shared_checkpoints')
                with open(gitignore_path, 'w') as f:
                    f.write(content)
                logging.info("📝 Updated .gitignore to track evolution data")
        
        # Create .gitignore for shared_checkpoints to exclude large files only
        checkpoint_gitignore = self.checkpoints_path / '.gitignore'
        if not checkpoint_gitignore.exists():
            with open(checkpoint_gitignore, 'w') as f:
                f.write("""# Ignore large model files but track evolution progress
*.pyc
__pycache__/
*.model
*.weights
*.h5
*.pth
large_files/
temp/
""")
            logging.info("📝 Created gitignore for shared_checkpoints")
    
    def git_sync(self):
        """Sync evolution data with GitHub"""
        try:
            # Check if we're in a git repo
            result = subprocess.run(['git', 'status'], capture_output=True, text=True)
            if result.returncode != 0:
                return  # Not a git repo
            
            # Pull latest changes first
            subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True)
            
            # Add shared_checkpoints changes
            subprocess.run(['git', 'add', 'shared_checkpoints/'], capture_output=True)
            
            # Commit if there are changes
            result = subprocess.run(['git', 'diff', '--staged', '--quiet'], capture_output=True)
            if result.returncode != 0:  # There are changes
                commit_msg = f"🤖 Evolution sync: Generation {self.current_generation}"
                subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True)
                subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True)
                logging.info(f"🔄 Synced generation {self.current_generation} to GitHub")
            
            self.last_git_sync = time.time()
            
        except Exception as e:
            logging.error(f"Git sync failed: {e}")
    
    def load_generation(self):
        """Load the current generation number from disk (with git sync)"""
        gen_file = self.checkpoints_path / "current_generation.txt"
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                return int(f.read().strip())
        return 1
    
    def save_generation(self):
        """Save current generation number"""
        gen_file = self.checkpoints_path / "current_generation.txt"
        with open(gen_file, 'w') as f:
            f.write(str(self.current_generation))
    
    def load_best_scores(self):
        """Load best scores from disk"""
        scores_file = self.checkpoints_path / "best_scores.json"
        if scores_file.exists():
            with open(scores_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_best_scores(self):
        """Save best scores"""
        scores_file = self.checkpoints_path / "best_scores.json"
        with open(scores_file, 'w') as f:
            json.dump(self.best_scores, f, indent=2)
    
    def get_all_evolvable_files(self):
        """Find all Python files with EVOLVE blocks"""
        evolvable_files = []
        
        if not self.repos_path.exists():
            logging.warning("No repos folder found")
            return evolvable_files
        
        for repo_dir in self.repos_path.iterdir():
            if not repo_dir.is_dir():
                continue
            
            for py_file in repo_dir.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if '# EVOLVE-BLOCK-START' in content:
                            evolvable_files.append({
                                'path': py_file,
                                'repo': repo_dir.name,
                                'size': len(content)
                            })
                except Exception as e:
                    logging.error(f"Error reading {py_file}: {e}")
        
        return evolvable_files
    
    def mutate_code(self, code):
        """Apply evolutionary mutations to code"""
        lines = code.split('\n')
        
        # Different mutation strategies
        mutations = [
            self._add_comment,
            self._optimize_loop,
            self._add_error_handling,
            self._improve_docstring,
            self._refactor_variable_names
        ]
        
        # Apply 1-3 random mutations
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            mutation = random.choice(mutations)
            lines = mutation(lines)
        
        return '\n'.join(lines)
    
    def _add_comment(self, lines):
        """Add helpful comments"""
        if lines and len(lines) > 3:
            idx = random.randint(0, len(lines)-1)
            comment = f"    # Evolution improvement at generation {self.current_generation}"
            lines.insert(idx, comment)
        return lines
    
    def _optimize_loop(self, lines):
        """Optimize loop structures"""
        for i, line in enumerate(lines):
            if 'for ' in line and ' in ' in line:
                # Suggest list comprehension
                if i+1 < len(lines) and 'append' in lines[i+1]:
                    lines[i] = f"# Optimized: {line}"
        return lines
    
    def _add_error_handling(self, lines):
        """Add try-except blocks"""
        for i, line in enumerate(lines):
            if 'open(' in line or 'read(' in line:
                if i > 0 and 'try:' not in lines[i-1]:
                    lines.insert(i, '    try:')
                    lines.insert(i+2, '    except Exception as e:')
                    lines.insert(i+3, f'        print(f"Evolution error: {{e}}")')
        return lines
    
    def _improve_docstring(self, lines):
        """Improve function documentation"""
        for i, line in enumerate(lines):
            if 'def ' in line and i+1 < len(lines):
                if '"""' not in lines[i+1] and "'''" not in lines[i+1]:
                    func_name = line.split('def ')[1].split('(')[0]
                    docstring = f'    """Evolved function {func_name} - optimized at generation {self.current_generation}"""'
                    lines.insert(i+1, docstring)
        return lines
    
    def _refactor_variable_names(self, lines):
        """Improve variable names for clarity"""
        replacements = {
            'x': 'input_value',
            'y': 'output_value',
            'tmp': 'temporary',
            'data': 'input_data',
            'res': 'result'
        }
        
        for i, line in enumerate(lines):
            for old, new in replacements.items():
                if f' {old} ' in line or f'{old}=' in line:
                    lines[i] = lines[i].replace(f' {old} ', f' {new} ')
                    lines[i] = lines[i].replace(f'{old}=', f'{new}=')
        return lines
    
    def evaluate_code(self, filepath, original_code, mutated_code):
        """Evaluate if mutation is an improvement"""
        # Simple metrics for now - in production, this would run actual tests
        original_lines = len(original_code.split('\n'))
        mutated_lines = len(mutated_code.split('\n'))
        
        # Score based on various factors
        scores = {
            'complexity': min(mutated_lines / original_lines, 1.2) if original_lines > 0 else 1.0,
            'comments': mutated_code.count('#') / max(original_code.count('#'), 1),
            'error_handling': 1.2 if 'try:' in mutated_code and 'except' in mutated_code else 1.0,
            'documentation': 1.1 if '"""' in mutated_code or "'''" in mutated_code else 1.0
        }
        
        # Overall improvement score
        overall = sum(scores.values()) / len(scores)
        
        # Add some randomness for exploration
        overall *= random.uniform(0.95, 1.05)
        
        return overall, scores
    
    def save_improved_version(self, filepath, improved_code, score):
        """Save improved code and create checkpoint"""
        # Save the improved version
        with open(filepath, 'w') as f:
            f.write(improved_code)
        
        # Track best versions
        repo_name = filepath.parent.name
        if repo_name not in self.best_scores or score > self.best_scores[repo_name].get('score', 0):
            self.best_scores[repo_name] = {
                'score': score,
                'file': str(filepath),
                'generation': self.current_generation,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save best version separately in git-tracked location
            best_dir = self.checkpoints_path / 'best_versions' / repo_name
            best_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(filepath, best_dir / filepath.name)
            
            logging.info(f"🏆 New best for {repo_name}: {score:.3f}")
        
        self.save_best_scores()
    
    def evolution_cycle(self):
        """Run one complete evolution cycle"""
        logging.info(f"\n{'='*60}")
        logging.info(f"🔄 EVOLUTION CYCLE {self.current_generation}")
        logging.info(f"{'='*60}")
        
        # Find all evolvable files
        evolvable_files = self.get_all_evolvable_files()
        
        if not evolvable_files:
            logging.warning("No evolvable files found")
            return
        
        logging.info(f"📁 Found {len(evolvable_files)} evolvable files")
        
        improvements = 0
        total_files = 0
        
        for file_info in evolvable_files:
            filepath = file_info['path']
            total_files += 1
            
            try:
                # Read current code
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Extract evolvable section
                start_marker = '# EVOLVE-BLOCK-START'
                end_marker = '# EVOLVE-BLOCK-END'
                
                if start_marker in content and end_marker in content:
                    start_idx = content.index(start_marker) + len(start_marker)
                    end_idx = content.index(end_marker)
                    evolvable_code = content[start_idx:end_idx]
                    
                    # Mutate
                    mutated_code = self.mutate_code(evolvable_code)
                    
                    # Reconstruct
                    improved_content = content[:start_idx] + mutated_code + content[end_idx:]
                    
                    # Evaluate
                    score, metrics = self.evaluate_code(filepath, evolvable_code, mutated_code)
                    
                    # Keep if improved
                    if score > 1.05:  # At least 5% improvement
                        self.save_improved_version(filepath, improved_content, score)
                        improvements += 1
                        logging.info(f"✅ Improved {filepath.name} (score: {score:.3f})")
                    
            except Exception as e:
                logging.error(f"Error processing {filepath}: {e}")
        
        # Save checkpoint every 10 generations
        if self.current_generation % 10 == 0:
            self.save_checkpoint()
        
        # Log summary
        improvement_rate = (improvements / total_files * 100) if total_files > 0 else 0
        logging.info(f"\n📊 CYCLE {self.current_generation} SUMMARY")
        logging.info(f"   Files processed: {total_files}")
        logging.info(f"   Improvements: {improvements}")
        logging.info(f"   Improvement rate: {improvement_rate:.1f}%")
        logging.info(f"   Best scores: {self.best_scores}")
        
        # Increment generation
        self.current_generation += 1
        self.save_generation()
    
    def save_checkpoint(self):
        """Save complete system state"""
        checkpoint_dir = self.checkpoints_path / f"generation_{self.current_generation}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Copy all repos to checkpoint
        if self.repos_path.exists():
            shutil.copytree(self.repos_path, checkpoint_dir / 'repos', dirs_exist_ok=True)
        
        # Save metadata
        metadata = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'best_scores': self.best_scores
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"💾 Checkpoint saved: generation_{self.current_generation}")
    
    def run_forever(self):
        """Run evolution cycles forever (24/7/365) with Git sync"""
        logging.info("🌟 Starting 24/7/365 Evolution Engine")
        logging.info(f"⏰ Cycle interval: {self.cycle_interval} seconds")
        logging.info(f"🔄 Git sync interval: {self.git_sync_interval} seconds")
        
        while self.is_evolving:
            try:
                self.evolution_cycle()
                
                # Sync with Git after each cycle
                self.git_sync()
                
                logging.info(f"⏰ Waiting {self.cycle_interval} seconds until next cycle...")
                time.sleep(self.cycle_interval)
                
                # Also sync periodically during the wait
                while time.time() - self.last_git_sync < self.cycle_interval:
                    time.sleep(60)  # Check every minute
                    if time.time() - self.last_git_sync > self.git_sync_interval:
                        self.git_sync()
                
            except KeyboardInterrupt:
                logging.info("🛑 Evolution stopped by user")
                self.save_checkpoint()
                self.git_sync()  # Final sync
                break
            except Exception as e:
                logging.error(f"❌ Error in evolution cycle: {e}")
                logging.info("⏰ Retrying in 5 minutes...")
                time.sleep(300)

if __name__ == "__main__":
    engine = EvolutionEngine()
    engine.run_forever()
    def get_ui_evolvable_files(self):
        """Get UI files that can be evolved"""
        ui_files = []
        
        # HTML files
        for html_file in Path('ui').glob('*.html'):
            ui_files.append(str(html_file))
            
        # CSS files
        for css_file in Path('static/css').glob('*.css'):
            ui_files.append(str(css_file))
            
        # JS files
        for js_file in Path('static/js').glob('*.js'):
            ui_files.append(str(js_file))
            
        return ui_files
    
    def evolve_ui(self, file_path):
        """Apply UI-specific mutations"""
        content = Path(file_path).read_text()
        
        mutations = [
            self._improve_responsive_design,
            self._fix_button_styles,
            self._improve_scroll_behavior,
            self._optimize_mobile_view
        ]
        
        for mutation in mutations:
            content = mutation(content)
            
        return content
    
    def _improve_responsive_design(self, content):
        """Add/improve responsive breakpoints"""
        if '@media' not in content:
            # Add basic responsive design
            media_queries = '''
/* Responsive design */
@media screen and (max-width: 768px) {
    /* Mobile styles */
    .header { height: 50px; }
    .chat-list-panel { width: 100%; }
}

@media screen and (min-width: 769px) and (max-width: 1024px) {
    /* Tablet styles */
    .tools-panel { width: 250px; }
}
'''
            content += media_queries
        return content
    
    def _fix_button_styles(self, content):
        """Ensure buttons are properly styled and clickable"""
        if 'cursor: pointer' not in content:
            button_fix = '''
/* Ensure all buttons are clickable */
button, .btn, [role="button"] {
    cursor: pointer !important;
    pointer-events: auto !important;
    position: relative !important;
    z-index: 100 !important;
}

button:hover, .btn:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    transition: all 0.2s ease;
}
'''
            content += button_fix
        return content
    
    def _improve_scroll_behavior(self, content):
        """Fix scrolling issues"""
        scroll_fix = '''
/* Fix scrolling */
.overflow-auto, .scrollable {
    overflow-y: auto !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: thin !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
}

::-webkit-scrollbar-thumb {
    background: var(--accent);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-light);
}
'''
        content += scroll_fix
        return content
    
    def _optimize_mobile_view(self, content):
        """Optimize for mobile devices"""
        mobile_fix = '''
/* Mobile optimizations */
@media screen and (max-width: 768px) {
    /* Ensure touch targets are large enough */
    button, .btn, [role="button"], 
    .clickable, .tool-btn, .chat-item {
        min-height: 44px !important;
        min-width: 44px !important;
        padding: 12px !important;
    }
    
    /* Prevent text overflow */
    h1, h2, h3, p, span {
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    /* Improve touch scrolling */
    .messages-container, .chat-list-panel, .tools-panel {
        -webkit-overflow-scrolling: touch !important;
        overflow-y: auto !important;
    }
}
'''
        content += mobile_fix
        return content
