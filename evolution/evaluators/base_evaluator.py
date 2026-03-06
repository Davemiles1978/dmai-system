#!/usr/bin/env python3
"""
Base Evaluator Template for DMAI Evolution System
All AI evaluators inherit from this base class
"""

import os
import sys
import json
import time
import hashlib
import importlib.util
from datetime import datetime
from pathlib import Path

class BaseEvaluator:
    def __init__(self, name, version="1.0.0"):
        self.name = name
        self.version = version
        self.evolution_dir = "/Users/davidmiles/Desktop/AI-Evolution-System/evolution"
        self.log_file = f"{self.evolution_dir}/logs/{name}_evaluations.log"
        self.ensure_logs()
    
    def ensure_logs(self):
        """Ensure log directory exists"""
        log_dir = Path(f"{self.evolution_dir}/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_evaluation(self, target, score, suggestions):
        """Log evaluation results"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'evaluator': self.name,
            'target': target,
            'score': score,
            'suggestions': suggestions,
            'version': self.version
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def analyze_code(self, filepath):
        """Basic code analysis - override for AI-specific analysis"""
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            
            lines = code.split('\n')
            
            # Basic metrics
            analysis = {
                'file': filepath,
                'total_lines': len(lines),
                'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
                'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
                'blank_lines': len([l for l in lines if not l.strip()]),
                'functions': len([l for l in lines if 'def ' in l]),
                'classes': len([l for l in lines if 'class ' in l]),
                'imports': len([l for l in lines if 'import ' in l]),
                'file_hash': hashlib.md5(code.encode()).hexdigest()
            }
            
            return analysis
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return None
    
    def generate_suggestions(self, analysis):
        """Generate improvement suggestions - override with AI logic"""
        suggestions = []
        
        if analysis:
            if analysis['functions'] > 20:
                suggestions.append("Consider breaking down large functions")
            if analysis['classes'] > 10:
                suggestions.append("Consider separating classes into modules")
            if analysis['comment_lines'] < analysis['code_lines'] * 0.1:
                suggestions.append("Add more comments for clarity")
            if analysis['blank_lines'] < 10:
                suggestions.append("Add whitespace for readability")
        
        return suggestions
    
    def calculate_score(self, analysis):
        """Calculate improvement score - override with AI scoring"""
        if not analysis:
            return 50
        
        score = 70  # Base score
        
        # Adjust based on metrics
        if analysis['functions'] > 5:
            score += 5
        if analysis['classes'] > 2:
            score += 5
        if analysis['comment_lines'] > analysis['code_lines'] * 0.15:
            score += 10
        if analysis['blank_lines'] > analysis['total_lines'] * 0.1:
            score += 5
        
        return min(score, 100)
    
    def improve_code(self, filepath, suggestions):
        """Apply improvements to code - override with AI improvement logic"""
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            
            # Create backup
            backup_path = f"{filepath}.{self.name}.backup"
            with open(backup_path, 'w') as f:
                f.write(code)
            
            # Apply improvements (basic example - override for real improvements)
            improved_code = code
            
            # Add metadata about improvement
            header = f"\n# Improved by {self.name} Evaluator v{self.version}\n"
            header += f"# Date: {datetime.now().isoformat()}\n"
            header += f"# Suggestions applied: {len(suggestions)}\n"
            
            if '# IMPROVEMENT METADATA' not in code:
                improved_code = header + improved_code
            
            # Write improved code
            with open(filepath, 'w') as f:
                f.write(improved_code)
            
            return True
        except Exception as e:
            print(f"Error improving {filepath}: {e}")
            return False
    
    def evaluate(self, target_path):
        """Main evaluation method"""
        print(f"🤖 {self.name} evaluating: {target_path}")
        
        # Analyze
        analysis = self.analyze_code(target_path)
        
        # Generate suggestions
        suggestions = self.generate_suggestions(analysis)
        
        # Calculate score
        score = self.calculate_score(analysis)
        
        # Apply improvements if score is below threshold
        improved = False
        if score < 85 and suggestions:
            improved = self.improve_code(target_path, suggestions)
        
        # Log evaluation
        self.log_evaluation(target_path, score, len(suggestions))
        
        result = {
            'evaluator': self.name,
            'target': target_path,
            'score': score,
            'suggestions': suggestions,
            'improved': improved,
            'timestamp': datetime.now().isoformat(),
            'version': self.version
        }
        
        return result
    
    def __repr__(self):
        return f"<Evaluator {self.name} v{self.version}>"
