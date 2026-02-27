#!/usr/bin/env python3
"""
API Server - Connects UI to Evolution Engine
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import threading
import time

class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/evolution-stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Read current generation from evolution engine
            stats = self.get_evolution_stats()
            self.wfile.write(json.dumps(stats).encode())
            
        elif self.path == '/api/best-version':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get the best evolved version
            best = self.get_best_version()
            self.wfile.write(json.dumps(best).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/submit-guidance':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            guidance = json.loads(post_data)
            
            # Save guidance for next evolution cycle
            self.save_guidance(guidance)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'received'}).encode())
    
    def get_evolution_stats(self):
        """Get current evolution stats from engine"""
        stats = {
            'generation': 1,
            'bestScore': 0.0,
            'repos': 0,
            'status': 'running'
        }
        
        # Read from evolution engine files
        checkpoints = Path('checkpoints')
        if checkpoints.exists():
            # Get current generation
            gen_file = checkpoints / 'current_generation.txt'
            if gen_file.exists():
                with open(gen_file, 'r') as f:
                    stats['generation'] = int(f.read().strip())
            
            # Get best scores
            scores_file = checkpoints / 'best_scores.json'
            if scores_file.exists():
                with open(scores_file, 'r') as f:
                    scores = json.load(f)
                    if scores:
                        stats['bestScore'] = max([s.get('score', 0) for s in scores.values()])
        
        # Count repos
        repos = Path('repos')
        if repos.exists():
            stats['repos'] = len([d for d in repos.iterdir() if d.is_dir()])
        
        return stats
    
    def get_best_version(self):
        """Get the best evolved version of each AI"""
        best = {
            'generation': 0,
            'models': []
        }
        
        best_dir = Path('checkpoints/best_versions')
        if best_dir.exists():
            for repo_dir in best_dir.iterdir():
                if repo_dir.is_dir():
                    py_files = list(repo_dir.glob('*.py'))
                    if py_files:
                        best['models'].append({
                            'name': repo_dir.name,
                            'files': [f.name for f in py_files],
                            'path': str(repo_dir)
                        })
        
        return best
    
    def save_guidance(self, guidance):
        """Save human guidance for next evolution cycle"""
        guidance_file = Path('guidance/queue.json')
        guidance_file.parent.mkdir(exist_ok=True)
        
        if guidance_file.exists():
            with open(guidance_file, 'r') as f:
                queue = json.load(f)
        else:
            queue = []
        
        queue.append({
            'timestamp': time.time(),
            'guidance': guidance,
            'processed': False
        })
        
        with open(guidance_file, 'w') as f:
            json.dump(queue[-100:], f, indent=2)  # Keep last 100

def run_api_server():
    server = HTTPServer(('0.0.0.0', 8889), APIHandler)
    print(f"🌐 API Server running on port 8889")
    server.serve_forever()

if __name__ == "__main__":
    run_api_server()
