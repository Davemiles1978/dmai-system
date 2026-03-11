#!/usr/bin/env python3
"""Fresh Blood Harvester - Finds new AI systems, ideas, and capabilities from the internet"""

import requests
import json
import time
import random
import os
import logging
from pathlib import Path
from datetime import datetime
from pathlib import Path
import hashlib

class FreshBloodHarvester:
    """Harvests fresh blood from GitHub, HuggingFace, ArXiv, and Dark Web sources"""
    
    def __init__(self):
        self.seen_file = Path("data/evolution/fresh_blood_seen.json")
        self.results_file = Path("data/evolution/fresh_blood_candidates.json")
        self.load_state()
        
        # GitHub tokens (if available)
        self.github_token = os.environ.get('GITHUB_TOKEN', '')
        
    def load_state(self):
        """Load seen repositories to avoid duplicates"""
        if self.seen_file.exists():
            with open(self.seen_file) as f:
                self.seen = json.load(f)
        else:
            self.seen = {"github": [], "huggingface": [], "arxiv": [], "darkweb": []}
            
        if self.results_file.exists():
            with open(self.results_file) as f:
                self.candidates = json.load(f)
        else:
            self.candidates = []
    
    def save_state(self):
        """Save seen repositories"""
        with open(self.seen_file, 'w') as f:
            json.dump(self.seen, f, indent=2)
        with open(self.results_file, 'w') as f:
            json.dump(self.candidates, f, indent=2)
    
    def harvest_github(self, max_items=5):
        """Harvest trending AI repositories from GitHub"""
        print("🌐 Harvesting GitHub...")
        
        # Search queries for AI/ML repositories
        queries = [
            "topic:ai topic:machine-learning stars:>100",
            "topic:llm topic:language-model",
            "topic:agents topic:autonomous",
            "topic:rag topic:retrieval",
            "topic:embeddings topic:vector",
        ]
        
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        new_items = []
        for query in random.sample(queries, min(len(queries), 3)):
            try:
                url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.ok:
                    data = response.json()
                    for repo in data.get('items', [])[:max_items]:
                        repo_id = repo['full_name']
                        if repo_id not in self.seen['github']:
                            capabilities = self._infer_github_capabilities(repo)
                            
                            new_items.append({
                                "id": f"github_{repo_id}".replace('/', '_'),
                                "name": f"📦 {repo['name']}",
                                "full_name": repo_id,
                                "type": "external",
                                "source": "github",
                                "url": repo['html_url'],
                                "description": repo['description'][:200] if repo['description'] else "",
                                "stars": repo['stargazers_count'],
                                "freshness": "new",
                                "capabilities": capabilities,
                                "complexity": min(0.5, repo['stargazers_count'] / 10000),
                                "discovered": datetime.now().isoformat()
                            })
                            self.seen['github'].append(repo_id)
                            
            except Exception as e:
                print(f"⚠️ GitHub harvest error: {e}")
        
        return new_items
    
    def _infer_github_capabilities(self, repo):
        """Infer capabilities from repo name, description, and topics"""
        caps = []
        text = (repo.get('name', '') + ' ' + 
                repo.get('description', '') + ' ' + 
                ' '.join(repo.get('topics', []))).lower()
        
        # Language models
        if any(word in text for word in ['llm', 'language model', 'gpt', 'transformer']):
            caps.append("language_model")
        if any(word in text for word in ['rag', 'retrieval', 'context']):
            caps.append("retrieval")
        if any(word in text for word in ['agent', 'autonomous', 'tool']):
            caps.append("agent_framework")
        if any(word in text for word in ['embedding', 'vector']):
            caps.append("embeddings")
        if any(word in text for word in ['vision', 'image', 'clip']):
            caps.append("vision")
        if any(word in text for word in ['audio', 'speech', 'voice']):
            caps.append("audio")
        if any(word in text for word in ['code', 'programming', 'coder']):
            caps.append("coding")
        if any(word in text for word in ['optimization', 'efficient', 'fast']):
            caps.append("optimization")
        if any(word in text for word in ['testing', 'evaluation', 'benchmark']):
            caps.append("evaluation")
        
        return caps if caps else ["general_ai"]
    
    def harvest_huggingface(self, max_items=3):
        """Harvest trending models from HuggingFace"""
        print("🤗 Harvesting HuggingFace...")
        
        try:
            response = requests.get(
                "https://huggingface.co/api/trending",
                timeout=10,
                headers={"User-Agent": "DMAI-FreshBlood/1.0"}
            )
            
            if response.ok:
                models = response.json()[:max_items]
                new_items = []
                
                for model in models:
                    model_id = model.get('id', '')
                    if model_id and model_id not in self.seen['huggingface']:
                        # Extract capabilities from model tags
                        tags = model.get('tags', [])
                        caps = []
                        if 'text-generation' in tags:
                            caps.append('language_model')
                        if 'image-classification' in tags:
                            caps.append('vision')
                        if 'automatic-speech-recognition' in tags:
                            caps.append('audio')
                        
                        new_items.append({
                            "id": f"hf_{model_id.replace('/', '_')}",
                            "name": f"🤗 {model_id.split('/')[-1]}",
                            "full_name": model_id,
                            "type": "external",
                            "source": "huggingface",
                            "url": f"https://huggingface.co/{model_id}",
                            "description": model.get('description', '')[:200],
                            "downloads": model.get('downloads', 0),
                            "freshness": "new",
                            "capabilities": caps if caps else ["ml_model"],
                            "complexity": 0.4,
                            "discovered": datetime.now().isoformat()
                        })
                        self.seen['huggingface'].append(model_id)
                
                return new_items
        except Exception as e:
            print(f"⚠️ HuggingFace harvest error: {e}")
        
        return []
    
    def harvest_arxiv(self, max_items=2):
        """Harvest recent AI papers from ArXiv"""
        print("📄 Harvesting ArXiv...")
        
        # Search for recent AI/ML papers
        categories = ['cs.AI', 'cs.LG', 'cs.CL']
        new_items = []
        
        try:
            for cat in categories:
                url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results=2"
                response = requests.get(url, timeout=15)
                
                if response.ok:
                    # Parse XML (simplified - in production use proper XML parser)
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.text)
                    
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                        
                        if arxiv_id not in self.seen['arxiv']:
                            title = entry.find('{http://www.w3.org/2005/Atom}title').text
                            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                            
                            # Infer capabilities from title/abstract
                            caps = []
                            if any(word in title.lower() for word in ['llm', 'language', 'transformer']):
                                caps.append('language_model')
                            if any(word in title.lower() for word in ['agent', 'autonomous']):
                                caps.append('agent_framework')
                            if any(word in title.lower() for word in ['efficient', 'optimization']):
                                caps.append('optimization')
                            
                            new_items.append({
                                "id": f"arxiv_{arxiv_id}",
                                "name": f"📄 {title[:50]}...",
                                "type": "idea",
                                "source": "arxiv",
                                "url": f"https://arxiv.org/abs/{arxiv_id}",
                                "title": title,
                                "summary": summary[:500],
                                "freshness": "new",
                                "capabilities": caps,
                                "complexity": 0.6,
                                "discovered": datetime.now().isoformat()
                            })
                            self.seen['arxiv'].append(arxiv_id)
                            
        except Exception as e:
            print(f"⚠️ ArXiv harvest error: {e}")
        
        return new_items
    
    def harvest_darkweb(self, max_items=1):
        """Harvest from dark web sources (conceptual - would need Tor integration)"""
        print("🌑 Dark web harvesting would require Tor...")
        # This is a placeholder - actual dark web integration would need Tor
        return []
    
    def run_harvest_cycle(self):
        """Run one harvest cycle and return new candidates"""
        print("\n" + "="*60)
        print("🌐 FRESH BLOOD HARVEST CYCLE")
        print("="*60)
        
        all_new = []
        
        # Harvest from all sources
        all_new.extend(self.harvest_github())
        all_new.extend(self.harvest_huggingface())
        all_new.extend(self.harvest_arxiv())
        # all_new.extend(self.harvest_darkweb())  # Disabled for now
        
        # Add to candidates
        self.candidates.extend(all_new)
        
        # Keep only last 50 candidates
        if len(self.candidates) > 50:
            self.candidates = self.candidates[-50:]
        
        self.save_state()
        
        print(f"\n📊 Harvest complete: {len(all_new)} new candidates found")
        print(f"📈 Total candidates in queue: {len(self.candidates)}")
        
        return all_new
    
    def get_fresh_blood_for_evolution(self, max_items=2):
        """Get fresh blood candidates for the evolution engine"""
        if not self.candidates:
            return []
        
        # Return the newest candidates
        result = self.candidates[-max_items:]
        # Remove them from queue
        self.candidates = self.candidates[:-max_items]
        self.save_state()
        
        return result

if __name__ == "__main__":
    harvester = FreshBloodHarvester()
    harvester.run_harvest_cycle()
