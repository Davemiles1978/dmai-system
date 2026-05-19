"""
Deep Research Integration - Connects autonomous research to DMAI's learning loop
"""
import requests
import json
import time
import threading
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

class DeepResearchIntegrator:
    """Integrates deep research with DMAI's evolution system"""
    
    def __init__(self, si_core=None, stage_learner=None):
        self.si_core = si_core
        self.stage_learner = stage_learner
        self.research_queue = []
        self.research_results = {}
        self.is_running = False
        self.research_thread = None
        
    def search_arxiv(self, topic: str, max_results: int = 3) -> List[Dict]:
        """Search ArXiv for research papers"""
        try:
            import arxiv
            client = arxiv.Client()
            search = arxiv.Search(
                query=f"ti:{topic} OR abs:{topic}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for paper in client.results(search):
                papers.append({
                    'title': paper.title,
                    'summary': paper.summary[:800],
                    'url': paper.entry_id,
                    'published': paper.published.isoformat() if paper.published else None
                })
            return papers
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []
    
    def search_github(self, topic: str) -> List[Dict]:
        """Search GitHub for repositories"""
        try:
            url = f"https://api.github.com/search/repositories?q={topic}&sort=stars&order=desc&per_page=3"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                repos = response.json().get('items', [])
                return [{
                    'name': repo['full_name'],
                    'description': repo['description'][:500] if repo['description'] else '',
                    'stars': repo['stargazers_count'],
                    'url': repo['html_url']
                } for repo in repos]
        except Exception as e:
            print(f"GitHub search error: {e}")
        return []
    
    def search_huggingface(self, topic: str) -> List[Dict]:
        """Search HuggingFace for models"""
        try:
            url = f"https://huggingface.co/api/models?search={topic}&limit=3"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                models = response.json()
                return [{
                    'name': m.get('modelId', ''),
                    'downloads': m.get('downloads', 0),
                    'likes': m.get('likes', 0),
                    'url': f"https://huggingface.co/{m.get('modelId', '')}"
                } for m in models if isinstance(m, dict)]
        except Exception as e:
            print(f"HuggingFace search error: {e}")
        return []
    
    def research_topic(self, topic: str) -> Dict:
        """Perform deep research on a topic"""
        print(f"🔬 Deep researching: {topic}")
        
        # Gather from multiple sources
        sources = {
            'arxiv': self.search_arxiv(topic),
            'github': self.search_github(topic),
            'huggingface': self.search_huggingface(topic)
        }
        
        # Synthesize knowledge
        knowledge = self.synthesize_knowledge(topic, sources)
        
        # Store in SI Core
        if self.si_core and hasattr(self.si_core, 'add_insight'):
            self.si_core.add_insight(
                insight_text=knowledge['text'][:2000],
                entity_type='research',
                entities=[topic, 'deep_research'],
                relationship='mastered_knowledge',
                source_topic=topic,
                source_url=sources['arxiv'][0]['url'] if sources['arxiv'] else None,
                confidence=0.85
            )
        
        # Store result
        result = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'sources': {
                'arxiv_count': len(sources['arxiv']),
                'github_count': len(sources['github']),
                'huggingface_count': len(sources['huggingface'])
            },
            'knowledge': knowledge,
            'mastery_score': knowledge['mastery_score']
        }
        
        self.research_results[topic] = result
        return result
    
    def synthesize_knowledge(self, topic: str, sources: Dict) -> Dict:
        """Synthesize research into substantive knowledge"""
        knowledge_parts = []
        
        # Add topic introduction
        knowledge_parts.append(f"# {topic} - Deep Research Summary\n")
        
        # Add ArXiv findings
        if sources['arxiv']:
            knowledge_parts.append(f"\n## Research Papers ({len(sources['arxiv'])} found)\n")
            for paper in sources['arxiv'][:2]:
                knowledge_parts.append(f"**{paper['title']}**\n")
                knowledge_parts.append(f"{paper['summary'][:300]}...\n")
                knowledge_parts.append(f"Source: {paper['url']}\n\n")
        
        # Add GitHub findings
        if sources['github']:
            knowledge_parts.append(f"\n## GitHub Projects ({len(sources['github'])} found)\n")
            for repo in sources['github'][:2]:
                knowledge_parts.append(f"**{repo['name']}** ({repo['stars']} ⭐)\n")
                knowledge_parts.append(f"{repo['description'][:200]}...\n")
                knowledge_parts.append(f"Source: {repo['url']}\n\n")
        
        # Add HuggingFace findings
        if sources['huggingface']:
            knowledge_parts.append(f"\n## AI Models ({len(sources['huggingface'])} found)\n")
            for model in sources['huggingface'][:2]:
                knowledge_parts.append(f"**{model['name']}** ({model['downloads']} downloads)\n")
                knowledge_parts.append(f"Source: {model['url']}\n\n")
        
        # Calculate mastery score
        total_sources = len(sources['arxiv']) + len(sources['github']) + len(sources['huggingface'])
        mastery_score = min(1.0, total_sources / 15)  # 15 sources = 100% mastery
        
        if total_sources == 0:
            knowledge_parts.append(f"\n## Foundational Knowledge\n")
            knowledge_parts.append(f"{topic} is an important concept for DMAI's development. ")
            knowledge_parts.append(f"Further research will be conducted as more sources become available.\n")
            mastery_score = 0.3
        
        return {
            'text': ''.join(knowledge_parts),
            'mastery_score': mastery_score,
            'source_count': total_sources
        }
    
    def process_queue(self):
        """Process research queue continuously"""
        while self.is_running:
            if self.research_queue:
                topic = self.research_queue.pop(0)
                result = self.research_topic(topic)
                
                # Update stage learner if available
                if self.stage_learner and result['mastery_score'] >= 0.6:
                    # Mark topic as mastered if research depth sufficient
                    current_mastery = self.stage_learner.get_topic_mastery(topic)
                    threshold = self.stage_learner._get_topic_threshold(topic)
                    if current_mastery < threshold:
                        # Increment mastery
                        pass
            
            time.sleep(5)  # Rate limiting
    
    def start_research_loop(self):
        """Start the background research loop"""
        if self.is_running:
            return {"status": "already_running"}
        
        self.is_running = True
        self.research_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.research_thread.start()
        
        return {"status": "started", "message": "Deep research loop active"}
    
    def stop_research_loop(self):
        """Stop the research loop"""
        self.is_running = False
        return {"status": "stopped"}
    
    def queue_research(self, topic: str) -> Dict:
        """Add a topic to research queue"""
        if topic not in self.research_results:
            self.research_queue.append(topic)
            return {"status": "queued", "topic": topic, "queue_position": len(self.research_queue)}
        return {"status": "already_researched", "topic": topic}

def initialize_deep_research(si_core=None, stage_learner=None):
    """Initialize deep research system"""
    return DeepResearchIntegrator(si_core, stage_learner)
