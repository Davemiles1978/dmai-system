"""
Deep Research Integration - Fixed version
"""
import requests
import json
import time
import threading
from typing import Dict, List, Optional
from datetime import datetime

class DeepResearchIntegrator:
    """Deep research system for DMAI"""
    
    def __init__(self, synthetic_network=None, stage_learner=None):
        self.synthetic_network = synthetic_network  # Changed from si_core
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
    
    def research_topic(self, topic: str) -> Dict:
        """Perform deep research on a topic"""
        print(f"🔬 Deep researching: {topic}")
        
        sources = {
            'arxiv': self.search_arxiv(topic),
            'github': self.search_github(topic),
        }
        
        knowledge = self.synthesize_knowledge(topic, sources)
        
        # Store in synthetic network if available
        if self.synthetic_network and hasattr(self.synthetic_network, 'add_insight'):
            self.synthetic_network.add_insight(
                insight_text=knowledge['text'][:2000],
                entity_type='research',
                entities=[topic, 'deep_research'],
                relationship='mastered_knowledge',
                source_topic=topic,
                source_url=sources['arxiv'][0]['url'] if sources['arxiv'] else None,
                confidence=0.85
            )
        
        result = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'sources': {
                'arxiv_count': len(sources['arxiv']),
                'github_count': len(sources['github']),
            },
            'knowledge': knowledge,
            'mastery_score': knowledge['mastery_score']
        }
        
        self.research_results[topic] = result
        return result
    
    def synthesize_knowledge(self, topic: str, sources: Dict) -> Dict:
        """Synthesize research into knowledge"""
        knowledge_parts = []
        knowledge_parts.append(f"# {topic} - Deep Research Summary\n")
        
        if sources['arxiv']:
            knowledge_parts.append(f"\n## Research Papers ({len(sources['arxiv'])} found)\n")
            for paper in sources['arxiv'][:2]:
                knowledge_parts.append(f"**{paper['title']}**\n")
                knowledge_parts.append(f"{paper['summary'][:300]}...\n\n")
        
        if sources['github']:
            knowledge_parts.append(f"\n## GitHub Projects ({len(sources['github'])} found)\n")
            for repo in sources['github'][:2]:
                knowledge_parts.append(f"**{repo['name']}** ({repo['stars']} ⭐)\n")
                knowledge_parts.append(f"{repo['description'][:200]}...\n\n")
        
        total_sources = len(sources['arxiv']) + len(sources['github'])
        mastery_score = min(1.0, total_sources / 15)
        
        if total_sources == 0:
            knowledge_parts.append(f"\n## Foundational Knowledge\n")
            knowledge_parts.append(f"{topic} is being researched. Results will appear here.\n")
            mastery_score = 0.3
        
        return {
            'text': ''.join(knowledge_parts),
            'mastery_score': mastery_score,
            'source_count': total_sources
        }
    
    def queue_research(self, topic: str) -> Dict:
        """Add topic to research queue"""
        if topic not in self.research_results:
            self.research_queue.append(topic)
            return {"status": "queued", "topic": topic, "queue_position": len(self.research_queue)}
        return {"status": "already_researched", "topic": topic}

def initialize_deep_research(synthetic_network=None, stage_learner=None):
    """Initialize deep research system"""
    return DeepResearchIntegrator(synthetic_network, stage_learner)
