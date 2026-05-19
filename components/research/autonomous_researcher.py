"""
Autonomous researcher - DMAI studies topics in depth
"""
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime

class AutonomousResearcher:
    """DMAI's autonomous research system for deep topic mastery"""
    
    def __init__(self, si_core=None):
        self.si_core = si_core
        self.research_queue = []
        self.completed_research = []
        
    def search_github(self, topic: str) -> List[Dict]:
        """Search GitHub for repositories related to topic"""
        url = f"https://api.github.com/search/repositories?q={topic}&sort=stars&order=desc&per_page=5"
        try:
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
        """Search HuggingFace for models related to topic"""
        url = f"https://huggingface.co/api/models?search={topic}&limit=5"
        try:
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
    
    def research_topic_deep(self, topic: str, depth: str = 'comprehensive') -> Dict:
        """Deep research on a topic using multiple sources"""
        print(f"🔬 Researching: {topic} (Depth: {depth})")
        
        sources = {
            'github': self.search_github(topic),
            'huggingface': self.search_huggingface(topic),
        }
        
        synthesized = self.synthesize_knowledge(topic, sources)
        
        # Store in SI Core if available
        if self.si_core and hasattr(self.si_core, 'add_insight'):
            self.si_core.add_insight(
                insight_text=synthesized['summary'][:2000],
                entity_type='research',
                entities=[topic, 'autonomous_research'],
                relationship='deep_knowledge',
                source_topic=topic,
                confidence=0.85
            )
        
        research_result = {
            'topic': topic,
            'sources': sources,
            'synthesis': synthesized,
            'completed_at': datetime.now().isoformat(),
            'depth': depth
        }
        
        self.completed_research.append(research_result)
        return research_result
    
    def synthesize_knowledge(self, topic: str, sources: Dict) -> Dict:
        """Synthesize knowledge from multiple sources"""
        summary_parts = []
        sources_list = []
        
        if sources.get('github'):
            sources_list.extend([r['url'] for r in sources['github'][:3]])
            summary_parts.append(f"💻 GitHub Projects: {len(sources['github'])} repositories")
            for repo in sources['github'][:2]:
                summary_parts.append(f"   • {repo['name']} ({repo['stars']}⭐): {repo['description'][:100]}...")
        
        if sources.get('huggingface'):
            sources_list.extend([m['url'] for m in sources['huggingface'][:3]])
            summary_parts.append(f"\n🤗 HuggingFace Models: {len(sources['huggingface'])} models")
            for model in sources['huggingface'][:2]:
                summary_parts.append(f"   • {model['name']} ({model['downloads']} downloads)")
        
        mastery_score = min(1.0, (len(sources['github']) + len(sources['huggingface'])) / 20)
        
        return {
            'summary': '\n'.join(summary_parts),
            'sources': sources_list,
            'mastery_score': mastery_score,
            'confidence': 0.7 + (mastery_score * 0.3)
        }
    
    def run_continuous_research(self, topics: List[str] = None):
        """Run continuous research loop"""
        if topics is None:
            topics = [
                "Python programming", "Machine learning algorithms", 
                "Algorithmic trading strategies", "Content generation AI",
                "Self-modifying code", "Autonomous systems"
            ]
        
        print(f"🔬 Starting autonomous research on {len(topics)} topics...")
        
        for topic in topics:
            result = self.research_topic_deep(topic)
            print(f"✅ Completed: {topic} (Mastery: {result['synthesis']['mastery_score']:.2f})")
            time.sleep(5)
        
        return self.completed_research

def start_autonomous_research(si_core=None):
    """Start autonomous research"""
    researcher = AutonomousResearcher(si_core)
    return researcher
