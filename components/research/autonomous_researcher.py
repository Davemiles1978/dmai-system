"""
Autonomous researcher - DMAI studies topics in depth
"""
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime

_DOMAIN_KEYWORDS = {
    "machine_learning":       ["neural network", "deep learning", "gradient descent", "backpropagation", "transformer", "attention", "bert", "gpt", "llm", "fine-tuning", "pre-training"],
    "reinforcement_learning": ["reward", "policy", "q-learning", "ppo", "actor-critic", "mdp", "environment", "agent", "bandit", "mcts"],
    "autonomous_agents":      ["agent", "autonomous", "tool use", "planning", "agentic", "multi-agent", "swarm", "orchestration", "reasoning"],
    "trading":                ["stock", "trading", "portfolio", "alpha", "backtest", "market", "hedge", "options", "crypto", "arbitrage"],
    "content_generation":     ["content", "generation", "creative", "writing", "summarisation", "kdp", "publish", "blog", "social media"],
    "computer_vision":        ["image", "vision", "cnn", "detection", "segmentation", "ocr", "diffusion", "stable diffusion", "clip"],
    "nlp":                    ["natural language", "nlp", "sentiment", "classification", "ner", "embeddings", "vector", "semantic"],
    "self_improvement":       ["self-improvement", "recursive", "meta-learning", "self-play", "self-modify", "evolution", "kaizen"],
    "knowledge_systems":      ["knowledge graph", "ontology", "reasoning", "inference", "semantic web", "rdf", "sparql", "retrieval"],
    "robotics":               ["robot", "embodied", "manipulation", "locomotion", "sim2real", "ros"],
    "cybersecurity":          ["security", "vulnerability", "exploit", "cve", "penetration", "red team", "malware"],
    "web_technologies":       ["api", "fastapi", "flask", "react", "javascript", "typescript", "web scraping", "http"],
    "data_science":           ["data", "pandas", "statistics", "visualisation", "etl", "pipeline", "analytics", "dashboard"],
    "cloud_devops":           ["docker", "kubernetes", "ci/cd", "github actions", "render", "aws", "gcp", "azure", "terraform"],
}


def _classify_domain(text: str) -> str:
    """Return the best-matching domain for a given text string."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "knowledge_systems"


def _extract_entities(text: str, max_entities: int = 5) -> list[str]:
    """Extract named entities (model names, repo names, technique names) from text."""
    import re
    # Match CamelCase words, hyphenated-phrases, and ALLCAPS acronyms
    patterns = [
        r"\b[A-Z][a-zA-Z]{2,}(?:[A-Z][a-z]+)+\b",   # CamelCase
        r"\b[A-Za-z][\w]*-[\w][\w-]+\b",               # hyphenated
        r"\b[A-Z]{2,6}\b",                              # acronyms
        r"\bgpt-[\d\.]+\b",                              # model versions
        r"\bllama[\d\.-]*\b",
        r"\bclaude[\d\.-]*\b",
        r"\bgemini[\d\.-]*\b",
    ]
    entities = []
    for pattern in patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))
    # Deduplicate preserving order
    seen = set()
    result = []
    for e in entities:
        if e.lower() not in seen and len(e) > 2:
            seen.add(e.lower())
            result.append(e)
    return result[:max_entities]


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

        # Domain classification + entity extraction over the synthesized text
        classify_text = f"{topic} {synthesized.get('summary', '')}"
        domain = _classify_domain(classify_text)
        entities = _extract_entities(classify_text)

        # Emit a structured discovery for this research cycle
        discovery = {
            'topic': topic,
            'domain': domain,
            'entities': entities,
            'source': 'autonomous_researcher',
            'summary': synthesized.get('summary', ''),
        }
        self._persist_discovery(domain, entities, 'autonomous_researcher', synthesized.get('summary', ''))

        # Store in SI Core if available
        if self.si_core and hasattr(self.si_core, 'add_insight'):
            self.si_core.add_insight(
                domain=domain,
                concept=topic,
                source='autonomous_researcher',
                confidence=synthesized.get('confidence', 0.85),
                metadata={'entities': entities, 'mastery_score': synthesized.get('mastery_score')},
            )

        research_result = {
            'topic': topic,
            'sources': sources,
            'synthesis': synthesized,
            'discovery': discovery,
            'completed_at': datetime.now().isoformat(),
            'depth': depth
        }

        self.completed_research.append(research_result)
        return research_result

    def _persist_discovery(self, domain: str, entities: list, source: str, summary: str = "") -> None:
        """Write a structured discovery event to data/research/discoveries.jsonl."""
        import json
        from datetime import datetime, timezone
        from pathlib import Path
        discovery = {
            "id": f"disc_{int(datetime.now(timezone.utc).timestamp())}",
            "domain": domain,
            "entities": entities,
            "source": source,
            "summary": summary[:200],
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = Path("data/research/discoveries.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(discovery) + "\n")
    
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
        """Run perpetual research loop - cycles through topics continuously, ingests nightly data."""
        import json as _json
        from pathlib import Path as _Path

        DEFAULT_TOPICS = [
            "autonomous AI agents tool use",
            "large language model reasoning chains",
            "reinforcement learning from human feedback",
            "self-improving neural networks meta-learning",
            "algorithmic trading portfolio optimisation",
            "content generation AI revenue",
            "recursive self-improvement AGI",
            "knowledge graph neural reasoning",
            "multi-agent coordination swarm intelligence",
            "code generation LLM fine-tuning",
        ]

        if topics is None:
            topics = list(DEFAULT_TOPICS)

        print(f"Autonomous research loop started on {len(topics)} base topics (cycling forever)")
        cycle = 0
        while True:
            try:
                # Rotate through base topics
                topic = topics[cycle % len(topics)]
                result = self.research_topic_deep(topic)
                print(f"Research cycle {cycle}: {topic} (mastery={result['synthesis']['mastery_score']:.2f})")

                # Every 5 cycles, ingest any new nightly training data
                if cycle % 5 == 0:
                    self._ingest_nightly_training()

                cycle += 1
                time.sleep(300)  # 5 minutes between topics
            except Exception as e:
                print(f"Researcher loop error (cycle {cycle}): {e}")
                time.sleep(60)

    def _ingest_nightly_training(self):
        """Read data/training/*.json and add insights to SI core from new entries."""
        import json as _json
        from pathlib import Path as _Path
        training_path = _Path("data/training")
        if not training_path.exists():
            return
        seen_file = _Path("data/research/ingested_training.json")
        seen = set()
        if seen_file.exists():
            try:
                seen = set(_json.loads(seen_file.read_text()))
            except Exception:
                pass
        new_seen = set(seen)
        ingested = 0
        for tf in sorted(training_path.glob("*.json")):
            try:
                entries = _json.loads(tf.read_text())
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    src_url = entry.get("source", "")
                    if src_url in seen:
                        continue
                    new_seen.add(src_url)
                    domain = entry.get("domain", "knowledge_systems")
                    technique = entry.get("technique", entry.get("description", "")[:60])
                    confidence = 0.80
                    if self.si_core and hasattr(self.si_core, "add_insight"):
                        self.si_core.add_insight(
                            domain=domain,
                            concept=technique,
                            source=src_url or "nightly_training",
                            confidence=confidence,
                            metadata={"training_prompt": entry.get("training_prompt", ""),
                                      "expected_improvement": entry.get("expected_improvement", "")},
                        )
                    self._persist_discovery(domain, [technique], "nightly_training", technique)
                    ingested += 1
            except Exception as e:
                print(f"Nightly ingest error ({tf.name}): {e}")
        if ingested:
            print(f"Nightly training ingest: {ingested} new entries absorbed")
            seen_file.parent.mkdir(parents=True, exist_ok=True)
            seen_file.write_text(_json.dumps(sorted(new_seen)))

    def run_continuous_research_once(self, topics: List[str] = None):
        """Single-pass research (original behaviour, kept for tests)."""
        if topics is None:
            topics = [
                "Python programming", "Machine learning algorithms",
                "Algorithmic trading strategies", "Content generation AI",
                "Self-modifying code", "Autonomous systems"
            ]
        print(f"Single-pass research on {len(topics)} topics...")
        for topic in topics:
            result = self.research_topic_deep(topic)
            print(f"Completed: {topic} (Mastery: {result['synthesis']['mastery_score']:.2f})")
            time.sleep(5)
        return self.completed_research

def start_autonomous_research(si_core=None):
    """Start autonomous research"""
    researcher = AutonomousResearcher(si_core)
    return researcher
