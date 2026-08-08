"""
Autonomous researcher - DMAI studies topics in depth
"""
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from components.db import safe_open_kdb

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
    "linguistics":            ["etymology", "morphology", "phonology", "syntax", "semantics", "grammar", "lexicography", "rhetoric", "linguistics", "language", "vocabulary", "dialect", "corpus", "pragmatics", "semiotics"],
    "encyclopaedic_knowledge":["history", "philosophy", "civilisation", "empire", "renaissance", "enlightenment", "culture", "literature", "science history", "mathematics", "economics", "society", "ancient", "medieval", "classical"],
    "vocabulary_mastery":     ["word", "definition", "etymology", "meaning", "usage", "dictionary", "thesaurus", "oed", "lexicon", "nomenclature", "terminology", "jargon", "idiom", "expression", "figurative"],
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
    """DMAI's autonomous research system for deep topic mastery.

    Memory-first: before any external search, DMAI queries her own
    knowledge base. Only goes external if memory confidence < 0.55.
    """

    def __init__(self, si_core=None):
        self.si_core = si_core
        self.research_queue = []
        self.completed_research = []
        # ── Status surface ────────────────────────────────────────────────
        self.is_active = False
        self.last_run_at = None
        self.total_runs = 0
        self.total_items_produced = 0
        self.last_error = None
        self.interval_seconds = 300
        # Load memory retrieval at init
        self._memory_recall = None
        try:
            import sys, os
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from components.memory_retrieval import recall as _recall
            self._memory_recall = _recall
            print("AutonomousResearcher: MemoryRetrieval loaded — memory-first mode active")
        except Exception as _me:
            print(f"AutonomousResearcher: MemoryRetrieval unavailable ({_me}) — external-only mode")
        
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
        """Deep research on a topic — memory-first, external fallback."""
        print(f"🔬 Researching: {topic} (Depth: {depth})")

        # ── MEMORY FIRST ──────────────────────────────────────────────────────
        if self._memory_recall is not None:
            try:
                mem = self._memory_recall(topic)
                if mem.sufficient:
                    print(f"   ✅ Memory HIT (conf={mem.confidence:.2f}, src={mem.source}): {topic[:50]} — skipping external")
                    domain = _classify_domain(topic)
                    entities = _extract_entities(mem.best_text() + " " + topic)
                    self._persist_discovery(domain, entities, "memory_recall", mem.best_text()[:200])
                    if self.si_core and hasattr(self.si_core, 'add_insight'):
                        self.si_core.add_insight(
                            domain=domain, concept=topic, source='memory_recall',
                            confidence=mem.confidence,
                            metadata={'entities': entities, 'mastery_score': mem.confidence},
                        )
                    from datetime import datetime as _dt
                    self._record_run(len(entities) or 1)
                    return {
                        'topic': topic,
                        'sources': {'memory': mem.to_dict()},
                        'synthesis': {
                            'summary': mem.best_text()[:300],
                            'mastery_score': min(mem.confidence, 0.95),
                            'confidence': mem.confidence,
                            'memory_hit': True,
                        },
                        'discovery': {'topic': topic, 'domain': domain, 'from_memory': True},
                        'completed_at': _dt.now().isoformat(),
                        'depth': depth,
                        'from_memory': True,
                    }
            except Exception as _me:
                print(f"   ⚠️ Memory recall error: {_me} — falling back to external")

        # ── EXTERNAL SEARCH ───────────────────────────────────────────────────
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
        self._record_run(len(entities) or 1)
        return research_result

    def _record_run(self, items: int = 1) -> None:
        """Update status counters after a completed research cycle."""
        self.total_runs += 1
        self.total_items_produced += max(items, 0)
        self.last_run_at = datetime.now().isoformat()

    def get_status(self) -> dict:
        """Status surface for /api/research/autonomous/status. Never raises."""
        try:
            return {
                "available": True,
                "is_active": self.is_active,
                "last_run_at": self.last_run_at,
                "total_runs": self.total_runs,
                "total_items_produced": self.total_items_produced,
                "last_error": self.last_error,
                "interval_seconds": self.interval_seconds,
                "completed_research": len(self.completed_research),
            }
        except Exception as e:
            return {"available": True, "error": str(e)}

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

        # Immediately grow knowledge graph with the new domain + entities
        try:
            from components.graph_writer import GraphWriter as _GW
            _GW().add_discovery_node(domain, entities, source)
        except Exception:
            pass  # non-fatal
    
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
            # Language & encyclopaedic domains
            "etymology word origins English language",
            "history of philosophy enlightenment",
            "classical literature Shakespeare Dostoevsky",
            "linguistics morphology syntax semantics",
            "encyclopaedic knowledge civilisations empires",
            "rhetoric persuasion language philosophy",
            "mathematics number theory combinatorics",
            "history of science Newton Darwin Einstein",
        ]

        if topics is None:
            topics = list(DEFAULT_TOPICS)

        # Cross-process dedup: skip topics already researched today
        # Dedup: skip topics already researched today; resets at midnight automatically
        import json as _rj
        from datetime import datetime as _dt
        seen_file = _Path("data/research/seen_topics.json")
        seen_file.parent.mkdir(parents=True, exist_ok=True)

        def _load_seen_for_today():
            if not seen_file.exists():
                return set(), _dt.utcnow().strftime('%Y-%m-%d')
            try:
                data = _rj.loads(seen_file.read_text())
                today = _dt.utcnow().strftime('%Y-%m-%d')
                if isinstance(data, dict):
                    if data.get("date") == today:
                        return set(data.get("topics", [])), today
                    return set(), today  # new day — reset
                # Legacy plain-list format
                day_keys = {k for k in data if isinstance(k, str) and k.endswith(today)}
                return (day_keys if day_keys else set()), today
            except Exception:
                return set(), _dt.utcnow().strftime('%Y-%m-%d')

        def _save_seen(seen: set, today: str):
            try:
                seen_file.write_text(_rj.dumps({"date": today, "topics": sorted(seen)}))
            except Exception:
                pass

        def _expand_pool(base: list) -> list:
            """Grow topic pool from insights + capabilities so research never exhausts."""
            extra = []
            seen_labels = {t.lower().strip() for t in base}
            # From insights.jsonl
            ins_path = _Path("data/research/insights.jsonl")
            if ins_path.exists():
                try:
                    for line in ins_path.read_text().splitlines()[-500:]:
                        if not line.strip(): continue
                        try:
                            rec = _rj.loads(line)
                            concept = rec.get("concept", "").strip()
                            if concept and concept.lower() not in seen_labels and len(concept) > 8:
                                extra.append(concept[:120])
                                seen_labels.add(concept.lower())
                        except Exception:
                            pass
                except Exception:
                    pass
            # From capabilities DB
            try:
                import sqlite3 as _sq
                conn = safe_open_kdb("data/dmai_knowledge.db")
                cur = conn.cursor()
                cols = [r[1] for r in cur.fetchall()]
                name_col = next((c for c in ["name","capability","title"] if c in cols), None)
                if name_col:
                    cur.execute(f"SELECT {name_col} FROM capabilities ORDER BY rowid DESC LIMIT 200")
                    for (cap,) in cur.fetchall():
                        if cap and cap.strip().lower() not in seen_labels:
                            extra.append(str(cap).strip()[:120])
                            seen_labels.add(cap.strip().lower())
                conn.close()
            except Exception:
                pass
            return base + extra

        seen_topics, current_day = _load_seen_for_today()
        all_topics = _expand_pool(list(topics))
        print(f"Autonomous research loop: {len(all_topics)} topics ({len(topics)} base + {len(all_topics)-len(topics)} dynamic), cycling forever")

        # PR QQ: cooperative-stop pattern. ``self.is_active`` now actually
        # gates the loop; ``self._stop_event`` (if present) interrupts sleeps.
        import threading as _threading
        if not hasattr(self, "_stop_event") or self._stop_event is None:
            self._stop_event = _threading.Event()
        self.is_active = True
        cycle = 0
        while self.is_active and not self._stop_event.is_set():
            try:
                today = _dt.utcnow().strftime('%Y-%m-%d')

                # Midnight reset
                if today != current_day:
                    seen_topics, current_day = set(), today
                    all_topics = _expand_pool(list(topics))
                    print(f"[researcher] New day — reset seen, pool now {len(all_topics)} topics")

                topic = all_topics[cycle % len(all_topics)]
                dedup_key = f"{topic.lower().strip()}::{today}"

                if dedup_key in seen_topics:
                    cycle += 1
                    if cycle % len(all_topics) == 0:
                        # Full pool done for today — sleep 30 min then re-expand.
                        # Interruptible: returns True if stop was requested.
                        print(f"[researcher] Full pool done for today. Sleeping 30 min.")
                        if self._stop_event.wait(1800):
                            break
                        all_topics = _expand_pool(list(topics))
                    else:
                        if self._stop_event.wait(10):
                            break
                    continue

                result = self.research_topic_deep(topic)
                from_mem = result.get('from_memory', False)
                print(f"Research cycle {cycle}: {topic[:60]} (mastery={result['synthesis']['mastery_score']:.2f}, mem={'yes' if from_mem else 'no'})")

                seen_topics.add(dedup_key)
                _save_seen(seen_topics, today)

                if cycle % 5 == 0:
                    self._ingest_nightly_training()

                # Re-expand pool every 20 cycles — new insights may have arrived
                if cycle % 20 == 0:
                    prev = len(all_topics)
                    all_topics = _expand_pool(list(topics))
                    if len(all_topics) > prev:
                        print(f"[researcher] Pool expanded {prev} → {len(all_topics)}")

                cycle += 1
                if self._stop_event.wait(120 if from_mem else 300):
                    break
            except Exception as e:
                self.last_error = str(e)
                print(f"Researcher loop error (cycle {cycle}): {e}")
                if self._stop_event.wait(60):
                    break
        print("[researcher] run_continuous_research stopped cleanly")

    def stop(self, join_timeout: float = 5.0) -> None:
        """Cooperatively stop the continuous research loop (PR QQ)."""
        import threading as _threading
        if not hasattr(self, "_stop_event") or self._stop_event is None:
            self._stop_event = _threading.Event()
        self.is_active = False
        self._stop_event.set()

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
