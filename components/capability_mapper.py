"""
CapabilityMapper — reverse-engineers Perplexity.ai capabilities and maintains
target_capabilities.json as the benchmark DMAI must match.
"""
import os, json, logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CAPABILITIES = {
    "web_search": {
        "description": "Search web via Tavily and synthesise a cited answer with source URLs",
        "implemented": False, "priority": 1,
        "component": "components/chat_engine.py"
    },
    "citation_synthesis": {
        "description": "Extract URLs from search results and format as numbered citations",
        "implemented": False, "priority": 1,
        "component": "components/chat_engine.py"
    },
    "follow_up_generation": {
        "description": "Generate 3 contextual follow-up questions after each answer",
        "implemented": False, "priority": 2,
        "component": "components/chat_engine.py"
    },
    "streaming_responses": {
        "description": "Stream tokens in real-time to UI via Flask stream_with_context",
        "implemented": False, "priority": 2,
        "component": "components/chat_engine.py"
    },
    "conversation_memory": {
        "description": "Store and retrieve conversation history in SQLite for multi-turn context",
        "implemented": False, "priority": 2,
        "component": "components/conversation_memory.py"
    },
    "source_ranking": {
        "description": "Rank search results by relevance before synthesis",
        "implemented": False, "priority": 3,
        "component": "components/chat_engine.py"
    },
    "image_understanding": {
        "description": "Accept image inputs and reason about visual content",
        "implemented": False, "priority": 4,
        "component": "components/multimodal_handler.py"
    },
    "autonomous_research": {
        "description": "Proactively research topics without user prompting",
        "implemented": True, "priority": 1,
        "component": "components/autonomous_researcher.py"
    },
    "self_improvement": {
        "description": "Generate and execute kaizen improvements to own code via self-evolution loop",
        "implemented": False, "priority": 1,
        "component": "components/self_evolution_orchestrator.py"
    },
    "content_generation": {
        "description": "Generate social media posts as Alex Riviera from research insights",
        "implemented": False, "priority": 2,
        "component": "components/alex_riviera_content.py"
    },
    "social_posting": {
        "description": "Post content to Twitter/X and LinkedIn automatically",
        "implemented": False, "priority": 2,
        "component": "components/social_media_poster.py"
    },
    "self_scanning": {
        "description": "Audit own routes, threads, KPIs, and DB tables to find gaps",
        "implemented": False, "priority": 1,
        "component": "components/self_scanner.py"
    },
    "code_generation": {
        "description": "Use LLM providers to generate working Flask/Python code for gap items",
        "implemented": False, "priority": 1,
        "component": "components/code_generator.py"
    }
}

class CapabilityMapper:
    def __init__(self, data_path="data"):
        self.data_path = data_path.rstrip("/")
        self.target_caps_path = os.path.join(self.data_path, "target_capabilities.json")

    def run(self) -> dict:
        os.makedirs(self.data_path, exist_ok=True)

        # Load existing or start fresh
        if os.path.exists(self.target_caps_path):
            with open(self.target_caps_path) as f:
                caps = json.load(f)
            # Merge in new defaults not yet tracked
            for k, v in DEFAULT_CAPABILITIES.items():
                if k not in caps:
                    caps[k] = dict(v)
        else:
            caps = {k: dict(v) for k, v in DEFAULT_CAPABILITIES.items()}

        # Auto-detect implemented status from component file existence
        for name, info in caps.items():
            if name.startswith("_") or not isinstance(info, dict):
                continue
            component_path = info.get("component", "")
            if component_path and Path(component_path).exists():
                try:
                    source = Path(component_path).read_text(errors="ignore")
                    if len(source) > 200 and "NotImplementedError" not in source:
                        caps[name]["implemented"] = True
                except Exception:
                    pass

        caps["_last_updated"] = datetime.now(timezone.utc).isoformat()

        with open(self.target_caps_path, "w") as f:
            json.dump(caps, f, indent=2)

        implemented = sum(1 for k, v in caps.items() if not k.startswith("_") and isinstance(v, dict) and v.get("implemented"))
        total = sum(1 for k in caps if not k.startswith("_"))
        logger.info(f"CapabilityMapper: {implemented}/{total} capabilities implemented")
        return caps

    def mark_implemented(self, capability_name: str):
        if not os.path.exists(self.target_caps_path):
            return
        try:
            with open(self.target_caps_path) as f:
                caps = json.load(f)
            if capability_name in caps and isinstance(caps[capability_name], dict):
                caps[capability_name]["implemented"] = True
                caps[capability_name]["implemented_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.target_caps_path, "w") as f:
                json.dump(caps, f, indent=2)
        except Exception as e:
            logger.warning(f"CapabilityMapper.mark_implemented: {e}")
