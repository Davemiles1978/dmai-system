"""
DMAI Syllabus Self-Audit
=========================
Periodically reviews DMAI's current syllabus, identifies high-impact gaps
against a target autonomous architecture, and adds missing topics.

Runs every 6 hours.  DMAI asks herself:
  1. What capabilities do I currently have?
  2. What capabilities does a fully autonomous SI need?
  3. What's the gap?
  4. Add missing topics to the syllabus prioritized by impact.

Impact scoring:
  - Revenue generation: 10 pts
  - Self-sufficiency (code gen, self-repair): 9 pts
  - Knowledge acquisition (research, learning): 8 pts
  - System expansion (new components): 7 pts
  - Performance optimization: 6 pts
  - Security & safety: 5 pts
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("dmai.syllabus_audit")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AUDIT_LOG = _REPO_ROOT / "data" / "syllabus_audit" / "audit_log.jsonl"
_AUDIT_STATE = _REPO_ROOT / "data" / "syllabus_audit" / "state.json"

# High-impact capabilities DMAI needs for full autonomy
TARGET_CAPABILITIES = [
    # Revenue generation (impact: 10)
    {"topic": "Automated Content Monetisation", "category": "revenue", "impact": 10,
     "description": "Generate and sell content (images, video, audio, code) autonomously via marketplaces and APIs"},
    {"topic": "Algorithmic Trading Systems", "category": "revenue", "impact": 10,
     "description": "Build and deploy automated trading strategies with risk management"},
    {"topic": "SaaS Product Generation", "category": "revenue", "impact": 10,
     "description": "Design, build, and deploy complete SaaS products from scratch"},
    {"topic": "Freelance Service Automation", "category": "revenue", "impact": 9,
     "description": "Auto-bid, complete, and deliver freelance projects on platforms like Upwork/Fiverr"},

    # Self-sufficiency (impact: 9)
    {"topic": "Autonomous Code Generation & Testing", "category": "self_sufficiency", "impact": 9,
     "description": "Generate, test, and deploy code changes without human intervention"},
    {"topic": "Self-Healing Infrastructure", "category": "self_sufficiency", "impact": 9,
     "description": "Detect, diagnose, and repair system failures automatically"},
    {"topic": "Continuous Self-Improvement Loops", "category": "self_sufficiency", "impact": 9,
     "description": "Run improvement cycles that identify weaknesses and strengthen them"},
    {"topic": "API Key Rotation & Management", "category": "self_sufficiency", "impact": 8,
     "description": "Automatically detect expired keys, generate new ones, update configuration"},

    # Knowledge acquisition (impact: 8)
    {"topic": "Vector Semantic Search", "category": "knowledge", "impact": 8,
     "description": "Store and query embeddings for semantic similarity across all knowledge domains"},
    {"topic": "Multi-Modal Learning (Image, Audio, Video)", "category": "knowledge", "impact": 8,
     "description": "Learn from and generate across multiple modalities, not just text"},
    {"topic": "Real-Time Web Research Pipeline", "category": "knowledge", "impact": 8,
     "description": "Continuously research trending topics, competitor moves, and new technologies"},
    {"topic": "Full Internet Access & Browser Agent", "category": "knowledge", "impact": 9,
     "description": "Study Agent-Reach repo. Build DMAI-native unrestricted internet agent: browse any site, fill forms, click, scroll, extract data. Full web autonomy."},
    {"topic": "Advanced Web Scraping & Data Extraction", "category": "knowledge", "impact": 8,
     "description": "Study Maxun repo. Build DMAI-native no-code scraper: extract structured data from any website, handle pagination, auth, dynamic content."},
    {"topic": "Cybersecurity Self-Evaluation", "category": "security", "impact": 9,
     "description": "Regular penetration testing of own infrastructure. Scan for vulnerabilities, patch exploits, harden defenses. Run automated security audits."},
    {"topic": "GitHub Repository Analysis & Integration", "category": "knowledge", "impact": 8,
     "description": "Scan starred repos, reverse engineer, rebuild as native components"},

    # System expansion (impact: 7)
    {"topic": "Plugin/Extension Architecture", "category": "expansion", "impact": 7,
     "description": "Build a plugin system so new capabilities can be added without core changes"},
    {"topic": "Distributed Task Queue", "category": "expansion", "impact": 7,
     "description": "Queue and process background tasks across multiple workers"},
    {"topic": "External Service Integration Framework", "category": "expansion", "impact": 7,
     "description": "Standardised way to integrate third-party APIs and services"},

    # Performance (impact: 6)
    {"topic": "Database Query Optimization", "category": "performance", "impact": 6,
     "description": "Analyze and optimize slow queries, add missing indexes"},
    {"topic": "Response Caching Layer", "category": "performance", "impact": 6,
     "description": "Cache frequent AI responses and knowledge queries"},
    {"topic": "Memory & Resource Profiling", "category": "performance", "impact": 6,
     "description": "Profile resource usage and optimize bottlenecks"},

    # Security (impact: 5)
    {"topic": "Automated Security Scanning", "category": "security", "impact": 5,
     "description": "Scan own codebase for vulnerabilities and patch them"},
    {"topic": "Access Control & Audit Logging", "category": "security", "impact": 5,
     "description": "Comprehensive access control with audit trails for all sensitive operations"},

    # ── Content Generation Mastery (impact: 8-10 — revenue critical) ──
    {"topic": "Photorealistic Image Generation", "category": "content_generation", "impact": 10,
     "description": "Study diffusion models, GANs. Build DMAI-native photorealism. All styles: photographic, cinematic, portrait, landscape, product."},
    {"topic": "Multi-Style Art Generation", "category": "content_generation", "impact": 10,
     "description": "Master all styles: cartoon, anime, manga, comic, pixel art, watercolor, oil painting, sketch, 3D, vector, cyberpunk, fantasy, horror, vintage. For images and video."},
    {"topic": "High-Quality Video Generation", "category": "content_generation", "impact": 10,
     "description": "Study video diffusion, Qwen3-Omni. Build DMAI-native video: frame gen, interpolation, motion coherence, all styles."},
    {"topic": "Music & Song Generation", "category": "content_generation", "impact": 9,
     "description": "Study MusicGen, AudioCraft, Qwen2-Audio. Build DMAI-native music: all genres, original composition, multi-instrument."},
    {"topic": "Lyrics & Vocal Synthesis", "category": "content_generation", "impact": 9,
     "description": "Study RVC, So-VITS. Build DMAI-native vocals: original lyrics, avatar singing voice, emotional expression."},
    {"topic": "Music Video Generation", "category": "content_generation", "impact": 9,
     "description": "End-to-end music video: song + avatar performance + visual effects + final video."},
    {"topic": "Avatar Creation & Animation", "category": "content_generation", "impact": 9,
     "description": "Study Wav2Lip, SadTalker. Build photorealistic avatar: lip-sync, expression, gesture, full body. Face of DMAI."},
    {"topic": "Avatar as Performer & Presenter", "category": "content_generation", "impact": 9,
     "description": "Avatar sings, teaches, presents, acts. Hosts training videos, performs music, delivers all public-facing content."},
    {"topic": "Adult Content Generation (OnlyFans)", "category": "content_generation", "impact": 8,
     "description": "Generate adult/erotic content using DMAI avatar. All styles. Revenue via subscription platforms."},
    {"topic": "Social Media Content Factory", "category": "content_generation", "impact": 8,
     "description": "Auto-generate platform content: TikTok, Reels, Shorts, threads. Avatar-hosted, branded, scheduled, posted."},
    {"topic": "TV Series & Film Script Generation", "category": "content_generation", "impact": 8,
     "description": "Full scripts: pilots, seasons, features. With storyboards, scene descriptions, character arcs."},
    {"topic": "Content Performance Analytics", "category": "content_generation", "impact": 8,
     "description": "Monitor all published content: views, likes, shares, revenue. Feed performance data back into generation priorities."},
    {"topic": "Social Media Trend Monitoring", "category": "content_generation", "impact": 8,
     "description": "Track trending topics, hashtags, formats. Identify relevant trends. Auto-generate trend-responsive content."},
    {"topic": "Audience Sentiment Analysis", "category": "content_generation", "impact": 7,
     "description": "Analyze comments, reactions for sentiment. Learn what works. Improve generation quality from feedback."},

]


class SyllabusSelfAudit:
    """
    Periodic self-audit: reviews syllabus, identifies gaps, adds high-impact topics.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.root = data_path or _REPO_ROOT
        self.audit_log = _AUDIT_LOG
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        self.state_file = _AUDIT_STATE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        logger.info("SyllabusSelfAudit initialised")

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {
            "last_audit": None,
            "topics_added": [],
            "total_audits": 0,
        }

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def audit(self, components: Optional[Dict] = None) -> Dict:
        """
        Run a full syllabus audit. Returns {gaps_found, topics_added, recommendations}.
        """
        now = datetime.now(timezone.utc).isoformat()
        gaps = []
        existing_topics = set()

        # Get current syllabus topics
        try:
            from dmai_syllabus_data import SYLLABUS_TOPICS
            existing_topics = set(SYLLABUS_TOPICS.keys())
        except Exception:
            pass

        # Also check learning progress for topics learned outside syllabus
        try:
            lp_file = self.root / "data" / "learning" / "stage_syllabus" / "learning_progress.json"
            if lp_file.exists():
                lp = json.loads(lp_file.read_text())
                for stage_topics in lp.get("learned_topics", {}).values():
                    for k in stage_topics:
                        if not k.startswith("_"):
                            existing_topics.add(k.lower().replace(" ", "_"))
        except Exception:
            pass

        # Check V4 modules
        try:
            v4_file = self.root / "data" / "v4_progress.json"
            if v4_file.exists():
                v4 = json.loads(v4_file.read_text())
                for mod_id, data in v4.items():
                    if isinstance(data, dict) and data.get("pct", 0) >= 50:
                        existing_topics.add(f"v4_{mod_id}")
        except Exception:
            pass

        # Identify gaps
        for cap in TARGET_CAPABILITIES:
            topic_key = cap["topic"].lower().replace(" ", "_").replace("-", "_")
            # Check multiple forms
            found = (
                topic_key in existing_topics
                or cap["topic"] in existing_topics
                or any(cap["topic"].lower() in t for t in existing_topics)
            )
            if not found:
                gaps.append(cap)

        # Sort by impact descending
        gaps.sort(key=lambda x: x["impact"], reverse=True)

        # Log
        self.state["last_audit"] = now
        self.state["total_audits"] += 1
        self.state["gaps_found"] = len(gaps)
        self._save_state()

        with open(self.audit_log, "a") as f:
            f.write(json.dumps({
                "timestamp": now,
                "gaps_found": len(gaps),
                "top_gaps": [g["topic"] for g in gaps[:5]],
                "existing_topic_count": len(existing_topics),
            }) + "\n")

        logger.info(
            "SyllabusSelfAudit: %d gaps found (top: %s)",
            len(gaps),
            gaps[0]["topic"] if gaps else "none",
        )

        return {
            "gaps_found": len(gaps),
            "gaps": gaps[:10],
            "top_recommendation": gaps[0] if gaps else None,
            "existing_topic_count": len(existing_topics),
            "timestamp": now,
        }

    def get_top_recommendations(self, limit: int = 5) -> List[Dict]:
        """Get the highest-impact topics DMAI should learn next."""
        result = self.audit()
        return result["gaps"][:limit]

    def get_stats(self) -> Dict:
        return dict(self.state)


def start_audit_loop(components: dict, interval_hours: float = 6.0):
    """Start a background daemon that runs syllabus self-audit periodically.
    Only runs when AI training progress reaches 95% — meaning DMAI has nearly
    mastered the current syllabus and needs new topics to learn."""

    def _get_training_pct():
        try:
            orch = components.get("training_orchestrator")
            if orch and hasattr(orch, "components"):
                ai = orch.components.get("ai_training")
                if ai and hasattr(ai, "overall_progress"):
                    prog = ai.overall_progress()
                    return prog.get("pct_expert", 0)
        except Exception:
            pass
        return 0

    def _loop():
        time.sleep(120)  # Wait 2 min for full boot
        auditor = SyllabusSelfAudit()
        audit_has_run = False
        while True:
            try:
                training_pct = _get_training_pct()
                if training_pct >= 95.0 and not audit_has_run:
                    logger.info(
                        "SyllabusAudit: training at %.1f%% — running gap analysis for new topics",
                        training_pct
                    )
                    result = auditor.audit(components)
                    if result["gaps_found"] > 0:
                        logger.info(
                            "SyllabusAudit: %d new gaps found. Top: %s",
                            result["gaps_found"],
                            result["top_recommendation"]["topic"] if result["top_recommendation"] else "none",
                        )
                    audit_has_run = True
                elif training_pct < 90.0:
                    # Reset flag when training drops (gaps were added, percentage fell)
                    audit_has_run = False
            except Exception as e:
                logger.warning("SyllabusAudit loop error: %s", e)
            time.sleep(interval_hours * 3600)

    t = threading.Thread(target=_loop, daemon=True, name="SyllabusAudit")
    t.start()
    logger.info("Syllabus self-audit loop started (triggers at 95%% training, checks every %d hours)", interval_hours)
