"""Alex Riviera Content Pipeline — multi-platform social media automation.

Platforms:
  - Instagram: lifestyle, behind-the-scenes, "letting hair down"
  - TikTok: trending formats, authentic moments
  - LinkedIn: professional expertise, case studies
  - X (Twitter): thought leadership, quick insights

All content is auto-generated, queued for approval, then posted on click.
"""
import os, json, logging, time, threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

logger = logging.getLogger("dmai.alex_content_pipeline")

CONTENT_QUEUE_FILE = "data/alex_content_queue.json"
APPROVAL_STATE_FILE = "data/alex_approvals.json"

PLATFORM_CONFIG = {
    "instagram": {
        "persona": "alex_public",
        "tone": "lifestyle, authentic, aspirational",
        "content_types": ["photo_post", "carousel", "story", "reel"],
        "hashtags": ["#ConfidenceCoach", "#WomenInBusiness", "#LifestyleCreator", "#BuildInPublic"],
    },
    "tiktok": {
        "persona": "alex_public",
        "tone": "trending, relatable, quick-hitting",
        "content_types": ["trend_challenge", "day_in_life", "transformation", "advice_stitch"],
        "hashtags": ["#ConfidenceTips", "#GlowUp", "#WomenEmpowerment", "#FYP"],
    },
    "linkedin": {
        "persona": "alex_public",
        "tone": "professional, insightful, results-driven",
        "content_types": ["case_study", "industry_insight", "client_win", "thought_piece"],
        "hashtags": ["#Leadership", "#Confidence", "#CareerGrowth", "#Coaching"],
    },
    "x": {
        "persona": "alex_public",
        "tone": "sharp, concise, thought-leader",
        "content_types": ["thread", "hot_take", "insight_share"],
        "hashtags": ["#BuildInPublic", "#AI", "#Confidence", "#Entrepreneurship"],
    },
}


class AlexContentPipeline:
    """Generates and manages multi-platform content for Alex Riviera."""

    def __init__(self):
        self.queue_file = Path(CONTENT_QUEUE_FILE)
        self.approval_file = Path(APPROVAL_STATE_FILE)
        self.queue = self._load_queue()
        self.approvals = self._load_approvals()

    def _load_queue(self) -> List[Dict]:
        if self.queue_file.exists():
            try:
                return json.loads(self.queue_file.read_text())
            except Exception:
                pass
        return []

    def _save_queue(self):
        self.queue_file.write_text(json.dumps(self.queue, indent=2))

    def _load_approvals(self) -> Dict:
        if self.approval_file.exists():
            try:
                return json.loads(self.approval_file.read_text())
            except Exception:
                pass
        return {}

    def _save_approvals(self):
        self.approval_file.write_text(json.dumps(self.approvals, indent=2))

    def generate_content(self, platform: str, content_type: str = None, topic: str = None) -> Optional[Dict]:
        """Generate content for a specific platform."""
        config = PLATFORM_CONFIG.get(platform)
        if not config:
            logger.error(f"Unknown platform: {platform}")
            return None

        # Use provided topic or pick from insights
        if not topic:
            topic = self._pick_topic_from_insights()

        if not content_type:
            content_type = config["content_types"][0]

        prompt = self._build_content_prompt(platform, config, content_type, topic)
        content_text = self._generate_with_llm(prompt)

        if not content_text:
            return None

        entry = {
            "id": f"{platform}_{int(time.time())}",
            "platform": platform,
            "content_type": content_type,
            "topic": topic,
            "content": content_text,
            "hashtags": config["hashtags"],
            "persona": config["persona"],
            "status": "pending_approval",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approved_at": None,
            "posted_at": None,
        }
        self.queue.append(entry)
        self._save_queue()
        logger.info(f"Queued {platform} content: {content_type} — {topic}")
        return entry

    def _pick_topic_from_insights(self) -> str:
        """Pick a topic from recent insights."""
        try:
            from components.db import safe_open_kdb
            conn = safe_open_kdb("data/dmai_knowledge.db")
            rows = conn.execute(
                "SELECT insight_text FROM insights "
                "WHERE insight_text IS NOT NULL AND LENGTH(insight_text) > 100 "
                "ORDER BY id DESC LIMIT 10"
            ).fetchall()
            conn.close()
            if rows:
                # Pick a random insight
                import random
                row = random.choice(rows)
                text = str(row[0]) if isinstance(row, tuple) else str(row.get("insight_text", ""))
                return text[:200]
        except Exception:
            pass
        return "confidence and personal growth"

    def _build_content_prompt(self, platform: str, config: Dict, content_type: str, topic: str) -> str:
        """Build LLM prompt for content generation."""
        persona = "Alex Riviera, 28-year-old American confidence coach and educator"

        if platform == "instagram":
            return f"""Create an Instagram caption for {persona}.
Tone: {config['tone']}
Content type: {content_type}
Topic: {topic}

The post should feel aspirational but authentic — showing the lifestyle Alex has built through her professional success.
Include a hook, 2-3 short paragraphs, and a call to action.
Max 150 words.
Hashtags: {' '.join(config['hashtags'])}"""

        elif platform == "tiktok":
            return f"""Create a TikTok script for {persona}.
Tone: {config['tone']}
Content type: {content_type}
Topic: {topic}

Write a 15-30 second video script with:
- Opening hook (first 2 seconds grab attention)
- Main content
- Call to action
Keep it punchy, relatable, and conversational.
Hashtags: {' '.join(config['hashtags'])}"""

        elif platform == "linkedin":
            return f"""Create a LinkedIn post for {persona}.
Tone: {config['tone']}
Content type: {content_type}
Topic: {topic}

Professional, results-driven, showing expertise without being salesy.
Structure: Hook → Insight → Application → Question for engagement.
Max 200 words.
Hashtags: {' '.join(config['hashtags'])}"""

        elif platform == "x":
            return f"""Create an X (Twitter) post for {persona}.
Tone: {config['tone']}
Content type: {content_type}
Topic: {topic}

Sharp, concise, thought-leader style.
Max 280 characters.
Hashtags: {' '.join(config['hashtags'])}"""

        return ""

    def _generate_with_llm(self, prompt: str) -> Optional[str]:
        """Generate content using available LLM."""
        import requests
        providers = [
            ("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1/chat/completions", "mistral-small"),
            ("google", "GOOGLE_AI_STUDIO_KEY", None, "gemini-2.0-flash-lite"),
            ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
        ]
        for name, env_key, url, model in providers:
            key = os.environ.get(env_key)
            if not key:
                continue
            try:
                if name == "google":
                    resp = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    resp = requests.post(
                        url,
                        headers={"Authorization": f"Bearer {key}"},
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 500,
                        },
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.debug(f"{name} generation failed: {e}")
        return None

    def get_pending_approvals(self) -> List[Dict]:
        """Get all content awaiting approval."""
        return [c for c in self.queue if c["status"] == "pending_approval"]

    def approve_content(self, content_id: str) -> bool:
        """Approve a piece of content for posting."""
        for entry in self.queue:
            if entry["id"] == content_id and entry["status"] == "pending_approval":
                entry["status"] = "approved"
                entry["approved_at"] = datetime.now(timezone.utc).isoformat()
                self._save_queue()
                return True
        return False

    def reject_content(self, content_id: str) -> bool:
        """Reject a piece of content."""
        for entry in self.queue:
            if entry["id"] == content_id and entry["status"] == "pending_approval":
                entry["status"] = "rejected"
                self._save_queue()
                return True
        return False

    def get_queue_stats(self) -> Dict:
        """Get queue statistics."""
        stats = {"pending": 0, "approved": 0, "rejected": 0, "posted": 0}
        for entry in self.queue:
            stats[entry["status"]] = stats.get(entry["status"], 0) + 1
        return stats


def run_daily_content_generation():
    """Generate daily content for all platforms."""
    pipeline = AlexContentPipeline()
    topics = [
        "confidence building techniques",
        "work-life balance for entrepreneurs",
        "personal growth mindset",
        "overcoming self-doubt",
        "building a successful coaching business",
    ]
    for platform in PLATFORM_CONFIG:
        for topic in topics[:2]:  # 2 posts per platform per day
            pipeline.generate_content(platform, topic=topic)
    logger.info("Daily content generation complete")
    return pipeline.get_queue_stats()
