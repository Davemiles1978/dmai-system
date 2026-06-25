"""
AlexRivieraContentEngine — generates social media content as Alex Riviera
from DMAI's research insights, KPI progress, and kaizen improvements.
Public persona: Alex Riviera — AI researcher and autonomous systems developer.
"""
import os, json, sqlite3, logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

ALEX_SYSTEM_PROMPT = """You are Alex Riviera, a cutting-edge AI researcher and autonomous systems developer.
You write in first person as a real person sharing genuine discoveries and achievements.
Voice: confident, data-driven, accessible. Short punchy sentences.
Audience: tech-savvy professionals interested in AI, automation, and the future of work.
Topics: AI/AGI breakthroughs, autonomous systems, self-improving AI, trading automation, future of work.
IMPORTANT: Never mention you are an AI or machine. Write as a real human researcher."""


class AlexRivieraContentEngine:
    def __init__(self, data_path="data"):
        self.data_path = data_path.rstrip("/")
        self.db_path = os.path.join(self.data_path, "dmai_knowledge.db")
        self.queue_path = os.path.join(self.data_path, "content_queue.jsonl")
        self._ensure_schema()

    def _ensure_schema(self):
        """Make sure the insights table exists. Idempotent."""
        try:
            os.makedirs(self.data_path, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    title       TEXT,
                    description TEXT,
                    content     TEXT,
                    source      TEXT,
                    url         TEXT,
                    tags        TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at)")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"AlexRiviera._ensure_schema: {e}")

    def run_daily_cycle(self):
        """Generate and queue today's content — called every 6 hours"""
        post = self.generate_daily_insight_post()
        if post:
            self.queue_post(post)
            logger.info("AlexRiviera: queued daily insight post")
        else:
            logger.info("AlexRiviera: no insights available for post today")

    def generate_daily_insight_post(self) -> dict | None:
        """Pull top insights from last 24h and write Twitter thread + LinkedIn post"""
        try:
            if not os.path.exists(self.db_path):
                return None
            conn = sqlite3.connect(self.db_path)
            yesterday = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT COALESCE(content, description, title, '') as content FROM insights WHERE created_at >= ? ORDER BY id DESC LIMIT 5",
                (yesterday,)
            ).fetchall()
            conn.close()

            if not rows:
                # Fall back to most recent insights regardless of date
                conn = sqlite3.connect(self.db_path)
                rows = conn.execute(
                    "SELECT COALESCE(content, description, title, '') as content FROM insights ORDER BY id DESC LIMIT 3"
                ).fetchall()
                conn.close()

            if not rows:
                return None

            insights_text = "\n".join([f"- {str(r[0])[:200]}" for r in rows[:3]])
            prompt = f"""{ALEX_SYSTEM_PROMPT}

Based on my AI research today, write a Twitter thread and LinkedIn post about these findings:
{insights_text}

Return ONLY valid JSON in this exact format:
{{"tweets": ["tweet1 (max 280 chars)", "tweet2 (max 280 chars)", "tweet3 (max 280 chars)"], "linkedin_post": "150-word professional LinkedIn post"}}

Tweet 1 must be a scroll-stopping hook. Include 1-2 relevant hashtags in the last tweet."""

            content_str = self._generate_with_llm(prompt)
            if content_str:
                return {
                    "type": "daily_insight",
                    "content": content_str,
                    "platform": ["twitter", "linkedin"],
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.warning(f"AlexRiviera.daily_insight: {e}")
        return None

    def generate_kpi_progress_post(self) -> dict | None:
        """Weekly KPI progress framed as Alex's personal learning journey"""
        try:
            state_path = os.path.join(self.data_path, "si_core_state.json")
            if not os.path.exists(state_path):
                return None
            with open(state_path) as f:
                state = json.load(f)

            caps, insights_count = 0, 0
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                try:
                    caps = conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0]
                    insights_count = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
                except Exception:
                    pass
                conn.close()

            skill_rate = round(state.get("skill_acquisition_rate", 0) * 100, 1)
            rsir = round(state.get("recursive_self_improvement_rate", 0) * 100, 1)

            prompt = f"""{ALEX_SYSTEM_PROMPT}

Write a single Friday progress tweet about my AI development week:
- Processed {insights_count:,} total research insights  
- Acquired {caps:,} capabilities
- Skill acquisition rate: {skill_rate}%
- Self-improvement rate: {rsir}%

One tweet only, max 280 chars. Personal founder-style metrics update. Include #AI #BuildInPublic"""

            content_str = self._generate_with_llm(prompt)
            if content_str:
                return {
                    "type": "kpi_progress",
                    "content": content_str,
                    "platform": ["twitter"],
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.warning(f"AlexRiviera.kpi_progress: {e}")
        return None

    def generate_kaizen_post(self, improvement_description: str) -> dict | None:
        """Post about a self-improvement that just executed"""
        try:
            prompt = f"""{ALEX_SYSTEM_PROMPT}

I just made this improvement to my AI system: {improvement_description}

Write a LinkedIn post (100-120 words) about this self-improvement milestone.
Structure: what wasn't working → what I changed → what improved.
Personal and specific. No hype."""

            content_str = self._generate_with_llm(prompt)
            if content_str:
                return {
                    "type": "kaizen",
                    "content": content_str,
                    "platform": ["linkedin"],
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.warning(f"AlexRiviera.kaizen_post: {e}")
        return None

    def _generate_with_llm(self, prompt: str) -> str | None:
        """Generate content using best available provider"""
        try:
            import requests
        except ImportError:
            logger.warning("AlexRiviera: requests not available")
            return None

        providers = [
            ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile", "openai"),
            ("google", "GOOGLE_AI_STUDIO_KEY", None, "gemini-2.0-flash-lite", "google"),
            ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions", "gpt-4o-mini", "openai"),
            ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1/chat/completions", "deepseek-chat", "openai"),
        ]

        for name, env_key, url, model, style in providers:
            key = os.environ.get(env_key)
            if not key:
                continue
            try:
                if style == "google":
                    resp = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=20
                    )
                    if resp.status_code == 200:
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    resp = requests.post(
                        url,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 600},
                        timeout=20
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"AlexRiviera LLM {name}: {e}")
        return None

    def queue_post(self, post: dict, scheduled_at: datetime = None):
        """Add post to content queue"""
        if scheduled_at is None:
            scheduled_at = datetime.now(timezone.utc)
        entry = {**post, "scheduled_at": scheduled_at.isoformat(), "status": "pending"}
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.queue_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_queue_count(self) -> int:
        if not os.path.exists(self.queue_path):
            return 0
        count = 0
        with open(self.queue_path) as f:
            for line in f:
                try:
                    if json.loads(line.strip()).get("status") == "pending":
                        count += 1
                except Exception:
                    pass
        return count
