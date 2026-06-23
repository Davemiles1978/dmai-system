"""
SocialMediaPoster — posts queued Alex Riviera content to Twitter/X and LinkedIn.
Required env vars (optional — system degrades gracefully if missing):
  TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
  LINKEDIN_ACCESS_TOKEN, LINKEDIN_PERSON_URN
"""
import os, json, logging, time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SocialMediaPoster:
    def __init__(self, data_path="data"):
        self.data_path = data_path.rstrip("/")
        self.queue_path = os.path.join(self.data_path, "content_queue.jsonl")
        self.posted_path = os.path.join(self.data_path, "posted_content.jsonl")

    def post_pending(self):
        """Post all pending queue items that are due now"""
        if not os.path.exists(self.queue_path):
            return

        pending, still_pending = [], []
        with open(self.queue_path) as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if item.get("status") == "pending":
                        pending.append(item)
                except Exception:
                    pass

        now = datetime.now(timezone.utc)
        posted = []

        for item in pending:
            try:
                sched_str = item.get("scheduled_at", now.isoformat())
                scheduled = datetime.fromisoformat(sched_str.replace("Z", "+00:00"))
            except Exception:
                scheduled = now

            if scheduled <= now:
                platforms = item.get("platform", [])
                content = item.get("content", "")
                success = False

                for platform in platforms:
                    if platform == "twitter":
                        ok = self._post_twitter(content, item)
                        success = success or ok
                    elif platform == "linkedin":
                        ok = self._post_linkedin(content, item)
                        success = success or ok

                item["status"] = "posted" if success else "failed"
                item["posted_at"] = now.isoformat()
                posted.append(item)
            else:
                still_pending.append(item)

        if posted:
            with open(self.queue_path, "w") as f:
                for item in still_pending:
                    f.write(json.dumps(item) + "\n")
            with open(self.posted_path, "a") as f:
                for item in posted:
                    f.write(json.dumps(item) + "\n")

    def _post_twitter(self, content: str, item: dict) -> bool:
        api_key = os.environ.get("TWITTER_API_KEY")
        api_secret = os.environ.get("TWITTER_API_SECRET")
        access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
        access_secret = os.environ.get("TWITTER_ACCESS_SECRET")

        if not all([api_key, api_secret, access_token, access_secret]):
            logger.info("SocialMediaPoster: Twitter keys not configured — skipping")
            return False

        try:
            from requests_oauthlib import OAuth1
            import requests
            auth = OAuth1(api_key, api_secret, access_token, access_secret)
            tweets = self._extract_tweets(content)

            last_id = None
            for tweet_text in tweets[:5]:
                tweet_text = str(tweet_text)[:280]
                payload = {"text": tweet_text}
                if last_id:
                    payload["reply"] = {"in_reply_to_tweet_id": last_id}

                resp = requests.post(
                    "https://api.twitter.com/2/tweets",
                    auth=auth,
                    json=payload,
                    timeout=15
                )
                if resp.status_code == 201:
                    last_id = resp.json().get("data", {}).get("id")
                    logger.info(f"SocialMediaPoster: Twitter posted tweet {last_id}")
                    time.sleep(1)
                else:
                    logger.warning(f"SocialMediaPoster: Twitter error {resp.status_code}: {resp.text[:200]}")
                    return False
            return True

        except ImportError:
            logger.warning("SocialMediaPoster: requests_oauthlib not installed — run: pip install requests-oauthlib")
            return False
        except Exception as e:
            logger.warning(f"SocialMediaPoster: Twitter error: {e}")
            return False

    def _post_linkedin(self, content: str, item: dict) -> bool:
        token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
        person_urn = os.environ.get("LINKEDIN_PERSON_URN")

        if not token or not person_urn:
            logger.info("SocialMediaPoster: LinkedIn keys not configured — skipping")
            return False

        try:
            import requests
            post_text = self._extract_linkedin_text(content)

            resp = requests.post(
                "https://api.linkedin.com/v2/ugcPosts",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "X-Restli-Protocol-Version": "2.0.0"
                },
                json={
                    "author": f"urn:li:person:{person_urn}",
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {"text": post_text[:3000]},
                            "shareMediaCategory": "NONE"
                        }
                    },
                    "visibility": {
                        "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                    }
                },
                timeout=15
            )
            if resp.status_code in [200, 201]:
                logger.info("SocialMediaPoster: LinkedIn posted successfully")
                return True
            else:
                logger.warning(f"SocialMediaPoster: LinkedIn error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"SocialMediaPoster: LinkedIn error: {e}")
        return False

    def _extract_tweets(self, content: str) -> list:
        """Extract tweets array from JSON content or split on double newline"""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "tweets" in data:
                return [str(t) for t in data["tweets"]]
            if isinstance(data, list):
                return [str(t) for t in data]
        except Exception:
            pass
        parts = [p.strip() for p in content.split("\n\n") if p.strip()]
        return parts if parts else [content[:280]]

    def _extract_linkedin_text(self, content: str) -> str:
        """Extract LinkedIn post text from JSON or return raw"""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return str(data.get("linkedin_post", data.get("text", content)))
        except Exception:
            pass
        return content[:3000]

    def get_queue_stats(self) -> dict:
        pending = 0
        posted_total = 0

        if os.path.exists(self.queue_path):
            with open(self.queue_path) as f:
                for line in f:
                    try:
                        if json.loads(line.strip()).get("status") == "pending":
                            pending += 1
                    except Exception:
                        pass

        if os.path.exists(self.posted_path):
            with open(self.posted_path) as f:
                for line in f:
                    if line.strip():
                        posted_total += 1

        return {
            "pending_posts": pending,
            "posted_total": posted_total,
            "twitter_configured": bool(os.environ.get("TWITTER_API_KEY")),
            "linkedin_configured": bool(os.environ.get("LINKEDIN_ACCESS_TOKEN"))
        }
