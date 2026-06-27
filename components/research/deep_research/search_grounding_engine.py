"""
DMAI SearchGroundingEngine
==========================
Executes real-time web searches via Tavily API (primary) or Brave Search API
(secondary). No mock data — if no API key is configured, returns an explicit
pending status so callers know to surface the missing-key message.
"""

import os
import re
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_BRAVE_ENDPOINT  = "https://api.search.brave.com/res/v1/web/search"
_USER_AGENT      = "DMAI-Research/1.0"


class SearchGroundingEngine:
    """
    Real-time web search with automatic provider fallback.

    Priority: Tavily → Brave → no-search (explicit error).
    """

    def __init__(self):
        self.tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
        self.brave_key  = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()

        if self.tavily_key:
            self.primary = "tavily"
        elif self.brave_key:
            self.primary = "brave"
        else:
            self.primary = "none"

        logger.info("SearchGroundingEngine initialised — primary provider: %s", self.primary)

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, num_results: int = 8) -> Dict:
        """
        Search the web for *query*.

        Returns
        -------
        {
            "success": bool,
            "query": str,
            "provider": "tavily" | "brave" | "none",
            "results": [{"title", "url", "snippet", "raw_content"}],
            "direct_answer": str   # Tavily only, empty string otherwise
        }
        """
        if self.primary == "none":
            return {
                "success": False,
                "query": query,
                "provider": "none",
                "error": (
                    "No search API key configured. "
                    "Set TAVILY_API_KEY or BRAVE_SEARCH_API_KEY to enable web search."
                ),
                "results": [],
                "direct_answer": "",
            }

        if self.primary == "tavily" or (self.primary == "brave" and not self.brave_key):
            result = self._search_tavily(query, num_results)
            if result["success"]:
                return result
            # Fall through to Brave
            logger.warning("Tavily search failed (%s), falling back to Brave", result.get("error"))

        if self.brave_key:
            result = self._search_brave(query, num_results)
            if result["success"]:
                return result
            logger.warning("Brave search also failed: %s", result.get("error"))

        return {
            "success": False,
            "query": query,
            "provider": self.primary,
            "error": "All search providers failed.",
            "results": [],
            "direct_answer": "",
        }

    def fetch_page_content(self, url: str, max_chars: int = 4000) -> str:
        """
        Fetch and clean the text content of *url*.
        Returns at most *max_chars* characters, empty string on any failure.
        Never raises.
        """
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=10,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                return ""

            html = resp.text
            text = self._extract_text(html)
            return text[:max_chars]
        except Exception as exc:
            logger.debug("fetch_page_content failed for %s: %s", url, exc)
            return ""

    def get_status(self) -> Dict:
        return {
            "primary_provider": self.primary,
            "tavily_configured": bool(self.tavily_key),
            "brave_configured":  bool(self.brave_key),
            "search_ready":      self.primary != "none",
        }

    # ── Provider implementations ─────────────────────────────────────────────

    def _search_tavily(self, query: str, num_results: int) -> Dict:
        try:
            payload = {
                "api_key":           self.tavily_key,
                "query":             query,
                "search_depth":      "advanced",
                "include_answer":    True,
                "include_raw_content": True,
                "max_results":       min(num_results, 10),
            }
            resp = requests.post(_TAVILY_ENDPOINT, json=payload, timeout=20)
            if resp.status_code != 200:
                return {"success": False, "error": f"Tavily HTTP {resp.status_code}"}

            data = resp.json()
            results = []
            for item in data.get("results", []):
                raw = (item.get("raw_content") or item.get("content") or "")[:3000]
                results.append({
                    "title":       item.get("title", ""),
                    "url":         item.get("url", ""),
                    "snippet":     item.get("content", "")[:500],
                    "raw_content": raw,
                    "score":       item.get("score", 0.0),
                })

            return {
                "success":        True,
                "query":          query,
                "provider":       "tavily",
                "results":        results,
                "direct_answer":  data.get("answer", ""),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _search_brave(self, query: str, num_results: int) -> Dict:
        try:
            resp = requests.get(
                _BRAVE_ENDPOINT,
                headers={
                    "Accept":               "application/json",
                    "Accept-Encoding":      "gzip",
                    "X-Subscription-Token": self.brave_key,
                },
                params={"q": query, "count": min(num_results, 20), "text_decorations": "false"},
                timeout=15,
            )
            if resp.status_code != 200:
                return {"success": False, "error": f"Brave HTTP {resp.status_code}"}

            data = resp.json()
            raw_results = data.get("web", {}).get("results", [])
            results = []
            for item in raw_results:
                results.append({
                    "title":       item.get("title", ""),
                    "url":         item.get("url", ""),
                    "snippet":     item.get("description", "")[:500],
                    "raw_content": item.get("description", "")[:3000],
                    "score":       0.0,
                })

            return {
                "success":        True,
                "query":          query,
                "provider":       "brave",
                "results":        results,
                "direct_answer":  "",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Text extraction ───────────────────────────────────────────────────────

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML — uses BeautifulSoup if available,
        otherwise falls back to regex stripping."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove script/style noise
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            parts = []
            for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
                t = tag.get_text(separator=" ", strip=True)
                if len(t) > 20:
                    parts.append(t)
            return " ".join(parts)
        except ImportError:
            pass

        # Regex fallback
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
