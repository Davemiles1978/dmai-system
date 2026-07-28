"""
DMAI Live Web Search — Tavily primary, DuckDuckGo fallback.
Provides real-time internet access for queries outside DMAI's knowledge.
"""
import os
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed — DDG fallback disabled")


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the live web. Uses Tavily if API key is set, falls back to DuckDuckGo.
    Returns list of {title, url, snippet}.
    """
    results = []

    # Primary: Tavily
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            import requests
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": max_results},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for r in data.get("results", [])[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                    })
                if results:
                    logger.info("Tavily search returned %d results for: %s", len(results), query[:60])
                    return results
        except Exception as e:
            logger.warning("Tavily search failed: %s", e)

    # Fallback: DuckDuckGo
    if DDGS_AVAILABLE and not results:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            if results:
                logger.info("DuckDuckGo search returned %d results for: %s", len(results), query[:60])
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)

    return results


def search_and_summarize(query: str) -> Optional[str]:
    """
    Search the web and return a concise summary of findings.
    Used by DMAI when she needs real-time information.
    """
    results = search_web(query, max_results=3)
    if not results:
        return None

    lines = [f"Live web search results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['snippet'][:300]}")
        lines.append(f"   Source: {r['url']}\n")

    return "\n".join(lines)
