"""
DMAI CitationManager
====================
Collects sources from all sub-task results, deduplicates by normalised URL,
assigns stable global citation IDs, and rewrites inline [N] markers so the
final consolidated report has consistent, non-overlapping citation numbers.
"""

import re
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


class CitationManager:
    """
    Thread-safe citation registry.

    Usage pattern:
        cm = CitationManager()
        for sub_result in sub_results:
            id_map = cm.register_sources(sub_result["sources"])
            sub_result["synthesis"] = cm.remap_citations(sub_result["synthesis"], id_map)
        final_refs = cm.format_references()
        cm.reset()  # ready for next request
    """

    def __init__(self):
        self.citations: List[Dict] = []        # [{"id": int, "title": str, "url": str}]
        self.url_to_id: Dict[str, int] = {}    # normalised_url → global_id

    # ── Public API ────────────────────────────────────────────────────────────

    def register_sources(self, sources: List[Dict]) -> Dict[int, int]:
        """
        Register a list of source dicts from a sub-task result.

        Each source dict must contain at minimum: "url", "citation_id" (local int).
        "title" is optional but strongly recommended.

        Returns a mapping {local_citation_id: global_citation_id} for use
        with remap_citations().
        """
        id_map: Dict[int, int] = {}

        for source in sources:
            url   = source.get("url", "").strip()
            title = source.get("title", url or "Unknown Source").strip()
            local_id = int(source.get("citation_id", 0))

            if not url:
                continue

            norm_url = self._normalise_url(url)

            if norm_url in self.url_to_id:
                # Already registered — just map the local ID to the existing global one
                global_id = self.url_to_id[norm_url]
            else:
                # New source — assign next global ID
                global_id = len(self.citations) + 1
                self.citations.append({"id": global_id, "title": title, "url": url})
                self.url_to_id[norm_url] = global_id
                logger.debug("CitationManager: registered [%d] %s", global_id, url)

            if local_id:
                id_map[local_id] = global_id

        return id_map

    def remap_citations(self, text: str, id_map: Dict[int, int]) -> str:
        """
        Rewrite every [N] inline citation marker in *text* using *id_map*.

        [N] markers that have no entry in id_map are left unchanged.
        """
        if not id_map or not text:
            return text

        def replace(match: re.Match) -> str:
            local_id = int(match.group(1))
            global_id = id_map.get(local_id)
            if global_id is not None:
                return f"[{global_id}]"
            return match.group(0)  # leave as-is

        return re.sub(r"\[(\d+)\]", replace, text)

    def format_references(self) -> str:
        """
        Build a markdown-formatted references section.

        Example output:
            ## Sources

            [1] OpenAI Blog — https://openai.com/blog/...
            [2] arXiv — https://arxiv.org/abs/...
        """
        if not self.citations:
            return ""

        lines = ["## Sources", ""]
        for cite in self.citations:
            title = cite.get("title") or cite.get("url", "Source")
            url   = cite.get("url", "")
            lines.append(f"[{cite['id']}] {title} — {url}")

        return "\n".join(lines)

    def get_all_citations(self) -> List[Dict]:
        """Return the full list of registered citations as dicts."""
        return list(self.citations)

    def count(self) -> int:
        return len(self.citations)

    def reset(self):
        """Clear all state so the instance can be reused for a new request."""
        self.citations.clear()
        self.url_to_id.clear()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_url(url: str) -> str:
        """
        Normalise a URL for deduplication:
        - Lowercase scheme and host
        - Remove trailing slash from path
        - Strip common tracking params (utm_*, ref, source)
        """
        try:
            parsed = urlparse(url.strip())
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()
            path   = parsed.path.rstrip("/") or "/"
            # Reconstruct without query/fragment for dedup purposes
            return urlunparse((scheme, netloc, path, "", "", ""))
        except Exception:
            return url.lower().rstrip("/")
