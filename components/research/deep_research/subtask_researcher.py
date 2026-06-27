"""
DMAI SubTaskResearcher
======================
Researches a single sub-question end-to-end:
  1. Web search via SearchGroundingEngine
  2. Fetch full content for top N results
  3. Synthesise a mini-report with inline [N] citation markers

One instance per sub-question; re-instantiate for each call or reuse — it
carries no persistent state between research() calls.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
You are a precise research analyst. Based on the search results below, write a \
2-3 paragraph synthesis that directly and factually answers the question. \
Use inline citation markers like [1], [2] matching the source numbers below. \
Be concise, factual, and do not invent information not present in the sources.

Question: {sub_question}
Broader context: {parent_query}

Sources:
{sources_block}

Write a synthesis with inline citations (2-3 paragraphs):"""


class SubTaskResearcher:
    """
    Researches one sub-question and returns a structured mini-report.
    """

    def __init__(self, search_engine, ai_hub=None):
        self.search_engine = search_engine
        self.ai_hub        = ai_hub

    # ── Public API ────────────────────────────────────────────────────────────

    def research(self, sub_question: str, parent_query: str = "",
                 fetch_pages: int = 2) -> Dict:
        """
        Research *sub_question*.

        Parameters
        ----------
        fetch_pages : int
            Number of top results to fetch full page content for (0 = skip).

        Returns
        -------
        {
            "success": bool,
            "sub_question": str,
            "sources": [{"title", "url", "citation_id"}],
            "synthesis": str,       # with inline [N] markers
            "raw_search_results": [...],
            "confidence": "high" | "medium" | "low",
            "search_provider": str
        }
        """
        # ── Step 1: Search ────────────────────────────────────────────────────
        search_result = self.search_engine.search(sub_question, num_results=6)

        if not search_result.get("success"):
            return {
                "success":            False,
                "sub_question":       sub_question,
                "sources":            [],
                "synthesis":          f"Search unavailable: {search_result.get('error', 'unknown')}",
                "raw_search_results": [],
                "confidence":         "low",
                "search_provider":    search_result.get("provider", "none"),
            }

        raw_results = search_result.get("results", [])
        if not raw_results:
            return {
                "success":            False,
                "sub_question":       sub_question,
                "sources":            [],
                "synthesis":          "No search results returned.",
                "raw_search_results": [],
                "confidence":         "low",
                "search_provider":    search_result.get("provider", "none"),
            }

        # ── Step 2: Optionally fetch full page content ─────────────────────
        enriched = list(raw_results)  # shallow copy
        for i, item in enumerate(enriched[:fetch_pages]):
            url = item.get("url", "")
            if url and not item.get("raw_content"):
                page_text = self.search_engine.fetch_page_content(url, max_chars=3000)
                if page_text:
                    enriched[i] = dict(item, raw_content=page_text)

        # ── Step 3: Build source list (citation IDs start at 1 locally) ───
        sources = []
        for idx, item in enumerate(enriched[:8], start=1):
            sources.append({
                "title":       item.get("title", item.get("url", "Source")),
                "url":         item.get("url", ""),
                "citation_id": idx,
            })

        # ── Step 4: Build context for synthesis ───────────────────────────
        context_budget = 6000
        sources_block_parts = []
        used = 0
        for src in sources:
            idx  = src["citation_id"]
            item = enriched[idx - 1] if idx - 1 < len(enriched) else {}
            body = (item.get("raw_content") or item.get("snippet") or "")
            # Trim per source so we never blow past context budget
            remaining = max(0, context_budget - used - 200)
            body = body[:min(800, remaining)]
            if not body:
                continue
            sources_block_parts.append(f"[{idx}] {src['title']} ({src['url']}):\n{body}")
            used += len(body)
            if used >= context_budget:
                break

        sources_block = "\n\n".join(sources_block_parts)

        # ── Step 5: Synthesise ────────────────────────────────────────────
        direct_answer = search_result.get("direct_answer", "")
        synthesis, method = self._synthesise(
            sub_question, parent_query, sources_block, sources, direct_answer
        )

        confidence = "high" if method == "llm" else "medium"

        return {
            "success":            True,
            "sub_question":       sub_question,
            "sources":            sources,
            "synthesis":          synthesis,
            "raw_search_results": raw_results,
            "confidence":         confidence,
            "search_provider":    search_result.get("provider", "none"),
        }

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesise(self, sub_question: str, parent_query: str,
                    sources_block: str, sources: List[Dict],
                    direct_answer: str) -> tuple:
        """
        Returns (synthesis_text, method_used).
        method_used: "llm" | "fallback"
        """
        if sources_block and self.ai_hub is not None:
            prompt = _SYNTHESIS_PROMPT.format(
                sub_question=sub_question,
                parent_query=parent_query or sub_question,
                sources_block=sources_block,
            )
            raw = self._call_llm(prompt)
            if raw and len(raw) > 50:
                return raw.strip(), "llm"

        # Fallback: concatenate direct answer + top snippets with citation markers
        parts = []
        if direct_answer:
            parts.append(direct_answer)

        for src in sources[:3]:
            idx  = src["citation_id"]
            item_idx = idx - 1
            # We don't have enriched here, reconstruct from sources_block
            parts.append(f"According to [{idx}] {src['title']}: relevant information found at {src['url']}.")

        synthesis = " ".join(parts) if parts else "No synthesis available — search returned results but no content could be extracted."
        return synthesis, "fallback"

    # ── LLM caller ───────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.ai_hub is None:
            return None

        for method_name in ("_query_openai", "_query_perplexity", "_query_anthropic", "_query_gemini"):
            method = getattr(self.ai_hub, method_name, None)
            if method is None:
                continue
            try:
                result = method(prompt)
                if result and result.get("success"):
                    text = result.get("response", "")
                    if text and len(text) > 20:
                        return text
            except Exception as exc:
                logger.debug("SubTaskResearcher LLM %s failed: %s", method_name, exc)

        return None
