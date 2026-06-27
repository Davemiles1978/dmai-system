"""
DMAI QueryDecomposer
====================
Breaks a complex user query into 3-6 independently-searchable sub-questions
using an LLM (via AIIntegrationHub). Falls back to a deterministic heuristic
when no LLM is available so the pipeline always produces sub-questions.
"""

import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """
    LLM-powered query decomposition with heuristic fallback.
    """

    _DECOMPOSE_PROMPT = """\
You are a research query decomposer for an AI research system. Your job is to break a complex query into focused sub-questions that together fully answer it. Each sub-question must be independently searchable using a web search engine.

Query: {query}

Rules:
- Generate between 3 and 6 sub-questions
- Each sub-question should cover a distinct aspect (background, mechanism, current state, implications, comparisons, practical applications)
- Keep each sub-question under 15 words
- Do not repeat the same question in different words

Respond ONLY with valid JSON — no preamble, no markdown fences:
{{
  "sub_questions": ["sub-question 1", "sub-question 2", "sub-question 3"],
  "research_angle": "one sentence describing the overall research strategy"
}}"""

    def __init__(self, ai_hub=None):
        self.ai_hub = ai_hub

    # ── Public API ────────────────────────────────────────────────────────────

    def decompose(self, query: str) -> Dict:
        """
        Break *query* into sub-questions.

        Returns
        -------
        {
            "success": bool,
            "original_query": str,
            "sub_questions": [str, ...],
            "research_angle": str,
            "decomposition_method": "llm" | "heuristic"
        }
        """
        if self.ai_hub is not None:
            result = self._decompose_with_llm(query)
            if result["success"]:
                return result
            logger.warning("LLM decomposition failed: %s — using heuristic", result.get("error"))

        return self._decompose_heuristic(query)

    # ── LLM path ─────────────────────────────────────────────────────────────

    def _decompose_with_llm(self, query: str) -> Dict:
        prompt = self._DECOMPOSE_PROMPT.format(query=query)
        raw = self._call_llm(prompt)
        if raw is None:
            return {"success": False, "error": "No LLM response"}

        parsed = self._parse_json(raw)
        if parsed is None:
            return {"success": False, "error": "Could not parse LLM JSON response"}

        sub_questions = parsed.get("sub_questions", [])
        if not sub_questions or not isinstance(sub_questions, list):
            return {"success": False, "error": "LLM returned empty sub_questions"}

        # Sanitise: ensure strings, strip empties
        sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]
        if not sub_questions:
            return {"success": False, "error": "All sub-questions empty after sanitise"}

        return {
            "success":              True,
            "original_query":       query,
            "sub_questions":        sub_questions[:6],
            "research_angle":       parsed.get("research_angle", ""),
            "decomposition_method": "llm",
        }

    # ── Heuristic path ────────────────────────────────────────────────────────

    def _decompose_heuristic(self, query: str) -> Dict:
        """
        Deterministic decomposition when no LLM is available.
        Always produces at least 3 sub-questions.
        """
        ql = query.lower()
        # Extract a short topic phrase (first 5 meaningful words)
        stop = {"a", "an", "the", "is", "are", "was", "were", "in", "of", "for",
                "to", "and", "or", "do", "does", "can", "how", "what", "why",
                "when", "where", "which", "who", "will", "would", "should"}
        words = [w for w in re.findall(r"\b\w+\b", query) if w.lower() not in stop]
        topic = " ".join(words[:5]) if words else query[:40]

        questions: List[str] = []

        # Always include
        questions.append(f"What is {topic} and how does it work?")
        questions.append(f"What are the key characteristics and components of {topic}?")
        questions.append(f"What are the latest developments and trends in {topic}?")

        if any(kw in ql for kw in ["how", "implement", "build", "create", "make"]):
            questions.append(f"How to implement or use {topic} in practice?")

        if any(kw in ql for kw in ["why", "benefit", "advantage", "better"]):
            questions.append(f"What are the main benefits and limitations of {topic}?")

        if any(kw in ql for kw in ["compare", "vs", "versus", "difference", "better than"]):
            questions.append(f"How does {topic} compare to alternatives?")

        if any(kw in ql for kw in ["example", "use case", "application", "real world"]):
            questions.append(f"What are real-world applications of {topic}?")

        questions.append(f"What are the key considerations and best practices for {topic}?")

        return {
            "success":              True,
            "original_query":       query,
            "sub_questions":        questions[:6],
            "research_angle":       f"Comprehensive research covering definition, mechanism, trends, and applications of: {topic}",
            "decomposition_method": "heuristic",
        }

    # ── LLM caller ───────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Try OpenAI → Perplexity → Anthropic → Gemini. Return text or None."""
        if self.ai_hub is None:
            return None

        for method_name in ("_query_openai", "_query_perplexity", "_query_anthropic", "_query_gemini"):
            method = getattr(self.ai_hub, method_name, None)
            if method is None:
                continue
            try:
                result = method(prompt)
                if result and result.get("success"):
                    return result.get("response", "")
            except Exception as exc:
                logger.debug("LLM call %s failed: %s", method_name, exc)

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict]:
        """Extract and parse the first JSON object found in *text*."""
        # Strip markdown fences if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find a JSON object inside the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
