"""
DMAI DeepResearchOrchestrator
==============================
Top-level coordinator for multi-hop research. Equivalent to Perplexity Pro Search:
  1. QueryDecomposer  → break query into 3-6 sub-questions
  2. SubTaskResearcher (per sub-question) → search + fetch + synthesise
  3. CitationManager  → deduplicate + renumber all sources globally
  4. Consolidation LLM pass → unified markdown report with inline citations

Depth modes
-----------
  quick    — 3 sub-questions, 0 page fetches  (~5s with keys)
  standard — 5 sub-questions, 1 page fetch   (~15s with keys)
  deep     — 6 sub-questions, 2 page fetches (~30s with keys)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .search_grounding_engine import SearchGroundingEngine
from .query_decomposer        import QueryDecomposer
from .subtask_researcher      import SubTaskResearcher
from .citation_manager        import CitationManager

logger = logging.getLogger(__name__)

_CONSOLIDATION_PROMPT = """\
You are a senior research analyst. You have received independent research reports \
for several sub-questions that together answer a complex user query. Your task is \
to synthesise them into ONE coherent, well-structured markdown research report.

Rules:
- Structure the report with ## headers for each major theme
- Use inline citation markers [N] from the sub-reports wherever you reference a fact
- Do NOT invent facts not present in the sub-reports
- Do NOT repeat the same information twice
- Write in a clear, professional style
- Aim for 400-800 words in the body

Original Query: {query}

Sub-Reports:
{sub_reports_block}

Write the unified research report (markdown, starting with # {title}):"""

_DEPTH_CONFIG = {
    "quick":    {"num_questions": 3, "fetch_pages": 0},
    "standard": {"num_questions": 5, "fetch_pages": 1},
    "deep":     {"num_questions": 6, "fetch_pages": 2},
}


class DeepResearchOrchestrator:
    """
    Perplexity Pro Search equivalent for DMAI.
    """

    def __init__(self, ai_hub=None, data_path: str = "data/research/deep"):
        self.ai_hub        = ai_hub
        self.data_path     = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.search_engine = SearchGroundingEngine()
        self.decomposer    = QueryDecomposer(ai_hub=ai_hub)
        self.citation_mgr  = CitationManager()

        logger.info(
            "DeepResearchOrchestrator ready — search: %s, llm: %s",
            self.search_engine.primary,
            "available" if ai_hub else "none",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def research(self, query: str, depth: str = "standard") -> Dict:
        """
        Run the full multi-hop research pipeline on *query*.

        Parameters
        ----------
        query : str
            The user's complex research question.
        depth : "quick" | "standard" | "deep"

        Returns
        -------
        {
            "success": bool,
            "query": str,
            "depth": str,
            "sub_questions_researched": int,
            "final_report": str,        # full markdown including ## Sources
            "sources": [...],
            "sub_task_reports": [...],
            "processing_time_seconds": float,
            "search_provider": str,
            "llm_used": str,
            "decomposition_method": str,
            "status": "complete" | "partial" | "failed"
        }
        """
        start_time = time.time()
        cfg = _DEPTH_CONFIG.get(depth, _DEPTH_CONFIG["standard"])
        num_questions = cfg["num_questions"]
        fetch_pages   = cfg["fetch_pages"]

        self.citation_mgr.reset()
        llm_used = "none"

        # ── Step 1: Decompose ─────────────────────────────────────────────
        decomp = self.decomposer.decompose(query)
        if not decomp["success"]:
            return self._failed_result(query, depth, "Query decomposition failed", start_time)

        sub_questions: List[str] = decomp["sub_questions"][:num_questions]
        decomp_method = decomp.get("decomposition_method", "heuristic")
        logger.info(
            "DeepResearch: %d sub-questions (method=%s) for: %s",
            len(sub_questions), decomp_method, query[:80],
        )

        # ── Step 2: Research each sub-question ────────────────────────────
        sub_task_reports: List[Dict] = []
        successful_count = 0

        for i, sq in enumerate(sub_questions, start=1):
            logger.info("  [%d/%d] Researching: %s", i, len(sub_questions), sq[:80])
            researcher = SubTaskResearcher(
                search_engine=self.search_engine,
                ai_hub=self.ai_hub,
            )
            report = researcher.research(sq, parent_query=query, fetch_pages=fetch_pages)

            # Register sources + remap inline citations globally
            if report.get("success") and report.get("sources"):
                id_map = self.citation_mgr.register_sources(report["sources"])
                report["synthesis"] = self.citation_mgr.remap_citations(
                    report.get("synthesis", ""), id_map
                )
                # Update source citation_ids to global values
                for src in report["sources"]:
                    src["citation_id"] = id_map.get(src["citation_id"], src["citation_id"])
                successful_count += 1

                if report.get("confidence") == "high":
                    llm_used = "LLM-synthesised"

            sub_task_reports.append(report)

        if successful_count == 0:
            return self._failed_result(
                query, depth,
                "All sub-question searches failed — no search API key may be configured.",
                start_time,
            )

        # ── Step 3: Consolidate ───────────────────────────────────────────
        final_body, consolidation_method = self._consolidate(query, sub_task_reports)

        if consolidation_method == "llm":
            llm_used = "LLM-synthesised + LLM-consolidated"

        # ── Step 4: Append references ─────────────────────────────────────
        references_block = self.citation_mgr.format_references()
        final_report = final_body.strip()
        if references_block:
            final_report = final_report + "\n\n" + references_block

        # ── Step 5: Persist ───────────────────────────────────────────────
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = self.data_path / f"report_{timestamp}.json"
        result = {
            "success":                   True,
            "query":                     query,
            "depth":                     depth,
            "sub_questions_researched":  successful_count,
            "final_report":              final_report,
            "sources":                   self.citation_mgr.get_all_citations(),
            "sub_task_reports":          sub_task_reports,
            "processing_time_seconds":   round(time.time() - start_time, 2),
            "search_provider":           self.search_engine.primary,
            "llm_used":                  llm_used,
            "decomposition_method":      decomp_method,
            "research_angle":            decomp.get("research_angle", ""),
            "status":                    "complete" if successful_count == len(sub_questions) else "partial",
        }

        try:
            with open(report_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("DeepResearch report saved: %s", report_file)
        except Exception as exc:
            logger.warning("Could not save report: %s", exc)

        logger.info(
            "DeepResearch complete — %d/%d sub-questions, %.1fs, status=%s",
            successful_count, len(sub_questions),
            result["processing_time_seconds"], result["status"],
        )
        return result

    def get_status(self) -> Dict:
        """Quick health check — no API calls made."""
        return {
            "search_provider":  self.search_engine.primary,
            "search_ready":     self.search_engine.primary != "none",
            "llm_ready":        self.ai_hub is not None,
            "data_path":        str(self.data_path),
            "tavily_configured": self.search_engine.get_status()["tavily_configured"],
            "brave_configured":  self.search_engine.get_status()["brave_configured"],
        }

    def list_past_reports(self, limit: int = 10) -> List[Dict]:
        """List the most recent saved research reports (metadata only)."""
        reports = []
        try:
            files = sorted(self.data_path.glob("report_*.json"), reverse=True)
            for f in files[:limit]:
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    reports.append({
                        "file":                f.name,
                        "query":               d.get("query", "")[:100],
                        "depth":               d.get("depth", ""),
                        "status":              d.get("status", ""),
                        "processing_time_seconds": d.get("processing_time_seconds"),
                        "sources_count":       len(d.get("sources", [])),
                        "timestamp":           f.stem.replace("report_", ""),
                    })
                except Exception:
                    pass
        except Exception:
            pass
        return reports

    # ── Consolidation ─────────────────────────────────────────────────────────

    def _consolidate(self, query: str, sub_task_reports: List[Dict]) -> tuple:
        """
        Build a single unified report from all sub-task syntheses.
        Returns (markdown_text, method) where method = "llm" | "concat".
        """
        successful = [r for r in sub_task_reports if r.get("success")]
        if not successful:
            return "No research data could be gathered.", "concat"

        # Build sub-reports block (truncate each to keep prompt manageable)
        sub_reports_parts = []
        for r in successful:
            sq        = r.get("sub_question", "")
            synthesis = r.get("synthesis", "")[:800]
            sub_reports_parts.append(f"### {sq}\n{synthesis}")
        sub_reports_block = "\n\n".join(sub_reports_parts)

        # Try LLM consolidation
        if self.ai_hub is not None:
            title = self._make_title(query)
            prompt = _CONSOLIDATION_PROMPT.format(
                query=query,
                sub_reports_block=sub_reports_block,
                title=title,
            )
            raw = self._call_llm(prompt)
            if raw and len(raw) > 100:
                return raw.strip(), "llm"

        # Fallback: concatenate under headers
        lines = [f"# {self._make_title(query)}", ""]
        for r in successful:
            sq = r.get("sub_question", "")
            synthesis = r.get("synthesis", "")
            lines.append(f"## {sq}")
            lines.append(synthesis)
            lines.append("")

        return "\n".join(lines), "concat"

    # ── Helpers ───────────────────────────────────────────────────────────────

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
                    if text and len(text) > 50:
                        return text
            except Exception as exc:
                logger.debug("Consolidation LLM %s failed: %s", method_name, exc)
        return None

    @staticmethod
    def _make_title(query: str) -> str:
        """Convert a query string into a clean report title."""
        title = query.strip().rstrip("?").strip()
        if len(title) > 80:
            title = title[:77] + "..."
        return title

    @staticmethod
    def _failed_result(query: str, depth: str, reason: str, start_time: float) -> Dict:
        return {
            "success":                   False,
            "query":                     query,
            "depth":                     depth,
            "sub_questions_researched":  0,
            "final_report":              f"Research failed: {reason}",
            "sources":                   [],
            "sub_task_reports":          [],
            "processing_time_seconds":   round(time.time() - start_time, 2),
            "search_provider":           "none",
            "llm_used":                  "none",
            "decomposition_method":      "none",
            "research_angle":            "",
            "status":                    "failed",
            "error":                     reason,
        }
