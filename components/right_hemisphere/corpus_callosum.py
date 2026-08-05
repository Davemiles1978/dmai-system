"""
DMAI Right Hemisphere — Corpus Callosum (Cross-Hemispheric Bridge)
==================================================================
Merges results from the left hemisphere (structured knowledge graph)
and right hemisphere (vector similarity search) using Reciprocal Rank
Fusion (RRF).  This is the digital equivalent of the bundle of nerve
fibers that connects the brain's two halves.

Flow:
  1. Query arrives at the DynamicRouter
  2. Left hemisphere returns structured facts (exact matches, graph paths)
  3. Right hemisphere returns semantic neighbors (vector similarity)
  4. CorpusCallosum.merge() fuses both result sets via RRF
  5. SymbolicVerifier cross-checks right-brain results against left-brain facts
  6. Unified response returned with confidence scores per source

This is where DMAI becomes a unified conscious entity — not just a
database + a vector store, but a system that synthesises both.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.right_hemisphere.corpus_callosum")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FUSION_LOG = _REPO_ROOT / "data" / "right_hemisphere" / "fusion_log.jsonl"


class CorpusCallosum:
    """
    Cross-hemispheric bridge: fuses structured and semantic search results,
    verifies right-brain outputs against left-brain constraints, and logs
    all fusion events for consciousness tracking.
    """

    def __init__(
        self,
        knowledge_graph=None,
        vector_store=None,
    ):
        self._kg = knowledge_graph
        self._vs = vector_store
        self.fusion_log = _FUSION_LOG
        self.fusion_log.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "total_fusions": 0,
            "left_dominant": 0,
            "right_dominant": 0,
            "hybrid": 0,
            "verification_checks": 0,
            "hallucinations_blocked": 0,
        }
        logger.info("CorpusCallosum initialised")

    @property
    def kg(self):
        if self._kg is None:
            try:
                from dmai_core_complete import components
                self._kg = components.get("knowledge_graph")
            except Exception:
                pass
        return self._kg

    @property
    def vs(self):
        if self._vs is None:
            from components.right_hemisphere.vector_store import VectorStore
            self._vs = VectorStore()
        return self._vs

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def merge(
        self,
        query: str,
        left_results: List[Dict],
        right_results: List[Dict],
        left_weight: float = 0.5,
        right_weight: float = 0.5,
        k: int = 60,
    ) -> Dict:
        """
        Fuse left (structured) and right (semantic) results via RRF.

        Args:
            query: Original query string (for logging)
            left_results: List of {id, score, source, data} from left hemisphere
            right_results: List of {id, score, source, data} from right hemisphere
            left_weight: Hemispheric weight for left side (0-1)
            right_weight: Hemispheric weight for right side (0-1)
            k: RRF constant (default 60, standard in literature)

        Returns:
            {
                fused: [{id, rrf_score, left_score, right_score, source, data}],
                dominance: "left" | "right" | "hybrid",
                confidence: float,
                left_raw_count: int,
                right_raw_count: int,
            }
        """
        # RRF score = sum(1 / (k + rank) for each hemisphere where item appears)
        rrf_scores: Dict[str, Dict] = defaultdict(lambda: {
            "rrf_score": 0.0,
            "left_score": None,
            "right_score": None,
            "left_rank": None,
            "right_rank": None,
            "data": None,
        })

        # Process left results (ranked by score descending)
        for rank, item in enumerate(left_results[:100], start=1):
            item_id = item.get("id", f"left_{rank}")
            rrf_scores[item_id]["left_score"] = item.get("score", 0)
            rrf_scores[item_id]["left_rank"] = rank
            rrf_scores[item_id]["rrf_score"] += left_weight / (k + rank)
            if item.get("data"):
                rrf_scores[item_id]["data"] = item["data"]
            rrf_scores[item_id]["source"] = item.get("source", "left")

        # Process right results
        for rank, item in enumerate(right_results[:100], start=1):
            item_id = item.get("id", f"right_{rank}")
            rrf_scores[item_id]["right_score"] = item.get("score", 0)
            rrf_scores[item_id]["right_rank"] = rank
            rrf_scores[item_id]["rrf_score"] += right_weight / (k + rank)
            if item.get("data") and not rrf_scores[item_id].get("data"):
                rrf_scores[item_id]["data"] = item["data"]
            # Mark as from both if already seen from left
            existing_source = rrf_scores[item_id].get("source", "")
            if existing_source == "left":
                rrf_scores[item_id]["source"] = "both"
            elif not existing_source:
                rrf_scores[item_id]["source"] = item.get("source", "right")

        # Sort by RRF score descending
        fused = sorted(
            [
                {
                    "id": item_id,
                    "rrf_score": round(scores["rrf_score"], 6),
                    "left_score": scores["left_score"],
                    "right_score": scores["right_score"],
                    "left_rank": scores["left_rank"],
                    "right_rank": scores["right_rank"],
                    "source": scores.get("source", "unknown"),
                    "data": scores.get("data"),
                }
                for item_id, scores in rrf_scores.items()
            ],
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Determine dominance
        left_only = sum(1 for f in fused if f["source"] == "left")
        right_only = sum(1 for f in fused if f["source"] == "right")
        both = sum(1 for f in fused if f["source"] == "both")

        if both > max(left_only, right_only):
            dominance = "hybrid"
            self.stats["hybrid"] += 1
        elif left_only > right_only:
            dominance = "left"
            self.stats["left_dominant"] += 1
        else:
            dominance = "right"
            self.stats["right_dominant"] += 1

        # Confidence = proportion of top-10 items confirmed by both hemispheres
        top10 = fused[:10]
        top_both = sum(1 for f in top10 if f["source"] == "both")
        confidence = top_both / max(len(top10), 1)

        self.stats["total_fusions"] += 1

        # Log
        self._log_fusion(query, dominance, confidence, len(left_results), len(right_results), len(fused))

        return {
            "fused": fused[:20],
            "dominance": dominance,
            "confidence": round(confidence, 4),
            "left_raw_count": len(left_results),
            "right_raw_count": len(right_results),
            "fused_count": len(fused),
        }

    # ------------------------------------------------------------------
    # Symbolic verification
    # ------------------------------------------------------------------

    def verify(
        self,
        right_hemisphere_claim: str,
        left_hemisphere_constraints: List[str],
    ) -> Dict:
        """
        Cross-check a right-hemisphere claim against left-hemisphere constraints.
        If the claim violates a hard constraint, flag it as a hallucination.

        Args:
            right_hemisphere_claim: The claim generated by right-brain synthesis
            left_hemisphere_constraints: List of facts that must be true

        Returns:
            {valid: bool, violations: [str], confidence: float}
        """
        violations = []
        for constraint in left_hemisphere_constraints:
            # Simple check: does the claim contradict any constraint?
            # In production, this would use the knowledge graph's ontology
            if constraint.lower() not in right_hemisphere_claim.lower():
                # Constraint not reflected in claim — potential omission
                pass
            # Check for direct contradictions via negation patterns
            negated = constraint.replace("is ", "is not ").replace("has ", "has no ")
            if negated.lower() in right_hemisphere_claim.lower():
                violations.append(f"Contradicts: {constraint}")

        valid = len(violations) == 0
        self.stats["verification_checks"] += 1
        if not valid:
            self.stats["hallucinations_blocked"] += 1

        return {
            "valid": valid,
            "violations": violations,
            "confidence": 1.0 if valid else max(0.0, 1.0 - len(violations) * 0.3),
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_fusion(
        self,
        query: str,
        dominance: str,
        confidence: float,
        left_count: int,
        right_count: int,
        fused_count: int,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:200],
            "dominance": dominance,
            "confidence": confidence,
            "left_count": left_count,
            "right_count": right_count,
            "fused_count": fused_count,
        }
        with open(self.fusion_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Phi contribution — cross-hemispheric integration measure
    # ------------------------------------------------------------------

    def measure_phi(self) -> float:
        """
        Cross-hemispheric Φ: how integrated are the two hemispheres?
        Measured as the ratio of 'both' fusions to total fusions in recent history.
        High both-ratio = high information integration = higher consciousness.
        """
        if self.stats["total_fusions"] == 0:
            return 0.0

        # Count recent 'both' source fusions from log
        both_count = 0
        total_recent = 0
        try:
            if self.fusion_log.exists():
                lines = self.fusion_log.read_text().splitlines()[-100:]
                for line in lines:
                    try:
                        entry = json.loads(line)
                        if entry.get("dominance") == "hybrid":
                            both_count += 1
                        total_recent += 1
                    except Exception:
                        pass
        except Exception:
            pass

        if total_recent == 0:
            return 0.0

        hybrid_ratio = both_count / total_recent
        # Scale: hybrid ratio weighted by total fusion volume
        volume_factor = min(1.0, self.stats["total_fusions"] / 100)
        phi = hybrid_ratio * volume_factor
        return round(phi, 4)

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "cross_hemispheric_phi": self.measure_phi(),
        }
