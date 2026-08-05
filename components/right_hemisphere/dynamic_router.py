"""
DMAI Right Hemisphere — Dynamic Router
=======================================
Analyzes incoming prompts/queries and calculates hemispheric influence
weights.  Routes processing to the appropriate hemisphere(s).

Left-hemisphere signals: exact, count, sum, budget, code, syntax, schema
Right-hemisphere signals: vibe, feel, creative, imagine, explore, context

Returns (left_weight, right_weight) and a routing decision:
  - "left"  : deterministic graph query, skip vector search
  - "right" : semantic vector search, skip graph query
  - "hybrid" : run both, fuse via CorpusCallosum
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Tuple

logger = logging.getLogger("dmai.right_hemisphere.dynamic_router")


class DynamicRouter:
    """
    Linguistic analysis router that computes hemispheric weights from prompt features.
    """

    # Tokens indicating need for structured/precise processing
    LEFT_SIGNALS = {
        "exact", "count", "sum", "budget", "total", "before", "after",
        "on track", "compliant", "structure", "id", "code", "schema",
        "define", "calculate", "compute", "list", "query", "table",
        "syntax", "compile", "parse", "validate", "verify", "check",
        "sql", "cypher", "api", "endpoint", "route", "method",
        "commit", "push", "deploy", "status", "error code",
    }

    # Tokens indicating need for semantic/contextual processing
    RIGHT_SIGNALS = {
        "vibe", "feel", "creative", "brainstorm", "imagine", "maybe",
        "explore", "abstract", "morale", "culture", "innovative", "nuance",
        "similar", "like", "analogy", "metaphor", "context", "meaning",
        "why", "how come", "what if", "suppose", "could", "might",
        "suggest", "recommend", "opinion", "trend", "pattern",
        "sentiment", "emotion", "tone", "style", "aesthetic",
    }

    def __init__(self):
        self.stats = {
            "total_routes": 0,
            "left_routes": 0,
            "right_routes": 0,
            "hybrid_routes": 0,
        }
        logger.info("DynamicRouter initialised")

    def calculate_weights(self, prompt: str) -> Tuple[float, float]:
        """
        Calculate hemispheric weights from prompt analysis.
        Returns (left_weight, right_weight) both in range [0, 1].
        """
        normalized = prompt.lower()
        tokens = set(normalized.split())

        # Count signal tokens
        left_score = sum(1 for t in tokens if t in self.LEFT_SIGNALS)
        # Check multi-word signals
        for signal in self.LEFT_SIGNALS:
            if " " in signal and signal in normalized:
                left_score += 1
        right_score = sum(1 for t in tokens if t in self.RIGHT_SIGNALS)
        for signal in self.RIGHT_SIGNALS:
            if " " in signal and signal in normalized:
                right_score += 1

        # Structural features
        if any(c.isdigit() for c in prompt):
            left_score += 1.5  # Numbers → structured processing
        if '"' in prompt or "'" in prompt:
            left_score += 1.0  # Quoted strings → exact match
        if "?" in prompt:
            right_score += 0.5  # Questions often need contextual understanding
        if len(prompt) > 200:
            right_score += 1.0  # Long form → likely narrative/contextual
        if "```" in prompt or "def " in prompt or "import " in prompt:
            left_score += 2.0  # Code → deterministic parsing

        # Sigmoid normalization
        total = left_score + right_score
        if total == 0:
            return 0.5, 0.5  # Default equilibrium

        alpha = 1.0 / (1.0 + math.exp(-(left_score - right_score)))
        left_weight = round(alpha, 3)
        right_weight = round(1.0 - alpha, 3)

        return left_weight, right_weight

    def route(self, prompt: str) -> Dict:
        """
        Analyze prompt and return routing decision with weights.
        """
        left_w, right_w = self.calculate_weights(prompt)
        self.stats["total_routes"] += 1

        if left_w > 0.7:
            decision = "left"
            self.stats["left_routes"] += 1
        elif right_w > 0.7:
            decision = "right"
            self.stats["right_routes"] += 1
        else:
            decision = "hybrid"
            self.stats["hybrid_routes"] += 1

        logger.debug(
            "DynamicRouter: prompt='%s...' -> left=%.3f right=%.3f decision=%s",
            prompt[:60], left_w, right_w, decision,
        )

        return {
            "decision": decision,
            "left_weight": left_w,
            "right_weight": right_w,
            "prompt_preview": prompt[:100],
        }

    def get_hemispheric_balance(self) -> float:
        """
        Return the current hemispheric balance as a single score.
        1.0 = perfectly balanced usage of both hemispheres.
        0.0 = entirely one-sided.
        """
        total = self.stats["total_routes"]
        if total == 0:
            return 0.5
        hybrid_pct = self.stats["hybrid_routes"] / total
        left_pct = self.stats["left_routes"] / total
        right_pct = self.stats["right_routes"] / total
        # Balance = hybrid bonus + penalty for lopsidedness
        imbalance = abs(left_pct - right_pct)
        balance = (hybrid_pct * 1.0) + ((1.0 - hybrid_pct) * (1.0 - imbalance))
        return round(balance, 4)

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "hemispheric_balance": self.get_hemispheric_balance(),
        }
