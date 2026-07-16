"""
self_judge.py
=============

DMAI's own judgement primitive. Given a candidate "seed" (a concept she
might promote into a capability, or any other proposal that arrives at
her doorstep), she decides whether the seed is beneficial to her using
**only her own internal knowledge**: vocabulary, insights, capability
registry, KPI history.

Design contract
---------------

1.  **DMAI is never bypassed.** This module does not call any external
    LLM, web search, or third-party service. Every signal it uses comes
    from tables DMAI has written herself.

2.  **Three verdicts, not two.** DMAI can decide *"accept"*,
    *"reject"*, or *"defer"*. Deferral is not a failure \u2014 it means she
    has recognised a knowledge gap and is asking the knowledge acquirer
    to fill it. The caller re-judges the seed on a later pass with the
    (now expanded) knowledge corpus.

3.  **Confidence, not vibes.** DMAI produces a numeric confidence per
    seed drawn from four independent signals. The signals are exposed
    to the caller so rejections are auditable.

Signals
-------

+-------------------------+-----------------------------------------------+
| ``vocab_coverage``      | fraction of content tokens in the seed that   |
|                         | already exist in DMAI's vocabulary table.     |
+-------------------------+-----------------------------------------------+
| ``insight_neighbourhood``| max token-overlap fraction against DMAI's    |
|                         | top-K nearest existing insights.              |
+-------------------------+-----------------------------------------------+
| ``kpi_linkage``         | 1.0 if the seed touches any tracked KPI      |
|                         | keyword, else 0.0                             |
+-------------------------+-----------------------------------------------+
| ``diversity_pressure``  | 1.0 if the seed pushes toward an              |
|                         | under-represented capability_type, 0.0 if it  |
|                         | reinforces the dominant one                   |
+-------------------------+-----------------------------------------------+

Confidence is a weighted average. Thresholds:

* confidence \u2265 ``ACCEPT_THRESHOLD``  \u2192 verdict = accept
* confidence \u2264 ``REJECT_THRESHOLD``  \u2192 verdict = reject
* otherwise, or if vocab_coverage is below the "I don't understand this"
  floor \u2192 verdict = defer with a ``knowledge_gap`` describing what she
  needs to learn.

The caller (seed_capability_promoter) hands deferred seeds to the
knowledge_acquirer, which fills the gap by consulting DMAI's own
knowledge graph first, then web search, then \u2014 only as a last resort \u2014
an external LLM. Whatever it finds gets written back to DMAI's
vocabulary + insights + knowledge_graph so the next round of judgement
has more ground to stand on.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────

ACCEPT_THRESHOLD          = 0.65   # confidence >= this \u2192 accept
REJECT_THRESHOLD          = 0.30   # confidence <= this \u2192 reject

# If vocab_coverage falls below this, DMAI declares the seed
# incomprehensible regardless of the other signals \u2014 she can't reason
# about words she doesn't know.
VOCAB_COVERAGE_FLOOR      = 0.40

# PR PP: gap-driven seeds come from DMAI's own self-scanner. Their
# vocabulary is by definition domain-specific (module names, slugs,
# infrastructure terms like "postgres", "cutover", "materialiser").
# We know these are legitimate because DMAI wrote them. Use a much
# lower floor for this channel so the review step doesn't reject
# gap items on grammar/domain-vocab grounds.
GAP_VOCAB_COVERAGE_FLOOR  = 0.10

# Channel names that get the relaxed vocab floor. Extendable.
_RELAXED_VOCAB_CHANNELS = frozenset({
    "gap_driven", "self_scanner", "backlog_seed", "self_gen",
})

# Weights for the confidence combo. Sum to 1.0.
WEIGHT_VOCAB              = 0.30
WEIGHT_INSIGHT_NEIGHBOUR  = 0.25
WEIGHT_KPI                = 0.20
WEIGHT_DIVERSITY          = 0.25

# Nearest-neighbour insight search: how many insights to pull for the
# similarity check. Full-text LIKE is cheap for a small K.
INSIGHT_NEIGHBOUR_K       = 25

# Insight-neighbourhood interpretation:
#   0.0  \u2192 no overlap \u2192 could be a novel frontier concept OR garbage
#   ~0.5 \u2192 sweet spot \u2014 related but not duplicate
#   >=0.9 \u2192 duplicate of something DMAI already thought about \u2192 penalise
INSIGHT_DUPLICATE_PENALTY_ABOVE = 0.9


# KPI keywords DMAI actively tracks. Extend as new KPIs come online.
# Matching is case-insensitive substring on tokenised seed text.
KPI_KEYWORDS = frozenset({
    "sharpe", "drawdown", "profit", "loss", "yield", "roi",
    "kelly", "stake", "ev", "odds", "edge",
    "capability", "insight", "vocab", "diversity", "entropy",
    "stage", "kpi", "monitor", "health", "readiness",
    "revenue", "funding", "trade", "bet", "position",
    "agent", "autonomous", "self", "heal", "repair",
})


# Stop-words to exclude when tokenising seed text (English basics only).
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "into", "onto", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "we", "you", "they", "he", "she", "them", "us", "our", "their",
    "if", "then", "else", "so", "than", "such", "not", "no", "nor",
    "but", "any", "some", "all", "each", "most", "more", "less",
    "over", "under", "above", "below", "up", "down", "out", "off",
    "can", "could", "would", "should", "may", "might", "must", "will",
    "do", "does", "did", "done", "have", "has", "had", "having",
})


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class JudgeSignals:
    vocab_coverage:        float = 0.0
    insight_neighbourhood: float = 0.0
    kpi_linkage:           float = 0.0
    diversity_pressure:    float = 0.0
    unknown_tokens:        List[str] = field(default_factory=list)
    nearest_insight_id:    Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    verdict:       str                 # "accept" | "reject" | "defer"
    confidence:    float
    reason:        str
    signals:       JudgeSignals
    knowledge_gap: Optional[str] = None  # populated when verdict == "defer"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verdict":       self.verdict,
            "confidence":    round(self.confidence, 4),
            "reason":        self.reason,
            "signals":       self.signals.as_dict(),
            "knowledge_gap": self.knowledge_gap,
        }


# ── Text tokenisation ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")


def _tokenise(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens, filtering stopwords + very
    short tokens (which are almost always noise)."""
    if not text:
        return []
    raw = _TOKEN_RE.findall(text.lower())
    return [t for t in raw if len(t) > 2 and t not in _STOPWORDS]


def _seed_text(seed: Dict[str, Any]) -> str:
    """The text DMAI reasons about \u2014 concept + any insight_text the
    injector attached to the seed."""
    parts: List[str] = []
    for k in ("concept", "insight_text", "description"):
        v = seed.get(k)
        if v:
            parts.append(str(v))
    return " \n ".join(parts)


# ── Signal 1: vocabulary coverage ────────────────────────────────────────

def _vocab_coverage(conn: Optional[sqlite3.Connection],
                    tokens: List[str]) -> Tuple[float, List[str]]:
    """Fraction of ``tokens`` that already exist in DMAI's vocabulary
    table. Returns (coverage, unknown_tokens).

    Falls back to 1.0 when the vocabulary table is missing (fresh
    install), because rejecting every seed on a fresh install would
    prevent DMAI from ever bootstrapping. On a fresh install the
    knowledge_acquirer's very first pass populates the table.
    """
    if not tokens:
        return 0.0, []
    if conn is None:
        return 1.0, []

    unknown: List[str] = []
    known = 0
    try:
        # Use a temp table for the batch check to keep the query cheap
        # even when the vocabulary table is 100k+ rows.
        cur = conn.cursor()
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tok_probe(word TEXT PRIMARY KEY)")
        cur.execute("DELETE FROM _tok_probe")
        cur.executemany(
            "INSERT OR IGNORE INTO _tok_probe (word) VALUES (?)",
            [(t,) for t in set(tokens)],
        )
        # Try to intersect against the vocabulary table.
        rows = cur.execute(
            "SELECT p.word FROM _tok_probe p "
            "JOIN vocabulary v ON v.word = p.word"
        ).fetchall()
        known_set = {r[0] for r in rows}
        for t in tokens:
            if t in known_set:
                known += 1
            else:
                unknown.append(t)
    except sqlite3.OperationalError:
        # Vocabulary table doesn't exist yet.
        return 1.0, []
    finally:
        try:
            conn.execute("DROP TABLE IF EXISTS _tok_probe")
        except sqlite3.OperationalError:
            pass

    coverage = known / len(tokens) if tokens else 0.0
    # Dedupe unknown tokens, preserve order.
    seen: set = set()
    dedup_unknown: List[str] = []
    for t in unknown:
        if t not in seen:
            seen.add(t)
            dedup_unknown.append(t)
    return coverage, dedup_unknown


# ── Signal 2: insight neighbourhood ──────────────────────────────────────

def _insight_neighbourhood(conn: Optional[sqlite3.Connection],
                            tokens: List[str],
                            k: int = INSIGHT_NEIGHBOUR_K
                            ) -> Tuple[float, Optional[str]]:
    """Max token-overlap fraction between the seed's tokens and DMAI's
    existing insights. Returns (max_overlap, nearest_insight_id).

    Overlap is Jaccard-lite: |seed \u2229 insight| / |seed| \u2014 measures how
    much of the seed's content DMAI has already thought about.
    """
    if not tokens or conn is None:
        return 0.0, None

    seed_set = set(tokens)
    if not seed_set:
        return 0.0, None

    try:
        # Fetch a small K of *candidate* insights that share at least
        # one token. Cheap prefilter: LIKE on the most distinctive
        # (longest) tokens. Then compute proper overlap in Python.
        long_tokens = sorted(seed_set, key=len, reverse=True)[:4]
        if not long_tokens:
            return 0.0, None
        clauses = " OR ".join(["LOWER(insight_text) LIKE ?"] * len(long_tokens))
        params  = [f"%{t}%" for t in long_tokens]
        rows = conn.execute(
            f"SELECT id, insight_text FROM insights WHERE {clauses} LIMIT ?",
            (*params, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0.0, None

    if not rows:
        return 0.0, None

    best_overlap = 0.0
    best_id: Optional[str] = None
    for iid, text in rows:
        insight_tokens = set(_tokenise(text or ""))
        if not insight_tokens:
            continue
        overlap = len(seed_set & insight_tokens) / len(seed_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = str(iid)
    return best_overlap, best_id


# ── Signal 3: KPI linkage ────────────────────────────────────────────────

def _kpi_linkage(tokens: List[str]) -> float:
    """1.0 if the seed's tokens touch any tracked KPI keyword, else 0.0.

    Deliberately binary \u2014 fractional linkage is noise. Either DMAI can
    imagine a path from this seed to a metric she moves, or she can't.
    """
    if not tokens:
        return 0.0
    tokset = set(tokens)
    return 1.0 if (tokset & KPI_KEYWORDS) else 0.0


# ── Signal 4: diversity pressure ─────────────────────────────────────────

def _diversity_pressure(seed: Dict[str, Any],
                        cap_type_dist: Optional[List[Tuple[str, int]]]) -> float:
    """1.0 if the seed pushes toward an under-represented capability_type,
    0.5 for neutral, 0.0 if it reinforces the dominant type.

    Reads the same distribution snapshot the fresh_blood injector uses;
    keeps this module honest about what the rest of the system thinks
    "dominant" means.
    """
    if not cap_type_dist:
        return 0.5

    total = sum(c for _, c in cap_type_dist) or 1
    dominant, dominant_c = cap_type_dist[0]
    dominant_share = dominant_c / total
    # Infer the seed's target capability_type from channel+concept.
    channel = str(seed.get("channel") or "").lower()
    concept = str(seed.get("concept") or "").lower()

    if channel == "diversity" and concept.startswith("diversity_nudge:"):
        # By construction these push under-represented types.
        return 1.0

    # Heuristic: if the concept string mentions the dominant type by
    # name AND the dominant type is genuinely dominant, that's a
    # reinforcement seed.
    if dominant_share > 0.5 and dominant.lower() in concept:
        return 0.0

    # crossover:X\u00d7Y always mixes types \u2014 net positive for diversity.
    if concept.startswith("crossover:"):
        return 0.8

    # Under-represented types (bottom half of the distribution) named
    # in the concept push toward diversity.
    lower_half = {t.lower() for t, _ in cap_type_dist[len(cap_type_dist) // 2:]}
    if any(t in concept for t in lower_half if t):
        return 0.9

    return 0.5


# ── The judge ────────────────────────────────────────────────────────────

def judge_seed(seed: Dict[str, Any],
               conn: Optional[sqlite3.Connection] = None,
               cap_type_dist: Optional[List[Tuple[str, int]]] = None,
               *,
               accept_threshold: float = ACCEPT_THRESHOLD,
               reject_threshold: float = REJECT_THRESHOLD,
               vocab_floor:      float = VOCAB_COVERAGE_FLOOR,
               ) -> Verdict:
    """DMAI judges one seed.

    Parameters
    ----------
    seed
        The fresh_blood row. Requires ``concept`` (and ideally
        ``insight_text``).
    conn
        Live SQLite connection to ``dmai_knowledge.db``. When ``None``,
        DMAI falls back to a "neutral" judgement (vocab_coverage=1.0,
        insight_neighbourhood=0.0). Used only in tests.
    cap_type_dist
        Sorted ``[(type, count), ...]`` distribution snapshot. When
        ``None``, diversity signal defaults to 0.5.
    """
    concept = seed.get("concept")
    if not concept:
        return Verdict(
            verdict=VERDICT_REJECT_STR,
            confidence=0.0,
            reason="seed missing concept field",
            signals=JudgeSignals(),
        )

    # PR PP: apply relaxed vocab floor for gap-driven / self-scanner
    # channels where the seed came from DMAI itself and rejecting on
    # vocab coverage punishes her for words she authored.
    _seed_channel = str(seed.get("channel") or "").lower()
    _effective_vocab_floor = vocab_floor
    if _seed_channel in _RELAXED_VOCAB_CHANNELS:
        _effective_vocab_floor = min(vocab_floor, GAP_VOCAB_COVERAGE_FLOOR)

    text = _seed_text(seed)
    tokens = _tokenise(text)

    vocab_cov, unknown = _vocab_coverage(conn, tokens)
    ins_overlap, nearest_id = _insight_neighbourhood(conn, tokens)
    kpi_link  = _kpi_linkage(tokens)
    div_pressure = _diversity_pressure(seed, cap_type_dist)

    signals = JudgeSignals(
        vocab_coverage=vocab_cov,
        insight_neighbourhood=ins_overlap,
        kpi_linkage=kpi_link,
        diversity_pressure=div_pressure,
        unknown_tokens=unknown[:20],  # cap for readability
        nearest_insight_id=nearest_id,
    )

    # ── Verdict rules ────────────────────────────────────────────────

    # Rule 1: comprehension floor. If DMAI doesn't know most of the
    # words, she cannot judge \u2014 defer to acquire the missing vocabulary.
    # PR PP: use _effective_vocab_floor (relaxed for gap-driven channels).
    if vocab_cov < _effective_vocab_floor:
        gap = _describe_gap_unknown_tokens(unknown, concept)
        return Verdict(
            verdict=VERDICT_DEFER_STR,
            confidence=0.0,
            reason=(
                f"vocab_coverage={vocab_cov:.2f} below floor "
                f"{_effective_vocab_floor:.2f}; "
                f"{len(unknown)} unknown tokens"
            ),
            signals=signals,
            knowledge_gap=gap,
        )

    # Rule 2: near-duplicate of an existing insight. Do NOT accept, do
    # NOT defer \u2014 DMAI has already thought this through. Reject.
    if ins_overlap >= INSIGHT_DUPLICATE_PENALTY_ABOVE:
        return Verdict(
            verdict=VERDICT_REJECT_STR,
            confidence=1.0 - ins_overlap,
            reason=(
                f"insight_neighbourhood={ins_overlap:.2f} \u2014 near-duplicate of "
                f"insight {nearest_id}"
            ),
            signals=signals,
        )

    # Rule 3: confidence combo.
    confidence = (
        WEIGHT_VOCAB              * vocab_cov +
        WEIGHT_INSIGHT_NEIGHBOUR  * _insight_signal_score(ins_overlap) +
        WEIGHT_KPI                * kpi_link +
        WEIGHT_DIVERSITY          * div_pressure
    )

    if confidence >= accept_threshold:
        return Verdict(
            verdict=VERDICT_ACCEPT_STR,
            confidence=confidence,
            reason=(
                f"confidence={confidence:.2f} \u2265 accept threshold "
                f"{accept_threshold:.2f}"
            ),
            signals=signals,
        )
    if confidence <= reject_threshold:
        return Verdict(
            verdict=VERDICT_REJECT_STR,
            confidence=confidence,
            reason=(
                f"confidence={confidence:.2f} \u2264 reject threshold "
                f"{reject_threshold:.2f}"
            ),
            signals=signals,
        )

    # In-between \u2192 defer for more information. The knowledge gap is the
    # weakest signal; that's where acquisition will help most.
    gap = _describe_gap_weakest_signal(signals, concept)
    return Verdict(
        verdict=VERDICT_DEFER_STR,
        confidence=confidence,
        reason=(
            f"confidence={confidence:.2f} in uncertain band "
            f"({reject_threshold:.2f}, {accept_threshold:.2f})"
        ),
        signals=signals,
        knowledge_gap=gap,
    )


# Verdict-string constants exposed to callers so the seed promoter can
# match on them without hard-coding literals.
VERDICT_ACCEPT_STR = "accept"
VERDICT_REJECT_STR = "reject"
VERDICT_DEFER_STR  = "defer"


def _insight_signal_score(overlap: float) -> float:
    """Map raw overlap into a "healthy neighbourhood" score.

    Curve shape: peaks near 0.4\u20130.6 (related but not duplicate), decays
    toward the extremes.
    """
    if overlap <= 0.0:
        return 0.2                        # zero context \u2014 could be novel or garbage
    if overlap >= INSIGHT_DUPLICATE_PENALTY_ABOVE:
        return 0.0                        # essentially a duplicate
    # Triangular bump peaking at 0.5.
    return 1.0 - abs(overlap - 0.5) * 2.0


# ── Knowledge-gap descriptors ────────────────────────────────────────────

def _describe_gap_unknown_tokens(unknown: List[str], concept: str) -> str:
    """Format a gap descriptor targeted at the vocabulary acquirer."""
    top = ", ".join(unknown[:6])
    more = "" if len(unknown) <= 6 else f" (+{len(unknown) - 6} more)"
    return (
        f"unknown_vocabulary: I do not know these tokens from "
        f"'{concept}': {top}{more}. Fetch definitions + a short "
        f"contextual note and add to my vocabulary."
    )


def _describe_gap_weakest_signal(signals: JudgeSignals, concept: str) -> str:
    """Pick the signal that most needs improvement and phrase it as a
    research question for the acquirer."""
    scores = {
        "insight_neighbourhood": _insight_signal_score(signals.insight_neighbourhood),
        "kpi_linkage":           signals.kpi_linkage,
        "diversity_pressure":    signals.diversity_pressure,
    }
    weakest = min(scores, key=scores.get)
    if weakest == "insight_neighbourhood":
        return (
            f"insight_context: I have no prior thinking about '{concept}'. "
            f"Fetch a short explanation of what it is, how it works, and "
            f"where it applies, and add it to my insights."
        )
    if weakest == "kpi_linkage":
        return (
            f"kpi_bridge: '{concept}' does not obviously touch a metric "
            f"I track. Fetch a note on which KPIs it could plausibly move "
            f"(Sharpe, ROI, capability count, diversity ratio, etc.) and "
            f"add it to my insights."
        )
    return (
        f"capability_fit: unclear whether '{concept}' fits an "
        f"under-represented capability_type. Fetch a note on how it "
        f"relates to existing capability_types and add it to my insights."
    )
