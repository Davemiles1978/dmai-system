"""
StatisticalGreyhoundModel — Microfish-free predictor for greyhound markets.

User decision (per handoff):
    Microfish extrapolates curves and is unsuitable for sports.
    Greyhound markets (trap_winner / greyhound_winner) MUST use a deterministic
    statistical model. Microfish remains for stocks/FOREX/ETF.

Drop-in shape:
    predict(requirement: str, seed_data: str = "", **kwargs) -> dict
returning {probability, confidence, rationale, id, signals}

Algorithm:
    Weighted logistic combination of four sub-signals:
      1. Timeform Master Rating z-score within the field          (weight 0.45)
      2. Trap bias by track (favour low traps at most UK tracks)   (weight 0.15)
      3. Recent form positions (lower position numbers = better)   (weight 0.25)
      4. Trainer 3-week implied strike rate                        (weight 0.15)

    Each sub-signal is normalised to [-1, 1]. The weighted sum is mapped
    through a logistic to get probability. Confidence is the agreement between
    sub-signals (1 - normalised stdev of signed sub-signal directions).

All inputs are parsed from the `seed_data` block produced by
GreyhoundRunner._build_seed():
    Track, Off, Trap, Timeform Master Rating, Recent form, Trainer,
    Field implied probability
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Track-level trap-1 win-rate priors (rough UK industry averages — refined when
# we have enough settled history to learn per-track). Used as a small +/- nudge.
# Most UK 4-bend tracks favour low traps on standard distances.
_TRACK_TRAP_PRIORS: Dict[int, float] = {
    1: +0.20, 2: +0.10, 3: 0.00, 4: 0.00, 5: -0.10, 6: -0.20,
}


def _parse_seed(seed_data: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not seed_data:
        return out
    for line in seed_data.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip().lower()] = val.strip()
    return out


def _parse_form_positions(form: str) -> List[int]:
    """'3142' or '3-1-4-2' -> [3,1,4,2]. Letters and dashes ignored."""
    if not form:
        return []
    digits = re.findall(r"[1-9]", form)
    return [int(d) for d in digits]


def _form_score(form_positions: List[int]) -> float:
    """Lower positions = better. Map mean position to signed signal in [-1, 1].

    Scale assumption: 6-runner races, position 1 best, position 6 worst.
    Signal = (3.5 - mean_pos) / 2.5  -> position 1 -> +1.0, position 6 -> -1.0.
    """
    if not form_positions:
        return 0.0
    mean_pos = sum(form_positions[:6]) / min(len(form_positions), 6)
    return max(-1.0, min(1.0, (3.5 - mean_pos) / 2.5))


def _trap_score(trap: Optional[int]) -> float:
    if not trap:
        return 0.0
    return _TRACK_TRAP_PRIORS.get(trap, 0.0)


def _rating_zscore(my_rating: float, field_ratings: List[float]) -> float:
    """Z-score against field. Clamped to [-1, 1] via tanh."""
    if not field_ratings or my_rating <= 0:
        return 0.0
    mean = sum(field_ratings) / len(field_ratings)
    var = sum((r - mean) ** 2 for r in field_ratings) / max(len(field_ratings) - 1, 1)
    sd = math.sqrt(var) if var > 0 else 1.0
    z = (my_rating - mean) / max(sd, 0.5)
    return math.tanh(z / 1.5)


def _logistic(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class StatisticalGreyhoundModel:
    """Deterministic per-runner predictor for greyhound markets.

    Field context (mean / sd of master ratings, etc.) is normally supplied by
    GreyhoundRunner; if absent, the model falls back to single-runner mode
    using the implied-probability line as the rating anchor.
    """

    WEIGHTS: Dict[str, float] = {
        "rating": 0.45,
        "trap": 0.15,
        "form": 0.25,
        "trainer": 0.15,
    }

    def __init__(self, db_path: str = "data/dmai_knowledge.db"):
        self.db_path = db_path
        self._trainer_strike_cache: Dict[str, Tuple[float, float]] = {}  # name -> (rate, ts)

    # ------------------------------------------------------------------
    # Trainer strike rate from mon_tips history (3-week window).
    # ------------------------------------------------------------------

    def _trainer_strike(self, trainer: str) -> float:
        if not trainer:
            return 0.0
        now = time.time()
        cached = self._trainer_strike_cache.get(trainer)
        if cached and (now - cached[1]) < 3600:
            return cached[0]
        rate = 0.0
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                cur = conn.execute(
                    "SELECT status, COUNT(*) FROM mon_tips "
                    "WHERE rationale LIKE ? AND created_at > ? "
                    "AND status IN ('won','lost') GROUP BY status",
                    (f"%{trainer}%", now - 21 * 86400),
                )
                won, lost = 0, 0
                for status, n in cur.fetchall():
                    if status == "won":
                        won = n
                    elif status == "lost":
                        lost = n
                total = won + lost
                if total >= 5:
                    rate = won / total
            finally:
                conn.close()
        except Exception as e:
            logger.debug("trainer strike lookup failed for %s: %s", trainer, e)
        self._trainer_strike_cache[trainer] = (rate, now)
        return rate

    def _trainer_score(self, trainer: str) -> float:
        """Signed signal vs the population average (~14% for greyhound winners).

        Map [0..0.30] strike rate to [-1..+1] tanh-style.
        """
        rate = self._trainer_strike(trainer)
        if rate <= 0:
            return 0.0  # no evidence — neutral
        return math.tanh((rate - 0.14) / 0.10)

    # ------------------------------------------------------------------
    # Microfish-compatible API
    # ------------------------------------------------------------------

    def predict(self, requirement: str, seed_data: str = "",
                max_rounds: int = 2, agent_count: int = 4,
                field_ratings: Optional[List[float]] = None,
                **_) -> Dict[str, Any]:
        """Return verdict dict matching Microfish PredictionEngine.predict shape.

        Args:
            requirement: free-text question (unused — interface compat only)
            seed_data:   formatted key/value block from GreyhoundRunner._build_seed
            field_ratings: optional list of all master ratings in the race,
                           used to compute the focal runner's z-score.
        """
        parsed = _parse_seed(seed_data)

        # Extract field ratings from seed if not passed explicitly.
        if field_ratings is None:
            field_str = parsed.get("field master ratings", "")
            if field_str:
                try:
                    field_ratings = [
                        float(x) for x in field_str.split(",") if x.strip()
                    ]
                except ValueError:
                    field_ratings = None

        try:
            master_rating = float(parsed.get("timeform master rating", "0") or 0)
        except ValueError:
            master_rating = 0.0
        try:
            trap = int(parsed.get("trap", "0") or 0)
        except ValueError:
            trap = 0
        try:
            implied_p = float(parsed.get("field implied probability", "0") or 0)
        except ValueError:
            implied_p = 0.0

        form_positions = _parse_form_positions(parsed.get("recent form", ""))
        trainer = parsed.get("trainer", "")

        # ── Sub-signals ────────────────────────────────────────────────
        s_rating = _rating_zscore(master_rating, field_ratings or [master_rating])
        s_trap = _trap_score(trap)
        s_form = _form_score(form_positions)
        s_trainer = self._trainer_score(trainer)

        # ── Weighted logistic ─────────────────────────────────────────
        # If we have field implied prob, anchor the logit there; sub-signals
        # then nudge it up/down. This keeps single-runner mode (no field data)
        # honest about the underlying market.
        if 0.001 < implied_p < 0.999:
            anchor_logit = math.log(implied_p / (1.0 - implied_p))
        else:
            anchor_logit = 0.0

        sig_logit = (
            self.WEIGHTS["rating"] * s_rating
            + self.WEIGHTS["trap"] * s_trap
            + self.WEIGHTS["form"] * s_form
            + self.WEIGHTS["trainer"] * s_trainer
        ) * 1.5  # scale: a max signal of +1.0 lifts logit by 1.5 (-> ~80% from 50%)

        prob = _logistic(anchor_logit + sig_logit)
        prob = max(0.01, min(0.99, prob))

        # ── Confidence = signal agreement ─────────────────────────────
        signals = [s_rating, s_trap, s_form, s_trainer]
        signed = [s for s in signals if s != 0.0]
        if signed:
            mean = sum(signed) / len(signed)
            disagreement = sum((s - mean) ** 2 for s in signed) / len(signed)
            agreement = max(0.0, 1.0 - math.sqrt(disagreement))
            # Magnitude bonus — strong unanimous signals are more confident
            magnitude = min(1.0, sum(abs(s) for s in signed) / len(signed) * 1.2)
            confidence = round(0.4 + 0.4 * agreement + 0.2 * magnitude, 3)
        else:
            confidence = 0.4

        signals_struct = [
            {"signal": "timeform_master_rating_z", "direction":
                "supports" if s_rating >= 0 else "opposes",
             "weight": round(abs(s_rating) * self.WEIGHTS["rating"], 3)},
            {"signal": "trap_bias", "direction":
                "supports" if s_trap >= 0 else "opposes",
             "weight": round(abs(s_trap) * self.WEIGHTS["trap"], 3)},
            {"signal": "recent_form", "direction":
                "supports" if s_form >= 0 else "opposes",
             "weight": round(abs(s_form) * self.WEIGHTS["form"], 3)},
            {"signal": "trainer_strike_rate", "direction":
                "supports" if s_trainer >= 0 else "opposes",
             "weight": round(abs(s_trainer) * self.WEIGHTS["trainer"], 3)},
        ]

        rationale = (
            f"Master rating {master_rating} z-score signal {s_rating:+.2f}; "
            f"trap {trap} bias {s_trap:+.2f}; "
            f"recent form {form_positions[:6]} signal {s_form:+.2f}; "
            f"trainer {trainer or '(unknown)'} strike-rate signal {s_trainer:+.2f}. "
            f"Field implied prob {implied_p:.2%} -> model prob {prob:.2%} "
            f"at confidence {confidence:.2f}."
        )

        pid = uuid.uuid4().hex[:16]
        return {
            "id": pid,
            "probability": round(prob, 4),
            "confidence": confidence,
            "rationale": rationale,
            "signals": signals_struct,
            "engine": "statistical_greyhound_v1",
            "elapsed_seconds": 0.01,
        }
