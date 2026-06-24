"""
SkillAssessor — multi-axis assessment of long-form creative output.

For each submission to the review queue, computes a skill score across
craft dimensions tailored to the work type. Tracks a rolling skill curve
per work type and decides when auto-publish can be unlocked.

Work types and their assessment axes:
  book_chapter / book_manuscript: hook, prose, structure, voice, dialogue,
    pacing, originality, market_fit
  research_paper: thesis, methodology, evidence, citation_quality,
    structure, originality, clarity, contribution
  article: hook, structure, evidence, voice, originality, seo, length_fit
  tv_script / screenplay: logline, structure, character, dialogue, scene_work,
    visual_writing, market_fit, page_count
  course_lesson: clarity, demonstration, exercise, takeaway, runtime_fit,
    portfolio_quality
  newsletter_essay: hook_scene, voice, specificity, arc, payoff, length_fit

Skill curve graduation rule: N consecutive submissions of a given type
score >= GRADUATION_THRESHOLD AND user has explicitly approved the
work-type via `mark_graduated`. The user's explicit consent is required;
the system never auto-graduates on metrics alone.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import statistics
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Thresholds — tunable
SKILL_PASS = 7.5          # out of 10
GRADUATION_THRESHOLD = 8.0
GRADUATION_REQUIRED_CONSECUTIVE = 5
GRADUATION_MIN_TOTAL = 10  # min total reviewed before consideration

WORK_TYPES = {
    "book_chapter":   ["hook", "prose", "structure", "voice", "dialogue", "pacing", "originality", "market_fit"],
    "book_manuscript": ["hook", "prose", "structure", "voice", "dialogue", "pacing", "originality", "market_fit"],
    "research_paper": ["thesis", "methodology", "evidence", "citation_quality", "structure", "originality", "clarity", "contribution"],
    "article":        ["hook", "structure", "evidence", "voice", "originality", "seo", "length_fit"],
    "tv_script":      ["logline", "structure", "character", "dialogue", "scene_work", "visual_writing", "market_fit", "page_count"],
    "screenplay":     ["logline", "structure", "character", "dialogue", "scene_work", "visual_writing", "market_fit", "page_count"],
    "course_lesson":  ["clarity", "demonstration", "exercise", "takeaway", "runtime_fit", "portfolio_quality"],
    "newsletter_essay": ["hook_scene", "voice", "specificity", "arc", "payoff", "length_fit"],
}


class SkillAssessor:
    def __init__(self, data_path: str | Path = "data") -> None:
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.data_path / "dmai_knowledge.db")
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS skill_assessments ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT DEFAULT (datetime('now')), "
                "submission_id TEXT NOT NULL, "
                "work_type TEXT NOT NULL, "
                "scores_json TEXT, "
                "overall REAL, "
                "passed INTEGER, "
                "notes TEXT, "
                "assessor TEXT DEFAULT 'auto')"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_skill_work_type "
                "ON skill_assessments(work_type, ts DESC)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS skill_graduation ("
                "work_type TEXT PRIMARY KEY, "
                "graduated INTEGER DEFAULT 0, "
                "graduated_at TEXT, "
                "graduated_by TEXT, "
                "notes TEXT)"
            )
            c.commit()

    # ── Assessment ────────────────────────────────────────────────────────────
    def assess(
        self,
        submission_id: str,
        work_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess a submission. payload is the full work content + metadata.
        Returns scores per axis + overall + pass/fail.
        """
        with self._lock:
            return self._assess_inner(submission_id, work_type, payload)

    def _assess_inner(
        self,
        submission_id: str,
        work_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        axes = WORK_TYPES.get(work_type)
        if not axes:
            return {
                "error": f"unknown work_type: {work_type}",
                "supported": list(WORK_TYPES.keys()),
            }

        scores: Dict[str, float] = {}
        notes: List[str] = []

        # Extract text for analysis
        text = self._extract_text(payload)
        word_count = len(text.split())
        sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
        avg_sentence_len = word_count / sentence_count

        for axis in axes:
            score, axis_notes = self._score_axis(axis, work_type, payload, text)
            scores[axis] = score
            if axis_notes:
                notes.append(f"{axis}: {axis_notes}")

        overall = round(statistics.mean(scores.values()), 2) if scores else 0.0
        passed = overall >= SKILL_PASS

        with self._conn() as c:
            c.execute(
                "INSERT INTO skill_assessments(submission_id, work_type, "
                "scores_json, overall, passed, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    work_type,
                    json.dumps(scores),
                    overall,
                    1 if passed else 0,
                    " | ".join(notes)[:2000],
                ),
            )
            c.commit()

        return {
            "submission_id": submission_id,
            "work_type": work_type,
            "scores": scores,
            "overall": overall,
            "passed": passed,
            "word_count": word_count,
            "avg_sentence_len": round(avg_sentence_len, 1),
            "notes": notes,
        }

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        """Best-effort text extraction from various payload shapes."""
        if isinstance(payload, str):
            return payload
        for key in ("prose", "body", "script", "content", "full_content", "manuscript", "essay"):
            v = payload.get(key)
            if isinstance(v, str) and len(v) > 50:
                return v
            if isinstance(v, list):
                joined = "\n\n".join(str(x) for x in v if x)
                if len(joined) > 50:
                    return joined
        if "chapters" in payload and isinstance(payload["chapters"], list):
            return "\n\n".join(
                ch.get("prose", "") if isinstance(ch, dict) else str(ch)
                for ch in payload["chapters"]
            )
        return json.dumps(payload)[:4000]

    def _score_axis(
        self,
        axis: str,
        work_type: str,
        payload: Dict[str, Any],
        text: str,
    ) -> Tuple[float, str]:
        """
        Score a single craft axis. Real, structural heuristics — no random
        numbers. Conservative — scores rarely exceed 8.5 from heuristics
        alone, which is correct: high scores should reflect human review.
        """
        text_lower = text.lower()
        word_count = len(text.split())
        sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
        avg_sent = word_count / sentence_count

        # Default base: 5.0 (neutral)
        score = 5.0
        notes: List[str] = []

        if axis == "hook":
            first_para = text.split("\n\n")[0] if "\n\n" in text else text[:400]
            first_sent = first_para.split(".")[0] if "." in first_para else first_para[:160]
            if len(first_sent) < 12:
                score = 4.0; notes.append("very short opener")
            elif any(p in first_sent.lower() for p in ["it was", "this is a story", "in this article", "today we"]):
                score = 4.0; notes.append("generic opener")
            elif any(c in first_sent for c in ['"', "—", "?"]) or first_sent[0].isupper() and len(first_sent) > 30:
                score = 7.0; notes.append("specific opener")
            else:
                score = 5.5

        elif axis == "prose":
            # Sentence-length variance is a craft proxy
            sent_lens = [len(s.split()) for s in re.split(r"[.!?]+", text) if s.strip()]
            if not sent_lens:
                score = 3.0
            else:
                var = statistics.stdev(sent_lens) if len(sent_lens) > 1 else 0
                if var < 3:
                    score = 4.5; notes.append("monotonous sentence length")
                elif var > 14:
                    score = 7.5; notes.append("varied sentence rhythm")
                else:
                    score = 6.5
            # Penalise common AI-tells
            tells = ["delve", "tapestry", "in conclusion", "furthermore", "moreover", "it is important to note"]
            tell_hits = sum(text_lower.count(t) for t in tells)
            if tell_hits >= 3:
                score = max(3.0, score - 1.5)
                notes.append(f"{tell_hits} stock phrases — voice flat")

        elif axis == "structure":
            # Look for paragraph breaks, scene markers, section headings
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            if len(paragraphs) < 3:
                score = 4.0; notes.append("under-structured")
            elif len(paragraphs) > 4:
                score = 7.0
            # Headings (markdown or numbered) help
            if re.search(r"^(#+|\d+\.)\s", text, re.MULTILINE):
                score = min(8.0, score + 0.5)

        elif axis == "voice":
            # First-person presence + idiosyncratic word choice
            fp = sum(text_lower.count(f" {w} ") for w in ["i ", "i'm", "i've", "my", "me "])
            density = fp / max(1, word_count) * 1000
            if density > 8:
                score = 7.5; notes.append("strong personal voice")
            elif density > 2:
                score = 6.5
            else:
                score = 5.0; notes.append("impersonal voice")

        elif axis == "dialogue":
            quote_count = text.count('"') + text.count("'")
            ratio = quote_count / max(1, sentence_count)
            if work_type in ("tv_script", "screenplay"):
                # Expect format like CHARACTER\nLine
                cue_hits = len(re.findall(r"^[A-Z][A-Z ]{2,}\s*$", text, re.MULTILINE))
                if cue_hits >= 5:
                    score = 7.5; notes.append(f"{cue_hits} character cues")
                else:
                    score = 4.5; notes.append("no/few character cues")
            else:
                if ratio < 0.2:
                    score = 5.0
                elif ratio > 4:
                    score = 7.0
                else:
                    score = 6.0

        elif axis == "pacing":
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                score = 3.0
            else:
                lens = [len(p.split()) for p in paragraphs]
                if len(lens) > 1:
                    var = statistics.stdev(lens)
                    score = 7.0 if 20 < var < 120 else 5.5
                else:
                    score = 4.5

        elif axis == "originality":
            # Anti-cliché check
            cliches = [
                "little did they know", "in a world where", "ever after",
                "as a language model", "as an ai", "i am just an ai",
                "embark on a journey", "navigate the complexities",
            ]
            hits = sum(text_lower.count(c) for c in cliches)
            if hits > 0:
                score = max(2.0, 6.0 - hits * 1.5)
                notes.append(f"{hits} cliché phrase(s)")
            else:
                score = 6.5

        elif axis == "market_fit":
            # Length proxy for genre fit
            if work_type == "book_manuscript":
                if 60000 <= word_count <= 110000:
                    score = 8.0
                elif 30000 <= word_count < 60000 or 110000 < word_count <= 140000:
                    score = 6.5
                else:
                    score = 4.5; notes.append(f"{word_count}w outside typical range")
            elif work_type == "book_chapter":
                if 2000 <= word_count <= 5000:
                    score = 8.0
                elif 1000 <= word_count < 2000 or 5000 < word_count <= 8000:
                    score = 6.5
                else:
                    score = 5.0
            elif work_type in ("tv_script", "screenplay"):
                # ~1 page per minute, 250 words/page
                if work_type == "tv_script" and 6000 <= word_count <= 15000:
                    score = 7.5
                elif work_type == "screenplay" and 22000 <= word_count <= 32000:
                    score = 7.5
                else:
                    score = 5.0
            else:
                score = 6.0

        elif axis == "thesis":
            # Research-paper: look for "we propose", "this paper", "our hypothesis"
            markers = ["we propose", "we argue", "this paper", "our hypothesis", "we show", "we demonstrate"]
            hits = sum(text_lower.count(m) for m in markers)
            score = 7.0 if hits >= 2 else 5.0 if hits == 1 else 3.5

        elif axis == "methodology":
            markers = ["methodology", "we used", "dataset", "experiment", "control group", "sample size", "n =", "p <"]
            hits = sum(text_lower.count(m) for m in markers)
            score = min(8.0, 4.0 + hits * 0.6)

        elif axis == "evidence":
            citation_count = len(re.findall(r"\[\d+\]|\(\w+,?\s*\d{4}\)|https?://", text))
            score = min(8.5, 4.0 + citation_count * 0.3)
            if citation_count == 0:
                notes.append("zero citations")

        elif axis == "citation_quality":
            urls = re.findall(r"https?://[\w./\-?=&%]+", text)
            primary_hits = sum(
                1 for u in urls
                if any(d in u for d in [".gov", ".edu", ".org", "doi.org", "arxiv.org", "ncbi"])
            )
            score = min(8.5, 4.0 + primary_hits * 0.7)
            if urls and primary_hits == 0:
                notes.append("no primary/authoritative sources")

        elif axis == "clarity":
            if avg_sent > 32:
                score = 4.0; notes.append("sentences too long")
            elif avg_sent < 8:
                score = 5.5; notes.append("sentences very short")
            else:
                score = 7.0

        elif axis == "contribution":
            markers = ["novel", "first to", "we contribute", "this work extends", "new approach", "to our knowledge"]
            hits = sum(text_lower.count(m) for m in markers)
            score = 7.0 if hits >= 1 else 5.0

        elif axis == "seo":
            h2 = len(re.findall(r"^##\s", text, re.MULTILINE))
            h3 = len(re.findall(r"^###\s", text, re.MULTILINE))
            score = min(8.0, 4.0 + (h2 * 0.6) + (h3 * 0.3))
            if h2 == 0:
                notes.append("no H2 headings")

        elif axis == "length_fit":
            if work_type == "article":
                score = 8.0 if 1200 <= word_count <= 3000 else 5.5 if 800 <= word_count <= 5000 else 4.0
            elif work_type == "newsletter_essay":
                score = 8.0 if 1500 <= word_count <= 4000 else 5.5
            else:
                score = 6.5

        elif axis == "logline":
            logline = payload.get("logline") or payload.get("premise")
            if isinstance(logline, str):
                if 15 <= len(logline.split()) <= 40:
                    score = 7.5
                elif logline:
                    score = 5.5; notes.append("logline length off")
                else:
                    score = 3.0
            else:
                score = 3.5; notes.append("no logline")

        elif axis == "character":
            # Distinct name count + capitalised dialogue cues
            name_pattern = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
            distinct = len(set(name_pattern))
            if distinct >= 4:
                score = 7.0
            elif distinct >= 2:
                score = 5.5
            else:
                score = 3.5; notes.append("under-populated cast")

        elif axis == "scene_work":
            # Look for INT./EXT. slug lines in scripts
            slugs = len(re.findall(r"^(INT\.|EXT\.)", text, re.MULTILINE))
            if work_type in ("tv_script", "screenplay"):
                score = min(8.0, 4.0 + slugs * 0.4)
                if slugs == 0:
                    notes.append("no scene slug lines")
            else:
                score = 6.0

        elif axis == "visual_writing":
            # Action-line nouns/verbs vs dialogue ratio
            action_lines = sum(1 for l in text.split("\n") if l and not l.startswith('"') and not re.match(r"^[A-Z][A-Z ]+$", l))
            score = 6.5 if action_lines > 20 else 5.0

        elif axis == "page_count":
            pages = max(1, word_count / 250)
            if work_type == "tv_script":
                score = 8.0 if 25 <= pages <= 60 else 5.0
            elif work_type == "screenplay":
                score = 8.0 if 90 <= pages <= 120 else 5.0
            else:
                score = 6.0

        elif axis == "demonstration":
            # Course lessons should have worked examples
            markers = ["example:", "for example", "here's how", "watch:", "demo:", "let's"]
            hits = sum(text_lower.count(m) for m in markers)
            score = min(8.0, 4.0 + hits * 0.8)

        elif axis == "exercise":
            markers = ["exercise:", "your turn", "try this", "practice:", "homework", "challenge:"]
            hits = sum(text_lower.count(m) for m in markers)
            score = 7.5 if hits >= 1 else 4.0

        elif axis == "takeaway":
            markers = ["takeaway", "in summary", "remember:", "key point", "the key is"]
            hits = sum(text_lower.count(m) for m in markers)
            score = 7.5 if hits >= 1 else 5.0

        elif axis == "runtime_fit":
            runtime = payload.get("runtime_minutes") or 0
            try:
                runtime = float(runtime)
            except Exception:
                runtime = 0
            score = 8.0 if 5 <= runtime <= 15 else 5.0

        elif axis == "portfolio_quality":
            project = payload.get("class_project") or payload.get("project")
            score = 7.5 if project else 5.0

        elif axis == "hook_scene":
            first_para = text.split("\n\n")[0] if "\n\n" in text else text[:500]
            sensory = sum(first_para.lower().count(w) for w in ["smell", "taste", "sound", "saw", "felt", "looked", "heard"])
            score = 7.5 if sensory >= 1 else 5.0

        elif axis == "specificity":
            # Proper nouns + concrete numbers = specificity
            nouns = len(re.findall(r"\b[A-Z][a-z]+\b", text))
            numbers = len(re.findall(r"\b\d+\b", text))
            score = min(8.0, 4.0 + (nouns / max(1, word_count) * 80) + (numbers / max(1, word_count) * 40))

        elif axis == "arc":
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            score = 7.0 if len(paragraphs) >= 6 else 5.0

        elif axis == "payoff":
            last_para = text.strip().split("\n\n")[-1] if "\n\n" in text else text[-300:]
            if any(w in last_para.lower() for w in ["because", "and that's why", "the lesson", "the point", "what i learned"]):
                score = 7.0
            else:
                score = 5.5

        return round(max(1.0, min(10.0, score)), 2), "; ".join(notes)

    # ── Skill curve + graduation ──────────────────────────────────────────────
    def skill_curve(self, work_type: str, limit: int = 30) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, submission_id, overall, passed FROM skill_assessments "
                "WHERE work_type = ? ORDER BY id DESC LIMIT ?",
                (work_type, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def stats(self, work_type: Optional[str] = None) -> Dict[str, Any]:
        with self._conn() as c:
            if work_type:
                rows = c.execute(
                    "SELECT overall, passed FROM skill_assessments "
                    "WHERE work_type = ? ORDER BY id DESC LIMIT 50",
                    (work_type,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT work_type, AVG(overall) AS avg_score, COUNT(*) AS n, "
                    "SUM(passed) AS passed_n FROM skill_assessments "
                    "GROUP BY work_type"
                ).fetchall()
                grad = c.execute(
                    "SELECT work_type, graduated FROM skill_graduation"
                ).fetchall()
                grad_map = {r["work_type"]: bool(r["graduated"]) for r in grad}
                return {
                    "by_work_type": [
                        {
                            "work_type": r["work_type"],
                            "avg_score": round(r["avg_score"] or 0, 2),
                            "submissions": r["n"],
                            "passed": r["passed_n"],
                            "graduated": grad_map.get(r["work_type"], False),
                        }
                        for r in rows
                    ],
                }
        if not rows:
            return {"work_type": work_type, "submissions": 0}
        scores = [float(r["overall"] or 0) for r in rows]
        recent = scores[: GRADUATION_REQUIRED_CONSECUTIVE]
        return {
            "work_type": work_type,
            "submissions": len(rows),
            "avg_score": round(sum(scores) / len(scores), 2),
            "recent_avg": round(sum(recent) / len(recent), 2) if recent else 0,
            "consecutive_above_threshold": self._consecutive_above(scores, GRADUATION_THRESHOLD),
            "eligible_for_graduation": self.eligible_for_graduation(work_type),
            "graduated": self.is_graduated(work_type),
        }

    @staticmethod
    def _consecutive_above(scores: List[float], threshold: float) -> int:
        count = 0
        for s in scores:
            if s >= threshold:
                count += 1
            else:
                break
        return count

    def eligible_for_graduation(self, work_type: str) -> bool:
        with self._conn() as c:
            rows = c.execute(
                "SELECT overall FROM skill_assessments WHERE work_type = ? "
                "ORDER BY id DESC LIMIT ?",
                (work_type, GRADUATION_REQUIRED_CONSECUTIVE),
            ).fetchall()
            total = c.execute(
                "SELECT COUNT(*) AS n FROM skill_assessments WHERE work_type = ?",
                (work_type,),
            ).fetchone()
        if not rows or total["n"] < GRADUATION_MIN_TOTAL:
            return False
        if len(rows) < GRADUATION_REQUIRED_CONSECUTIVE:
            return False
        return all(float(r["overall"] or 0) >= GRADUATION_THRESHOLD for r in rows)

    def is_graduated(self, work_type: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT graduated FROM skill_graduation WHERE work_type = ?",
                (work_type,),
            ).fetchone()
            return bool(row and row["graduated"])

    def mark_graduated(self, work_type: str, by: str = "user", notes: str = "") -> Dict[str, Any]:
        """User explicitly graduates a work type — auto-publish unlocks."""
        if work_type not in WORK_TYPES:
            return {"error": f"unknown work_type {work_type}"}
        if not self.eligible_for_graduation(work_type):
            return {
                "error": "not eligible for graduation yet",
                "stats": self.stats(work_type),
            }
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO skill_graduation("
                "work_type, graduated, graduated_at, graduated_by, notes) "
                "VALUES (?, 1, datetime('now'), ?, ?)",
                (work_type, by, notes),
            )
            c.commit()
        logger.info("SkillAssessor: %s GRADUATED by %s", work_type, by)
        return {"work_type": work_type, "graduated": True, "by": by}

    def revoke_graduation(self, work_type: str, by: str = "user", notes: str = "") -> Dict[str, Any]:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO skill_graduation("
                "work_type, graduated, graduated_at, graduated_by, notes) "
                "VALUES (?, 0, datetime('now'), ?, ?)",
                (work_type, by, "REVOKED: " + notes),
            )
            c.commit()
        return {"work_type": work_type, "graduated": False, "by": by}


def get_skill_assessor(data_path: str = "data") -> SkillAssessor:
    return SkillAssessor(data_path=data_path)
