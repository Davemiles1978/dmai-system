"""
KnowledgeIntegrityChecker — DMAI's hallucination prevention engine.

Runs cross-reference checks between:
  - vocabulary table (Wiktionary ground truth)
  - insights table  (DMAI's accumulated beliefs)
  - graph_schema.json (knowledge graph neurons)
  - capabilities table (registered skills)

Detects:
  1. DEFINITION_CONFLICT   — insight_text contradicts Wiktionary definition
  2. ORPHANED_NEURON       — graph neuron with no supporting insight or vocabulary entry
  3. LOW_CONFIDENCE_CLUSTER — group of insights on same topic all below threshold
  4. STALE_ENTRY           — insight not referenced or updated in >30 days
  5. DUPLICATE_CONCEPT     — two+ insights with near-identical source_topic but different text
  6. SEMANTIC_DRIFT        — insight relationship label doesn't match its text content
  7. OVERCLAIMED_CAPABILITY — capability registered but no supporting insights exist

Produces structured JSON reports stored in data/integrity/reports/ and in the
integrity_reports SQLite table.
"""

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH      = Path("data/dmai_knowledge.db")
GRAPH_PATH   = Path("aevora-training/dashboard/data/graph_schema.json")
REPORTS_DIR  = Path("data/integrity/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
CONFIDENCE_THRESHOLD     = 0.40   # below this = low confidence flag
STALE_DAYS               = 30     # days without use = stale
DUPLICATE_SIMILARITY     = 0.75   # topic string similarity threshold
MIN_SUPPORTING_INSIGHTS  = 1      # minimum insights to support a capability


# ── String similarity (no external deps) ──────────────────────────────────
def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity on word sets."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _contains_negation(text: str) -> bool:
    negators = ["not ", "never ", "isn't", "is not", "does not", "doesn't",
                "no ", "without ", "opposite", "contrary", "unlike", "incorrect"]
    t = text.lower()
    return any(n in t for n in negators)


def _short_overlap(a: str, b: str, min_len: int = 4) -> float:
    """Token overlap ratio for short texts."""
    ta = set(w for w in _normalise(a).split() if len(w) >= min_len)
    tb = set(w for w in _normalise(b).split() if len(w) >= min_len)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


# ── DB helpers ─────────────────────────────────────────────────────────────
def _db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Table init ─────────────────────────────────────────────────────────────
def _ensure_tables():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS integrity_reports (
            id TEXT PRIMARY KEY,
            run_at TEXT NOT NULL,
            total_checked INTEGER DEFAULT 0,
            total_flags INTEGER DEFAULT 0,
            critical INTEGER DEFAULT 0,
            warning INTEGER DEFAULT 0,
            info INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            report_json TEXT NOT NULL,
            summary TEXT
        );

        CREATE TABLE IF NOT EXISTS integrity_flags (
            id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            flag_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            entity_id TEXT,
            entity_type TEXT,
            title TEXT NOT NULL,
            detail TEXT NOT NULL,
            suggested_action TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            resolved_at TEXT,
            resolution_note TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_flags_report ON integrity_flags(report_id);
        CREATE INDEX IF NOT EXISTS idx_flags_type ON integrity_flags(flag_type);
        CREATE INDEX IF NOT EXISTS idx_flags_resolved ON integrity_flags(resolved);
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════
class KnowledgeIntegrityChecker:
    """
    Full cross-reference integrity scan of DMAI's knowledge base.
    Deterministic — no LLM calls. Pure data analysis.
    """

    def __init__(self):
        _ensure_tables()
        self.flags = []
        self.stats = {
            "insights_checked": 0,
            "vocab_entries": 0,
            "neurons_checked": 0,
            "capabilities_checked": 0,
        }

    # ── Public entry point ─────────────────────────────────────────────────
    def run(self) -> dict:
        """Run full integrity check. Returns report dict."""
        self.flags = []
        run_id   = str(uuid.uuid4())
        run_at   = _now_str()

        logger.info("IntegrityChecker: starting run %s", run_id)

        conn = _db()

        # Load all data
        try:
            insights     = [dict(r) for r in conn.execute("SELECT * FROM insights").fetchall()]
        except Exception:
            insights = []
        try:
            vocab        = {r["word"]: dict(r) for r in conn.execute("SELECT * FROM vocabulary").fetchall()}
        except Exception:
            vocab = {}
        try:
            capabilities = [dict(r) for r in conn.execute("SELECT * FROM capabilities").fetchall()]
        except Exception:
            capabilities = []
        conn.close()

        graph_neurons   = self._load_graph_neurons()

        self.stats["insights_checked"]     = len(insights)
        self.stats["vocab_entries"]        = len(vocab)
        self.stats["neurons_checked"]      = len(graph_neurons)
        self.stats["capabilities_checked"] = len(capabilities)

        total_checked = (len(insights) + len(vocab) +
                         len(graph_neurons) + len(capabilities))

        # ── Run all checks ─────────────────────────────────────────────────
        self._check_definition_conflicts(insights, vocab)
        self._check_orphaned_neurons(graph_neurons, insights, vocab)
        self._check_low_confidence_clusters(insights)
        self._check_stale_entries(insights)
        self._check_duplicate_concepts(insights)
        self._check_semantic_drift(insights)
        self._check_overclaimed_capabilities(capabilities, insights)

        # ── Severity counts ────────────────────────────────────────────────
        critical = sum(1 for f in self.flags if f["severity"] == "critical")
        warning  = sum(1 for f in self.flags if f["severity"] == "warning")
        info     = sum(1 for f in self.flags if f["severity"] == "info")

        # ── Build report ───────────────────────────────────────────────────
        report = {
            "id": run_id,
            "run_at": run_at,
            "total_checked": total_checked,
            "total_flags": len(self.flags),
            "critical": critical,
            "warning": warning,
            "info": info,
            "stats": self.stats,
            "flags": self.flags,
            "summary": self._build_summary(critical, warning, info),
        }

        # ── Persist ────────────────────────────────────────────────────────
        self._persist_report(run_id, run_at, report, total_checked)

        # Save JSON to disk
        report_path = REPORTS_DIR / f"integrity_{run_at[:10]}_{run_id[:8]}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        logger.info("IntegrityChecker: run %s complete — %d flags (%d critical, %d warning, %d info)",
                    run_id, len(self.flags), critical, warning, info)
        return report

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1 — Definition conflicts
    # ══════════════════════════════════════════════════════════════════════
    def _check_definition_conflicts(self, insights: list, vocab: dict):
        """
        For each insight whose source_topic matches a vocabulary word,
        compare the insight_text against the Wiktionary definition.
        Flag if the overlap is very low (< 0.15) — suggests DMAI's belief
        contradicts or is unrelated to the authoritative definition.
        """
        checked = 0
        for ins in insights:
            topic = (ins.get("source_topic") or "").lower().strip()
            if topic not in vocab:
                continue
            vword = vocab[topic]
            vdef  = vword.get("definition", "")
            itext = ins.get("insight_text", "")
            if not vdef or not itext or len(itext) < 20:
                continue

            overlap = _short_overlap(itext, vdef)
            checked += 1

            # Hard contradiction: insight contains negation AND vocab does not
            insight_negated = _contains_negation(itext)
            vocab_negated   = _contains_negation(vdef)
            is_contradiction = insight_negated != vocab_negated and overlap < 0.3

            if is_contradiction:
                self.flags.append(self._flag(
                    flag_type   = "DEFINITION_CONFLICT",
                    severity    = "critical",
                    entity_id   = ins["id"],
                    entity_type = "insight",
                    title       = f"Definition conflict: '{topic}'",
                    detail      = (
                        f"Insight says: \"{itext[:200]}\"\n"
                        f"Wiktionary says: \"{vdef[:200]}\"\n"
                        f"Overlap score: {overlap:.2f}. One contains negation, the other does not."
                    ),
                    action      = f"Review insight {ins['id']} — consider deleting or correcting it. "
                                  f"Wiktionary definition: \"{vdef[:100]}\""
                ))
            elif overlap < 0.10 and len(itext) > 40 and len(vdef) > 40:
                self.flags.append(self._flag(
                    flag_type   = "DEFINITION_CONFLICT",
                    severity    = "warning",
                    entity_id   = ins["id"],
                    entity_type = "insight",
                    title       = f"Weak definition match: '{topic}'",
                    detail      = (
                        f"Insight: \"{itext[:180]}\"\n"
                        f"Wiktionary: \"{vdef[:180]}\"\n"
                        f"Overlap score: {overlap:.2f} — possible semantic drift or wrong context."
                    ),
                    action      = f"Verify insight is using '{topic}' in the correct sense. "
                                  f"Confidence of insight: {ins.get('confidence', 0):.2f}."
                ))

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2 — Orphaned neurons
    # ══════════════════════════════════════════════════════════════════════
    def _check_orphaned_neurons(self, neurons: list, insights: list, vocab: dict):
        """
        Flag graph neurons that have no supporting insight OR vocabulary entry.
        These are beliefs DMAI acts on that have no evidential backing.
        """
        insight_topics = {(ins.get("source_topic") or "").lower() for ins in insights}
        vocab_words    = set(vocab.keys())
        all_known      = insight_topics | vocab_words

        for neuron in neurons:
            nid    = neuron.get("id", "")
            label  = (neuron.get("label") or nid).lower()
            # Check if label OR id matches any known topic
            matched = (label in all_known or nid in all_known or
                       any(label in t or t in label for t in all_known if len(t) > 4))
            if not matched:
                activation = neuron.get("activation", 0)
                severity = "warning" if activation >= 0.7 else "info"
                self.flags.append(self._flag(
                    flag_type   = "ORPHANED_NEURON",
                    severity    = severity,
                    entity_id   = nid,
                    entity_type = "graph_neuron",
                    title       = f"Orphaned neuron: '{neuron.get('label', nid)}'",
                    detail      = (
                        f"Neuron '{nid}' (cluster: {neuron.get('cluster','?')}, "
                        f"activation: {activation:.2f}) has no supporting entry in the "
                        f"insights or vocabulary tables. It was likely seeded manually "
                        f"and has not been reinforced by study."
                    ),
                    action      = ("Let DMAI study this topic to generate supporting insights, "
                                   "or manually verify the neuron is correct and add a "
                                   "corroborating insight entry.")
                ))

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3 — Low confidence clusters
    # ══════════════════════════════════════════════════════════════════════
    def _check_low_confidence_clusters(self, insights: list):
        """
        Group insights by source_topic. If an entire topic cluster has
        average confidence below threshold, flag it as unreliable.
        """
        from collections import defaultdict
        topic_groups = defaultdict(list)
        for ins in insights:
            t = (ins.get("source_topic") or "unknown").strip()
            topic_groups[t].append(ins)

        for topic, group in topic_groups.items():
            if len(group) < 2:
                continue
            avg_conf = sum(i.get("confidence", 0) for i in group) / len(group)
            if avg_conf < CONFIDENCE_THRESHOLD:
                worst = min(group, key=lambda x: x.get("confidence", 1))
                self.flags.append(self._flag(
                    flag_type   = "LOW_CONFIDENCE_CLUSTER",
                    severity    = "warning",
                    entity_id   = topic,
                    entity_type = "topic_cluster",
                    title       = f"Low-confidence cluster: '{topic}'",
                    detail      = (
                        f"{len(group)} insights on '{topic}' average only "
                        f"{avg_conf:.2f} confidence (threshold: {CONFIDENCE_THRESHOLD}). "
                        f"Lowest entry: \"{(worst.get('insight_text') or '')[:150]}\" "
                        f"(confidence: {worst.get('confidence', 0):.2f})."
                    ),
                    action      = (f"Re-research '{topic}' to reinforce these beliefs, "
                                   f"or purge the {len(group)} weak entries if the topic "
                                   f"is outside DMAI's core domains.")
                ))

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 4 — Stale entries
    # ══════════════════════════════════════════════════════════════════════
    def _check_stale_entries(self, insights: list):
        """Flag insights not used or updated in > STALE_DAYS days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
        stale_count = 0
        stale_sample = []

        for ins in insights:
            last = ins.get("last_used") or ins.get("created_at") or ""
            try:
                dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    stale_count += 1
                    if len(stale_sample) < 5:
                        stale_sample.append({
                            "id": ins["id"],
                            "topic": ins.get("source_topic", "?"),
                            "last_used": last[:10],
                            "confidence": ins.get("confidence", 0),
                        })
            except Exception:
                continue

        if stale_count > 0:
            self.flags.append(self._flag(
                flag_type   = "STALE_ENTRIES",
                severity    = "info" if stale_count < 50 else "warning",
                entity_id   = "insights_table",
                entity_type = "bulk",
                title       = f"{stale_count} stale insight(s) not used in >{STALE_DAYS} days",
                detail      = (
                    f"{stale_count} insights have not been accessed or updated in over "
                    f"{STALE_DAYS} days. Sample: " +
                    "; ".join(f"'{s['topic']}' ({s['last_used']}, conf:{s['confidence']:.2f})"
                              for s in stale_sample)
                ),
                action      = ("Run a targeted research pass on these topics to refresh them, "
                               "or prune entries with confidence < 0.3 that are not in core domains.")
            ))

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5 — Duplicate concepts
    # ══════════════════════════════════════════════════════════════════════
    def _check_duplicate_concepts(self, insights: list):
        """
        Find pairs of insights where source_topic strings are very similar
        (>= DUPLICATE_SIMILARITY) but insight_text differs significantly.
        This indicates fragmented knowledge about the same concept.
        """
        topics = [(ins["id"], ins.get("source_topic", ""), ins.get("insight_text", ""))
                  for ins in insights if ins.get("source_topic")]

        seen_pairs = set()
        dupes_found = 0

        for i in range(min(len(topics), 500)):  # cap at 500 for performance
            for j in range(i + 1, min(len(topics), 500)):
                id_a, topic_a, text_a = topics[i]
                id_b, topic_b, text_b = topics[j]
                pair_key = tuple(sorted([id_a, id_b]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                topic_sim = _jaccard(topic_a, topic_b)
                if topic_sim >= DUPLICATE_SIMILARITY:
                    text_sim = _short_overlap(text_a, text_b)
                    if text_sim < 0.3:  # similar topic, divergent text = potential conflict
                        dupes_found += 1
                        if dupes_found <= 10:  # cap flag count
                            self.flags.append(self._flag(
                                flag_type   = "DUPLICATE_CONCEPT",
                                severity    = "warning",
                                entity_id   = f"{id_a}|{id_b}",
                                entity_type = "insight_pair",
                                title       = f"Fragmented concept: '{topic_a}' ≈ '{topic_b}'",
                                detail      = (
                                    f"Two insights share a similar topic (similarity: {topic_sim:.2f}) "
                                    f"but have divergent content (text overlap: {text_sim:.2f}).\n"
                                    f"A: \"{text_a[:150]}\"\n"
                                    f"B: \"{text_b[:150]}\""
                                ),
                                action      = (f"Merge or reconcile these two entries. "
                                               f"Keep the higher-confidence one or combine into "
                                               f"a single canonical insight.")
                            ))

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6 — Semantic drift
    # ══════════════════════════════════════════════════════════════════════
    def _check_semantic_drift(self, insights: list):
        """
        Flag insights where the relationship label is inconsistent with the
        insight text content. E.g. relationship='defines' but text contains
        causal language like 'causes' or 'leads to'.
        """
        RELATIONSHIP_SIGNALS = {
            "defines":     ["is ", "means ", "refers to", "known as", "definition", "term for"],
            "causes":      ["causes", "leads to", "results in", "triggers", "produces"],
            "part_of":     ["part of", "component", "subset", "belongs to", "within"],
            "enables":     ["enables", "allows", "makes possible", "facilitates", "supports"],
            "contradicts": ["contradicts", "opposes", "conflicts", "unlike", "contrary"],
        }

        drifted = 0
        for ins in insights:
            rel   = (ins.get("relationship") or "").lower()
            text  = (ins.get("insight_text") or "").lower()
            if not rel or not text or rel not in RELATIONSHIP_SIGNALS:
                continue

            declared_signals = RELATIONSHIP_SIGNALS[rel]
            has_declared = any(s in text for s in declared_signals)

            # Check if text matches a DIFFERENT relationship instead
            for other_rel, other_signals in RELATIONSHIP_SIGNALS.items():
                if other_rel == rel:
                    continue
                if any(s in text for s in other_signals) and not has_declared:
                    drifted += 1
                    if drifted <= 8:
                        self.flags.append(self._flag(
                            flag_type   = "SEMANTIC_DRIFT",
                            severity    = "warning",
                            entity_id   = ins["id"],
                            entity_type = "insight",
                            title       = f"Semantic drift: labelled '{rel}' but reads as '{other_rel}'",
                            detail      = (
                                f"Insight on '{ins.get('source_topic', '?')}' is categorised as "
                                f"relationship='{rel}' but the text content matches '{other_rel}' signals.\n"
                                f"Text: \"{text[:200]}\""
                            ),
                            action      = (f"Update the relationship field from '{rel}' to '{other_rel}', "
                                           f"or rephrase the insight text to match its declared relationship.")
                        ))
                    break

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 7 — Overclaimed capabilities
    # ══════════════════════════════════════════════════════════════════════
    def _check_overclaimed_capabilities(self, capabilities: list, insights: list):
        """
        Flag capabilities that have no supporting insights.
        These are things DMAI claims to be able to do with no knowledge backing.
        """
        insight_topics_lower = {(i.get("source_topic") or "").lower() for i in insights}
        insight_texts_lower  = " ".join((i.get("insight_text") or "") for i in insights[:2000]).lower()

        for cap in capabilities:
            name = (cap.get("name") or "").strip()
            desc = (cap.get("description") or "").lower()
            name_lower = name.lower()
            cap_type   = cap.get("capability_type", "")

            # Skip self-built capabilities (those are fine by definition)
            if cap_type == "self_built":
                continue

            # Check if any insight mentions this capability by name or description
            name_words = set(w for w in _normalise(name_lower).split() if len(w) > 3)
            supported = (
                name_lower in insight_topics_lower or
                any(name_lower in t for t in insight_topics_lower) or
                (name_words and len(name_words & set(_normalise(insight_texts_lower).split())) >= 2)
            )

            if not supported:
                self.flags.append(self._flag(
                    flag_type   = "OVERCLAIMED_CAPABILITY",
                    severity    = "info",
                    entity_id   = cap.get("id", name),
                    entity_type = "capability",
                    title       = f"Unsupported capability: '{name}'",
                    detail      = (
                        f"Capability '{name}' (type: {cap_type}) is registered but "
                        f"has no corroborating entries in the insights table. "
                        f"This may indicate the capability was seeded without DMAI "
                        f"actually studying or verifying its underlying knowledge."
                    ),
                    action      = (f"Have DMAI research '{name}' to generate supporting insights, "
                                   f"or remove the capability if it is no longer valid.")
                ))

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _flag(self, flag_type: str, severity: str, entity_id: str,
              entity_type: str, title: str, detail: str, action: str) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "flag_type": flag_type,
            "severity": severity,
            "entity_id": str(entity_id),
            "entity_type": entity_type,
            "title": title,
            "detail": detail,
            "suggested_action": action,
            "resolved": False,
            "created_at": _now_str(),
        }

    def _load_graph_neurons(self) -> list:
        try:
            if GRAPH_PATH.exists():
                return json.loads(GRAPH_PATH.read_text(encoding="utf-8")).get("neurons", [])
        except Exception as e:
            logger.warning("Could not load graph_schema.json: %s", e)
        return []

    def _build_summary(self, critical: int, warning: int, info: int) -> str:
        total = critical + warning + info
        if total == 0:
            return "Knowledge graph is clean. No integrity issues detected."
        parts = []
        if critical:
            parts.append(f"{critical} critical conflict(s) require immediate attention")
        if warning:
            parts.append(f"{warning} warning(s) may affect decision quality")
        if info:
            parts.append(f"{info} informational item(s) for review")
        return ". ".join(parts) + "."

    def _persist_report(self, run_id: str, run_at: str, report: dict, total_checked: int):
        try:
            conn = _db()
            conn.execute(
                "INSERT INTO integrity_reports "
                "(id, run_at, total_checked, total_flags, critical, warning, info, "
                "resolved, status, report_json, summary) VALUES (?,?,?,?,?,?,?,0,'completed',?,?)",
                (run_id, run_at, total_checked,
                 report["total_flags"], report["critical"],
                 report["warning"], report["info"],
                 json.dumps(report), report["summary"])
            )
            for flag in report["flags"]:
                conn.execute(
                    "INSERT INTO integrity_flags "
                    "(id, report_id, flag_type, severity, entity_id, entity_type, "
                    "title, detail, suggested_action, resolved, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,0,?)",
                    (flag["id"], run_id, flag["flag_type"], flag["severity"],
                     flag["entity_id"], flag["entity_type"], flag["title"],
                     flag["detail"], flag["suggested_action"], flag["created_at"])
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to persist integrity report: %s", e)

    # ── Resolve a flag ────────────────────────────────────────────────────
    @staticmethod
    def resolve_flag(flag_id: str, note: str = "") -> bool:
        try:
            conn = _db()
            conn.execute(
                "UPDATE integrity_flags SET resolved=1, resolved_at=?, resolution_note=? WHERE id=?",
                (_now_str(), note, flag_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("resolve_flag failed: %s", e)
            return False

    # ── Purge stale insights (if David approves) ──────────────────────────
    @staticmethod
    def purge_low_confidence(min_confidence: float = 0.2, dry_run: bool = True) -> dict:
        """Delete insights below confidence threshold. dry_run=True just counts."""
        try:
            conn = _db()
            count = conn.execute(
                "SELECT COUNT(*) FROM insights WHERE confidence < ?", (min_confidence,)
            ).fetchone()[0]
            if not dry_run:
                conn.execute("DELETE FROM insights WHERE confidence < ?", (min_confidence,))
                conn.commit()
            conn.close()
            return {"would_delete" if dry_run else "deleted": count,
                    "threshold": min_confidence, "dry_run": dry_run}
        except Exception as e:
            return {"error": str(e)}
