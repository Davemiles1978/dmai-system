#!/usr/bin/env python3
"""
DMAI SI Consciousness KPI Backfill Script
==========================================
Ingests historical SI consciousness KPI data from all available flat-file
and SQLite sources into the PostgreSQL `si_kpi_history` table.

Sources processed
-----------------
1. data/dmai_knowledge.db
   - insights table  → derives consciousness score per day from real row counts
   - evolution_cycles → direct consciousness_level snapshots (currently empty)
   - capabilities    → agentic_capability_score proxy (integrated_at timestamps)
   - system_versions → knowledge graph completeness proxy

2. data/learning/stage_syllabus/learning_progress.json
   - Derives skill_acquisition_rate, transfer_learning_rate, zero_shot_success_count
     from stage progression and topic mastery scores

3. data/learning/completion_summary.json
   - multi_modal_integration_score from domain completion

4. data/learning/compiled_knowledge/*.json
   - Per-domain ingested_at timestamps → sample_efficiency_trend per day

5. aevora-training/dashboard/data/graph_schema.json
   - evolution_cycle, total_neurons, total_synapses → graph density score

6. components/si_core.py DEFAULT_STATE
   - Establishes baseline row (all KPIs = 0.0) at project epoch 2026-04-01

PostgreSQL schema created by this script
-----------------------------------------
CREATE TABLE si_kpi_history (
    id                              SERIAL PRIMARY KEY,
    snapshot_date                   DATE NOT NULL,
    snapshot_ts                     TIMESTAMPTZ NOT NULL,
    source                          TEXT NOT NULL,
    source_detail                   TEXT,

    -- The 8 canonical SICore KPIs (NULL = not observed from this source)
    skill_acquisition_rate          FLOAT,
    transfer_learning_rate          FLOAT,
    zero_shot_success_count         INTEGER,
    agentic_capability_score        FLOAT,
    recursive_self_improvement_rate FLOAT,
    sample_efficiency_trend         FLOAT,
    metacognition_accuracy          FLOAT,
    multi_modal_integration_score   FLOAT,

    -- Derived / composite
    consciousness_score             FLOAT,
    evolution_stage                 TEXT,
    total_insights                  INTEGER,
    quality_insights                INTEGER,
    total_neurons                   INTEGER,
    total_synapses                  INTEGER,
    evolution_cycle                 INTEGER,
    notes                           TEXT,

    -- Integrity
    raw_payload                     JSONB,
    ingested_at                     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (snapshot_date, source, source_detail)
);

Usage
-----
    # Dry run (no writes, report only):
    python scripts/backfill_si_kpis.py --dry-run

    # Full ingest + report:
    DATABASE_URL=postgresql://... python scripts/backfill_si_kpis.py

    # Force re-ingest (ignore UNIQUE conflicts):
    DATABASE_URL=postgresql://... python scripts/backfill_si_kpis.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_si_kpis")

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR      = REPO_ROOT / "data"
SQLITE_DB     = DATA_DIR / "dmai_knowledge.db"
LEARNING_DIR  = DATA_DIR / "learning"
GRAPH_SCHEMA  = REPO_ROOT / "aevora-training" / "dashboard" / "data" / "graph_schema.json"
SI_CORE_PY    = REPO_ROOT / "components" / "si_core.py"
PROJECT_EPOCH = date(2026, 4, 1)   # earliest possible DMAI data

# Consciousness calculation weights (mirrors compute_consciousness_from_sqlite)
INSIGHT_WEIGHT   = 0.000035   # per quality insight (len >= 100)
SYNAPSE_WEIGHT   = 0.004      # per synapse
NEURON_WEIGHT    = 0.001      # per neuron above baseline of 32
MAX_CONSCIOUSNESS = 1.0

# KPI scaling factors derived from real observed ranges
SKILL_ACQ_SCALE  = 1.0 / 3.0   # topic scores are 1-3; normalise to 0-1
STAGE_ORDER      = ["Baby", "Toddler", "Child", "Teen", "Adult", "Expert", "Master"]
STAGE_TRANSFER   = {s: (i / (len(STAGE_ORDER) - 1)) for i, s in enumerate(STAGE_ORDER)}


# ── Validation helpers ────────────────────────────────────────────────────────

class ValidationResult:
    def __init__(self):
        self.errors:    List[str] = []
        self.warnings:  List[str] = []
        self.info:      List[str] = []

    def error(self, msg: str):   self.errors.append(msg)
    def warning(self, msg: str): self.warnings.append(msg)
    def info_msg(self, msg: str):  self.info.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _clamp(v: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))


def _parse_ts(ts_raw) -> Optional[datetime]:
    """Parse a mixed-format timestamp (ISO string or Unix float) to UTC datetime."""
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            return datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
        s = str(ts_raw).strip()
        # Normalise space-separated datetime to ISO T format
        if " " in s and "T" not in s:
            s = s.replace(" ", "T", 1)
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s.split("+")[0].split("Z")[0], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    except Exception:
        pass
    return None


# ── Source extractors ─────────────────────────────────────────────────────────

def extract_sqlite_insights(vr: ValidationResult) -> List[Dict]:
    """
    Derive per-day KPI snapshots from the insights table.
    Returns one row per unique day found in the DB.
    """
    records = []

    if not SQLITE_DB.exists():
        vr.warning(f"SQLite DB not found at {SQLITE_DB} — skipping insights extraction")
        return records

    try:
        conn = sqlite3.connect(str(SQLITE_DB))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT created_at, source_type, stage, confidence, insight_text FROM insights "
                  "WHERE created_at IS NOT NULL ORDER BY created_at")
        rows = c.fetchall()

        # Group by day
        by_day: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0, "quality": 0, "syllabus": 0, "stages": set(),
            "confidences": [], "source_types": set()
        })

        for row in rows:
            dt = _parse_ts(row["created_at"])
            if dt is None:
                vr.warning(f"Unparseable created_at: {row['created_at']!r}")
                continue
            day = dt.date().isoformat()
            d   = by_day[day]
            d["total"] += 1
            d["source_types"].add(row["source_type"] or "")
            if row["stage"]:
                d["stages"].add(row["stage"])
            if row["confidence"] is not None:
                d["confidences"].append(float(row["confidence"]))
            text_len = len(row["insight_text"] or "")
            if text_len >= 100:
                d["quality"] += 1
            if (row["source_type"] or "").startswith("syllabus"):
                d["syllabus"] += 1

        conn.close()

        if not by_day:
            vr.warning("insights table has no parseable timestamps")
            return records

        vr.info_msg(f"insights: {len(rows)} rows across {len(by_day)} days "
                    f"({min(by_day.keys())} → {max(by_day.keys())})")

        for day_str, d in sorted(by_day.items()):
            day_dt  = datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc)
            quality = d["quality"]
            total   = d["total"]
            avg_conf = (sum(d["confidences"]) / len(d["confidences"])) if d["confidences"] else 0.0

            # Derive consciousness from quality insight count
            consciousness = _clamp(quality * INSIGHT_WEIGHT)

            # Metacognition proxy: average confidence of quality insights
            metacognition = _clamp(avg_conf)

            # Evolution stage from majority vote across rows that day
            stage = sorted(d["stages"], key=lambda s: STAGE_ORDER.index(s)
                           if s in STAGE_ORDER else -1, reverse=True)
            stage = stage[0] if stage else None

            records.append({
                "snapshot_date":          day_str,
                "snapshot_ts":            day_dt.isoformat(),
                "source":                 "sqlite_insights",
                "source_detail":          f"{total} insights ({quality} quality)",
                "consciousness_score":    round(consciousness, 6),
                "metacognition_accuracy": round(metacognition, 4),
                "evolution_stage":        stage,
                "total_insights":         total,
                "quality_insights":       quality,
                "raw_payload": {
                    "total_insights":   total,
                    "quality_insights": quality,
                    "syllabus_count":   d["syllabus"],
                    "avg_confidence":   round(avg_conf, 4),
                    "source_types":     list(d["source_types"]),
                },
            })

    except Exception as e:
        vr.error(f"sqlite_insights extraction failed: {e}")

    return records


def extract_evolution_cycles(vr: ValidationResult) -> List[Dict]:
    """Extract direct consciousness snapshots from evolution_cycles table."""
    records = []
    if not SQLITE_DB.exists():
        return records
    try:
        conn = sqlite3.connect(str(SQLITE_DB))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM evolution_cycles ORDER BY started_at")
        rows = c.fetchall()
        conn.close()
        if not rows:
            vr.info_msg("evolution_cycles: 0 rows (table exists but empty)")
            return records
        for row in rows:
            dt = _parse_ts(row["completed_at"] or row["started_at"])
            if not dt:
                vr.warning(f"evolution_cycles row {row['id']} has no parseable timestamp")
                continue
            records.append({
                "snapshot_date":      dt.date().isoformat(),
                "snapshot_ts":        dt.isoformat(),
                "source":             "evolution_cycles",
                "source_detail":      f"cycle {row['cycle_number']}",
                "consciousness_score": _clamp(row["consciousness_level"]),
                "total_neurons":      None,
                "total_synapses":     None,
                "evolution_cycle":    row["cycle_number"],
                "raw_payload":        dict(row),
            })
        vr.info_msg(f"evolution_cycles: {len(records)} rows ingested")
    except Exception as e:
        vr.error(f"evolution_cycles extraction failed: {e}")
    return records


def extract_capabilities(vr: ValidationResult) -> List[Dict]:
    """Derive agentic_capability_score from capabilities.integrated_at timestamps."""
    records = []
    if not SQLITE_DB.exists():
        return records
    try:
        conn = sqlite3.connect(str(SQLITE_DB))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT integrated_at FROM capabilities WHERE integrated_at IS NOT NULL ORDER BY integrated_at")
        rows = c.fetchall()
        conn.close()
        if not rows:
            vr.info_msg("capabilities: no rows with integrated_at")
            return records

        # Cumulative capability count per day → score = count / total
        total = len(rows)
        by_day: Dict[str, int] = defaultdict(int)
        for row in rows:
            dt = _parse_ts(row["integrated_at"])
            if dt:
                by_day[dt.date().isoformat()] += 1

        running = 0
        for day_str in sorted(by_day.keys()):
            running += by_day[day_str]
            score = _clamp(running / total)
            day_dt = datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc)
            records.append({
                "snapshot_date":          day_str,
                "snapshot_ts":            day_dt.isoformat(),
                "source":                 "capabilities_table",
                "source_detail":          f"{running}/{total} capabilities integrated",
                "agentic_capability_score": round(score, 4),
                "raw_payload":            {"cumulative_capabilities": running, "total_capabilities": total},
            })
        vr.info_msg(f"capabilities: {total} rows → {len(records)} daily snapshots")
    except Exception as e:
        vr.error(f"capabilities extraction failed: {e}")
    return records


def extract_learning_progress(vr: ValidationResult) -> List[Dict]:
    """
    Derive skill_acquisition_rate, transfer_learning_rate, zero_shot_success_count
    from stage_syllabus/learning_progress.json.
    """
    records = []
    lp_path = LEARNING_DIR / "stage_syllabus" / "learning_progress.json"
    if not lp_path.exists():
        vr.warning(f"learning_progress.json not found at {lp_path}")
        return records

    try:
        data = json.loads(lp_path.read_text(encoding="utf-8"))
        learned_topics = data.get("learned_topics", {})

        # last_updated gives us the snapshot timestamp
        last_updated_str = data.get("last_updated") or data.get("last_learning_cycle")
        dt = _parse_ts(last_updated_str)
        if dt is None:
            dt = datetime.now(timezone.utc)
            vr.warning("learning_progress: no last_updated — using now()")

        # Compute KPIs from topic scores
        all_topics = {}
        for stage, topics in learned_topics.items():
            for topic, score in topics.items():
                if not topic.startswith("_"):
                    all_topics[topic] = max(all_topics.get(topic, 0), score)

        total_topics    = len(all_topics)
        mastered_topics = sum(1 for s in all_topics.values() if s >= 3)
        avg_score       = (sum(all_topics.values()) / total_topics) if total_topics else 0.0

        skill_acq   = _clamp(avg_score * SKILL_ACQ_SCALE)
        zero_shot   = mastered_topics

        current_stage = data.get("current_stage", "Baby")
        transfer_lr = STAGE_TRANSFER.get(current_stage, 0.0)

        # Metacognition from exam pass rates
        exam_passes = sum(1 for v in all_topics.values() if isinstance(v, bool) and v)
        # Actually count _phase_N_exam_passed
        exam_passes = 0
        for stage_topics in learned_topics.values():
            exam_passes += sum(1 for k, v in stage_topics.items() if k.endswith("_exam_passed") and v)

        metacognition = _clamp(exam_passes / 6.0)  # 6 phases in Baby stage

        # Recursive self-improvement: count EVOLUTION topics mastered
        evo_topics = [t for t in all_topics if t.startswith("EVOLUTION")]
        rsi = _clamp(len(evo_topics) / 10.0)  # normalise to expected ~10 evolution topics

        # Validate: check for format mismatches
        for stage, topics in learned_topics.items():
            for topic, score in topics.items():
                if not topic.startswith("_") and not isinstance(score, (int, float)):
                    vr.warning(f"learning_progress: non-numeric score for {stage}/{topic}: {score!r}")

        records.append({
            "snapshot_date":                  dt.date().isoformat(),
            "snapshot_ts":                    dt.isoformat(),
            "source":                         "learning_progress",
            "source_detail":                  f"stage={current_stage}, topics={total_topics}, mastered={mastered_topics}",
            "skill_acquisition_rate":         round(skill_acq, 4),
            "transfer_learning_rate":         round(transfer_lr, 4),
            "zero_shot_success_count":        zero_shot,
            "metacognition_accuracy":         round(metacognition, 4),
            "recursive_self_improvement_rate": round(rsi, 4),
            "evolution_stage":                current_stage,
            "raw_payload": {
                "total_topics":    total_topics,
                "mastered_topics": mastered_topics,
                "avg_score":       round(avg_score, 4),
                "current_stage":   current_stage,
                "exam_passes":     exam_passes,
                "evo_topics":      evo_topics,
            },
        })
        vr.info_msg(f"learning_progress: {total_topics} topics, stage={current_stage}, "
                    f"mastered={mastered_topics}")
    except Exception as e:
        vr.error(f"learning_progress extraction failed: {e}")
    return records


def extract_compiled_knowledge(vr: ValidationResult) -> List[Dict]:
    """
    Derive sample_efficiency_trend and multi_modal_integration_score
    from compiled_knowledge/*.json ingested_at timestamps.
    """
    records = []
    ck_dir = LEARNING_DIR / "compiled_knowledge"
    if not ck_dir.exists():
        vr.warning(f"compiled_knowledge dir not found at {ck_dir}")
        return records

    files = sorted(ck_dir.glob("*.json"))
    if not files:
        vr.warning("compiled_knowledge: no JSON files found")
        return records

    modules_by_day: Dict[str, List[str]] = defaultdict(list)
    total_files = 0
    for f in files:
        if f.stem == "master_knowledge":
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            ingested_at = d.get("ingested_at") or d.get("completed_at")
            dt = _parse_ts(ingested_at)
            if dt is None:
                vr.warning(f"compiled_knowledge/{f.name}: no ingested_at timestamp")
                continue
            modules_by_day[dt.date().isoformat()].append(d.get("module") or f.stem)
            total_files += 1
        except Exception as e:
            vr.warning(f"compiled_knowledge/{f.name}: parse error — {e}")

    if not modules_by_day:
        vr.warning("compiled_knowledge: no parseable ingested_at timestamps found")
        return records

    total_modules = total_files
    running = 0
    for day_str in sorted(modules_by_day.keys()):
        mods = modules_by_day[day_str]
        running += len(mods)
        mm_score = _clamp(running / max(total_modules, 1))
        # Sample efficiency: modules-per-day relative to a baseline of 1 per week
        sample_eff = _clamp(len(mods) / 7.0)
        day_dt = datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc)
        records.append({
            "snapshot_date":                day_str,
            "snapshot_ts":                  day_dt.isoformat(),
            "source":                       "compiled_knowledge",
            "source_detail":                f"modules={mods}",
            "multi_modal_integration_score": round(mm_score, 4),
            "sample_efficiency_trend":      round(sample_eff, 4),
            "raw_payload": {
                "modules_ingested_today": mods,
                "cumulative_modules":    running,
                "total_modules":         total_modules,
            },
        })
    vr.info_msg(f"compiled_knowledge: {total_files} modules across {len(modules_by_day)} days")
    return records


def extract_graph_schema(vr: ValidationResult) -> List[Dict]:
    """Extract neuron/synapse counts and evolution cycle from graph_schema.json."""
    records = []
    if not GRAPH_SCHEMA.exists():
        vr.warning(f"graph_schema.json not found at {GRAPH_SCHEMA}")
        return records
    try:
        gs = json.loads(GRAPH_SCHEMA.read_text(encoding="utf-8"))
        last_updated = gs.get("last_updated")
        dt = _parse_ts(last_updated)
        if dt is None:
            dt = datetime.now(timezone.utc)
            vr.warning("graph_schema: no last_updated — using now()")

        neurons   = int(gs.get("total_neurons", 0))
        synapses  = int(gs.get("total_synapses", 0))
        cycle     = int(gs.get("evolution_cycle", 0))

        # Graph density as a KPI proxy
        graph_density = _clamp((neurons * NEURON_WEIGHT) + (synapses * SYNAPSE_WEIGHT))
        # Recursive self-improvement ≈ evolution cycle normalised to expected ~50 cycles
        rsi = _clamp(cycle / 50.0)

        records.append({
            "snapshot_date":                  dt.date().isoformat(),
            "snapshot_ts":                    dt.isoformat(),
            "source":                         "graph_schema",
            "source_detail":                  f"cycle={cycle}, neurons={neurons}, synapses={synapses}",
            "total_neurons":                  neurons,
            "total_synapses":                 synapses,
            "evolution_cycle":                cycle,
            "recursive_self_improvement_rate": round(rsi, 4),
            "raw_payload":                    {k: gs[k] for k in ["evolution_cycle", "total_neurons",
                                                                    "total_synapses", "last_updated",
                                                                    "schema_version"]
                                               if k in gs},
        })
        vr.info_msg(f"graph_schema: cycle={cycle}, neurons={neurons}, synapses={synapses}")
    except Exception as e:
        vr.error(f"graph_schema extraction failed: {e}")
    return records


def build_baseline_record(vr: ValidationResult) -> List[Dict]:
    """Emit a zero-baseline row at the project epoch (2026-04-01)."""
    vr.info_msg(f"baseline: emitting zero-state row at {PROJECT_EPOCH}")
    return [{
        "snapshot_date":                  PROJECT_EPOCH.isoformat(),
        "snapshot_ts":                    datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        "source":                         "baseline",
        "source_detail":                  "SICore DEFAULT_STATE — project epoch",
        "skill_acquisition_rate":         0.0,
        "transfer_learning_rate":         0.0,
        "zero_shot_success_count":        0,
        "agentic_capability_score":       0.0,
        "recursive_self_improvement_rate": 0.0,
        "sample_efficiency_trend":        0.0,
        "metacognition_accuracy":         0.0,
        "multi_modal_integration_score":  0.0,
        "consciousness_score":            0.0,
        "evolution_stage":                "Baby",
        "notes":                          "Synthetic baseline row — all KPIs initialised to 0.0 per SICore DEFAULT_STATE",
        "raw_payload":                    {"source": "si_core.py _DEFAULT_STATE"},
    }]


# ── Integrity validation ───────────────────────────────────────────────────────

def validate_records(records: List[Dict], vr: ValidationResult) -> List[Dict]:
    """Validate all extracted records; flag format issues and filter out bad rows."""
    valid  = []
    KPI_COLS = [
        "skill_acquisition_rate", "transfer_learning_rate",
        "agentic_capability_score", "recursive_self_improvement_rate",
        "sample_efficiency_trend", "metacognition_accuracy",
        "multi_modal_integration_score", "consciousness_score",
    ]

    for i, rec in enumerate(records):
        issues = []
        # Required fields
        if not rec.get("snapshot_date"):
            issues.append("missing snapshot_date")
        if not rec.get("source"):
            issues.append("missing source")

        # Date validity
        try:
            d = date.fromisoformat(rec["snapshot_date"])
            if d < PROJECT_EPOCH:
                issues.append(f"date {d} precedes project epoch {PROJECT_EPOCH}")
            if d > date.today():
                issues.append(f"date {d} is in the future")
        except Exception as e:
            issues.append(f"invalid snapshot_date: {e}")

        # KPI range checks
        for col in KPI_COLS:
            v = rec.get(col)
            if v is not None:
                if not isinstance(v, (int, float)):
                    issues.append(f"{col} is non-numeric: {v!r}")
                elif not (0.0 <= float(v) <= 1.0):
                    issues.append(f"{col}={v} out of [0,1] range — clamping")
                    rec[col] = _clamp(float(v))

        if rec.get("zero_shot_success_count") is not None:
            z = rec["zero_shot_success_count"]
            if not isinstance(z, int) or z < 0:
                issues.append(f"zero_shot_success_count={z!r} invalid — setting to 0")
                rec["zero_shot_success_count"] = 0

        if issues:
            for issue in issues:
                if "precedes" in issue or "invalid" in issue or "missing" in issue:
                    vr.error(f"Record[{i}] ({rec.get('source')}/{rec.get('snapshot_date')}): {issue}")
                else:
                    vr.warning(f"Record[{i}] ({rec.get('source')}/{rec.get('snapshot_date')}): {issue}")

        # Only skip records with hard errors
        hard_errors = [iss for iss in issues if any(
            kw in iss for kw in ("missing snapshot_date", "missing source", "invalid snapshot_date",
                                  "precedes", "in the future")
        )]
        if not hard_errors:
            valid.append(rec)

    vr.info_msg(f"Validation: {len(valid)}/{len(records)} records passed")
    return valid


# ── PostgreSQL ingest ─────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS si_kpi_history (
    id                              SERIAL PRIMARY KEY,
    snapshot_date                   DATE NOT NULL,
    snapshot_ts                     TIMESTAMPTZ NOT NULL,
    source                          TEXT NOT NULL,
    source_detail                   TEXT,
    skill_acquisition_rate          FLOAT,
    transfer_learning_rate          FLOAT,
    zero_shot_success_count         INTEGER,
    agentic_capability_score        FLOAT,
    recursive_self_improvement_rate FLOAT,
    sample_efficiency_trend         FLOAT,
    metacognition_accuracy          FLOAT,
    multi_modal_integration_score   FLOAT,
    consciousness_score             FLOAT,
    evolution_stage                 TEXT,
    total_insights                  INTEGER,
    quality_insights                INTEGER,
    total_neurons                   INTEGER,
    total_synapses                  INTEGER,
    evolution_cycle                 INTEGER,
    notes                           TEXT,
    raw_payload                     JSONB,
    ingested_at                     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (snapshot_date, source, source_detail)
);
CREATE INDEX IF NOT EXISTS idx_si_kpi_history_date ON si_kpi_history (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_si_kpi_history_source ON si_kpi_history (source);
"""

# Useful longitudinal queries added as a comment in the report
LONGITUDINAL_QUERIES = """
-- Consciousness over time
SELECT snapshot_date, source, consciousness_score
  FROM si_kpi_history
 WHERE consciousness_score IS NOT NULL
 ORDER BY snapshot_date, source;

-- All 8 KPIs latest snapshot
SELECT * FROM si_kpi_history
 WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM si_kpi_history)
 ORDER BY source;

-- KPI trend (daily max across sources)
SELECT snapshot_date,
       MAX(consciousness_score)             AS consciousness,
       MAX(skill_acquisition_rate)          AS skill_acq,
       MAX(transfer_learning_rate)          AS transfer_lr,
       MAX(agentic_capability_score)        AS agentic,
       MAX(multi_modal_integration_score)   AS multimodal,
       MAX(metacognition_accuracy)          AS metacognition,
       MAX(recursive_self_improvement_rate) AS rsi
  FROM si_kpi_history
 GROUP BY snapshot_date
 ORDER BY snapshot_date;
"""


def ingest_to_postgres(records: List[Dict], force: bool, dry_run: bool, vr: ValidationResult) -> Tuple[int, int, int]:
    """
    Write records to Postgres si_kpi_history table.
    Returns (inserted, skipped_duplicate, errors).
    """
    if dry_run:
        vr.info_msg(f"DRY RUN: would insert {len(records)} records (no DB writes)")
        return (len(records), 0, 0)

    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        vr.warning("DATABASE_URL not set — writing to local SQLite si_kpi_history.db instead")
        return ingest_to_sqlite_fallback(records, force, vr)

    try:
        import psycopg2
        import psycopg2.extras
        if dsn.startswith("postgres://"):
            dsn = "postgresql://" + dsn[len("postgres://"):]
        conn = psycopg2.connect(dsn, connect_timeout=15)
        cur = conn.cursor()
    except Exception as e:
        vr.error(f"Postgres connection failed: {e}")
        vr.warning("Falling back to local SQLite si_kpi_history.db")
        return ingest_to_sqlite_fallback(records, force, vr)

    inserted = skipped = errors = 0
    try:
        cur.execute(DDL)
        conn.commit()

        COLS = [
            "snapshot_date", "snapshot_ts", "source", "source_detail",
            "skill_acquisition_rate", "transfer_learning_rate", "zero_shot_success_count",
            "agentic_capability_score", "recursive_self_improvement_rate",
            "sample_efficiency_trend", "metacognition_accuracy",
            "multi_modal_integration_score", "consciousness_score",
            "evolution_stage", "total_insights", "quality_insights",
            "total_neurons", "total_synapses", "evolution_cycle", "notes", "raw_payload",
        ]

        conflict_action = "DO UPDATE SET ingested_at = NOW(), notes = EXCLUDED.notes" if force else "DO NOTHING"

        for rec in records:
            vals = []
            for col in COLS:
                v = rec.get(col)
                if col == "raw_payload" and isinstance(v, dict):
                    v = json.dumps(v)
                vals.append(v)
            try:
                cur.execute(
                    f"INSERT INTO si_kpi_history ({', '.join(COLS)}) "
                    f"VALUES ({', '.join(['%s'] * len(COLS))}) "
                    f"ON CONFLICT (snapshot_date, source, source_detail) {conflict_action}",
                    vals,
                )
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                vr.error(f"Row insert error ({rec.get('source')}/{rec.get('snapshot_date')}): {e}")
                errors += 1
                conn.rollback()

        conn.commit()
        conn.close()
        vr.info_msg(f"Postgres: {inserted} inserted, {skipped} skipped (duplicates), {errors} errors")
    except Exception as e:
        vr.error(f"Postgres DDL/ingest error: {e}")
        errors += 1

    return inserted, skipped, errors


def ingest_to_sqlite_fallback(records: List[Dict], force: bool, vr: ValidationResult) -> Tuple[int, int, int]:
    """Write records to a local SQLite file when Postgres is unavailable."""
    out_db = DATA_DIR / "si_kpi_history.db"
    out_db.parent.mkdir(parents=True, exist_ok=True)

    DDL_SQLITE = DDL.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT") \
                    .replace("TIMESTAMPTZ", "TEXT") \
                    .replace("JSONB", "TEXT") \
                    .replace("FLOAT", "REAL")

    conn = sqlite3.connect(str(out_db))
    c = conn.cursor()
    for stmt in DDL_SQLITE.split(";"):
        stmt = stmt.strip()
        if stmt:
            try:
                c.execute(stmt)
            except Exception:
                pass
    conn.commit()

    inserted = skipped = errors = 0
    COLS = [
        "snapshot_date", "snapshot_ts", "source", "source_detail",
        "skill_acquisition_rate", "transfer_learning_rate", "zero_shot_success_count",
        "agentic_capability_score", "recursive_self_improvement_rate",
        "sample_efficiency_trend", "metacognition_accuracy",
        "multi_modal_integration_score", "consciousness_score",
        "evolution_stage", "total_insights", "quality_insights",
        "total_neurons", "total_synapses", "evolution_cycle", "notes", "raw_payload",
    ]

    for rec in records:
        vals = [json.dumps(rec.get(col)) if isinstance(rec.get(col), dict)
                else rec.get(col) for col in COLS]
        try:
            if force:
                c.execute(f"INSERT OR REPLACE INTO si_kpi_history ({', '.join(COLS)}) "
                          f"VALUES ({', '.join(['?'] * len(COLS))})", vals)
            else:
                c.execute(f"INSERT OR IGNORE INTO si_kpi_history ({', '.join(COLS)}) "
                          f"VALUES ({', '.join(['?'] * len(COLS))})", vals)
            if c.rowcount == 1:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            vr.error(f"SQLite fallback row error: {e}")
            errors += 1

    conn.commit()
    conn.close()
    vr.info_msg(f"SQLite fallback ({out_db}): {inserted} inserted, {skipped} skipped, {errors} errors")
    return inserted, skipped, errors


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    records: List[Dict],
    vr: ValidationResult,
    inserted: int,
    skipped: int,
    errors: int,
    dry_run: bool,
) -> str:
    """Generate a human-readable summary report."""

    lines = [
        "=" * 72,
        "DMAI SI CONSCIOUSNESS KPI — BACKFILL REPORT",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "=" * 72,
        "",
    ]

    # ── Ingest summary
    lines += [
        "## INGEST SUMMARY",
        f"  Mode:             {'DRY RUN (no writes)' if dry_run else 'LIVE'}",
        f"  Total records:    {len(records)}",
        f"  Inserted:         {inserted}",
        f"  Skipped (dupe):   {skipped}",
        f"  Errors:           {errors}",
        "",
    ]

    # ── Source breakdown
    by_source: Dict[str, int] = defaultdict(int)
    for r in records:
        by_source[r.get("source", "unknown")] += 1
    lines.append("## RECORDS BY SOURCE")
    for src, cnt in sorted(by_source.items()):
        lines.append(f"  {src:<40} {cnt:>4} records")
    lines.append("")

    # ── Date range
    dates = [r["snapshot_date"] for r in records if r.get("snapshot_date")]
    if dates:
        lines += [
            "## TEMPORAL COVERAGE",
            f"  Earliest snapshot: {min(dates)}",
            f"  Latest snapshot:   {max(dates)}",
            f"  Unique days:       {len(set(dates))}",
            "",
        ]

    # ── KPI coverage matrix
    KPI_COLS = [
        "skill_acquisition_rate", "transfer_learning_rate",
        "zero_shot_success_count", "agentic_capability_score",
        "recursive_self_improvement_rate", "sample_efficiency_trend",
        "metacognition_accuracy", "multi_modal_integration_score",
        "consciousness_score",
    ]
    lines.append("## KPI COVERAGE (records containing each KPI)")
    kpi_counts = {k: sum(1 for r in records if r.get(k) is not None) for k in KPI_COLS}
    all_covered = True
    for kpi, cnt in kpi_counts.items():
        status = "✓" if cnt > 0 else "✗ GAP"
        lines.append(f"  {kpi:<45} {cnt:>4} records  {status}")
        if cnt == 0:
            all_covered = False
    lines.append("")

    # ── Latest KPI values
    latest: Dict[str, Any] = {}
    for r in sorted(records, key=lambda x: x.get("snapshot_date", "")):
        for k in KPI_COLS:
            if r.get(k) is not None:
                latest[k] = r[k]
    if latest:
        lines.append("## LATEST OBSERVED KPI VALUES")
        for k, v in sorted(latest.items()):
            lines.append(f"  {k:<45} {v}")
        lines.append("")

    # ── Gaps and mismatches
    lines.append("## GAPS & FORMAT MISMATCHES")
    if vr.errors:
        lines.append(f"  ERRORS ({len(vr.errors)}):")
        for e in vr.errors:
            lines.append(f"    ✗ {e}")
    if vr.warnings:
        lines.append(f"  WARNINGS ({len(vr.warnings)}):")
        for w in vr.warnings:
            lines.append(f"    ⚠ {w}")

    # Specific gap analysis
    gaps = []
    missing_kpis = [k for k, c in kpi_counts.items() if c == 0]
    if missing_kpis:
        gaps.append(f"No historical data found for: {', '.join(missing_kpis)}")

    # Check for temporal gaps (> 7 days between consecutive snapshots)
    day_list = sorted(set(dates))
    for i in range(1, len(day_list)):
        d0 = date.fromisoformat(day_list[i - 1])
        d1 = date.fromisoformat(day_list[i])
        gap_days = (d1 - d0).days
        if gap_days > 7:
            gaps.append(f"Temporal gap: {gap_days} days between {day_list[i-1]} and {day_list[i]}")

    # Check for evolution_cycles being empty
    if not any(r["source"] == "evolution_cycles" for r in records):
        gaps.append("evolution_cycles table is empty — no direct consciousness_level snapshots available")

    # Check if si_core_state.json exists
    si_state = DATA_DIR / "si_core_state.json"
    if not si_state.exists():
        gaps.append(f"si_core_state.json not found at {si_state} — no persisted SICore KPI state to backfill")

    for gap in gaps:
        lines.append(f"    ⚠ {gap}")

    if not vr.errors and not vr.warnings and not gaps:
        lines.append("    ✓ No gaps or mismatches detected")
    lines.append("")

    # ── Recommendations
    lines.append("## RECOMMENDATIONS")
    recs_list = []
    if missing_kpis:
        recs_list.append(
            "Run a real training session via POST /api/training/run to generate "
            "non-zero KPI values — all KPIs currently derive from zero-baseline."
        )
    if not any(r["source"] == "evolution_cycles" for r in records):
        recs_list.append(
            "The evolution_cycles table is empty. Once DMAI completes its first "
            "knowledge-graph evolution run (Friday cron), a direct consciousness_level "
            "snapshot will be inserted there."
        )
    if (DATA_DIR / "si_kpi_history.db").exists() and not os.environ.get("DATABASE_URL"):
        recs_list.append(
            "Set DATABASE_URL on Render (pointing to dmai-harvester-db PostgreSQL) "
            "and re-run this script to migrate from the local SQLite fallback."
        )
    recs_list.append(
        "Use these SQL queries for longitudinal analysis:\n" + LONGITUDINAL_QUERIES
    )
    for i, r in enumerate(recs_list, 1):
        lines.append(f"  {i}. {r}")
    lines.append("")

    # ── Info log
    lines.append("## EXTRACTION LOG")
    for msg in vr.info:
        lines.append(f"  ℹ {msg}")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill DMAI SI Consciousness KPIs into PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction and validation only — no DB writes")
    parser.add_argument("--force",   action="store_true", help="Re-insert even if UNIQUE conflict exists")
    parser.add_argument("--report",  default="data/si_kpi_backfill_report.txt", help="Path to write the text report")
    args = parser.parse_args()

    vr = ValidationResult()
    all_records: List[Dict] = []

    logger.info("Starting SI KPI backfill...")

    # ── Extract from all sources
    logger.info("[1/6] Extracting SQLite insights...")
    all_records.extend(extract_sqlite_insights(vr))

    logger.info("[2/6] Extracting evolution_cycles...")
    all_records.extend(extract_evolution_cycles(vr))

    logger.info("[3/6] Extracting capabilities...")
    all_records.extend(extract_capabilities(vr))

    logger.info("[4/6] Extracting learning progress...")
    all_records.extend(extract_learning_progress(vr))

    logger.info("[5/6] Extracting compiled knowledge...")
    all_records.extend(extract_compiled_knowledge(vr))

    logger.info("[6/6] Extracting graph schema...")
    all_records.extend(extract_graph_schema(vr))

    # Always include baseline
    all_records.extend(build_baseline_record(vr))

    logger.info(f"Extracted {len(all_records)} raw records")

    # ── Validate
    valid_records = validate_records(all_records, vr)
    logger.info(f"Validated: {len(valid_records)}/{len(all_records)} records OK")

    # ── Ingest
    inserted, skipped, errors = ingest_to_postgres(valid_records, args.force, args.dry_run, vr)

    # ── Report
    report = generate_report(valid_records, vr, inserted, skipped, errors, args.dry_run)

    # Write report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report written to {report_path}")
    print(report)

    sys.exit(0 if not vr.errors else 1)


if __name__ == "__main__":
    main()
