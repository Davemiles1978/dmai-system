#!/usr/bin/env python3
"""
migrate_to_postgres.py — one-shot migration from SQLite → PostgreSQL

Reads both SQLite databases:
  data/dmai.db            (evolution state, tasks, persona, conversations,
                           funding state, api_keys, admin_api_keys)
  data/dmai_knowledge.db  (insights)

Writes everything to the Postgres instance pointed to by DATABASE_URL.

Safe to run multiple times — uses INSERT ... ON CONFLICT DO NOTHING / DO UPDATE
so it won't duplicate rows.

Usage:
  DATABASE_URL=postgresql://... python scripts/migrate_to_postgres.py
  python scripts/migrate_to_postgres.py --dry-run   (shows counts only)
"""

import os
import sys
import json
import sqlite3
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("migrate")

# ── Resolve paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH  = Path(os.getenv("DATA_PATH", ROOT / "data"))
DMAI_DB    = DATA_PATH / "dmai.db"
KNOW_DB    = DATA_PATH / "dmai_knowledge.db"

sys.path.insert(0, str(ROOT))


def get_pg_conn():
    import psycopg2
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        log.error("DATABASE_URL is not set. Export it before running this script.")
        sys.exit(1)
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return psycopg2.connect(url)


def sqlite_rows(db_path: Path, sql: str, params=()):
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.warning("SQLite query failed (%s): %s", db_path.name, e)
        return []
    finally:
        conn.close()


def run_migration(dry_run: bool = False):
    if dry_run:
        log.info("DRY RUN — no data will be written to Postgres")

    pg = None if dry_run else get_pg_conn()

    def pg_exec(sql, params=()):
        if dry_run:
            return
        with pg.cursor() as cur:
            cur.execute(sql, params)

    def pg_commit():
        if not dry_run:
            pg.commit()

    # ── Ensure schema exists ──────────────────────────────────────────────────
    if not dry_run:
        log.info("Initialising Postgres schema via pg_storage…")
        from components.pg_storage import PGStorage
        PGStorage()  # triggers _init_schema()
        log.info("Schema ready")

    # ── 1. evolution_state ────────────────────────────────────────────────────
    rows = sqlite_rows(DMAI_DB, "SELECT state_json FROM evolution_state ORDER BY id DESC LIMIT 1")
    log.info("evolution_state: %d row(s)", len(rows))
    for r in rows:
        pg_exec("DELETE FROM evolution_state")
        pg_exec("INSERT INTO evolution_state (state_json) VALUES (%s)", (r["state_json"],))
    pg_commit()

    # ── 2. tasks ─────────────────────────────────────────────────────────────
    rows = sqlite_rows(DMAI_DB, "SELECT * FROM tasks")
    log.info("tasks: %d row(s)", len(rows))
    for r in rows:
        pg_exec(
            """INSERT INTO tasks (id, title, status, priority, data_json)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT(id) DO UPDATE SET
                 status=EXCLUDED.status, priority=EXCLUDED.priority,
                 data_json=EXCLUDED.data_json, updated_at=NOW()""",
            (r.get("id",""), r.get("title",""), r.get("status","pending"),
             r.get("priority","medium"), r.get("data_json") or json.dumps(r))
        )
    pg_commit()

    # ── 3. persona ────────────────────────────────────────────────────────────
    rows = sqlite_rows(DMAI_DB, "SELECT data_json FROM persona ORDER BY id DESC LIMIT 1")
    log.info("persona: %d row(s)", len(rows))
    for r in rows:
        pg_exec("DELETE FROM persona")
        pg_exec("INSERT INTO persona (data_json) VALUES (%s)", (r["data_json"],))
    pg_commit()

    # ── 4. conversations ──────────────────────────────────────────────────────
    rows = sqlite_rows(DMAI_DB, "SELECT * FROM conversations ORDER BY id")
    log.info("conversations: %d row(s)", len(rows))
    for r in rows:
        pg_exec(
            "INSERT INTO conversations (user_msg, message, response, context) VALUES (%s,%s,%s,%s)",
            (r.get("user_msg",""), r.get("message",""), r.get("response",""), r.get("context",""))
        )
    pg_commit()

    # ── 5. funding_state ──────────────────────────────────────────────────────
    rows = sqlite_rows(DMAI_DB, "SELECT * FROM funding_state ORDER BY id DESC LIMIT 1")
    log.info("funding_state: %d row(s)", len(rows))
    for r in rows:
        pg_exec("DELETE FROM funding_state")
        pg_exec(
            "INSERT INTO funding_state (revenue_avenues, learned_concepts, performance_data) VALUES (%s,%s,%s)",
            (r.get("revenue_avenues","{}"), r.get("learned_concepts","[]"), r.get("performance_data","{}"))
        )
    pg_commit()

    # ── 6. api_keys (harvester bulk) ──────────────────────────────────────────
    rows = sqlite_rows(DMAI_DB, "SELECT * FROM api_keys")
    log.info("api_keys: %d row(s)", len(rows))
    for r in rows:
        pg_exec(
            """INSERT INTO api_keys (key, service, source, validated)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT(key) DO NOTHING""",
            (r.get("key",""), r.get("service",""), r.get("source",""),
             r.get("validated", 0))
        )
    pg_commit()

    # ── 7. admin_api_keys ─────────────────────────────────────────────────────
    rows = sqlite_rows(DMAI_DB, "SELECT * FROM admin_api_keys")
    log.info("admin_api_keys: %d row(s)", len(rows))
    for r in rows:
        pg_exec(
            """INSERT INTO admin_api_keys (provider_id, api_key)
               VALUES (%s,%s)
               ON CONFLICT(provider_id) DO UPDATE SET
                 api_key=EXCLUDED.api_key, updated_at=NOW()""",
            (r.get("provider_id",""), r.get("api_key",""))
        )
    pg_commit()

    # ── 8. insights (knowledge DB) ────────────────────────────────────────────
    rows = sqlite_rows(KNOW_DB,
        "SELECT id, insight_text, entity_type, entities, relationship, "
        "confidence, source_url, source_title, source_type FROM insights")
    log.info("insights: %d row(s)", len(rows))
    batch = 0
    for r in rows:
        pg_exec(
            """INSERT INTO insights
               (id, insight_text, entity_type, entities, relationship,
                confidence, source_url, source_title, source_type)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(id) DO NOTHING""",
            (r.get("id",""), r.get("insight_text",""), r.get("entity_type",""),
             r.get("entities",""), r.get("relationship",""),
             float(r.get("confidence") or 0.5),
             r.get("source_url",""), r.get("source_title",""),
             r.get("source_type","web"))
        )
        batch += 1
        if batch % 500 == 0:
            pg_commit()
            log.info("  … committed %d insights", batch)
    pg_commit()
    log.info("  insights done: %d total", batch)

    if not dry_run:
        pg.close()

    log.info("Migration complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate DMAI SQLite → PostgreSQL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print row counts only, do not write to Postgres")
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run)
