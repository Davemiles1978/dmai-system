#!/usr/bin/env python3
"""One-shot SQLite → PostgreSQL data migration. Run on Render via shell."""
import os, sys, sqlite3, psycopg2

DATA_PATH = os.environ.get("DATA_PATH", "data")
SQLITE_PATH = os.path.join(DATA_PATH, "dmai_knowledge.db")
PG_DSN = os.environ.get("DATABASE_URL", "")

if not PG_DSN:
    print("DATABASE_URL not set — aborting")
    sys.exit(1)
if PG_DSN.startswith("postgres://"):
    PG_DSN = "postgresql://" + PG_DSN[len("postgres://"):]
if not os.path.exists(SQLITE_PATH):
    print(f"SQLite DB not found at {SQLITE_PATH} — aborting")
    sys.exit(1)

FLAG_FILE = os.path.join(DATA_PATH, ".migration_complete")
if os.path.exists(FLAG_FILE):
    print("Migration already completed")
    sys.exit(0)

print(f"Source: {SQLITE_PATH}")
print(f"Target: {PG_DSN.split('@')[-1]}")  # hide credentials

sq = sqlite3.connect(SQLITE_PATH)
pg = psycopg2.connect(PG_DSN)
pg.autocommit = True

TABLES = [
    "capabilities", "insights", "system_state", "syllabus_content",
    "vocabulary", "encyclopaedia", "graph_neurons", "graph_synapses",
    "mon_wallets", "mon_tips", "at_state", "at_trades", "at_ticks",
    "work_review_queue", "skill_assessments", "brain_entries",
    "knowledge_graph", "suggestions", "topics", "sources",
    "learning_progress", "learning_queue", "conversations",
]

migrated = 0
for table in TABLES:
    try:
        # Check if PG already has data
        cur = pg.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        if cur.fetchone()[0] > 0:
            print(f"  {table}: already populated — skipping")
            cur.close()
            continue
        cur.close()
        
        # Check if SQLite has the table
        cur = sq.cursor()
        cur.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table}'")
        if cur.fetchone()[0] == 0:
            print(f"  {table}: not in SQLite — skipping")
            cur.close()
            continue
        
        # Read from SQLite
        cur.execute(f"SELECT * FROM {table}")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        cur.close()
        
        if not rows:
            print(f"  {table}: empty — skipping")
            continue
        
        # Write to PostgreSQL in batches
        ph = ', '.join(['%s'] * len(cols))
        cl = ', '.join(f'"{c}"' for c in cols)
        sql = f'INSERT INTO {table} ({cl}) VALUES ({ph}) ON CONFLICT DO NOTHING'
        
        pg_cur = pg.cursor()
        batch = []
        written = 0
        for row in rows:
            batch.append(row)
            if len(batch) >= 500:
                for r in batch:
                    try:
                        pg_cur.execute(sql, r)
                        written += 1
                    except Exception:
                        pass
                batch = []
        for r in batch:
            try:
                pg_cur.execute(sql, r)
                written += 1
            except Exception:
                pass
        pg_cur.close()
        print(f"  {table}: {written}/{len(rows)} rows migrated")
        migrated += 1
    except Exception as e:
        print(f"  {table}: ERROR — {e}")

sq.close()
pg.close()

with open(FLAG_FILE, 'w') as f:
    f.write(f"migrated {migrated} tables")
print(f"\nDone. {migrated} tables migrated. Flag: {FLAG_FILE}")
