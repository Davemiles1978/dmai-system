"""Test the JSONL backlog ingestion for self-generation."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from components.self_generation_seed_backlog import seed_backlog


class SeedBacklogTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test.db")
        self.jsonl_path = str(Path(self.tmpdir) / "backlog.jsonl")

        # Minimal post-PR-HH capabilities schema
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE capabilities (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                capability_type TEXT,
                description TEXT,
                provenance TEXT,
                judge_confidence REAL,
                runtime_mode TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Sample backlog
        rows = [
            {
                "id": "gap_test_alpha",
                "name": "Alpha test capability",
                "capability_type": "utility",
                "description": "Test capability alpha",
                "priority": 1,
                "provenance": "gap_driven",
                "runtime_mode": "stub",
                "judge_confidence": 0.75,
            },
            {
                "id": "gap_test_beta",
                "name": "Beta test capability",
                "capability_type": "analyser",
                "description": "Test capability beta",
                "priority": 2,
                "provenance": "gap_driven",
                "runtime_mode": "stub",
                "judge_confidence": 0.70,
            },
        ]
        with open(self.jsonl_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_dry_run_writes_nothing(self):
        summary = seed_backlog(
            jsonl_path=self.jsonl_path,
            db_path=self.db_path,
            dry_run=True,
        )
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["read"], 2)
        self.assertEqual(summary["valid"], 2)
        self.assertEqual(summary["inserted"], 0)
        self.assertEqual(summary["would_insert"], 2)

        # DB should be empty
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0]
        conn.close()
        self.assertEqual(n, 0)

    def test_live_insert(self):
        summary = seed_backlog(
            jsonl_path=self.jsonl_path,
            db_path=self.db_path,
            dry_run=False,
        )
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["inserted"], 2)
        self.assertEqual(summary["already_present"], 0)

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id, name, provenance, runtime_mode, judge_confidence "
            "FROM capabilities ORDER BY id"
        ).fetchall()
        conn.close()

        self.assertEqual(len(rows), 2)
        # Both should be gap_driven / stub with confidence >= 0.60
        for row in rows:
            self.assertEqual(row[2], "gap_driven")
            self.assertEqual(row[3], "stub")
            self.assertGreaterEqual(row[4], 0.60)

    def test_idempotent(self):
        # First insert
        seed_backlog(jsonl_path=self.jsonl_path, db_path=self.db_path)
        # Second insert: nothing new
        summary = seed_backlog(
            jsonl_path=self.jsonl_path, db_path=self.db_path,
        )
        self.assertEqual(summary["inserted"], 0)
        self.assertEqual(summary["already_present"], 2)

    def test_invalid_rows_reported(self):
        bad_path = str(Path(self.tmpdir) / "bad.jsonl")
        with open(bad_path, "w") as f:
            f.write(json.dumps({"id": "bad_1", "name": "missing type"}) + "\n")
            f.write(json.dumps({
                "id": "gap_ok",
                "name": "OK",
                "capability_type": "utility",
                "description": "d",
                "priority": 1,
                "judge_confidence": 0.7,
            }) + "\n")

        summary = seed_backlog(jsonl_path=bad_path, db_path=self.db_path)
        self.assertEqual(summary["read"], 2)
        self.assertEqual(summary["valid"], 1)
        self.assertEqual(summary["inserted"], 1)
        self.assertEqual(len(summary["invalid"]), 1)
        self.assertEqual(summary["invalid"][0]["id"], "bad_1")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            seed_backlog(jsonl_path="/nonexistent/nope.jsonl",
                         db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
