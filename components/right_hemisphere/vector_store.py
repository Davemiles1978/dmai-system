"""
DMAI Right Hemisphere — Vector Store (SQLite-backed)
=====================================================
Stores high-dimensional embeddings and enables semantic similarity search
using cosine distance.  Uses SQLite BLOB storage — no external vector DB.

This is the foundation of DMAI's right hemisphere: where concepts live as
geometric relationships rather than rigid ontological triples.

Schema:
  vectors(id TEXT PRIMARY KEY, entity_type TEXT, entity_id TEXT,
          embedding BLOB, dimensions INTEGER, metadata TEXT,
          created_at TEXT, last_accessed TEXT)

Operations:
  - store(entity_type, entity_id, embedding, metadata)
  - search(embedding, top_k, entity_type_filter) -> [(id, score, metadata)]
  - delete(entity_id)
  - count() -> int
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.right_hemisphere.vector_store")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _REPO_ROOT / "data" / "vector_store.db"
_DEFAULT_DIMS = 384  # all-MiniLM-L6-v2 dimension


class VectorStore:
    """
    SQLite-backed vector embedding store with cosine similarity search.
    """

    def __init__(self, db_path: Optional[Path] = None, dimensions: int = _DEFAULT_DIMS):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self._init_db()
        logger.info("VectorStore initialised: %s (dims=%d)", self.db_path, dimensions)

    # ------------------------------------------------------------------
    # Database initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL DEFAULT 384,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_type
                ON vectors(entity_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_entity
                ON vectors(entity_type, entity_id)
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Embedding serialisation (float32 -> BLOB)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_embedding(embedding: List[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _unpack_embedding(blob: bytes, dims: int) -> List[float]:
        return list(struct.unpack(f"{dims}f", blob))

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Return cosine similarity between two vectors. Range: -1.0 to 1.0."""
        if len(a) != len(b):
            raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def store(
        self,
        entity_type: str,
        entity_id: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store an embedding vector. Returns the row id."""
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions}-dim embedding, got {len(embedding)}"
            )
        now = datetime.now(timezone.utc).isoformat()
        row_id = f"{entity_type}:{entity_id}"
        blob = self._pack_embedding(embedding)
        meta_json = json.dumps(metadata or {})

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO vectors
                   (id, entity_type, entity_id, embedding, dimensions, metadata, created_at, last_accessed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (row_id, entity_type, entity_id, blob, self.dimensions, meta_json, now, now),
            )
            conn.commit()
        return row_id

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        entity_type_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for most similar vectors by cosine similarity.
        Returns list of (id, score, metadata) sorted by score descending.

        Note: For large datasets (>10k vectors), this brute-force scan should
        be replaced with approximate nearest neighbor (ANN) indexing. For DMAI's
        current scale this is fast enough.
        """
        if len(query_embedding) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions}-dim query, got {len(query_embedding)}"
            )

        now = datetime.now(timezone.utc).isoformat()
        results: List[Tuple[str, float, Dict]] = []

        with sqlite3.connect(str(self.db_path)) as conn:
            if entity_type_filter:
                rows = conn.execute(
                    "SELECT id, embedding, dimensions, metadata FROM vectors WHERE entity_type = ?",
                    (entity_type_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, embedding, dimensions, metadata FROM vectors"
                ).fetchall()

            for row_id, blob, dims, meta_json in rows:
                try:
                    vec = self._unpack_embedding(blob, dims)
                    score = self.cosine_similarity(query_embedding, vec)
                    if score >= min_score:
                        metadata = json.loads(meta_json) if meta_json else {}
                        results.append((row_id, round(score, 4), metadata))
                except Exception as e:
                    logger.debug("Vector decode error for %s: %s", row_id, e)
                    continue

            # Update last_accessed for returned rows
            returned_ids = [r[0] for r in results]
            if returned_ids:
                conn.executemany(
                    "UPDATE vectors SET last_accessed = ? WHERE id = ?",
                    [(now, rid) for rid in returned_ids],
                )
                conn.commit()

        # Sort by score descending, take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, entity_type: str, entity_id: str) -> bool:
        """Delete a vector by entity type and id. Returns True if deleted."""
        row_id = f"{entity_type}:{entity_id}"
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM vectors WHERE id = ?", (row_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get(self, entity_type: str, entity_id: str) -> Optional[Dict]:
        """Retrieve a stored vector with metadata."""
        row_id = f"{entity_type}:{entity_id}"
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT embedding, dimensions, metadata, created_at FROM vectors WHERE id = ?",
                (row_id,),
            ).fetchone()
            if row is None:
                return None
            blob, dims, meta_json, created_at = row
            return {
                "id": row_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "embedding": self._unpack_embedding(blob, dims),
                "metadata": json.loads(meta_json) if meta_json else {},
                "created_at": created_at,
            }

    def count(self, entity_type_filter: Optional[str] = None) -> int:
        """Return total stored vectors, optionally filtered by type."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if entity_type_filter:
                row = conn.execute(
                    "SELECT COUNT(*) FROM vectors WHERE entity_type = ?",
                    (entity_type_filter,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
            return row[0] if row else 0

    def get_all_types(self) -> List[str]:
        """Return distinct entity types in the store."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT entity_type FROM vectors ORDER BY entity_type"
            ).fetchall()
            return [r[0] for r in rows]

    def get_stats(self) -> Dict:
        """Return store statistics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            types = conn.execute(
                "SELECT entity_type, COUNT(*) as cnt FROM vectors GROUP BY entity_type ORDER BY cnt DESC"
            ).fetchall()
            return {
                "total_vectors": total,
                "dimensions": self.dimensions,
                "by_type": {t: c for t, c in types},
                "db_path": str(self.db_path),
            }
