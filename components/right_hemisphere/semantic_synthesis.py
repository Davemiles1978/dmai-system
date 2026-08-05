"""
DMAI Right Hemisphere — Semantic Synthesis
===========================================
Soft clustering, contextual grouping, and pattern recognition on top of
the VectorStore.  This is where DMAI discovers *emergent* relationships
that aren't encoded in the structured knowledge graph.

Capabilities:
  - cluster_concepts(): Louvain-inspired greedy clustering by embedding proximity
  - find_analogies(): Given concept A, find concepts with similar relationship patterns
  - detect_novelty(): Identify outlier concepts that don't fit existing clusters
  - contextual_group(): Group entities by a shared context prompt
  - cross_domain_bridge(): Find latent connections between different entity types
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("dmai.right_hemisphere.semantic_synthesis")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CLUSTER_FILE = _REPO_ROOT / "data" / "right_hemisphere" / "clusters.json"
_NOVELTY_FILE = _REPO_ROOT / "data" / "right_hemisphere" / "novelty_log.jsonl"


class SemanticSynthesis:
    """
    Soft clustering and pattern discovery over vector embeddings.
    Does NOT require an embedding model — works with pre-stored vectors
    in the VectorStore.  For embedding generation, use the existing
    AI hub providers.
    """

    def __init__(self, vector_store=None):
        self._vs = vector_store
        self.cluster_file = _CLUSTER_FILE
        self.cluster_file.parent.mkdir(parents=True, exist_ok=True)
        self.novelty_file = _NOVELTY_FILE
        self.novelty_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("SemanticSynthesis initialised")

    @property
    def vs(self):
        if self._vs is None:
            from components.right_hemisphere.vector_store import VectorStore
            self._vs = VectorStore()
        return self._vs

    # ------------------------------------------------------------------
    # Soft clustering (greedy, Louvain-inspired)
    # ------------------------------------------------------------------

    def cluster_concepts(
        self,
        entity_type: str = "concept",
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 2,
    ) -> List[Dict]:
        """
        Group concepts into soft clusters based on embedding proximity.
        Each concept can belong to multiple clusters (soft membership).

        Returns list of {cluster_id, members: [id], centroid, coherence}.
        """
        # Load all vectors of this type
        ids_and_vectors: List[Tuple[str, List[float], Dict]] = []
        with __import__('sqlite3').connect(str(self.vs.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, embedding, dimensions, metadata FROM vectors WHERE entity_type = ?",
                (entity_type,),
            ).fetchall()
            for row_id, blob, dims, meta_json in rows:
                try:
                    vec = self.vs._unpack_embedding(blob, dims)
                    meta = json.loads(meta_json) if meta_json else {}
                    ids_and_vectors.append((row_id, vec, meta))
                except Exception:
                    continue

        if len(ids_and_vectors) < min_cluster_size:
            return []

        # Greedy clustering: seed with first unassigned, assign all within threshold
        assigned: Set[str] = set()
        clusters: List[Dict] = []
        cluster_id = 0

        for i, (seed_id, seed_vec, _) in enumerate(ids_and_vectors):
            if seed_id in assigned:
                continue
            members = [seed_id]
            assigned.add(seed_id)
            for other_id, other_vec, _ in ids_and_vectors[i + 1:]:
                if other_id in assigned:
                    continue
                sim = self.vs.cosine_similarity(seed_vec, other_vec)
                if sim >= similarity_threshold:
                    members.append(other_id)
                    assigned.add(other_id)

            if len(members) >= min_cluster_size:
                # Compute centroid
                all_vecs = [v for vid, v, _ in ids_and_vectors if vid in members]
                centroid = [
                    sum(v[d] for v in all_vecs) / len(all_vecs)
                    for d in range(len(all_vecs[0]))
                ]
                # Coherence = mean pairwise similarity within cluster
                coherence = self._cluster_coherence(
                    [(vid, v) for vid, v, _ in ids_and_vectors if vid in members]
                )
                clusters.append({
                    "cluster_id": f"cluster_{cluster_id}",
                    "members": members,
                    "size": len(members),
                    "centroid_sample": centroid[:8],  # first 8 dims for logging
                    "coherence": round(coherence, 4),
                    "entity_type": entity_type,
                })
                cluster_id += 1

        # Persist
        self._save_clusters(entity_type, clusters)
        logger.info(
            "SemanticSynthesis: %d clusters found for %s (threshold=%.2f)",
            len(clusters), entity_type, similarity_threshold,
        )
        return clusters

    def _cluster_coherence(self, members: List[Tuple[str, List[float]]]) -> float:
        """Mean pairwise cosine similarity within a cluster."""
        if len(members) < 2:
            return 1.0
        sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sims.append(self.vs.cosine_similarity(members[i][1], members[j][1]))
        return sum(sims) / len(sims) if sims else 1.0

    def _save_clusters(self, entity_type: str, clusters: List[Dict]) -> None:
        data = {}
        if self.cluster_file.exists():
            try:
                data = json.loads(self.cluster_file.read_text())
            except Exception:
                pass
        data[entity_type] = {
            "clusters": clusters,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_clusters": len(clusters),
        }
        self.cluster_file.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Find analogies
    # ------------------------------------------------------------------

    def find_analogies(
        self,
        source_id: str,
        entity_type: str = "concept",
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find concepts most analogous to the given concept.
        Analogy = similar embedding vectors = related concepts.
        """
        source = self.vs.get(entity_type, source_id)
        if source is None:
            return []

        results = self.vs.search(
            source["embedding"],
            top_k=top_k + 1,  # +1 because source itself will match
            entity_type_filter=entity_type,
        )
        # Filter out the source itself
        analogies = [
            {"id": rid, "score": score, "metadata": meta}
            for rid, score, meta in results
            if rid != f"{entity_type}:{source_id}"
        ][:top_k]
        return analogies

    # ------------------------------------------------------------------
    # Novelty detection
    # ------------------------------------------------------------------

    def detect_novelty(
        self,
        entity_type: str = "concept",
        novelty_threshold: float = 0.3,
    ) -> List[Dict]:
        """
        Identify concepts that are outliers — low similarity to everything else.
        These are potential novel discoveries or errors that need attention.
        """
        ids_and_vectors = []
        with __import__('sqlite3').connect(str(self.vs.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, embedding, dimensions, metadata FROM vectors WHERE entity_type = ?",
                (entity_type,),
            ).fetchall()
            for row_id, blob, dims, meta_json in rows:
                try:
                    vec = self.vs._unpack_embedding(blob, dims)
                    meta = json.loads(meta_json) if meta_json else {}
                    ids_and_vectors.append((row_id, vec, meta))
                except Exception:
                    continue

        if len(ids_and_vectors) < 3:
            return []

        novelties = []
        for rid, vec, meta in ids_and_vectors:
            # Mean similarity to all other vectors
            sims = []
            for other_id, other_vec, _ in ids_and_vectors:
                if other_id == rid:
                    continue
                sims.append(self.vs.cosine_similarity(vec, other_vec))
            mean_sim = sum(sims) / len(sims) if sims else 1.0
            if mean_sim < novelty_threshold:
                novelties.append({
                    "id": rid,
                    "mean_similarity": round(mean_sim, 4),
                    "metadata": meta,
                    "is_novel": True,
                })

        novelties.sort(key=lambda x: x["mean_similarity"])
        # Log
        if novelties:
            self._log_novelties(novelties)
        return novelties

    def _log_novelties(self, novelties: List[Dict]) -> None:
        with open(self.novelty_file, "a") as f:
            for n in novelties:
                f.write(json.dumps({
                    **n,
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")

    # ------------------------------------------------------------------
    # Cross-domain bridging
    # ------------------------------------------------------------------

    def cross_domain_bridge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find latent connections between different entity types.
        E.g., map a 'capability' to related 'topics' via embedding proximity.
        """
        source = self.vs.get(source_type, source_id)
        if source is None:
            return []

        results = self.vs.search(
            source["embedding"],
            top_k=top_k,
            entity_type_filter=target_type,
        )
        return [
            {"id": rid, "score": score, "metadata": meta}
            for rid, score, meta in results
        ]

    # ------------------------------------------------------------------
    # Contextual grouping
    # ------------------------------------------------------------------

    def contextual_group(
        self,
        context_embedding: List[float],
        entity_types: Optional[List[str]] = None,
        top_k: int = 20,
        min_score: float = 0.5,
    ) -> Dict[str, List[Dict]]:
        """
        Group entities by their relevance to a contextual prompt embedding.
        Returns results grouped by entity_type.
        """
        grouped: Dict[str, List[Dict]] = defaultdict(list)

        if entity_types is None:
            entity_types = self.vs.get_all_types()

        for etype in entity_types:
            results = self.vs.search(
                context_embedding,
                top_k=top_k,
                entity_type_filter=etype,
                min_score=min_score,
            )
            for rid, score, meta in results:
                grouped[etype].append({"id": rid, "score": score, "metadata": meta})

        return dict(grouped)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        clusters = {}
        if self.cluster_file.exists():
            try:
                clusters = json.loads(self.cluster_file.read_text())
            except Exception:
                pass
        return {
            "vector_store": self.vs.get_stats(),
            "clusters": {
                etype: data.get("total_clusters", 0)
                for etype, data in clusters.items()
            },
            "cluster_file": str(self.cluster_file),
        }
