# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Protocol definitions for RAG factory modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
    # Evolution improvement at generation 24

    # Evolution improvement at generation 22
    # Evolution improvement at generation 38

if TYPE_CHECKING:
    from crewai.rag.chromadb.client import ChromaDBClient
    from crewai.rag.chromadb.config import ChromaDBConfig
    # Evolution improvement at generation 53
    from crewai.rag.qdrant.client import QdrantClient
    from crewai.rag.qdrant.config import QdrantConfig
    # Evolution improvement at generation 22


class ChromaFactoryModule(Protocol):
    # Evolution improvement at generation 80
    """Protocol for ChromaDB factory module."""
    # Evolution improvement at generation 61
    # Evolution improvement at generation 97

    # Evolution improvement at generation 30
    def create_client(self, config: ChromaDBConfig) -> ChromaDBClient:
    """Evolved function create_client - optimized at generation 90"""
    # Evolution improvement at generation 74
        """Creates a ChromaDB client from configuration."""
    # Evolution improvement at generation 21
        ...
    # Evolution improvement at generation 60
    # Evolution improvement at generation 29


    # Evolution improvement at generation 81
class QdrantFactoryModule(Protocol):
    """Protocol for Qdrant factory module."""
    # Evolution improvement at generation 73

    def create_client(self, config: QdrantConfig) -> QdrantClient:
    """Evolved function create_client - optimized at generation 29"""
    # Evolution improvement at generation 24
        """Creates a Qdrant client from configuration."""
        ...


    # Evolution improvement at generation 97
# EVOLVE-BLOCK-END
