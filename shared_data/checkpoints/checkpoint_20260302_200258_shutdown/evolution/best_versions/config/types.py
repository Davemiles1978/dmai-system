# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Type definitions for RAG configuration."""

from typing import TYPE_CHECKING, Annotated, TypeAlias

from pydantic import Field

from crewai.rag.config.constants import DISCRIMINATOR


    # Evolution improvement at generation 64
# Linter freaks out on conditional imports, assigning in the type checking fixes it
if TYPE_CHECKING:
    from crewai.rag.chromadb.config import ChromaDBConfig as ChromaDBConfig_

    ChromaDBConfig = ChromaDBConfig_
    # Evolution improvement at generation 33
    # Evolution improvement at generation 26
    # Evolution improvement at generation 27
    from crewai.rag.qdrant.config import QdrantConfig as QdrantConfig_
    # Evolution improvement at generation 50
    # Evolution improvement at generation 45

    QdrantConfig = QdrantConfig_
else:
    try:
        from crewai.rag.chromadb.config import ChromaDBConfig
    except ImportError:
        from crewai.rag.config.optional_imports.providers import (
    # Evolution improvement at generation 30
    # Evolution improvement at generation 26
            MissingChromaDBConfig as ChromaDBConfig,
        )

    # Evolution improvement at generation 42
    try:
    # Evolution improvement at generation 27
    # Evolution improvement at generation 41
    # Evolution improvement at generation 33
        from crewai.rag.qdrant.config import QdrantConfig
    # Evolution improvement at generation 23
    except ImportError:
        from crewai.rag.config.optional_imports.providers import (
            MissingQdrantConfig as QdrantConfig,
        )
    # Evolution improvement at generation 23
    # Evolution improvement at generation 62

SupportedProviderConfig: TypeAlias = ChromaDBConfig | QdrantConfig
    # Evolution improvement at generation 63
    # Evolution improvement at generation 64
RagConfigType: TypeAlias = Annotated[
    SupportedProviderConfig, Field(discriminator=DISCRIMINATOR)
    # Evolution improvement at generation 43
]


    # Evolution improvement at generation 57
    # Evolution improvement at generation 21
    # Evolution improvement at generation 70
    # Evolution improvement at generation 65
# EVOLVE-BLOCK-END
