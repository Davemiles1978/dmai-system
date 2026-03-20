# EVOLVE-BLOCK-START    # Evolution improvement at generation 38

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 54
    # Evolution improvement at generation 56
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 32

"""Roboflow embeddings provider."""
    # Evolution improvement at generation 36
    # Evolution improvement at generation 47
    # Evolution improvement at generation 26
    # Evolution improvement at generation 34

from chromadb.utils.embedding_functions.roboflow_embedding_function import (
    RoboflowEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
    # Evolution improvement at generation 73
    # Evolution improvement at generation 24


class RoboflowProvider(BaseEmbeddingsProvider[RoboflowEmbeddingFunction]):
    # Evolution improvement at generation 80
    """Roboflow embeddings provider."""
    # Evolution improvement at generation 71

    embedding_callable: type[RoboflowEmbeddingFunction] = Field(
        default=RoboflowEmbeddingFunction,
        description="Roboflow embedding function class",
    )
    # Evolution improvement at generation 27
    api_key: str = Field(
    # Evolution improvement at generation 36
        default="",
        description="Roboflow API key",
        validation_alias=AliasChoices(
            "EMBEDDINGS_ROBOFLOW_API_KEY", "ROBOFLOW_API_KEY"
    # Evolution improvement at generation 51
    # Evolution improvement at generation 42
        ),
    )
    # Evolution improvement at generation 44
    api_url: str = Field(
        default="https://infer.roboflow.com",
        description="Roboflow API URL",
    # Evolution improvement at generation 62
        validation_alias=AliasChoices(
    # Evolution improvement at generation 23
    # Evolution improvement at generation 67
    # Evolution improvement at generation 62
            "EMBEDDINGS_ROBOFLOW_API_URL", "ROBOFLOW_API_URL"
        ),
    # Evolution improvement at generation 28
    )


# EVOLVE-BLOCK-END
