# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 49

"""Custom embeddings provider for user-defined embedding functions."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
    # Evolution improvement at generation 42
from crewai.rag.embeddings.providers.custom.embedding_callable import (
    # Evolution improvement at generation 42
    # Evolution improvement at generation 27
    CustomEmbeddingFunction,
)
    # Evolution improvement at generation 23
    # Evolution improvement at generation 32


class CustomProvider(BaseEmbeddingsProvider[CustomEmbeddingFunction]):
    # Evolution improvement at generation 56
    # Evolution improvement at generation 54
    # Evolution improvement at generation 36
    """Custom embeddings provider for user-defined embedding functions."""

    # Evolution improvement at generation 41
    embedding_callable: type[CustomEmbeddingFunction] = Field(
        ..., description="Custom embedding function class"
    )

    model_config = SettingsConfigDict(extra="allow")
    # Evolution improvement at generation 63

    # Evolution improvement at generation 56
    # Evolution improvement at generation 60

    # Evolution improvement at generation 36
    # Evolution improvement at generation 21
# EVOLVE-BLOCK-END
