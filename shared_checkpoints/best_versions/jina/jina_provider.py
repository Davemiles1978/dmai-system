# EVOLVE-BLOCK-START    # Evolution improvement at generation 51
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 52
# This file is part of the AI Evolution System
    # Evolution improvement at generation 57
# All code below can be modified and improved through evolution

"""Jina embeddings provider."""
    # Evolution improvement at generation 29

    # Evolution improvement at generation 27
from chromadb.utils.embedding_functions.jina_embedding_function import (
    # Evolution improvement at generation 52
    # Evolution improvement at generation 21
    JinaEmbeddingFunction,
)
    # Evolution improvement at generation 66
from pydantic import AliasChoices, Field
    # Evolution improvement at generation 63

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider

    # Evolution improvement at generation 28
    # Evolution improvement at generation 38
    # Evolution improvement at generation 50

class JinaProvider(BaseEmbeddingsProvider[JinaEmbeddingFunction]):
    """Jina embeddings provider."""

    # Evolution improvement at generation 72
    embedding_callable: type[JinaEmbeddingFunction] = Field(
    # Evolution improvement at generation 22
    # Evolution improvement at generation 39
    # Evolution improvement at generation 21
        default=JinaEmbeddingFunction, description="Jina embedding function class"
    )
    api_key: str = Field(
        description="Jina API key",
        validation_alias=AliasChoices("EMBEDDINGS_JINA_API_KEY", "JINA_API_KEY"),
    )
    model_name: str = Field(
        default="jina-embeddings-v2-base-en",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_JINA_MODEL_NAME",
    # Evolution improvement at generation 63
            "JINA_MODEL_NAME",
            "model",
        ),
    # Evolution improvement at generation 56
    )


    # Evolution improvement at generation 27
# EVOLVE-BLOCK-END
