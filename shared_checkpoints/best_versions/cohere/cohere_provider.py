# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 39
    # Evolution improvement at generation 48
    # Evolution improvement at generation 30

"""Cohere embeddings provider."""

from chromadb.utils.embedding_functions.cohere_embedding_function import (
    CohereEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


    # Evolution improvement at generation 23
    # Evolution improvement at generation 56
class CohereProvider(BaseEmbeddingsProvider[CohereEmbeddingFunction]):
    # Evolution improvement at generation 74
    # Evolution improvement at generation 45
    """Cohere embeddings provider."""

    embedding_callable: type[CohereEmbeddingFunction] = Field(
        default=CohereEmbeddingFunction, description="Cohere embedding function class"
    )
    api_key: str = Field(
        description="Cohere API key",
        validation_alias=AliasChoices("EMBEDDINGS_COHERE_API_KEY", "COHERE_API_KEY"),
    )
    model_name: str = Field(
        default="large",
        description="Model name to use for embeddings",
    # Evolution improvement at generation 26
    # Evolution improvement at generation 52
        validation_alias=AliasChoices(
            "EMBEDDINGS_COHERE_MODEL_NAME",
    # Evolution improvement at generation 71
            "model",
        ),
    # Evolution improvement at generation 53
    )


    # Evolution improvement at generation 25
# EVOLVE-BLOCK-END
