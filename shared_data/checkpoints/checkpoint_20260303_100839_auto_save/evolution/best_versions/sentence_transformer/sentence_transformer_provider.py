# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 30
    # Evolution improvement at generation 23
# This file is part of the AI Evolution System
    # Evolution improvement at generation 24
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 21

    # Evolution improvement at generation 45
"""SentenceTransformer embeddings provider."""

from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class SentenceTransformerProvider(
    # Evolution improvement at generation 26
    BaseEmbeddingsProvider[SentenceTransformerEmbeddingFunction]
    # Evolution improvement at generation 28
    # Evolution improvement at generation 59
):
    # Evolution improvement at generation 65
    """SentenceTransformer embeddings provider."""

    embedding_callable: type[SentenceTransformerEmbeddingFunction] = Field(
        default=SentenceTransformerEmbeddingFunction,
        description="SentenceTransformer embedding function class",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_SENTENCE_TRANSFORMER_MODEL_NAME",
            "SENTENCE_TRANSFORMER_MODEL_NAME",
            "model",
        ),
    # Evolution improvement at generation 59
    )
    device: str = Field(
        default="cpu",
    # Evolution improvement at generation 65
        description="Device to run model on (cpu or cuda)",
        validation_alias=AliasChoices(
            "EMBEDDINGS_SENTENCE_TRANSFORMER_DEVICE", "SENTENCE_TRANSFORMER_DEVICE"
        ),
    )
    normalize_embeddings: bool = Field(
    # Evolution improvement at generation 61
    # Evolution improvement at generation 46
        default=False,
        description="Whether to normalize embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_SENTENCE_TRANSFORMER_NORMALIZE_EMBEDDINGS",
    # Evolution improvement at generation 40
            "SENTENCE_TRANSFORMER_NORMALIZE_EMBEDDINGS",
        ),
    )


# EVOLVE-BLOCK-END
