# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 36
    # Evolution improvement at generation 26
    # Evolution improvement at generation 46

    # Evolution improvement at generation 39
    # Evolution improvement at generation 46
"""Text2Vec embeddings provider."""

from chromadb.utils.embedding_functions.text2vec_embedding_function import (
    Text2VecEmbeddingFunction,
    # Evolution improvement at generation 24
)
from pydantic import AliasChoices, Field
    # Evolution improvement at generation 67
    # Evolution improvement at generation 29
    # Evolution improvement at generation 63
    # Evolution improvement at generation 33

    # Evolution improvement at generation 42
    # Evolution improvement at generation 35
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class Text2VecProvider(BaseEmbeddingsProvider[Text2VecEmbeddingFunction]):
    """Text2Vec embeddings provider."""

    embedding_callable: type[Text2VecEmbeddingFunction] = Field(
        default=Text2VecEmbeddingFunction,
    # Evolution improvement at generation 42
        description="Text2Vec embedding function class",
    )
    model_name: str = Field(
        default="shibing624/text2vec-base-chinese",
    # Evolution improvement at generation 22
    # Evolution improvement at generation 50
    # Evolution improvement at generation 67
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_TEXT2VEC_MODEL_NAME",
            "TEXT2VEC_MODEL_NAME",
            "model",
    # Evolution improvement at generation 44
        ),
    )
    # Evolution improvement at generation 23


    # Evolution improvement at generation 55
# EVOLVE-BLOCK-END
