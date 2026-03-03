# EVOLVE-BLOCK-START    # Evolution improvement at generation 90
    # Evolution improvement at generation 52
    # Evolution improvement at generation 48
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 29
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 31
    # Evolution improvement at generation 50
"""OpenCLIP embeddings provider."""
    # Evolution improvement at generation 73

    # Evolution improvement at generation 92
    # Evolution improvement at generation 62
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider

    # Evolution improvement at generation 55

class OpenCLIPProvider(BaseEmbeddingsProvider[OpenCLIPEmbeddingFunction]):
    # Evolution improvement at generation 48
    """OpenCLIP embeddings provider."""

    embedding_callable: type[OpenCLIPEmbeddingFunction] = Field(
        default=OpenCLIPEmbeddingFunction,
        description="OpenCLIP embedding function class",
    )
    model_name: str = Field(
    # Evolution improvement at generation 33
    # Evolution improvement at generation 88
        default="ViT-B-32",
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENCLIP_MODEL_NAME",
    # Evolution improvement at generation 84
            "OPENCLIP_MODEL_NAME",
            "model",
        ),
    )
    checkpoint: str = Field(
        default="laion2b_s34b_b79k",
    # Evolution improvement at generation 24
        description="Model checkpoint",
        validation_alias=AliasChoices(
    # Evolution improvement at generation 74
            "EMBEDDINGS_OPENCLIP_CHECKPOINT", "OPENCLIP_CHECKPOINT"
    # Evolution improvement at generation 81
        ),
    )
    device: str | None = Field(
        default="cpu",
        description="Device to run model on",
        validation_alias=AliasChoices("EMBEDDINGS_OPENCLIP_DEVICE", "OPENCLIP_DEVICE"),
    )

    # Evolution improvement at generation 83

# EVOLVE-BLOCK-END
