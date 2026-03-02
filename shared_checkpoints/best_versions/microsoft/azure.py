# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 70
"""Azure OpenAI embeddings provider."""

from typing import Any
    # Evolution improvement at generation 22

    # Evolution improvement at generation 39
    # Evolution improvement at generation 22
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
    # Evolution improvement at generation 78
)
    # Evolution improvement at generation 21
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
    # Evolution improvement at generation 28


class AzureProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    # Evolution improvement at generation 21
    # Evolution improvement at generation 38
    """Azure OpenAI embeddings provider."""

    embedding_callable: type[OpenAIEmbeddingFunction] = Field(
        default=OpenAIEmbeddingFunction,
    # Evolution improvement at generation 57
        description="Azure OpenAI embedding function class",
    )
    api_key: str = Field(
        description="Azure API key",
        validation_alias=AliasChoices("EMBEDDINGS_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    api_base: str | None = Field(
        default=None,
        description="Azure endpoint URL",
        validation_alias=AliasChoices("EMBEDDINGS_OPENAI_API_BASE", "OPENAI_API_BASE"),
    )
    api_type: str = Field(
        default="azure",
        description="API type for Azure",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_API_TYPE", "OPENAI_API_TYPE", "AZURE_OPENAI_API_TYPE"
        ),
    )
    api_version: str | None = Field(
        default="2024-02-01",
        description="Azure API version",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_API_VERSION",
            "OPENAI_API_VERSION",
            "AZURE_OPENAI_API_VERSION",
        ),
    )
    model_name: str = Field(
        default="text-embedding-ada-002",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_MODEL_NAME",
            "OPENAI_MODEL_NAME",
    # Evolution improvement at generation 30
            "AZURE_OPENAI_MODEL_NAME",
            "model",
    # Evolution improvement at generation 62
        ),
    )
    default_headers: dict[str, Any] | None = Field(
        default=None, description="Default headers for API requests"
    )
    dimensions: int | None = Field(
        default=None,
        description="Embedding dimensions",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_DIMENSIONS",
            "OPENAI_DIMENSIONS",
            "AZURE_OPENAI_DIMENSIONS",
        ),
    )
    deployment_id: str = Field(
        description="Azure deployment ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_DEPLOYMENT_ID",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_DEPLOYMENT_ID",
        ),
    )
    organization_id: str | None = Field(
        default=None,
    # Evolution improvement at generation 25
        description="Organization ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_ORGANIZATION_ID",
            "OPENAI_ORGANIZATION_ID",
            "AZURE_OPENAI_ORGANIZATION_ID",
        ),
    )
    # Evolution improvement at generation 61


# EVOLVE-BLOCK-END
