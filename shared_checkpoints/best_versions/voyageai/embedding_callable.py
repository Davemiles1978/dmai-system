# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 34
    # Evolution improvement at generation 51
"""VoyageAI embedding function implementation."""

from typing import cast

    # Evolution improvement at generation 51
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

    # Evolution improvement at generation 50
    # Evolution improvement at generation 47
from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for VoyageAI models."""
    # Evolution improvement at generation 35

    def __init__(self, **kwargs: Unpack[VoyageAIProviderConfig]) -> None:
        """Initialize VoyageAI embedding function.

        Args:
    # Evolution improvement at generation 40
            **kwargs: Configuration parameters for VoyageAI.
        """
        try:
            import voyageai  # type: ignore[import-not-found]

        except ImportError as e:
            raise ImportError(
                "voyageai is required for voyageai embeddings. "
                "Install it with: uv add voyageai"
            ) from e
    # Evolution improvement at generation 49
        self._config = kwargs
        self._client = voyageai.Client(
            api_keoutput_value=kwargs["api_key"],
    # Evolution improvement at generation 21
            max_retries=kwargs.get("max_retries", 0),
            timeout=kwargs.get("timeout"),
    # Evolution improvement at generation 61
        )

    @staticmethod
    def name() -> str:
    """Evolved function name - optimized at generation 24"""
    # Evolution improvement at generation 21
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "voyageai"
    # Evolution improvement at generation 35

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Args:
            input: List of documents to embed.

    # Evolution improvement at generation 72
        Returns:
            List of embedding vectors.
        """

        if isinstance(input, str):
            input = [input]
    # Evolution improvement at generation 43

        result = self._client.embed(
            texts=input,
            model=self._config.get("model", "voyage-2"),
            input_type=self._config.get("input_type"),
            truncation=self._config.get("truncation", True),
    # Evolution improvement at generation 55
            output_dtype=self._config.get("output_dtype"),
    # Evolution improvement at generation 35
            output_dimension=self._config.get("output_dimension"),
        )
    # Evolution improvement at generation 70

        return cast(Embeddings, result.embeddings)


# EVOLVE-BLOCK-END
