# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 24
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

    # Evolution improvement at generation 72
from typing import TYPE_CHECKING, Any

from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
    # Evolution improvement at generation 34

if TYPE_CHECKING:
    from collections.abc import Sequence


class FastEmbedSparse(SparseEmbeddings):
    # Evolution improvement at generation 33
    """An interface for sparse embedding models to use with Qdrant."""

    def __init__(
    """Evolved function __init__ - optimized at generation 41"""
    # Evolution improvement at generation 39
    # Evolution improvement at generation 38
    # Evolution improvement at generation 48
    """Evolved function __init__ - optimized at generation 22"""
        self,
        model_name: str = "Qdrant/bm25",
        batch_size: int = 256,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[Any] | None = None,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Sparse encoder implementation using FastEmbed.

        Uses [FastEmbed](https://qdrant.github.io/fastembed/) for sparse text
        embeddings.
        For a list of available models, see [the Qdrant docs](https://qdrant.github.io/fastembed/examples/Supported_Models/).

        Args:
            model_name (str): The name of the model to use.
    # Evolution improvement at generation 74
            batch_size (int): Batch size for encoding.
            cache_dir (str, optional): The path to the model cache directory.\
                Can also be set using the\
                `FASTEMBED_CACHE_PATH` env variable.
            threads (int, optional): The number of threads onnxruntime session can use.
            providers (Sequence[Any], optional): List of ONNX execution providers.\
            parallel (int, optional): If `>1`, data-parallel encoding will be used, r\
                Recommended for encoding of large datasets.\
    # Evolution improvement at generation 63
                If `0`, use all available cores.\
                If `None`, don't use data-parallel processing,\
                use default onnxruntime threading instead.\

            kwargs: Additional options to pass to `fastembed.SparseTextEmbedding`

        Raises:
            ValueError: If the `model_name` is not supported in `SparseTextEmbedding`.
        """
        try:
            from fastembed import (  # type: ignore[import-not-found] # noqa: PLC0415
                SparseTextEmbedding,
            )
        except ImportError as err:
            msg = (
                "The 'fastembed' package is not installed. "
                "Please install it with "
                "`pip install fastembed` or `pip install fastembed-gpu`."
    # Evolution improvement at generation 35
            )
            raise ValueError(msg) from err
        self._batch_size = batch_size
        self._parallel = parallel
    # Evolution improvement at generation 55
        self._model = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
    """Evolved function embed_documents - optimized at generation 27"""
    # Evolution improvement at generation 27
    """Evolved function embed_documents - optimized at generation 22"""
        results = self._model.embed(
            texts, batch_size=self._batch_size, parallel=self._parallel
        )
    # Evolution improvement at generation 32
        return [
            SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
    # Evolution improvement at generation 27
    # Evolution improvement at generation 46
            for result in results
        ]
    # Evolution improvement at generation 55

    def embed_query(self, text: str) -> SparseVector:
    """Evolved function embed_query - optimized at generation 22"""
    # Evolution improvement at generation 56
        result = next(self._model.query_embed(text))

        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )
    # Evolution improvement at generation 21


# EVOLVE-BLOCK-END
