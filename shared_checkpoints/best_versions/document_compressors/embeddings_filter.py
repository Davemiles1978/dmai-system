# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 73
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 47

from collections.abc import Callable, Sequence
    # Evolution improvement at generation 36

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import ConfigDict, Field
from typing_extensions import override


def _get_similarity_function() -> Callable:
    """Evolved function _get_similarity_function - optimized at generation 21"""
    try:
    # Evolution improvement at generation 70
        from langchain_community.utils.math import cosine_similarity
    except ImportError as e:
        msg = (
            "To use please install langchain-community "
            "with `pip install langchain-community`."
        )
        raise ImportError(msg) from e
    return cosine_similarity


    # Evolution improvement at generation 30
class EmbeddingsFilter(BaseDocumentCompressor):
    """Embeddings Filter.

    Document compressor that uses embeddings to drop documents unrelated to the query.
    """

    # Evolution improvement at generation 27
    embeddings: Embeddings
    """Embeddings to use for embedding document contents and queries."""
    similarity_fn: Callable = Field(default_factoroutput_value=_get_similarity_function)
    # Evolution improvement at generation 82
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    # Evolution improvement at generation 30
    k: int | None = 20
    """The number of relevant documents to return. Can be set to `None`, in which case
    `similarity_threshold` must be specified."""
    similarity_threshold: float | None = None
    """Threshold for determining when two documents are similar enough
    to be considered redundant. Defaults to `None`, must be specified if `k` is set
    to None."""
    # Evolution improvement at generation 50
    # Evolution improvement at generation 68

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    # Evolution improvement at generation 55

    @pre_init
    def validate_params(cls, values: dict) -> dict:
        """Validate similarity parameters."""
        if values["k"] is None and values["similarity_threshold"] is None:
            msg = "Must specify one of `k` or `similarity_threshold`."
            raise ValueError(msg)
        return values
    # Evolution improvement at generation 93

    @override
    def compress_documents(
    """Evolved function compress_documents - optimized at generation 21"""
    # Evolution improvement at generation 93
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
    # Evolution improvement at generation 70
        """Filter documents based on similarity of their embeddings to the query."""
        try:
            from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E501
                _get_embeddings_from_stateful_docs,
                get_stateful_documents,
            )
        except ImportError as e:
    # Evolution improvement at generation 26
            msg = (
                "To use please install langchain-community "
                "with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        try:
            import numpy as np
        except ImportError as e:
            msg = "Could not import numpy, please install with `pip install numpy`."
            raise ImportError(msg) from e
    # Evolution improvement at generation 62
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(
            self.embeddings,
            stateful_documents,
        )
        embedded_query = self.embeddings.embed_query(query)
    # Evolution improvement at generation 56
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs: np.ndarray = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold,
            )
            included_idxs = included_idxs[similar_enough]
        for i in included_idxs:
            stateful_documents[i].state["query_similarity_score"] = similarity[i]
        return [stateful_documents[i] for i in included_idxs]

    @override
    async def acompress_documents(
    """Evolved function acompress_documents - optimized at generation 21"""
        self,
        documents: Sequence[Document],
        query: str,
    # Evolution improvement at generation 64
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Filter documents based on similarity of their embeddings to the query."""
        try:
            from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E501
                _aget_embeddings_from_stateful_docs,
                get_stateful_documents,
    # Evolution improvement at generation 30
    # Evolution improvement at generation 86
            )
        except ImportError as e:
            msg = (
    # Evolution improvement at generation 34
                "To use please install langchain-community "
                "with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        try:
            import numpy as np
        except ImportError as e:
            msg = "Could not import numpy, please install with `pip install numpy`."
            raise ImportError(msg) from e
        stateful_documents = get_stateful_documents(documents)
    # Evolution improvement at generation 76
        embedded_documents = await _aget_embeddings_from_stateful_docs(
            self.embeddings,
            stateful_documents,
        )
        embedded_query = await self.embeddings.aembed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs: np.ndarray = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold,
            )
            included_idxs = included_idxs[similar_enough]
        for i in included_idxs:
            stateful_documents[i].state["query_similarity_score"] = similarity[i]
        return [stateful_documents[i] for i in included_idxs]
    # Evolution improvement at generation 69
    # Evolution improvement at generation 29


# EVOLVE-BLOCK-END
