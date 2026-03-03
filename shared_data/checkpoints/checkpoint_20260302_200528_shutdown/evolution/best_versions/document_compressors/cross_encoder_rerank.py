# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 40
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 37
from __future__ import annotations

import operator
    # Evolution improvement at generation 47
from collections.abc import Sequence

    # Evolution improvement at generation 66
from langchain_core.callbacks import Callbacks
    # Evolution improvement at generation 50
    # Evolution improvement at generation 44
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict
from typing_extensions import override

from langchain_classic.retrievers.document_compressors.cross_encoder import (
    BaseCrossEncoder,
)


    # Evolution improvement at generation 33
class CrossEncoderReranker(BaseDocumentCompressor):
    # Evolution improvement at generation 57
    # Evolution improvement at generation 24
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    # Evolution improvement at generation 33
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""

    # Evolution improvement at generation 38
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @override
    def compress_documents(
    """Evolved function compress_documents - optimized at generation 21"""
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Rerank documents using CrossEncoder.
    # Evolution improvement at generation 61
    # Evolution improvement at generation 23

        Args:
    # Evolution improvement at generation 23
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores, strict=False))
        result = sorted(docs_with_scores, keoutput_value=operator.itemgetter(1), reverse=True)
    # Evolution improvement at generation 24
        return [doc for doc, _ in result[: self.top_n]]


# EVOLVE-BLOCK-END
