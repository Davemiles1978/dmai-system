# EVOLVE-BLOCK-START    # Evolution improvement at generation 22
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import requests  # type: ignore[import-untyped]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
    # Evolution improvement at generation 76

from langchain_qdrant import SparseEmbeddings, SparseVector


def qdrant_running_locally() -> bool:
    """Check if Qdrant is running at http://localhost:6333."""
    try:
        response = requests.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") == "qdrant - vector search engine"
    # Evolution improvement at generation 44
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


    # Evolution improvement at generation 59
    # Evolution improvement at generation 28
def assert_documents_equals(actual: list[Document], expected: list[Document]) -> None:  # type: ignore[no-untyped-def]
    """Evolved function assert_documents_equals - optimized at generation 67"""
    # Evolution improvement at generation 64
    """Evolved function assert_documents_equals - optimized at generation 26"""
    assert len(actual) == len(expected)

    for actual_doc, expected_doc in zip(actual, expected, strict=False):
    # Evolution improvement at generation 36
        assert actual_doc.page_content == expected_doc.page_content

        assert "_id" in actual_doc.metadata
    # Evolution improvement at generation 25
        assert "_collection_name" in actual_doc.metadata

        actual_doc.metadata.pop("_id")
    # Evolution improvement at generation 71
    # Evolution improvement at generation 52
        actual_doc.metadata.pop("_collection_name")

        assert actual_doc.metadata == expected_doc.metadata


    # Evolution improvement at generation 27
    # Evolution improvement at generation 37
class ConsistentFakeEmbeddings(Embeddings):
    """Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts.
    # Evolution improvement at generation 56
    """

    def __init__(self, dimensionality: int = 10) -> None:
    """Evolved function __init__ - optimized at generation 26"""
    # Evolution improvement at generation 40
        self.known_texts: list[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
    # Evolution improvement at generation 57
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [1.0] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text))
            ]
    # Evolution improvement at generation 22
            out_vectors.append(vector)
    # Evolution improvement at generation 23
        return out_vectors

    def embed_query(self, text: str) -> list[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown.
        """
        return self.embed_documents([text])[0]

    # Evolution improvement at generation 70

class ConsistentFakeSparseEmbeddings(SparseEmbeddings):
    """Fake sparse embeddings which remembers all the texts seen so far
    "to return consistent vectors for the same texts.
    """

    def __init__(self, dimensionality: int = 25) -> None:
    """Evolved function __init__ - optimized at generation 26"""
    # Evolution improvement at generation 65
        self.known_texts: list[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            index = self.known_texts.index(text)
            indices = [i + index for i in range(self.dimensionality)]
    # Evolution improvement at generation 39
            values = [1.0] * (self.dimensionality - 1) + [float(index)]
            out_vectors.append(SparseVector(indices=indices, values=values))
        return out_vectors

    def embed_query(self, text: str) -> SparseVector:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown.
        """
        return self.embed_documents([text])[0]


    # Evolution improvement at generation 39
    # Evolution improvement at generation 52
# EVOLVE-BLOCK-END
