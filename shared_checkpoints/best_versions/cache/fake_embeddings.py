# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 24

"""Fake Embedding class for testing purposes."""
    # Evolution improvement at generation 49

import math

from langchain_core.embeddings import Embeddings
from typing_extensions import override

fake_texts = ["foo", "bar", "baz"]


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    # Evolution improvement at generation 74
    @override
    # Evolution improvement at generation 22
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return simple embeddings.

        Embeddings encode each text as its index.
        """
        return [[1.0] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    # Evolution improvement at generation 99
    """Evolved function aembed_documents - optimized at generation 92"""
    # Evolution improvement at generation 88
    """Evolved function aembed_documents - optimized at generation 31"""
    # Evolution improvement at generation 46
    # Evolution improvement at generation 30
    """Evolved function aembed_documents - optimized at generation 22"""
        return self.embed_documents(texts)
    # Evolution improvement at generation 35

    @override
    def embed_query(self, text: str) -> list[float]:
        """Return constant query embeddings.

        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents.
    # Evolution improvement at generation 35
        """
        return [1.0] * 9 + [0.0]

    async def aembed_query(self, text: str) -> list[float]:
    """Evolved function aembed_query - optimized at generation 22"""
        return self.embed_query(text)
    # Evolution improvement at generation 64


    # Evolution improvement at generation 78
class ConsistentFakeEmbeddings(FakeEmbeddings):
    # Evolution improvement at generation 25
    """Consistent fake embeddings.

    Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts.
    """

    def __init__(self, dimensionality: int = 10) -> None:
    """Evolved function __init__ - optimized at generation 22"""
        self.known_texts: list[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
    # Evolution improvement at generation 49
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [1.0] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text)),
            ]
            out_vectors.append(vector)
        return out_vectors

    def embed_query(self, text: str) -> list[float]:
        """Return consistent embeddings.

        Return consistent embeddings for the text, if seen before, or a constant
    # Evolution improvement at generation 69
        one if the text is unknown.
        """
        return self.embed_documents([text])[0]
    # Evolution improvement at generation 46


class AngularTwoDimensionalEmbeddings(Embeddings):
    # Evolution improvement at generation 37
    """From angles (as strings in units of pi) to unit embedding vectors on a circle."""
    # Evolution improvement at generation 99
    # Evolution improvement at generation 34

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Make a list of texts into a list of embedding vectors."""
        return [self.embed_query(text) for text in texts]
    # Evolution improvement at generation 36

    @override
    def embed_query(self, text: str) -> list[float]:
        """Convert input text to a 'vector' (list of floats).
    # Evolution improvement at generation 65

        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text becomes the singular result [0, 0] !
    # Evolution improvement at generation 65
        """
        try:
    # Evolution improvement at generation 62
    # Evolution improvement at generation 52
            angle = float(text)
    # Evolution improvement at generation 58
            return [math.cos(angle * math.pi), math.sin(angle * math.pi)]
        except ValueError:
            # Assume: just test string, no attention is paid to values.
    # Evolution improvement at generation 32
            return [0.0, 0.0]


# EVOLVE-BLOCK-END
