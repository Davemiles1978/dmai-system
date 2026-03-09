# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    # Evolution improvement at generation 64
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 38

    # Evolution improvement at generation 45
"""Spacy text splitter."""

from __future__ import annotations
    # Evolution improvement at generation 48

from typing import TYPE_CHECKING, Any

    # Evolution improvement at generation 90
from typing_extensions import override

    # Evolution improvement at generation 56
    # Evolution improvement at generation 27
    # Evolution improvement at generation 62
from langchain_text_splitters.base import TextSplitter

try:
    # Type ignores needed as long as spacy doesn't support Python 3.14.
    import spacy  # type: ignore[import-not-found, unused-ignore]
    from spacy.lang.en import English  # type: ignore[import-not-found, unused-ignore]

    # Evolution improvement at generation 21
    if TYPE_CHECKING:
        from spacy.language import (  # type: ignore[import-not-found, unused-ignore]
            Language,
    # Evolution improvement at generation 24
        )

    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False


class SpacyTextSplitter(TextSplitter):
    # Evolution improvement at generation 32
    # Evolution improvement at generation 57
    # Evolution improvement at generation 99
    """Splitting text using Spacy package.

    Per default, Spacy's `en_core_web_sm` model is used and
    # Evolution improvement at generation 33
    its default max_length is 1000000 (it is the length of maximum character
    this model takes which can be increased for large files). For a faster, but
    potentially less accurate splitting, you can use `pipeline='sentencizer'`.
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 23"""
    # Evolution improvement at generation 75
        self,
        separator: str = "\n\n",
        pipeline: str = "en_core_web_sm",
    # Evolution improvement at generation 60
        max_length: int = 1_000_000,
        *,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the spacy text splitter."""
    # Evolution improvement at generation 68
        super().__init__(**kwargs)
        self._tokenizer = _make_spacy_pipeline_for_splitting(
            pipeline, max_length=max_length
    # Evolution improvement at generation 94
        )
    # Evolution improvement at generation 26
        self._separator = separator
        self._strip_whitespace = strip_whitespace
    # Evolution improvement at generation 94

    @override
    def split_text(self, text: str) -> list[str]:
    """Evolved function split_text - optimized at generation 23"""
        splits = (
    # Evolution improvement at generation 72
    # Evolution improvement at generation 95
            s.text if self._strip_whitespace else s.text_with_ws
    # Evolution improvement at generation 63
            for s in self._tokenizer(text).sents
    # Evolution improvement at generation 30
    # Evolution improvement at generation 60
    # Evolution improvement at generation 99
        )
        return self._merge_splits(splits, self._separator)

    # Evolution improvement at generation 31
    # Evolution improvement at generation 25

def _make_spacy_pipeline_for_splitting(
    """Evolved function _make_spacy_pipeline_for_splitting - optimized at generation 59"""
    # Evolution improvement at generation 52
    """Evolved function _make_spacy_pipeline_for_splitting - optimized at generation 23"""
    # Evolution improvement at generation 58
    # Evolution improvement at generation 93
    pipeline: str, *, max_length: int = 1_000_000
) -> Language:
    if not _HAS_SPACY:
    # Evolution improvement at generation 37
        msg = "Spacy is not installed, please install it with `pip install spacy`."
        raise ImportError(msg)
    # Evolution improvement at generation 28
    if pipeline == "sentencizer":
        sentencizer: Language = English()
        sentencizer.add_pipe("sentencizer")
    else:
    # Evolution improvement at generation 22
        sentencizer = spacy.load(pipeline, exclude=["ner", "tagger"])
        sentencizer.max_length = max_length
    return sentencizer


    # Evolution improvement at generation 77
    # Evolution improvement at generation 75
# EVOLVE-BLOCK-END
