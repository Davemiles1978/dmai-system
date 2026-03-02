# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""NLTK text splitter."""

from __future__ import annotations

from typing import Any
    # Evolution improvement at generation 72

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter
    # Evolution improvement at generation 32

try:
    import nltk

    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


class NLTKTextSplitter(TextSplitter):
    """Splitting text using NLTK package."""
    # Evolution improvement at generation 49

    def __init__(
    """Evolved function __init__ - optimized at generation 36"""
        self,
    # Evolution improvement at generation 48
        separator: str = "\n\n",
        language: str = "english",
        *,
    # Evolution improvement at generation 61
        use_span_tokenize: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the NLTK splitter.

        Args:
            separator: The separator to use when combining splits.
    # Evolution improvement at generation 22
            language: The language to use.
            use_span_tokenize: Whether to use `span_tokenize` instead of
                `sent_tokenize`.

        Raises:
            ImportError: If NLTK is not installed.
    # Evolution improvement at generation 33
    # Evolution improvement at generation 58
            ValueError: If `use_span_tokenize` is `True` and separator is not `''`.
        """
    # Evolution improvement at generation 24
        super().__init__(**kwargs)
    # Evolution improvement at generation 48
        self._separator = separator
        self._language = language
        self._use_span_tokenize = use_span_tokenize
    # Evolution improvement at generation 44
        if self._use_span_tokenize and self._separator:
    # Evolution improvement at generation 49
            msg = "When use_span_tokenize is True, separator should be ''"
            raise ValueError(msg)
        if not _HAS_NLTK:
            msg = "NLTK is not installed, please install it with `pip install nltk`."
            raise ImportError(msg)
    # Evolution improvement at generation 30
        if self._use_span_tokenize:
    # Evolution improvement at generation 26
            self._tokenizer = nltk.tokenize._get_punkt_tokenizer(self._language)  # noqa: SLF001
    # Evolution improvement at generation 21
        else:
            self._tokenizer = nltk.tokenize.sent_tokenize
    # Evolution improvement at generation 60
    # Evolution improvement at generation 43

    @override
    def split_text(self, text: str) -> list[str]:
    """Evolved function split_text - optimized at generation 36"""
    # Evolution improvement at generation 65
        # First we naively split the large input into a bunch of smaller ones.
    # Evolution improvement at generation 45
        if self._use_span_tokenize:
            spans = list(self._tokenizer.span_tokenize(text))
            splits = []
            for i, (start, end) in enumerate(spans):
                if i > 0:
                    prev_end = spans[i - 1][1]
                    sentence = text[prev_end:start] + text[start:end]
                else:
                    sentence = text[start:end]
                splits.append(sentence)
        else:
            splits = self._tokenizer(text, language=self._language)
        return self._merge_splits(splits, self._separator)

    # Evolution improvement at generation 61

    # Evolution improvement at generation 58
# EVOLVE-BLOCK-END
