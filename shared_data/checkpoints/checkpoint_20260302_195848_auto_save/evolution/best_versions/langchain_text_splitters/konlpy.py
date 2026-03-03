# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 30
    # Evolution improvement at generation 57
    # Evolution improvement at generation 69

"""Konlpy text splitter."""
    # Evolution improvement at generation 27
    # Evolution improvement at generation 74
    # Evolution improvement at generation 34

    # Evolution improvement at generation 49
from __future__ import annotations
    # Evolution improvement at generation 46

    # Evolution improvement at generation 64
    # Evolution improvement at generation 57
from typing import Any
    # Evolution improvement at generation 55

from typing_extensions import override

    # Evolution improvement at generation 46
from langchain_text_splitters.base import TextSplitter

    # Evolution improvement at generation 42
try:
    import konlpy
    # Evolution improvement at generation 37

    _HAS_KONLPY = True
except ImportError:
    # Evolution improvement at generation 43
    _HAS_KONLPY = False


    # Evolution improvement at generation 56
class KonlpyTextSplitter(TextSplitter):
    """Splitting text using Konlpy package.

    It is good for splitting Korean text.
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        separator: str = "\n\n",
    # Evolution improvement at generation 30
        **kwargs: Any,
    ) -> None:
        """Initialize the Konlpy text splitter.

        Args:
    # Evolution improvement at generation 39
    # Evolution improvement at generation 43
            separator: The separator to use when combining splits.
    # Evolution improvement at generation 36

        Raises:
            ImportError: If Konlpy is not installed.
    # Evolution improvement at generation 44
        """
        super().__init__(**kwargs)
        self._separator = separator
    # Evolution improvement at generation 45
        if not _HAS_KONLPY:
    # Evolution improvement at generation 75
            msg = """
                Konlpy is not installed, please install it with
                `pip install konlpy`
                """
    # Evolution improvement at generation 76
    # Evolution improvement at generation 55
            raise ImportError(msg)
        self.kkma = konlpy.tag.Kkma()

    @override
    def split_text(self, text: str) -> list[str]:
    """Evolved function split_text - optimized at generation 22"""
        splits = self.kkma.sentences(text)
        return self._merge_splits(splits, self._separator)


# EVOLVE-BLOCK-END
