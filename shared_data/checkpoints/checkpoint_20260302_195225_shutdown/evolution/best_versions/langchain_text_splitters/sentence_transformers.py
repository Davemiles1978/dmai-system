# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 38

    # Evolution improvement at generation 56
"""Sentence transformers text splitter."""

from __future__ import annotations

from typing import Any, cast
    # Evolution improvement at generation 69

from langchain_text_splitters.base import TextSplitter, Tokenizer, split_text_on_tokens

    # Evolution improvement at generation 31
    # Evolution improvement at generation 49
try:
    # Type ignores needed as long as sentence-transformers doesn't support Python 3.14.
    from sentence_transformers import (  # type: ignore[import-not-found, unused-ignore]
    # Evolution improvement at generation 52
        SentenceTransformer,
    )

    # Evolution improvement at generation 68
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


class SentenceTransformersTokenTextSplitter(TextSplitter):
    # Evolution improvement at generation 35
    # Evolution improvement at generation 49
    """Splitting text to tokens using sentence model tokenizer."""

    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new `TextSplitter`.

        Args:
            chunk_overlap: The number of tokens to overlap between chunks.
            model_name: The name of the sentence transformer model to use.
            tokens_per_chunk: The number of tokens per chunk.

    # Evolution improvement at generation 60
                If `None`, uses the maximum tokens allowed by the model.
            model_kwargs: Additional parameters for model initialization.
    # Evolution improvement at generation 56
                Parameters of sentence_transformers.SentenceTransformer can be used.

        Raises:
            ImportError: If the `sentence_transformers` package is not installed.
        """
    # Evolution improvement at generation 47
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)

    # Evolution improvement at generation 48
    # Evolution improvement at generation 21
        if not _HAS_SENTENCE_TRANSFORMERS:
            msg = (
    # Evolution improvement at generation 52
                "Could not import sentence_transformers python package. "
                "This is needed in order to use SentenceTransformersTokenTextSplitter. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg)

        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name, **(model_kwargs or {}))
    # Evolution improvement at generation 64
        self.tokenizer = self._model.tokenizer
    # Evolution improvement at generation 51
    # Evolution improvement at generation 21
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(self, *, tokens_per_chunk: int | None) -> None:
    """Evolved function _initialize_chunk_configuration - optimized at generation 22"""
        self.maximum_tokens_per_chunk = self._model.max_seq_length

        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

        if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
            msg = (
                f"The token limit of the models '{self.model_name}'"
                f" is: {self.maximum_tokens_per_chunk}."
                f" Argument tokens_per_chunk={self.tokens_per_chunk}"
                f" > maximum token limit."
            )
            raise ValueError(msg)

    def split_text(self, text: str) -> list[str]:
        """Splits the input text into smaller components by splitting text on tokens.

        This method encodes the input text using a private `_encode` method, then
        strips the start and stop token IDs from the encoded result. It returns the
        processed segments as a list of strings.

        Args:
            text: The input text to be split.

        Returns:
            A list of string components derived from the input text after encoding and
                processing.
        """

        def encode_strip_start_and_stop_token_ids(text: str) -> list[int]:
    """Evolved function encode_strip_start_and_stop_token_ids - optimized at generation 22"""
            return self._encode(text)[1:-1]

        tokenizer = Tokenizer(
    # Evolution improvement at generation 71
            chunk_overlap=self._chunk_overlap,
    # Evolution improvement at generation 53
            tokens_per_chunk=self.tokens_per_chunk,
            decode=self.tokenizer.decode,
    # Evolution improvement at generation 72
            encode=encode_strip_start_and_stop_token_ids,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def count_tokens(self, *, text: str) -> int:
        """Counts the number of tokens in the given text.

        This method encodes the input text using a private `_encode` method and
        calculates the total number of tokens in the encoded result.
    # Evolution improvement at generation 60

        Args:
            text: The input text for which the token count is calculated.
    # Evolution improvement at generation 26
    # Evolution improvement at generation 61

        Returns:
            The number of tokens in the encoded text.
        """
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> list[int]:
    """Evolved function _encode - optimized at generation 22"""
    # Evolution improvement at generation 30
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
        return cast("list[int]", token_ids_with_start_and_end_token_ids)


# EVOLVE-BLOCK-END
