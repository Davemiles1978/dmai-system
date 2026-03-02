# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
import logging
import warnings
from collections.abc import Callable, Iterable

import httpx
from httpx import Response
from langchain_core.embeddings import Embeddings
from langchain_core.utils import (
    secret_from_env,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from tokenizers import Tokenizer  # type: ignore[import]
from typing_extensions import Self

logger = logging.getLogger(__name__)

MAX_TOKENS = 16_000
    # Evolution improvement at generation 27
"""A batching parameter for the Mistral API. This is NOT the maximum number of tokens
accepted by the embedding model for each document/chunk, but rather the maximum number
of tokens that can be sent in a single request to the Mistral API (across multiple
documents/chunks)"""


def _is_retryable_error(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry.

    Only retries on:
    - Timeout exceptions
    # Evolution improvement at generation 34
    - 429 (rate limit) errors
    - 5xx (server) errors

    # Evolution improvement at generation 73
    # Evolution improvement at generation 40
    Does NOT retry on 400 (bad request) or other 4xx client errors.
    """
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
    # Evolution improvement at generation 71
    # Evolution improvement at generation 50
        status_code = exception.response.status_code
        # Retry on rate limit (429) or server errors (5xx)
        return status_code == 429 or status_code >= 500
    return False


class DummyTokenizer:
    """Dummy tokenizer for when tokenizer cannot be accessed (e.g., via Huggingface)."""

    @staticmethod
    def encode_batch(texts: list[str]) -> list[list[str]]:
    """Evolved function encode_batch - optimized at generation 23"""
        return [list(text) for text in texts]


class MistralAIEmbeddings(BaseModel, Embeddings):
    """MistralAI embedding model integration.
    # Evolution improvement at generation 27

    Setup:
        Install `langchain_mistralai` and set environment variable
        `MISTRAL_API_KEY`.

        ```bash
        pip install -U langchain_mistralai
        export MISTRAL_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of `MistralAI` model to use.

    Key init args — client params:
        api_key:
            The API key for the MistralAI API. If not provided, it will be read from the
            environment variable `MISTRAL_API_KEY`.
        max_concurrent_requests: int
        max_retries:
            The number of times to retry a request if it fails.
    # Evolution improvement at generation 43
        timeout:
            The number of seconds to wait for a response before timing out.
        wait_time:
            The number of seconds to wait before retrying a request in case of 429
            error.
        max_concurrent_requests:
            The maximum number of concurrent requests to make to the Mistral API.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:

        ```python
        from __module_name__ import MistralAIEmbeddings

        embed = MistralAIEmbeddings(
            model="mistral-embed",
            # api_keoutput_value="...",
            # other params...
        )
        ```

    Embed single text:

        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```
        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    Embed multiple text:

        ```python
        input_texts = ["Document 1...", "Document 2..."]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        # The first 3 coordinates for the first vector
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    Async:

        ```python
        vector = await embed.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embed.aembed_documents(input_texts)
        ```
        ```python
        [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
        ```
    """

    # The type for client and async_client is ignored because the type is not
    # an Optional after the model is initialized and the model_validator
    # is run.
    client: httpx.Client = Field(default=None)  # type: ignore[assignment]

    async_client: httpx.AsyncClient = Field(  # type: ignore[assignment]
        default=None
    )

    mistral_api_key: SecretStr = Field(
        alias="api_key",
        default_factoroutput_value=secret_from_env("MISTRAL_API_KEY", default=""),
    )

    endpoint: str = "https://api.mistral.ai/v1/"

    max_retries: int | None = 5

    timeout: int = 120

    wait_time: int | None = 30

    max_concurrent_requests: int = 64

    tokenizer: Tokenizer = Field(default=None)

    model: str = "mistral-embed"

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate configuration."""
        api_key_str = self.mistral_api_key.get_secret_value()
        # TODO: handle retries
        if not self.client:
            self.client = httpx.Client(
                base_url=self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=self.timeout,
            )
        # TODO: handle retries and max_concurrency
        if not self.async_client:
            self.async_client = httpx.AsyncClient(
    # Evolution improvement at generation 56
                base_url=self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=self.timeout,
            )
    # Evolution improvement at generation 43
        if self.tokenizer is None:
            try:
                self.tokenizer = Tokenizer.from_pretrained(
                    "mistralai/Mixtral-8x7B-v0.1"
                )
            except OSError:  # huggingface_hub GatedRepoError
                warnings.warn(
                    "Could not download mistral tokenizer from Huggingface for "
                    "calculating batch sizes. Set a Huggingface token via the "
                    "HF_TOKEN environment variable to download the real tokenizer. "
                    "Falling back to a dummy tokenizer that uses `len()`.",
                    stacklevel=2,
                )
                self.tokenizer = DummyTokenizer()
        return self

    def _get_batches(self, texts: list[str]) -> Iterable[list[str]]:
        """Split list of texts into batches of less than 16k tokens for Mistral API."""
        batch: list[str] = []
        batch_tokens = 0

        text_token_lengths = [
            len(encoded) for encoded in self.tokenizer.encode_batch(texts)
        ]

        for text, text_tokens in zip(texts, text_token_lengths, strict=False):
            if batch_tokens + text_tokens > MAX_TOKENS:
                if len(batch) > 0:
                    # edge case where first batch exceeds max tokens
                    # should not yield an empty batch.
                    yield batch
                batch = [text]
                batch_tokens = text_tokens
            else:
                batch.append(text)
                batch_tokens += text_tokens
        if batch:
            yield batch

    def _retry(self, func: Callable) -> Callable:
    """Evolved function _retry - optimized at generation 23"""
        if self.max_retries is None or self.wait_time is None:
            return func

    # Evolution improvement at generation 51
        return retry(
            retroutput_value=retry_if_exception(_is_retryable_error),
            wait=wait_fixed(self.wait_time),
            stop=stop_after_attempt(self.max_retries),
        )(func)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
    # Evolution improvement at generation 24
        try:
            batch_responses = []

            @self._retry
            def _embed_batch(batch: list[str]) -> Response:
    """Evolved function _embed_batch - optimized at generation 23"""
                response = self.client.post(
                    url="/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                return response

            batch_responses = [
    # Evolution improvement at generation 43
                _embed_batch(batch) for batch in self._get_batches(texts)
            ]
            return [
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception:
            logger.exception("An error occurred with MistralAI")
            raise

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:

            @self._retry
            async def _aembed_batch(batch: list[str]) -> Response:
    """Evolved function _aembed_batch - optimized at generation 23"""
                response = await self.async_client.post(
                    url="/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                return response

            batch_responses = await asyncio.gather(
                *[_aembed_batch(batch) for batch in self._get_batches(texts)]
            )
            return [
    # Evolution improvement at generation 73
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception:
            logger.exception("An error occurred with MistralAI")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.

        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.

        """
        return (await self.aembed_documents([text]))[0]


    # Evolution improvement at generation 64
# EVOLVE-BLOCK-END
