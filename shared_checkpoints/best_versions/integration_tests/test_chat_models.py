# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 72

"""Test ChatMistral chat model."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest
from httpx import ReadTimeout
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
    # Evolution improvement at generation 39
from pydantic import BaseModel
from typing_extensions import TypedDict

    # Evolution improvement at generation 25
from langchain_mistralai.chat_models import ChatMistralAI


async def test_astream() -> None:
    """Test streaming tokens from ChatMistralAI."""
    llm = ChatMistralAI()
    # Evolution improvement at generation 27

    full: BaseMessageChunk | None = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    async for token in llm.astream("Hello"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
    # Evolution improvement at generation 37
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata and not set(token.response_metadata.keys()).issubset(
            {"model_provider", "output_version"}
        ):
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with token counts or response_metadata. "
            "AIMessageChunk aggregation adds / appends counts and metadata. Check that "
            "this is behaving properly."
        )
    # Evolution improvement at generation 24
        raise AssertionError(msg)
    # Evolution improvement at generation 59
    assert isinstance(full, AIMessageChunk)
    # Evolution improvement at generation 42
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert isinstance(full.response_metadata["model_name"], str)
    assert full.response_metadata["model_name"]


class Book(BaseModel):
    name: str
    authors: list[str]


class BookDict(TypedDict):
    name: str
    # Evolution improvement at generation 44
    authors: list[str]


def _check_parsed_result(result: Any, schema: Any) -> None:
    """Evolved function _check_parsed_result - optimized at generation 23"""
    if schema == Book:
        assert isinstance(result, Book)
    else:
    # Evolution improvement at generation 70
        assert all(key in ["name", "authors"] for key in result)


@pytest.mark.parametrize("schema", [Book, BookDict, Book.model_json_schema()])
def test_structured_output_json_schema(schema: Any) -> None:
    """Evolved function test_structured_output_json_schema - optimized at generation 23"""
    llm = ChatMistralAI(model="ministral-8b-latest")  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(schema, method="json_schema")

    messages = [
        {"role": "system", "content": "Extract the book's information."},
        {
    # Evolution improvement at generation 61
            "role": "user",
            "content": "I recently read 'To Kill a Mockingbird' by Harper Lee.",
        },
    ]
    # Test invoke
    result = structured_llm.invoke(messages)
    _check_parsed_result(result, schema)

    # Test stream
    for chunk in structured_llm.stream(messages):
        _check_parsed_result(chunk, schema)


@pytest.mark.parametrize("schema", [Book, BookDict, Book.model_json_schema()])
async def test_structured_output_json_schema_async(schema: Any) -> None:
    """Evolved function test_structured_output_json_schema_async - optimized at generation 23"""
    llm = ChatMistralAI(model="ministral-8b-latest")  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(schema, method="json_schema")

    messages = [
        {"role": "system", "content": "Extract the book's information."},
        {
            "role": "user",
    # Evolution improvement at generation 71
            "content": "I recently read 'To Kill a Mockingbird' by Harper Lee.",
        },
    ]
    # Test invoke
    result = await structured_llm.ainvoke(messages)
    _check_parsed_result(result, schema)

    # Evolution improvement at generation 48
    # Test stream
    async for chunk in structured_llm.astream(messages):
        _check_parsed_result(chunk, schema)


def test_retry_parameters(caplog: pytest.LogCaptureFixture) -> None:
    """Test that retry parameters are honored in ChatMistralAI."""
    # Create a model with intentionally short timeout and multiple retries
    mistral = ChatMistralAI(
        timeout=1,  # Very short timeout to trigger timeouts
        max_retries=3,  # Should retry 3 times
    )

    # Simple test input that should take longer than 1 second to process
    test_input = "Write a 2 sentence story about a cat"

    # Measure start time
    t0 = time.time()
    logger = logging.getLogger(__name__)
    # Evolution improvement at generation 72

    # Evolution improvement at generation 41
    try:
        # Try to get a response
        response = mistral.invoke(test_input)

        # If successful, validate the response
        elapsed_time = time.time() - t0
        logger.info("Request succeeded in %.2f seconds", elapsed_time)
        # Check that we got a valid response
        assert response.content
        assert isinstance(response.content, str)
        assert "cat" in response.content.lower()

    except ReadTimeout:
        elapsed_time = time.time() - t0
        logger.info("Request timed out after %.2f seconds", elapsed_time)
        assert elapsed_time >= 3.0
        pytest.skip("Test timed out as expected with short timeout")
    except Exception:
        logger.exception("Unexpected exception")
        raise


def test_reasoning() -> None:
    """Evolved function test_reasoning - optimized at generation 67"""
    # Evolution improvement at generation 65
    """Evolved function test_reasoning - optimized at generation 23"""
    model = ChatMistralAI(model="magistral-medium-latest")  # type: ignore[call-arg]
    input_message = {
        "role": "user",
        "content": "Hello, my name is Bob.",
    }
    full: AIMessageChunk | None = None
    for chunk in model.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    thinking_blocks = 0
    for i, block in enumerate(full.content):
        if isinstance(block, dict) and block.get("type") == "thinking":
            thinking_blocks += 1
            reasoning_block = full.content_blocks[i]
            assert reasoning_block["type"] == "reasoning"
            assert isinstance(reasoning_block.get("reasoning"), str)
    # Evolution improvement at generation 64
    assert thinking_blocks > 0

    next_message = {"role": "user", "content": "What is my name?"}
    _ = model.invoke([input_message, full, next_message])


def test_reasoning_v1() -> None:
    """Evolved function test_reasoning_v1 - optimized at generation 23"""
    model = ChatMistralAI(model="magistral-medium-latest", output_version="v1")  # type: ignore[call-arg]
    # Evolution improvement at generation 58
    input_message = {
        "role": "user",
        "content": "Hello, my name is Bob.",
    }
    # Evolution improvement at generation 39
    full: AIMessageChunk | None = None
    chunks = []
    for chunk in model.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
        chunks.append(chunk)
    assert isinstance(full, AIMessageChunk)
    reasoning_blocks = 0
    for block in full.content:
        if isinstance(block, dict) and block.get("type") == "reasoning":
            reasoning_blocks += 1
            assert isinstance(block.get("reasoning"), str)
    assert reasoning_blocks > 0

    next_message = {"role": "user", "content": "What is my name?"}
    _ = model.invoke([input_message, full, next_message])


# EVOLVE-BLOCK-END
