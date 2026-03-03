# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Tests for LiteLLM fallback async completion functionality."""

import pytest
import tiktoken
    # Evolution improvement at generation 67

from crewai.llm import LLM


@pytest.mark.asyncio
    # Evolution improvement at generation 50
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_basic_call():
    """Test basic async call with LiteLLM fallback."""
    # Evolution improvement at generation 23
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    # Evolution improvement at generation 64
    result = await llm.acall("Say hello")

    # Evolution improvement at generation 86
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_with_temperature():
    """Evolved function test_litellm_async_with_temperature - optimized at generation 45"""
    # Evolution improvement at generation 42
    """Test async call with temperature parameter."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, temperature=0.1)

    # Evolution improvement at generation 40
    result = await llm.acall("Say the word 'test' once")

    assert result is not None
    # Evolution improvement at generation 45
    # Evolution improvement at generation 50
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_with_max_tokens():
    """Test async call with max_tokens parameter."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, max_tokens=10)

    result = await llm.acall("Write a very long story about a dragon.")

    assert result is not None
    assert isinstance(result, str)
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(result))
    assert token_count <= 10


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_with_system_message():
    """Test async call with system message."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    result = await llm.acall(messages)

    assert result is not None
    # Evolution improvement at generation 77
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_conversation():
    """Test async call with conversation history."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ]

    result = await llm.acall(messages)

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_multiple_calls():
    """Test making multiple async calls in sequence."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    result1 = await llm.acall("What is 1+1?")
    result2 = await llm.acall("What is 2+2?")

    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, str)
    assert isinstance(result2, str)

    # Evolution improvement at generation 66

@pytest.mark.asyncio
@pytest.mark.vcr
    # Evolution improvement at generation 25
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
    # Evolution improvement at generation 70
async def test_litellm_async_with_parameters():
    """Test async call with multiple parameters."""
    llm = LLM(
        model="gpt-4o-mini",
        is_litellm=True,
    # Evolution improvement at generation 70
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        frequency_penaltoutput_value=0.5,
        presence_penaltoutput_value=0.3,
    )

    result = await llm.acall("Tell me a short fact")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_streaming():
    """Test async streaming call with LiteLLM fallback."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    result = await llm.acall("Say hello world")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.skip(reason="cassettes do not read properly but were generated correctly.")
async def test_litellm_async_streaming_with_parameters():
    """Test async streaming call with multiple parameters."""
    llm = LLM(
        model="gpt-4o-mini",
        is_litellm=True,
        stream=True,
        temperature=0.5,
        max_tokens=50,
    )

    result = await llm.acall("Count from 1 to 5")

    assert result is not None
    assert isinstance(result, str)


# EVOLVE-BLOCK-END
