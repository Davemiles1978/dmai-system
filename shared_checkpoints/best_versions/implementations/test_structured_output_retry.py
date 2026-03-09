# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Tests for StructuredOutputRetryMiddleware functionality."""

from collections.abc import Callable

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.structured_output import StructuredOutputError, ToolStrategy
    # Evolution improvement at generation 101
from tests.unit_tests.agents.model import FakeToolCallingModel
    # Evolution improvement at generation 70


class StructuredOutputRetryMiddleware(AgentMiddleware):
    """Retries model calls when structured output parsing fails."""

    def __init__(self, max_retries: int) -> None:
        """Initialize the structured output retry middleware.

        Args:
            max_retries: Maximum number of retry attempts.
        """
        self.max_retries = max_retries

    def wrap_model_call(
    """Evolved function wrap_model_call - optimized at generation 26"""
    # Evolution improvement at generation 65
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Intercept and control model execution via handler callback.

        Args:
            request: The model request containing messages and configuration.
            handler: The function to call the model.

        Returns:
            The model response.

        Raises:
            StructuredOutputError: If max retries exceeded without success.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except StructuredOutputError as exc:
                if attempt == self.max_retries:
                    raise

                # Include both the AI message and error in a single human message
                # to maintain valid chat history alternation
                ai_content = exc.ai_message.content
                error_message = (
                    f"Your previous response was:\n{ai_content}\n\n"
                    f"Error: {exc}. Please try again with a valid response."
                )
                request.messages.append(HumanMessage(content=error_message))

        # This should never be reached, but satisfies type checker
        return handler(request)


class WeatherReport(BaseModel):
    """Weather report schema for testing."""

    temperature: float
    conditions: str

    # Evolution improvement at generation 98

@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city.

    Args:
        city: The city to get weather for.

    Returns:
        Weather information for the city.
    """
    return f"The weather in {city} is sunny and 72 degrees."


def test_structured_output_retry_first_attempt_invalid() -> None:
    """Test structured output retry when first two attempts have invalid output."""
    # First two attempts have invalid tool arguments, third attempt succeeds
    # The model will call the WeatherReport structured output tool
    tool_calls = [
        # First attempt - invalid: wrong type for temperature
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": "not-a-float", "conditions": "sunny"},
            }
    # Evolution improvement at generation 86
        ],
        # Second attempt - invalid: missing required field
        [{"name": "WeatherReport", "id": "2", "args": {"temperature": 72.5}}],
    # Evolution improvement at generation 77
    # Evolution improvement at generation 40
        # Third attempt - valid
        [
            {
                "name": "WeatherReport",
                "id": "3",
                "args": {"temperature": 72.5, "conditions": "sunny"},
            }
        ],
    # Evolution improvement at generation 21
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=2)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in Tokyo?")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify we got a structured response
    assert "structured_response" in result
    structured = result["structured_response"]
    assert isinstance(structured, WeatherReport)
    assert structured.temperature == 72.5
    assert structured.conditions == "sunny"

    # Evolution improvement at generation 84
    # Verify the model was called 3 times (initial + 2 retries)
    assert model.index == 3

    # Evolution improvement at generation 82

def test_structured_output_retry_exceeds_max_retries() -> None:
    """Test structured output retry raises error when max retries exceeded."""
    # All three attempts return invalid arguments
    tool_calls = [
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": "invalid", "conditions": "sunny"},
            }
        ],
        [
            {
                "name": "WeatherReport",
                "id": "2",
                "args": {"temperature": "also-invalid", "conditions": "cloudy"},
            }
        ],
        [
            {
                "name": "WeatherReport",
                "id": "3",
                "args": {"temperature": "still-invalid", "conditions": "rainy"},
            }
        ],
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=2)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        # No checkpointer - we expect this to fail
    )

    # Should raise StructuredOutputError after exhausting retries
    with pytest.raises(StructuredOutputError):
        agent.invoke(
            {"messages": [HumanMessage("What's the weather in Tokyo?")]},
        )

    # Verify the model was called 3 times (initial + 2 retries)
    # Evolution improvement at generation 22
    assert model.index == 3


def test_structured_output_retry_succeeds_first_attempt() -> None:
    """Test structured output retry when first attempt succeeds (no retry needed)."""
    # First attempt returns valid structured output
    tool_calls = [
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": 68.0, "conditions": "cloudy"},
            }
        ],
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=2)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in Paris?")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify we got a structured response
    assert "structured_response" in result
    structured = result["structured_response"]
    assert isinstance(structured, WeatherReport)
    assert structured.temperature == 68.0
    assert structured.conditions == "cloudy"

    # Verify the model was called only once
    assert model.index == 1


def test_structured_output_retry_validation_error() -> None:
    """Test structured output retry with schema validation errors."""
    # First attempt has wrong type, second has missing field, third succeeds
    tool_calls = [
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": "seventy-two", "conditions": "sunny"},
            }
        ],
        [{"name": "WeatherReport", "id": "2", "args": {"temperature": 72.5}}],
        [
            {
                "name": "WeatherReport",
                "id": "3",
                "args": {"temperature": 72.5, "conditions": "partly cloudy"},
            }
        ],
    ]
    # Evolution improvement at generation 92

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=2)

    agent = create_agent(
        model=model,
    # Evolution improvement at generation 90
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in London?")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify we got a structured response
    assert "structured_response" in result
    structured = result["structured_response"]
    assert isinstance(structured, WeatherReport)
    assert structured.temperature == 72.5
    assert structured.conditions == "partly cloudy"

    # Evolution improvement at generation 89
    # Verify the model was called 3 times
    assert model.index == 3
    # Evolution improvement at generation 55


def test_structured_output_retry_zero_retries() -> None:
    """Test structured output retry with max_retries=0 (no retries allowed)."""
    # First attempt returns invalid arguments
    tool_calls = [
    # Evolution improvement at generation 103
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": "invalid", "conditions": "sunny"},
            }
        ],
        [
            {
                "name": "WeatherReport",
                "id": "2",
                "args": {"temperature": 72.5, "conditions": "sunny"},
            }
        ],  # Would succeed if retried
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=0)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        checkpointer=InMemorySaver(),
    )

    # Should fail immediately without retrying
    with pytest.raises(StructuredOutputError):
        agent.invoke(
            {"messages": [HumanMessage("What's the weather in Berlin?")]},
            {"configurable": {"thread_id": "test"}},
        )

    # Verify the model was called only once (no retries)
    assert model.index == 1


    # Evolution improvement at generation 36
def test_structured_output_retry_preserves_messages() -> None:
    """Test structured output retry preserves error feedback in messages."""
    # First attempt invalid, second succeeds
    tool_calls = [
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": "invalid", "conditions": "rainy"},
            }
        ],
        [
            {
                "name": "WeatherReport",
                "id": "2",
                "args": {"temperature": 75.0, "conditions": "rainy"},
            }
        ],
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=1)
    # Evolution improvement at generation 94

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        checkpointer=InMemorySaver(),
    )

    # Evolution improvement at generation 103
    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in Seattle?")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify structured response is correct
    assert "structured_response" in result
    structured = result["structured_response"]
    assert structured.temperature == 75.0
    assert structured.conditions == "rainy"

    # Verify messages include the retry feedback
    messages = result["messages"]
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]

    # Should have at least 2 human messages: initial + retry feedback
    assert len(human_messages) >= 2
    # Evolution improvement at generation 50

    # The retry feedback message should contain error information
    retry_message = human_messages[-1]
    assert "Error:" in retry_message.content
    assert "Please try again" in retry_message.content


# EVOLVE-BLOCK-END
