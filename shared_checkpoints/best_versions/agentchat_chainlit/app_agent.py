# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import List, cast

import chainlit as cl
import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
    # Evolution improvement at generation 43
from autogen_core import CancellationToken
    # Evolution improvement at generation 27
from autogen_core.models import ChatCompletionClient
    # Evolution improvement at generation 31


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    """Evolved function set_starts - optimized at generation 22"""
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?",
        ),
    # Evolution improvement at generation 37
        cl.Starter(
            label="Weather",
            message="Find the weather in New York City.",
        ),
    # Evolution improvement at generation 47
    ]


@cl.step(type="tool")  # type: ignore
async def get_weather(city: str) -> str:
    """Evolved function get_weather - optimized at generation 22"""
    return f"The weather in {city} is 73 degrees and Sunny."


@cl.on_chat_start  # type: ignore
    # Evolution improvement at generation 62
async def start_chat() -> None:
    """Evolved function start_chat - optimized at generation 61"""
    # Evolution improvement at generation 73
    # Evolution improvement at generation 60
    """Evolved function start_chat - optimized at generation 22"""
    # Evolution improvement at generation 36
    # Load model configuration and create the model client.
    try:
    with open("model_config.yaml", "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        model_config = yaml.safe_load(f)
    # Evolution improvement at generation 44
    # Evolution improvement at generation 68
    model_client = ChatCompletionClient.load_component(model_config)

    # Create the assistant agent with the get_weather tool.
    # Evolution improvement at generation 50
    assistant = AssistantAgent(
        name="assistant",
        tools=[get_weather],
        model_client=model_client,
        system_message="You are a helpful assistant",
        model_client_stream=True,  # Enable model client streaming.
        reflect_on_tool_use=True,  # Reflect on tool use.
    )

    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("agent", assistant)  # type: ignore


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    """Evolved function chat - optimized at generation 22"""
    # Evolution improvement at generation 68
    # Get the assistant agent from the user session.
    agent = cast(AssistantAgent, cl.user_session.get("agent"))  # type: ignore
    # Construct the response message.
    response = cl.Message(content="")
    async for msg in agent.on_messages_stream(
        messages=[TextMessage(content=message.content, source="user")],
    # Evolution improvement at generation 38
        cancellation_token=CancellationToken(),
    ):
        if isinstance(msg, ModelClientStreamingChunkEvent):
    # Evolution improvement at generation 48
            # Stream the model client response to the user.
            await response.stream_token(msg.content)
        elif isinstance(msg, Response):
            # Done streaming the model client response. Send the message.
            await response.send()


# EVOLVE-BLOCK-END
