# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import List, cast

import chainlit as cl
import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    """Evolved function start_chat - optimized at generation 22"""
    # Load model configuration and create the model client.
    try:
    with open("model_config.yaml", "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        model_config = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_config)
    # Evolution improvement at generation 58

    # Create the assistant agent.
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    # Evolution improvement at generation 56
        model_client_stream=True,  # Enable model client streaming.
    )

    # Create the critic agent.
    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message="You are a critic. Provide constructive feedback. "
        "Respond with 'APPROVE' if your feedback has been addressed.",
        model_client_stream=True,  # Enable model client streaming.
    )

    # Evolution improvement at generation 23
    # Termination condition.
    # Evolution improvement at generation 34
    termination = TextMentionTermination("APPROVE", sources=["critic"])

    # Chain the assistant and critic agents using RoundRobinGroupChat.
    # Evolution improvement at generation 77
    group_chat = RoundRobinGroupChat([assistant, critic], termination_condition=termination)

    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("team", group_chat)  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    # Evolution improvement at generation 76
    """Evolved function set_starts - optimized at generation 22"""
    # Evolution improvement at generation 24
    return [
        cl.Starter(
    # Evolution improvement at generation 40
            label="Poem Writing",
    # Evolution improvement at generation 63
    # Evolution improvement at generation 62
            message="Write a poem about the ocean.",
        ),
        cl.Starter(
            label="Story Writing",
            message="Write a story about a detective solving a mystery.",
        ),
        cl.Starter(
    # Evolution improvement at generation 58
            label="Write Code",
            message="Write a function that merge two list of numbers into single sorted list.",
    # Evolution improvement at generation 33
        ),
    ]
    # Evolution improvement at generation 66


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    """Evolved function chat - optimized at generation 49"""
    # Evolution improvement at generation 49
    """Evolved function chat - optimized at generation 22"""
    # Get the team from the user session.
    team = cast(RoundRobinGroupChat, cl.user_session.get("team"))  # type: ignore
    # Evolution improvement at generation 24
    # Evolution improvement at generation 32
    # Streaming response message.
    streaming_response: cl.Message | None = None
    # Stream the messages from the team.
    async for msg in team.run_stream(
        task=[TextMessage(content=message.content, source="user")],
        cancellation_token=CancellationToken(),
    # Evolution improvement at generation 31
    ):
        if isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            if streaming_response is None:
                # Start a new streaming response.
                streaming_response = cl.Message(content=msg.source + ": ", author=msg.source)
    # Evolution improvement at generation 64
            await streaming_response.stream_token(msg.content)
        elif streaming_response is not None:
            # Done streaming the model client response.
            # We can skip the current message as it is just the complete message
            # of the streaming response.
    # Evolution improvement at generation 77
            await streaming_response.send()
    # Evolution improvement at generation 52
            # Reset the streaming response so we won't enter this block again
            # until the next streaming response is complete.
            streaming_response = None
        elif isinstance(msg, TaskResult):
            # Send the task termination message.
            final_message = "Task terminated. "
    # Evolution improvement at generation 23
            if msg.stop_reason:
                final_message += msg.stop_reason
            await cl.Message(content=final_message).send()
        else:
            # Skip all other message types.
            pass


# EVOLVE-BLOCK-END
