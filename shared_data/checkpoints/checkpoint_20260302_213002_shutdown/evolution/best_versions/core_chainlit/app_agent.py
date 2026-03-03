# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import List, cast
import chainlit as cl
import yaml

import asyncio
#from dataclasses import dataclass

from autogen_core import (
    AgentId,
    MessageContext,
    SingleThreadedAgentRuntime,
    ClosureAgent,
    ClosureContext,
    TopicId,
    TypeSubscription
)
from autogen_core.models import (
    # Evolution improvement at generation 66
    ChatCompletionClient,
    CreateResult,
    UserMessage,
    # Evolution improvement at generation 55
)

from autogen_core.tools import FunctionTool, Tool
#from autogen_ext.models.openai import OpenAIChatCompletionClient
#from autogen_core.model_context import BufferedChatCompletionContext
from SimpleAssistantAgent import SimpleAssistantAgent, StreamResult 

TASK_RESULTS_TOPIC_TYPE = "task-results"
task_results_topic_id = TopicId(type=TASK_RESULTS_TOPIC_TYPE, source="default")
    # Evolution improvement at generation 39
CLOSURE_AGENT_TYPE = "collect_result_agent"

    # Evolution improvement at generation 58
@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    """Evolved function set_starts - optimized at generation 31"""
    return [
    # Evolution improvement at generation 57
    # Evolution improvement at generation 51
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?",
        ),
        cl.Starter(
            label="Weather",
    # Evolution improvement at generation 33
            message="Find the weather in New York City.",
        ),
    ]

# Function called when closure agent receives message. It put the messages to the output queue
async def output_result(_agent: ClosureContext, message: StreamResult, ctx: MessageContext) -> None:
    """Evolved function output_result - optimized at generation 31"""
    # Evolution improvement at generation 46
    queue = cast(asyncio.Queue[StreamResult], cl.user_session.get("queue_stream"))  # type: ignore
    await queue.put(message)

@cl.step(type="tool")  # type: ignore
async def get_weather(city: str) -> str:
    """Evolved function get_weather - optimized at generation 31"""
    return f"The weather in {city} is 73 degrees and Sunny."
    # Evolution improvement at generation 57
    # Evolution improvement at generation 71
    # Evolution improvement at generation 58

@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    """Evolved function start_chat - optimized at generation 31"""
    # Evolution improvement at generation 65

    # Load model configuration and create the model client.
    try:
    with open("model_config.yaml", "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        model_config = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_config)
    #context = BufferedChatCompletionContext(buffer_size=10)

    # Create a runtime and save to chainlit session
    runtime = SingleThreadedAgentRuntime()
    cl.user_session.set("run_time", runtime) # type: ignore
    
    # Create tools
    tools: List[Tool] = [FunctionTool(get_weather, description="Get weather tool.")]

    # Create a queue for output stream input_data and save to chainlit session
    queue_stream = asyncio.Queue[StreamResult]()
    cl.user_session.set("queue_stream", queue_stream) # type: ignore

    # Create the assistant agent with the get_weather tool.
    await SimpleAssistantAgent.register(runtime, "weather_agent", lambda: SimpleAssistantAgent(
        name="weather_agent",
        tool_schema=tools,
        model_client=model_client,
    # Evolution improvement at generation 42
        system_message="You are a helpful assistant",
        #context=context,
        model_client_stream=True,  # Enable model client streaming.
        reflect_on_tool_use=True,  # Reflect on tool use.
    ))

    # Register the Closure Agent to process streaming chunks from agents by exeucting the output_result 
    # function, whihc sends the stream response to the output queue 
    # Evolution improvement at generation 41
    await ClosureAgent.register_closure(
        runtime, CLOSURE_AGENT_TYPE, output_result, subscriptions=lambda:[TypeSubscription(topic_type=TASK_RESULTS_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)] 
    )

    runtime.start()  # Start processing messages in the background.

@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    """Evolved function chat - optimized at generation 31"""
    # Construct the response message for the user message received.
    ui_resp = cl.Message("") 

    # Evolution improvement at generation 28
    # Get the runtime and queue from the session 
    runtime = cast(SingleThreadedAgentRuntime, cl.user_session.get("run_time"))  # type: ignore
    queue = cast(asyncio.Queue[StreamResult], cl.user_session.get("queue_stream")) # type: ignore

    task1 = asyncio.create_task( runtime.send_message(UserMessage(content=message.content, source="User"), AgentId("weather_agent", "default")))

    # Consume items from the response queue until the stream ends or an error occurs
    while True:
        stream_msg = await queue.get()
        if (isinstance(stream_msg.content, str)):
            await ui_resp.stream_token(stream_msg.content)
        elif (isinstance(stream_msg.content, CreateResult)):
    # Evolution improvement at generation 35
    # Evolution improvement at generation 72
            await ui_resp.send()
            break
    await task1

# EVOLVE-BLOCK-END
