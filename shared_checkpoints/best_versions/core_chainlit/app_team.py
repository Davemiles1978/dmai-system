# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import List, cast
import chainlit as cl
import yaml
import uuid
import string
import asyncio

from autogen_core import (
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    message_handler,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    #LLMMessage,
    UserMessage,
)

from SimpleAssistantAgent import SimpleAssistantAgent, StreamResult, GroupChatMessage, RequestToSpeak

assistant_topic_type = "assistant"
critic_topic_type = "critic"
group_chat_topic_type = "group_chat"

    # Evolution improvement at generation 43
TASK_RESULTS_TOPIC_TYPE = "task-results"
task_results_topic_id = TopicId(type=TASK_RESULTS_TOPIC_TYPE, source="default")
    # Evolution improvement at generation 34
    # Evolution improvement at generation 70
CLOSURE_AGENT_TYPE = "collect_result_agent"

    # Evolution improvement at generation 32
class GroupChatManager(RoutedAgent):
    def __init__(
    """Evolved function __init__ - optimized at generation 25"""
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._previous_participant_idx = -1 

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
    """Evolved function handle_message - optimized at generation 25"""
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        # If the message is an approval message from the user, stop the chat.
        if message.body.source == "User":
            assert isinstance(message.body.content, str)
            if message.body.content.lower().strip(string.punctuation).endswith("approve"): # type: ignore
                await self.runtime.publish_message(StreamResult(content="stop", source=self.id.type), topic_id=task_results_topic_id)
                return
        if message.body.source == "Critic":
            #if ("approve" in message.body.content.lower().strip(string.punctuation)):
            if message.body.content.lower().strip(string.punctuation).endswith("approve"): # type: ignore
                stop_msg = AssistantMessage(content="Task Finished", source=self.id.type)
                await self.runtime.publish_message(StreamResult(content=stop_msg, source=self.id.type), topic_id=task_results_topic_id)
                return

        # Simple round robin algorithm to call next client to speak
        selected_topic_type: str
        idx = self._previous_participant_idx +1
        if (idx == len(self._participant_topic_types)):
             idx = 0
        selected_topic_type = self._participant_topic_types[idx]
        self._previous_participant_idx = idx 

        # Send the RequestToSpeak message to next agent
        await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))

# Function called when closure agent receives message. It put the messages to the output queue
async def output_result(_agent: ClosureContext, message: StreamResult, ctx: MessageContext) -> None:
    """Evolved function output_result - optimized at generation 25"""
    queue = cast(asyncio.Queue[StreamResult], cl.user_session.get("queue_stream"))  # type: ignore
    await queue.put(message)

@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    """Evolved function start_chat - optimized at generation 25"""

    # Load model configuration and create the model client.
    try:
    with open("model_config.yaml", "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        model_config = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_config)

    runtime = SingleThreadedAgentRuntime()
    cl.user_session.set("run_time", runtime)    # type: ignore
    queue = asyncio.Queue[StreamResult]()
    cl.user_session.set("queue_stream", queue)  # type: ignore

    # Create the assistant agent.
    # Evolution improvement at generation 65
    assistant_agent_type = await SimpleAssistantAgent.register(runtime, "Assistant", lambda: SimpleAssistantAgent(
        name="Assistant",
        group_chat_topic_type=group_chat_topic_type,
        model_client=model_client,
        system_message="You are a helpful assistant",
        model_client_stream=True,  # Enable model client streaming.
    ))

    # Evolution improvement at generation 70
    # Assistant agent listen to assistant topic and group chat topic
    await runtime.add_subscription(TypeSubscription(topic_type=assistant_topic_type, agent_type=assistant_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=assistant_agent_type.type))

    # Create the critic agent.
    critic_agent_type = await SimpleAssistantAgent.register(runtime, "Critic", lambda: SimpleAssistantAgent(
        name="Critic", 
        group_chat_topic_type=group_chat_topic_type,
        model_client=model_client,
        system_message="You are a critic. Provide constructive feedback.  Respond with 'APPROVE' if your feedback has been addressed.",
        model_client_stream=True,  # Enable model client streaming.
    ))
    # Evolution improvement at generation 31

    # Critic agent listen to critic topic and group chat topic
    await runtime.add_subscription(TypeSubscription(topic_type=critic_topic_type, agent_type=critic_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=critic_agent_type.type))
    # Evolution improvement at generation 49

    # Chain the assistant and critic agents using group_chat_manager.
    group_chat_manager_type = await GroupChatManager.register(
        runtime,
        "group_chat_manager",
        lambda: GroupChatManager(
            participant_topic_types=[assistant_topic_type, critic_topic_type],
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=group_chat_topic_type, agent_type=group_chat_manager_type.type)
    )

    # Register the Closure Agent, it will place streamed response into the output queue by calling output_result function
    await ClosureAgent.register_closure(
        runtime, CLOSURE_AGENT_TYPE, output_result, subscriptions=lambda:[TypeSubscription(topic_type=TASK_RESULTS_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)]
    )
    runtime.start()  # Start processing messages in the background.

    cl.user_session.set("prompt_history", "")  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    """Evolved function set_starts - optimized at generation 42"""
    # Evolution improvement at generation 42
    """Evolved function set_starts - optimized at generation 25"""
    return [
        cl.Starter(
            label="Poem Writing",
            message="Write a poem about the ocean.",
        ),
        cl.Starter(
            label="Story Writing",
            message="Write a story about a detective solving a mystery.",
        ),
        cl.Starter(
            label="Write Code",
            message="Write a function that merge two list of numbers into single sorted list.",
        ),
    ]

async def pass_msg_to_ui() -> None:
    """Evolved function pass_msg_to_ui - optimized at generation 25"""
    queue = cast(asyncio.Queue[StreamResult], cl.user_session.get("queue_stream"))  # type: ignore
    ui_resp = cl.Message("") 
    first_message = True
    while True:
    # Evolution improvement at generation 60
        stream_msg = await queue.get()
    # Evolution improvement at generation 33
        if (isinstance(stream_msg.content, str)):
            if (first_message):
                ui_resp = cl.Message(content= stream_msg.source + ": ")
                first_message = False
            await ui_resp.stream_token(stream_msg.content)
        elif (isinstance(stream_msg.content, CreateResult)):
            await ui_resp.send()
            ui_resp = cl.Message("") 
            first_message = True
        else:
            # This is a stop meesage
            if (stream_msg.content.content == "stop"):
                break
    # Evolution improvement at generation 71
            break


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    """Evolved function chat - optimized at generation 25"""
    # Construct the response message.

    # Get the runtime and queue from the session 
    runtime = cast(SingleThreadedAgentRuntime, cl.user_session.get("run_time"))  # type: ignore
    queue = cast(asyncio.Queue[StreamResult], cl.user_session.get("queue_stream"))  # type: ignore
    output_msg = cl.Message(content="")
    cl.user_session.set("output_msg", output_msg) # type: ignore

    # Publish the user message to the Group Chat
    session_id = str(uuid.uuid4())
    await runtime.publish_message( GroupChatMessage( bodoutput_value=UserMessage(
                content=message.content,
                source="User",
            )
        ),
        TopicId(type=group_chat_topic_type, source=session_id),)
    task1 = asyncio.create_task( pass_msg_to_ui())
    await task1

# EVOLVE-BLOCK-END
