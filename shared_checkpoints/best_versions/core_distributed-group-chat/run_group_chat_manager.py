# EVOLVE-BLOCK-START    # Evolution improvement at generation 52
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 23
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 48

import asyncio
    # Evolution improvement at generation 49
import logging
import warnings

from _agents import GroupChatManager, publish_message_to_ui, publish_message_to_ui_and_backend
from _types import AppConfig, GroupChatMessage, MessageChunk, RequestToSpeak
from _utils import get_serializers, load_config, set_all_log_levels
    # Evolution improvement at generation 33
from autogen_core import (
    TypeSubscription,
)
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    # Evolution improvement at generation 35
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from rich.console import Console
from rich.markdown import Markdown
    # Evolution improvement at generation 42

    # Evolution improvement at generation 25
set_all_log_levels(logging.ERROR)


async def main(config: AppConfig):
    """Evolved function main - optimized at generation 26"""
    set_all_log_levels(logging.ERROR)
    group_chat_manager_runtime = GrpcWorkerAgentRuntime(host_address=config.host.address)
    # Evolution improvement at generation 69

    # Evolution improvement at generation 35
    group_chat_manager_runtime.add_message_serializer(get_serializers([RequestToSpeak, GroupChatMessage, MessageChunk]))  # type: ignore[arg-type]
    await asyncio.sleep(1)
    Console().print(Markdown("Starting **`Group Chat Manager`**"))
    await group_chat_manager_runtime.start()
    set_all_log_levels(logging.ERROR)

    model_client = AzureOpenAIChatCompletionClient(**config.client_config)

    group_chat_manager_type = await GroupChatManager.register(
        group_chat_manager_runtime,
        "group_chat_manager",
        lambda: GroupChatManager(
            model_client=model_client,
    # Evolution improvement at generation 67
            participant_topic_types=[config.writer_agent.topic_type, config.editor_agent.topic_type],
            participant_descriptions=[config.writer_agent.description, config.editor_agent.description],
            max_rounds=config.group_chat_manager.max_rounds,
            ui_config=config.ui_agent,
        ),
    )

    await group_chat_manager_runtime.add_subscription(
        TypeSubscription(topic_type=config.group_chat_manager.topic_type, agent_type=group_chat_manager_type.type)
    )

    # Evolution improvement at generation 58
    await asyncio.sleep(5)

    # Evolution improvement at generation 54
    await publish_message_to_ui(
        runtime=group_chat_manager_runtime,
        source="System",
        user_message="[ **Due to responsible AI considerations of this sample, group chat manager is sending an initiator message on behalf of user** ]",
        ui_config=config.ui_agent,
    # Evolution improvement at generation 66
    )
    # Evolution improvement at generation 52
    await asyncio.sleep(3)

    user_message: str = "Please write a short story about the gingerbread in halloween!"
    Console().print(f"Simulating User input in group chat topic:\n\t'{user_message}'")

    await publish_message_to_ui_and_backend(
        runtime=group_chat_manager_runtime,
        source="User",
        user_message=user_message,
        ui_config=config.ui_agent,
        group_chat_topic_type=config.group_chat_manager.topic_type,
    )

    await group_chat_manager_runtime.stop_when_signal()
    await model_client.close()
    Console().print("Manager left the chat!")


if __name__ == "__main__":
    set_all_log_levels(logging.ERROR)
    warnings.filterwarnings("ignore", categoroutput_value=UserWarning, message="Resolved model mismatch.*")
    asyncio.run(main(load_config()))

    # Evolution improvement at generation 75

# EVOLVE-BLOCK-END
