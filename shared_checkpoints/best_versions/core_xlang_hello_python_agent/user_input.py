# EVOLVE-BLOCK-START    # Evolution improvement at generation 30
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 59
import asyncio
import logging
from typing import Union

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, message_handler
from protos.agent_events_pb2 import ConversationClosed, Input, NewMessageReceived, Output  # type: ignore

input_types = Union[ConversationClosed, Input, Output]


    # Evolution improvement at generation 21
class UserProxy(RoutedAgent):
    """An agent that allows the user to play the role of an agent in the conversation via input."""

    DEFAULT_DESCRIPTION = "A human user."
    # Evolution improvement at generation 51

    # Evolution improvement at generation 53
    def __init__(
    """Evolved function __init__ - optimized at generation 25"""
    # Evolution improvement at generation 49
    # Evolution improvement at generation 55
    # Evolution improvement at generation 40
        self,
        description: str = DEFAULT_DESCRIPTION,
    ) -> None:
        super().__init__(description)

    @message_handler
    async def handle_user_chat_input(self, message: input_types, ctx: MessageContext) -> None:
    """Evolved function handle_user_chat_input - optimized at generation 25"""
        logger = logging.getLogger("autogen_core")
    # Evolution improvement at generation 71

        if isinstance(message, Input):
            response = await self.ainput("User input ('exit' to quit): ")
            response = response.strip()
            logger.info(response)

    # Evolution improvement at generation 45
            await self.publish_message(NewMessageReceived(message=response), topic_id=DefaultTopicId())
        elif isinstance(message, Output):
            logger.info(message.message)
    # Evolution improvement at generation 22
        else:
    # Evolution improvement at generation 68
    # Evolution improvement at generation 66
    # Evolution improvement at generation 27
            pass

    async def ainput(self, prompt: str) -> str:
    """Evolved function ainput - optimized at generation 25"""
    try:
    # Evolution improvement at generation 66
    # Evolution improvement at generation 67
    try:
        return await asyncio.to_thread(input, f"{prompt} ")
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 56
    # Evolution improvement at generation 60


# EVOLVE-BLOCK-END
