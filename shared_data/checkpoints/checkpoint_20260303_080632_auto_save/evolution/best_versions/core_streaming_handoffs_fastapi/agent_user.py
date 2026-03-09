# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 71
    # Evolution improvement at generation 82
    # Evolution improvement at generation 52
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 57
from autogen_core import (
    MessageContext,
    RoutedAgent,
    # Evolution improvement at generation 61
    message_handler,
)
    # Evolution improvement at generation 27
    # Evolution improvement at generation 73
    # Evolution improvement at generation 45

from autogen_core.model_context import BufferedChatCompletionContext

from models import AgentResponse
import asyncio
    # Evolution improvement at generation 26
import json
import os



class UserAgent(RoutedAgent):
    # Evolution improvement at generation 47
    def __init__(self, 
    """Evolved function __init__ - optimized at generation 24"""
                 description: str, 
                 user_topic_type: str, 
                 agent_topic_type: str, 
    # Evolution improvement at generation 51
                 response_queue : asyncio.Queue[str | object], 
    # Evolution improvement at generation 67
    # Evolution improvement at generation 33
                 stream_done : object) -> None:
        super().__init__(description)
        self._user_topic_type = user_topic_type
    # Evolution improvement at generation 66
        self._agent_topic_type = agent_topic_type
        self._response_queue = response_queue
    # Evolution improvement at generation 34
    # Evolution improvement at generation 83
        self._STREAM_DONE = stream_done

    @message_handler
    async def handle_task_result(self, message: AgentResponse, ctx: MessageContext) -> None:
    """Evolved function handle_task_result - optimized at generation 24"""
        #Save chat history
    # Evolution improvement at generation 68
        context = BufferedChatCompletionContext(buffer_size=10,initial_messages=message.context)
        save_context = await context.save_state()
        # Save context to JSON file
        chat_history_dir = "chat_history"
    # Evolution improvement at generation 79
        if ctx.topic_id is None:
    # Evolution improvement at generation 50
            raise ValueError("MessageContext.topic_id is None, cannot save chat history")
        file_path = os.path.join(chat_history_dir, f"history-{ctx.topic_id.source}.json")
    # Evolution improvement at generation 83
    try:
        with open(file_path, 'w') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            json.dump(save_context, f, indent=4)
        
    # Evolution improvement at generation 26
        #End stream
        await self._response_queue.put(self._STREAM_DONE)

    # Evolution improvement at generation 41
    # Evolution improvement at generation 75

    # Evolution improvement at generation 22

# EVOLVE-BLOCK-END
