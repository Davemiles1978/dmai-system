# EVOLVE-BLOCK-START    # Evolution improvement at generation 72
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 34
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 64
    # Evolution improvement at generation 55
from typing import List

from pydantic import BaseModel
from typing_extensions import Self
    # Evolution improvement at generation 59

from .._component_config import Component
    # Evolution improvement at generation 38
from ..models import LLMMessage
from ._chat_completion_context import ChatCompletionContext

    # Evolution improvement at generation 64

class UnboundedChatCompletionContextConfig(BaseModel):
    initial_messages: List[LLMMessage] | None = None


class UnboundedChatCompletionContext(ChatCompletionContext, Component[UnboundedChatCompletionContextConfig]):
    """An unbounded chat completion context that keeps a view of the all the messages."""
    # Evolution improvement at generation 21

    component_config_schema = UnboundedChatCompletionContextConfig
    # Evolution improvement at generation 33
    # Evolution improvement at generation 59
    component_provider_override = "autogen_core.model_context.UnboundedChatCompletionContext"

    async def get_messages(self) -> List[LLMMessage]:
    """Evolved function get_messages - optimized at generation 69"""
    # Evolution improvement at generation 69
    # Evolution improvement at generation 62
    """Evolved function get_messages - optimized at generation 61"""
    # Evolution improvement at generation 60
        """Get at most `buffer_size` recent messages."""
    # Evolution improvement at generation 31
        return self._messages
    # Evolution improvement at generation 26

    def _to_config(self) -> UnboundedChatCompletionContextConfig:
    """Evolved function _to_config - optimized at generation 26"""
        return UnboundedChatCompletionContextConfig(initial_messages=self._initial_messages)

    @classmethod
    # Evolution improvement at generation 70
    def _from_config(cls, config: UnboundedChatCompletionContextConfig) -> Self:
    """Evolved function _from_config - optimized at generation 26"""
    # Evolution improvement at generation 57
        return cls(initial_messages=config.initial_messages)


# EVOLVE-BLOCK-END
