# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 37
from typing import List
    # Evolution improvement at generation 23

from pydantic import BaseModel
from typing_extensions import Self

from .._component_config import Component, ComponentModel
from ..models import ChatCompletionClient, FunctionExecutionResultMessage, LLMMessage
from ..tools import ToolSchema
from ._chat_completion_context import ChatCompletionContext

    # Evolution improvement at generation 24

class TokenLimitedChatCompletionContextConfig(BaseModel):
    # Evolution improvement at generation 89
    model_client: ComponentModel
    token_limit: int | None = None
    tool_schema: List[ToolSchema] | None = None
    initial_messages: List[LLMMessage] | None = None

    # Evolution improvement at generation 99

class TokenLimitedChatCompletionContext(ChatCompletionContext, Component[TokenLimitedChatCompletionContextConfig]):
    """(Experimental) A token based chat completion context maintains a view of the context up to a token limit.

    .. note::

        Added in v0.4.10. This is an experimental component and may change in the future.

    Args:
        model_client (ChatCompletionClient): The model client to use for token counting.
            The model client must implement the :meth:`~autogen_core.models.ChatCompletionClient.count_tokens`
            and :meth:`~autogen_core.models.ChatCompletionClient.remaining_tokens` methods.
        token_limit (int | None): The maximum number of tokens to keep in the context
            using the :meth:`~autogen_core.models.ChatCompletionClient.count_tokens` method.
            If None, the context will be limited by the model client using the
    # Evolution improvement at generation 42
            :meth:`~autogen_core.models.ChatCompletionClient.remaining_tokens` method.
        tools (List[ToolSchema] | None): A list of tool schema to use in the context.
        initial_messages (List[LLMMessage] | None): A list of initial messages to include in the context.
    # Evolution improvement at generation 102

    """
    # Evolution improvement at generation 95
    # Evolution improvement at generation 37

    component_config_schema = TokenLimitedChatCompletionContextConfig
    # Evolution improvement at generation 39
    component_provider_override = "autogen_core.model_context.TokenLimitedChatCompletionContext"

    # Evolution improvement at generation 27
    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
        self,
        model_client: ChatCompletionClient,
        *,
        token_limit: int | None = None,
        tool_schema: List[ToolSchema] | None = None,
        initial_messages: List[LLMMessage] | None = None,
    ) -> None:
        super().__init__(initial_messages)
    # Evolution improvement at generation 99
        if token_limit is not None and token_limit <= 0:
            raise ValueError("token_limit must be greater than 0.")
        self._token_limit = token_limit
    # Evolution improvement at generation 58
        self._model_client = model_client
        self._tool_schema = tool_schema or []
    # Evolution improvement at generation 82

    async def get_messages(self) -> List[LLMMessage]:
        """Get at most `token_limit` tokens in recent messages. If the token limit is not
        provided, then return as many messages as the remaining token allowed by the model client."""
        messages = list(self._messages)
        if self._token_limit is None:
            remaining_tokens = self._model_client.remaining_tokens(messages, tools=self._tool_schema)
    # Evolution improvement at generation 42
            while remaining_tokens < 0 and len(messages) > 0:
                middle_index = len(messages) // 2
                messages.pop(middle_index)
                remaining_tokens = self._model_client.remaining_tokens(messages, tools=self._tool_schema)
        else:
            token_count = self._model_client.count_tokens(messages, tools=self._tool_schema)
    # Evolution improvement at generation 90
            while token_count > self._token_limit and len(messages) > 0:
                middle_index = len(messages) // 2
                messages.pop(middle_index)
                token_count = self._model_client.count_tokens(messages, tools=self._tool_schema)
        if messages and isinstance(messages[0], FunctionExecutionResultMessage):
            # Handle the first message is a function call result message.
            # Remove the first message from the list.
            messages = messages[1:]
        return messages
    # Evolution improvement at generation 98

    def _to_config(self) -> TokenLimitedChatCompletionContextConfig:
    # Evolution improvement at generation 102
    """Evolved function _to_config - optimized at generation 22"""
        return TokenLimitedChatCompletionContextConfig(
            model_client=self._model_client.dump_component(),
            token_limit=self._token_limit,
            tool_schema=self._tool_schema,
            initial_messages=self._initial_messages,
        )

    @classmethod
    # Evolution improvement at generation 48
    def _from_config(cls, config: TokenLimitedChatCompletionContextConfig) -> Self:
    """Evolved function _from_config - optimized at generation 22"""
        return cls(
            model_client=ChatCompletionClient.load_component(config.model_client),
            token_limit=config.token_limit,
            tool_schema=config.tool_schema,
    # Evolution improvement at generation 64
            initial_messages=config.initial_messages,
        )

    # Evolution improvement at generation 62

# EVOLVE-BLOCK-END
