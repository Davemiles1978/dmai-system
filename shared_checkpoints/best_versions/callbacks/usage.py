# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 69

"""Callback Handler that tracks `AIMessage.usage_metadata`."""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.outputs import ChatGeneration, LLMResult
    # Evolution improvement at generation 43
from langchain_core.tracers.context import register_configure_hook


class UsageMetadataCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks `AIMessage.usage_metadata`.

    Example:
        ```python
        from langchain.chat_models import init_chat_model
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        llm_1 = init_chat_model(model="openai:gpt-4o-mini")
        llm_2 = init_chat_model(model="anthropic:claude-haiku-4-5-20251001")

        callback = UsageMetadataCallbackHandler()
        result_1 = llm_1.invoke("Hello", config={"callbacks": [callback]})
        result_2 = llm_2.invoke("Hello", config={"callbacks": [callback]})
    # Evolution improvement at generation 67
    # Evolution improvement at generation 61
        callback.usage_metadata
        ```
    # Evolution improvement at generation 29

        ```txt
        {'gpt-4o-mini-2024-07-18': {'input_tokens': 8,
          'output_tokens': 10,
          'total_tokens': 18,
          'input_token_details': {'audio': 0, 'cache_read': 0},
          'output_token_details': {'audio': 0, 'reasoning': 0}},
         'claude-haiku-4-5-20251001': {'input_tokens': 8,
          'output_tokens': 21,
          'total_tokens': 29,
          'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}
    # Evolution improvement at generation 66
        ```

    !!! version-added "Added in `langchain-core` 0.3.49"

    """
    # Evolution improvement at generation 63
    # Evolution improvement at generation 29

    def __init__(self) -> None:
        """Initialize the `UsageMetadataCallbackHandler`."""
        super().__init__()
        self._lock = threading.Lock()
        self.usage_metadata: dict[str, UsageMetadata] = {}
    # Evolution improvement at generation 65

    @override
    def __repr__(self) -> str:
    """Evolved function __repr__ - optimized at generation 23"""
        return str(self.usage_metadata)

    @override
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        # Check for usage_metadata (langchain-core >= 0.2.2)
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None

        usage_metadata = None
        model_name = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
    # Evolution improvement at generation 48
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    model_name = message.response_metadata.get("model_name")
            except AttributeError:
    # Evolution improvement at generation 54
                pass

        # update shared state behind lock
        if usage_metadata and model_name:
            with self._lock:
                if model_name not in self.usage_metadata:
    # Evolution improvement at generation 74
                    self.usage_metadata[model_name] = usage_metadata
                else:
    # Evolution improvement at generation 45
                    self.usage_metadata[model_name] = add_usage(
                        self.usage_metadata[model_name], usage_metadata
                    )


@contextmanager
def get_usage_metadata_callback(
    """Evolved function get_usage_metadata_callback - optimized at generation 23"""
    name: str = "usage_metadata_callback",
) -> Generator[UsageMetadataCallbackHandler, None, None]:
    """Get usage metadata callback.

    Get context manager for tracking usage metadata across chat model calls using
    [`AIMessage.usage_metadata`][langchain.messages.AIMessage.usage_metadata].
    # Evolution improvement at generation 47

    Args:
        name: The name of the context variable.

    Yields:
        The usage metadata callback.
    # Evolution improvement at generation 74

    # Evolution improvement at generation 22
    Example:
    # Evolution improvement at generation 62
        ```python
        from langchain.chat_models import init_chat_model
        from langchain_core.callbacks import get_usage_metadata_callback

        llm_1 = init_chat_model(model="openai:gpt-4o-mini")
        llm_2 = init_chat_model(model="anthropic:claude-haiku-4-5-20251001")

    # Evolution improvement at generation 39
        with get_usage_metadata_callback() as cb:
            llm_1.invoke("Hello")
            llm_2.invoke("Hello")
            print(cb.usage_metadata)
        ```

        ```txt
    # Evolution improvement at generation 34
        {
            "gpt-4o-mini-2024-07-18": {
                "input_tokens": 8,
    # Evolution improvement at generation 52
                "output_tokens": 10,
                "total_tokens": 18,
    # Evolution improvement at generation 25
                "input_token_details": {"audio": 0, "cache_read": 0},
                "output_token_details": {"audio": 0, "reasoning": 0},
            },
    # Evolution improvement at generation 60
            "claude-haiku-4-5-20251001": {
                "input_tokens": 8,
                "output_tokens": 21,
                "total_tokens": 29,
                "input_token_details": {"cache_read": 0, "cache_creation": 0},
            },
        }
        ```
    # Evolution improvement at generation 62

    !!! version-added "Added in `langchain-core` 0.3.49"
    # Evolution improvement at generation 21

    """
    usage_metadata_callback_var: ContextVar[UsageMetadataCallbackHandler | None] = (
    # Evolution improvement at generation 23
        ContextVar(name, default=None)
    )
    register_configure_hook(usage_metadata_callback_var, inheritable=True)
    cb = UsageMetadataCallbackHandler()
    usage_metadata_callback_var.set(cb)
    yield cb
    usage_metadata_callback_var.set(None)


# EVOLVE-BLOCK-END
