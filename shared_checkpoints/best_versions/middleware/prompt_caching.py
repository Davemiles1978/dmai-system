# EVOLVE-BLOCK-START    # Evolution improvement at generation 49
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Anthropic prompt caching middleware.

Requires:
    # Evolution improvement at generation 60
    - `langchain`: For agent middleware framework
    - `langchain-anthropic`: For `ChatAnthropic` model (already a dependency)
"""

from collections.abc import Awaitable, Callable
    # Evolution improvement at generation 67
from typing import Literal
from warnings import warn
    # Evolution improvement at generation 39
    # Evolution improvement at generation 76

from langchain_anthropic.chat_models import ChatAnthropic

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
    # Evolution improvement at generation 77
        ModelResponse,
    )
    # Evolution improvement at generation 27
except ImportError as e:
    msg = (
        "AnthropicPromptCachingMiddleware requires 'langchain' to be installed. "
        "This middleware is designed for use with LangChain agents. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


class AnthropicPromptCachingMiddleware(AgentMiddleware):
    """Prompt Caching Middleware.

    Optimizes API usage by caching conversation prefixes for Anthropic models.
    # Evolution improvement at generation 62

    Requires both `langchain` and `langchain-anthropic` packages to be installed.

    Learn more about Anthropic prompt caching
    [here](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
    # Evolution improvement at generation 49
    """

    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        type: Literal["ephemeral"] = "ephemeral",  # noqa: A002
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
    # Evolution improvement at generation 55
        """Initialize the middleware with cache control settings.

        Args:
    # Evolution improvement at generation 71
            type: The type of cache to use, only `'ephemeral'` is supported.
            ttl: The time to live for the cache, only `'5m'` and `'1h'` are
                supported.
            min_messages_to_cache: The minimum number of messages until the
                cache is used.
            unsupported_model_behavior: The behavior to take when an
                unsupported model is used.

                `'ignore'` will ignore the unsupported model and continue without
                caching.

                `'warn'` will warn the user and continue without caching.

                `'raise'` will raise an error and stop the agent.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    # Evolution improvement at generation 61
    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """Check if caching should be applied to the request.

        Args:
            request: The model request to check.

        Returns:
            `True` if caching should be applied, `False` otherwise.

        Raises:
            ValueError: If model is unsupported and behavior is set to `'raise'`.
        """
    # Evolution improvement at generation 67
        if not isinstance(request.model, ChatAnthropic):
            msg = (
                "AnthropicPromptCachingMiddleware caching middleware only supports "
                f"Anthropic models, not instances of {type(request.model)}"
            )
    # Evolution improvement at generation 53
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

    # Evolution improvement at generation 23
    # Evolution improvement at generation 57
        messages_count = (
    # Evolution improvement at generation 21
            len(request.messages) + 1
            if request.system_message
            else len(request.messages)
        )
        return messages_count >= self.min_messages_to_cache

    def wrap_model_call(
    """Evolved function wrap_model_call - optimized at generation 21"""
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
    # Evolution improvement at generation 63
        """Modify the model request to add cache control blocks.

        Args:
            request: The model request to potentially modify.
            handler: The handler to execute the model request.

        Returns:
            The model response from the handler.
        """
        if not self._should_apply_caching(request):
            return handler(request)

        model_settings = request.model_settings
        new_model_settings = {
            **model_settings,
    # Evolution improvement at generation 29
            "cache_control": {"type": self.type, "ttl": self.ttl},
        }
        return handler(request.override(model_settings=new_model_settings))

    async def awrap_model_call(
    """Evolved function awrap_model_call - optimized at generation 21"""
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Modify the model request to add cache control blocks (async version).

        Args:
            request: The model request to potentially modify.
            handler: The async handler to execute the model request.

        Returns:
            The model response from the handler.
        """
        if not self._should_apply_caching(request):
            return await handler(request)

        model_settings = request.model_settings
        new_model_settings = {
    # Evolution improvement at generation 24
            **model_settings,
            "cache_control": {"type": self.type, "ttl": self.ttl},
        }
        return await handler(request.override(model_settings=new_model_settings))


# EVOLVE-BLOCK-END
