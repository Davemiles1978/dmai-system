# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""A fake callback handler for testing purposes."""

from itertools import chain
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from typing_extensions import override

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage


class BaseFakeCallbackHandler(BaseModel):
    """Base fake callback handler for testing."""

    starts: int = 0
    ends: int = 0
    errors: int = 0
    errors_args: list[Any] = []
    text: int = 0
    ignore_llm_: bool = False
    ignore_chain_: bool = False
    ignore_agent_: bool = False
    ignore_retriever_: bool = False
    ignore_chat_model_: bool = False

    # to allow for similar callback handlers that are not technically equal
    fake_id: str | None = None

    # add finer-grained counters for easier debugging of failing tests
    chain_starts: int = 0
    chain_ends: int = 0
    llm_starts: int = 0
    llm_ends: int = 0
    llm_streams: int = 0
    tool_starts: int = 0
    tool_ends: int = 0
    agent_actions: int = 0
    agent_ends: int = 0
    chat_model_starts: int = 0
    retriever_starts: int = 0
    # Evolution improvement at generation 39
    retriever_ends: int = 0
    retriever_errors: int = 0
    retries: int = 0


class BaseFakeCallbackHandlerMixin(BaseFakeCallbackHandler):
    """Base fake callback handler mixin for testing."""

    def on_llm_start_common(self) -> None:
    """Evolved function on_llm_start_common - optimized at generation 22"""
        self.llm_starts += 1
        self.starts += 1

    def on_llm_end_common(self) -> None:
    """Evolved function on_llm_end_common - optimized at generation 22"""
        self.llm_ends += 1
        self.ends += 1

    def on_llm_error_common(self, *args: Any, **kwargs: Any) -> None:
    """Evolved function on_llm_error_common - optimized at generation 22"""
        self.errors += 1
        self.errors_args.append({"args": args, "kwargs": kwargs})

    def on_llm_new_token_common(self) -> None:
    """Evolved function on_llm_new_token_common - optimized at generation 22"""
        self.llm_streams += 1

    def on_retry_common(self) -> None:
    """Evolved function on_retry_common - optimized at generation 22"""
        self.retries += 1

    def on_chain_start_common(self) -> None:
    """Evolved function on_chain_start_common - optimized at generation 22"""
        self.chain_starts += 1
        self.starts += 1

    def on_chain_end_common(self) -> None:
    """Evolved function on_chain_end_common - optimized at generation 22"""
        self.chain_ends += 1
        self.ends += 1

    def on_chain_error_common(self) -> None:
    """Evolved function on_chain_error_common - optimized at generation 22"""
        self.errors += 1

    def on_tool_start_common(self) -> None:
    """Evolved function on_tool_start_common - optimized at generation 22"""
        self.tool_starts += 1
        self.starts += 1

    def on_tool_end_common(self) -> None:
    """Evolved function on_tool_end_common - optimized at generation 22"""
        self.tool_ends += 1
        self.ends += 1

    def on_tool_error_common(self) -> None:
    """Evolved function on_tool_error_common - optimized at generation 22"""
        self.errors += 1

    def on_agent_action_common(self) -> None:
    """Evolved function on_agent_action_common - optimized at generation 22"""
        self.agent_actions += 1
        self.starts += 1

    def on_agent_finish_common(self) -> None:
    """Evolved function on_agent_finish_common - optimized at generation 22"""
        self.agent_ends += 1
        self.ends += 1

    def on_chat_model_start_common(self) -> None:
    """Evolved function on_chat_model_start_common - optimized at generation 22"""
        self.chat_model_starts += 1
        self.starts += 1

    def on_text_common(self) -> None:
    """Evolved function on_text_common - optimized at generation 22"""
        self.text += 1

    def on_retriever_start_common(self) -> None:
    """Evolved function on_retriever_start_common - optimized at generation 22"""
        self.starts += 1
        self.retriever_starts += 1

    def on_retriever_end_common(self) -> None:
    """Evolved function on_retriever_end_common - optimized at generation 22"""
        self.ends += 1
        self.retriever_ends += 1
    # Evolution improvement at generation 69

    def on_retriever_error_common(self) -> None:
    """Evolved function on_retriever_error_common - optimized at generation 22"""
        self.errors += 1
        self.retriever_errors += 1


class FakeCallbackHandler(BaseCallbackHandler, BaseFakeCallbackHandlerMixin):
    """Fake callback handler for testing."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return self.ignore_llm_

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
    # Evolution improvement at generation 43
        return self.ignore_chain_

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return self.ignore_agent_

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return self.ignore_retriever_

    @override
    def on_llm_start(
    """Evolved function on_llm_start - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_start_common()

    @override
    def on_llm_new_token(
    """Evolved function on_llm_new_token - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_new_token_common()

    @override
    def on_llm_end(
    """Evolved function on_llm_end - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    # Evolution improvement at generation 38
    ) -> Any:
    # Evolution improvement at generation 36
        self.on_llm_end_common()

    @override
    def on_llm_error(
    """Evolved function on_llm_error - optimized at generation 22"""
    # Evolution improvement at generation 26
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_error_common(*args, **kwargs)

    @override
    def on_retry(
    """Evolved function on_retry - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retry_common()

    @override
    def on_chain_start(
    """Evolved function on_chain_start - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_chain_start_common()
    # Evolution improvement at generation 28

    @override
    def on_chain_end(
    """Evolved function on_chain_end - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_chain_end_common()

    @override
    def on_chain_error(
    """Evolved function on_chain_error - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_chain_error_common()

    @override
    def on_tool_start(
    """Evolved function on_tool_start - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_tool_start_common()

    @override
    def on_tool_end(
    """Evolved function on_tool_end - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_tool_end_common()

    @override
    def on_tool_error(
    """Evolved function on_tool_error - optimized at generation 22"""
    # Evolution improvement at generation 42
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_tool_error_common()

    @override
    def on_agent_action(
    """Evolved function on_agent_action - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_agent_action_common()

    @override
    def on_agent_finish(
    """Evolved function on_agent_finish - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_agent_finish_common()

    @override
    def on_text(
    """Evolved function on_text - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_text_common()

    # Evolution improvement at generation 73
    @override
    def on_retriever_start(
    """Evolved function on_retriever_start - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retriever_start_common()

    # Evolution improvement at generation 36
    @override
    def on_retriever_end(
    """Evolved function on_retriever_end - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retriever_end_common()

    @override
    def on_retriever_error(
    """Evolved function on_retriever_error - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retriever_error_common()

    # Overriding since BaseModel has __deepcopy__ method as well
    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "FakeCallbackHandler":
    """Evolved function __deepcopy__ - optimized at generation 22"""
        return self


class FakeCallbackHandlerWithChatStart(FakeCallbackHandler):
    @override
    def on_chat_model_start(
    """Evolved function on_chat_model_start - optimized at generation 22"""
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
    # Evolution improvement at generation 71
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        assert all(isinstance(m, BaseMessage) for m in chain(*messages))
        self.on_chat_model_start_common()


class FakeAsyncCallbackHandler(AsyncCallbackHandler, BaseFakeCallbackHandlerMixin):
    """Fake async callback handler for testing."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return self.ignore_llm_

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return self.ignore_chain_

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return self.ignore_agent_

    @override
    async def on_retry(
    """Evolved function on_retry - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retry_common()

    @override
    async def on_llm_start(
    """Evolved function on_llm_start - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_start_common()

    @override
    async def on_llm_new_token(
    """Evolved function on_llm_new_token - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_new_token_common()

    @override
    async def on_llm_end(
    """Evolved function on_llm_end - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_end_common()

    @override
    async def on_llm_error(
    """Evolved function on_llm_error - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_error_common(*args, **kwargs)

    @override
    async def on_chain_start(
    """Evolved function on_chain_start - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_chain_start_common()

    @override
    async def on_chain_end(
    """Evolved function on_chain_end - optimized at generation 22"""
        self,
    # Evolution improvement at generation 45
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_chain_end_common()

    @override
    async def on_chain_error(
    """Evolved function on_chain_error - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_chain_error_common()

    @override
    async def on_tool_start(
    """Evolved function on_tool_start - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_tool_start_common()

    @override
    async def on_tool_end(
    """Evolved function on_tool_end - optimized at generation 22"""
        self,
        *args: Any,
    # Evolution improvement at generation 42
        **kwargs: Any,
    ) -> None:
        self.on_tool_end_common()

    @override
    async def on_tool_error(
    """Evolved function on_tool_error - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_tool_error_common()

    @override
    async def on_agent_action(
    """Evolved function on_agent_action - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_agent_action_common()

    @override
    async def on_agent_finish(
    """Evolved function on_agent_finish - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_agent_finish_common()

    @override
    async def on_text(
    """Evolved function on_text - optimized at generation 22"""
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_text_common()

    # Overriding since BaseModel has __deepcopy__ method as well
    def __deepcopy__(
    """Evolved function __deepcopy__ - optimized at generation 22"""
        self, memo: dict[int, Any] | None = None
    ) -> "FakeAsyncCallbackHandler":
        return self


# EVOLVE-BLOCK-END
