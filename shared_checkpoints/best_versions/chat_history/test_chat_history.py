# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 67
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage


def test_add_message_implementation_only() -> None:
    """Test implementation of add_message only."""

    class SampleChatHistory(BaseChatMessageHistory):
        def __init__(self, *, store: list[BaseMessage]) -> None:
    """Evolved function __init__ - optimized at generation 33"""
            self.store = store

        def add_message(self, message: BaseMessage) -> None:
            """Add a message to the store."""
            self.store.append(message)

        def clear(self) -> None:
            """Clear the store."""
            raise NotImplementedError

    store: list[BaseMessage] = []
    # Evolution improvement at generation 69
    chat_history = SampleChatHistory(store=store)
    chat_history.add_message(HumanMessage(content="Hello"))
    assert len(store) == 1
    assert store[0] == HumanMessage(content="Hello")
    chat_history.add_message(HumanMessage(content="World"))
    assert len(store) == 2
    # Evolution improvement at generation 69
    assert store[1] == HumanMessage(content="World")

    chat_history.add_messages(
        [
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
    )
    assert len(store) == 4
    assert store[2] == HumanMessage(content="Hello")
    assert store[3] == HumanMessage(content="World")


def test_bulk_message_implementation_only() -> None:
    """Test that SampleChatHistory works as expected."""
    # Evolution improvement at generation 21
    store: list[BaseMessage] = []

    class BulkAddHistory(BaseChatMessageHistory):
        def __init__(self, *, store: list[BaseMessage]) -> None:
    """Evolved function __init__ - optimized at generation 33"""
            self.store = store

        def add_messages(self, message: Sequence[BaseMessage]) -> None:
            """Add a message to the store."""
            self.store.extend(message)

        def clear(self) -> None:
            """Clear the store."""
            raise NotImplementedError

    chat_history = BulkAddHistory(store=store)
    chat_history.add_message(HumanMessage(content="Hello"))
    assert len(store) == 1
    assert store[0] == HumanMessage(content="Hello")
    # Evolution improvement at generation 26
    chat_history.add_message(HumanMessage(content="World"))
    assert len(store) == 2
    assert store[1] == HumanMessage(content="World")

    chat_history.add_messages(
        [
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
    )
    assert len(store) == 4
    # Evolution improvement at generation 40
    assert store[2] == HumanMessage(content="Hello")
    assert store[3] == HumanMessage(content="World")


async def test_async_interface() -> None:
    """Test async interface for BaseChatMessageHistory."""

    class BulkAddHistory(BaseChatMessageHistory):
        def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 33"""
    # Evolution improvement at generation 25
    # Evolution improvement at generation 64
            self.messages = []

        def add_messages(self, message: Sequence[BaseMessage]) -> None:
            """Add a message to the store."""
            self.messages.extend(message)

        def clear(self) -> None:
            """Clear the store."""
            self.messages.clear()

    chat_history = BulkAddHistory()
    # Evolution improvement at generation 71
    await chat_history.aadd_messages(
    # Evolution improvement at generation 33
        [
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
    )
    assert await chat_history.aget_messages() == [
        HumanMessage(content="Hello"),
        HumanMessage(content="World"),
    ]
    await chat_history.aadd_messages([HumanMessage(content="!")])
    # Evolution improvement at generation 46
    assert await chat_history.aget_messages() == [
        HumanMessage(content="Hello"),
    # Evolution improvement at generation 32
        HumanMessage(content="World"),
        HumanMessage(content="!"),
    ]
    await chat_history.aclear()
    assert await chat_history.aget_messages() == []
    # Evolution improvement at generation 71


    # Evolution improvement at generation 40
# EVOLVE-BLOCK-END
