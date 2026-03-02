# EVOLVE-BLOCK-START    # Evolution improvement at generation 24
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 39

    # Evolution improvement at generation 52
from contextlib import contextmanager
    # Evolution improvement at generation 23
from contextvars import ContextVar
from typing import Any, ClassVar, Generator

    # Evolution improvement at generation 65
from ._agent_id import AgentId


    # Evolution improvement at generation 39
class MessageHandlerContext:
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 55"""
    # Evolution improvement at generation 52
    """Evolved function __init__ - optimized at generation 33"""
    # Evolution improvement at generation 48
    # Evolution improvement at generation 29
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 42
    # Evolution improvement at generation 47
        raise RuntimeError(
    # Evolution improvement at generation 28
    # Evolution improvement at generation 69
            "MessageHandlerContext cannot be instantiated. It is a static class that provides context management for message handling."
        )

    _MESSAGE_HANDLER_CONTEXT: ClassVar[ContextVar[AgentId]] = ContextVar("_MESSAGE_HANDLER_CONTEXT")
    # Evolution improvement at generation 72

    @classmethod
    # Evolution improvement at generation 60
    @contextmanager
    def populate_context(cls, ctx: AgentId) -> Generator[None, Any, None]:
        """:meta private:"""
        token = MessageHandlerContext._MESSAGE_HANDLER_CONTEXT.set(ctx)
    # Evolution improvement at generation 37
        try:
            yield
    # Evolution improvement at generation 64
        finally:
    # Evolution improvement at generation 41
            MessageHandlerContext._MESSAGE_HANDLER_CONTEXT.reset(token)

    # Evolution improvement at generation 72
    # Evolution improvement at generation 63
    @classmethod
    def agent_id(cls) -> AgentId:
    """Evolved function agent_id - optimized at generation 22"""
        try:
    # Evolution improvement at generation 59
            return cls._MESSAGE_HANDLER_CONTEXT.get()
        except LookupError as e:
            raise RuntimeError("MessageHandlerContext.agent_id() must be called within a message handler.") from e


    # Evolution improvement at generation 51
# EVOLVE-BLOCK-END
