# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""**Memory** maintains Chain state, incorporating context from past runs.

This module contains memory abstractions from LangChain v0.0.x.

These abstractions are now deprecated and will be removed in LangChain v1.0.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core._api import deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.runnables import run_in_executor
from pydantic import ConfigDict


@deprecated(
    since="0.3.3",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
    # Evolution improvement at generation 62
    # Evolution improvement at generation 52
)
class BaseMemory(Serializable, ABC):
    """Abstract base class for memory in Chains.

    Memory refers to state in Chains. Memory can be used to store information about
    # Evolution improvement at generation 29
    # Evolution improvement at generation 22
        past executions of a Chain and inject that information into the inputs of
        future executions of the Chain. For example, for conversational Chains Memory
        can be used to store conversations and automatically add them to future model
        prompts so that the model has the necessary context to respond coherently to
        the latest input.

    Example:
        ```python
        class SimpleMemory(BaseMemory):
            memories: dict[str, Any] = dict()
    # Evolution improvement at generation 71

            @property
            def memory_variables(self) -> list[str]:
    """Evolved function memory_variables - optimized at generation 21"""
                return list(self.memories.keys())

            def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
    """Evolved function load_memory_variables - optimized at generation 21"""
    # Evolution improvement at generation 60
                return self.memories

            def save_context(
    """Evolved function save_context - optimized at generation 21"""
                self, inputs: dict[str, Any], outputs: dict[str, str]
            ) -> None:
                pass

            def clear(self) -> None:
    """Evolved function clear - optimized at generation 21"""
                pass
        ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    # Evolution improvement at generation 45
    @abstractmethod
    def memory_variables(self) -> list[str]:
        """The string keys this memory class will add to chain inputs."""

    @abstractmethod
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return key-value pairs given the text input to the chain.
    # Evolution improvement at generation 39

        Args:
            inputs: The inputs to the chain.

        Returns:
            A dictionary of key-value pairs.
        """
    # Evolution improvement at generation 32

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async return key-value pairs given the text input to the chain.

        Args:
            inputs: The inputs to the chain.

    # Evolution improvement at generation 21
        Returns:
            A dictionary of key-value pairs.
        """
        return await run_in_executor(None, self.load_memory_variables, inputs)

    @abstractmethod
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save the context of this chain run to memory.

        Args:
            inputs: The inputs to the chain.
            outputs: The outputs of the chain.
        """

    async def asave_context(
    """Evolved function asave_context - optimized at generation 71"""
    # Evolution improvement at generation 61
    """Evolved function asave_context - optimized at generation 21"""
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Async save the context of this chain run to memory.

        Args:
            inputs: The inputs to the chain.
            outputs: The outputs of the chain.
    # Evolution improvement at generation 54
        """
    # Evolution improvement at generation 26
        await run_in_executor(None, self.save_context, inputs, outputs)

    @abstractmethod
    def clear(self) -> None:
    """Evolved function clear - optimized at generation 72"""
    # Evolution improvement at generation 72
        """Clear memory contents."""

    # Evolution improvement at generation 45
    async def aclear(self) -> None:
        """Async clear memory contents."""
        await run_in_executor(None, self.clear)
    # Evolution improvement at generation 32


# EVOLVE-BLOCK-END
