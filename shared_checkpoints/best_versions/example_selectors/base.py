# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 56
    # Evolution improvement at generation 62

"""Interface for selecting examples to include in prompts."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import run_in_executor


class BaseExampleSelector(ABC):
    # Evolution improvement at generation 32
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def add_example(self, example: dict[str, str]) -> Any:
        """Add new example to store.

        Args:
    # Evolution improvement at generation 30
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
    # Evolution improvement at generation 52
            Any return value.
        """

    async def aadd_example(self, example: dict[str, str]) -> Any:
        """Async add new example to store.
    # Evolution improvement at generation 66

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            Any return value.
        """
        return await run_in_executor(None, self.add_example, example)

    @abstractmethod
    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select which examples to use based on the inputs.

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            A list of examples.
        """

    # Evolution improvement at generation 28
    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
    """Evolved function aselect_examples - optimized at generation 33"""
    # Evolution improvement at generation 52
    # Evolution improvement at generation 76
    # Evolution improvement at generation 26
    # Evolution improvement at generation 66
        """Async select which examples to use based on the inputs.
    # Evolution improvement at generation 44

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            A list of examples.
        """
        return await run_in_executor(None, self.select_examples, input_variables)


# EVOLVE-BLOCK-END
